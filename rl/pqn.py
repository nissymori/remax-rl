"""
This code is copied and modified from https://github.com/mttga/purejaxql
"""

import copy
import os
import time
from functools import partial
from typing import Any, Dict, Optional

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pgx
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pgx.experimental.wrappers import auto_reset
from pydantic import BaseModel, Field


# ----------------------------
# Pydantic Config (OmegaConf CLI -> Pydantic -> dict), all-lowercase keys
# ----------------------------
class AppConfig(BaseModel):
    # algorithm
    algo: str = "pqn"  # or "pqn_softmax"

    # env
    env_name: str = "minatar-breakout"
    env_kwargs: Dict[str, Any] = Field(default_factory=dict)
    sticky_action_prob: float = 0.1

    # training sizes
    total_timesteps: float = 1e7
    total_timesteps_decay: float = 1e7  # decay基準に使う（短縮実験でも同じ比率保つ用）
    num_envs: int = 128
    num_steps: int = 32

    # updates / optimization
    num_minibatches: int = 32
    num_epochs: int = 2
    lr: float = 5e-4
    max_grad_norm: float = 10.0
    lr_linear_decay: bool = True
    norm_type: str = "layer_norm"
    norm_input: bool = False

    # RL hyperparams
    gamma: float = 0.99
    lambda_: float = 0.65 # allow CLI "lambda=..."
    rew_scale: float = 1.0

    # epsilon-greedy schedule
    eps_start: float = 1.0
    eps_finish: float = 0.05
    eps_decay: float = 0.1  # ratio of total updates

    # evaluation
    do_eval: bool = True
    num_eval_envs: int = 100

    # logging / misc
    wandb_project: str = "pgx-minatar-pqn"
    entity: str = "default"
    wandb_mode: str = "online"
    save_model: bool = False

    seed: int = 0

    # keep a nested dict like original (merged later): config = {**config, **config["alg"]}
    alg: Dict[str, Any] = Field(default_factory=lambda: {})

    class Config:
        extra = "allow"


conf_dict = OmegaConf.to_object(OmegaConf.from_cli())
config = AppConfig(**conf_dict)


# derive update counts
config.num_updates = int(config.total_timesteps // config.num_steps // config.num_envs)
config.num_updates_decay = int(config.total_timesteps_decay // config.num_steps // config.num_envs)

assert (config.num_steps * config.num_envs) % config.num_minibatches == 0, "num_minibatches must divide num_steps*num_envs"

env = pgx.make(config.env_name)

autoreset_step = auto_reset(env.step, env.init)
vmap_step = lambda n_envs: lambda env_state, action, rng: jax.vmap(
    autoreset_step, in_axes=(0, 0, 0)
)(env_state, action, jax.random.split(rng, n_envs))

# eps_decay is a ratio of total updates
eps_decay_updates = max(1, int(config.eps_decay * config.num_updates_decay))
eps_scheduler = optax.linear_schedule(
    config.eps_start,
    config.eps_finish,
    eps_decay_updates,
)

# ----------------------------
# Original code (logic unchanged; keys switched to lowercase)
# ----------------------------
class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            16,
            kernel_size=(3, 3),
            strides=1,
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            _ = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array
    epsilon: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


# epsilon-greedy exploration
def eps_greedy_exploration(rng, q_vals, eps):
    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    uniform_actions = jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1])
    random_actions = uniform_actions
    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        random_actions,
        greedy_actions,
    )
    return chosen_actions

def make_update_fn():
    # TRAINING LOOP
    def _update_step(runner_state):
        train_state, env_state, rng = runner_state

        # SAMPLE PHASE
        def _step_env(carry, _):
            env_state, rng = carry
            obs = env_state.observation
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = train_state.apply_fn(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                obs,
                train=False,
            )
            # different eps for each env
            _rngs = jax.random.split(rng_a, config.num_envs)
            eps = jnp.full(config.num_envs, eps_scheduler(train_state.n_updates))
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

            next_state = vmap_step(config.num_envs)(env_state, new_action, rng_s)

            transition = Transition(
                obs=obs,
                action=new_action,
                reward=config.rew_scale * next_state.rewards[:, 0],
                done=next_state.terminated,
                next_obs=next_state.observation,
                q_val=q_vals,
                epsilon=eps,
            )
            return (next_state, rng), (transition)

        # step the env
        rng, _rng = jax.random.split(rng)
        (env_state, rng), (transitions) = jax.lax.scan(
            _step_env, (env_state, _rng), None, config.num_steps
        )

        train_state = train_state.replace(
            timesteps=train_state.timesteps + config.num_steps * config.num_envs
        )

        last_q = train_state.apply_fn(
            {"params": train_state.params, "batch_stats": train_state.batch_stats},
            transitions.next_obs[-1],
            train=False,
        )
        last_q = jnp.max(last_q, axis=-1)

        def _get_target(lambda_returns_and_next_q, transition):
            lambda_returns, next_q = lambda_returns_and_next_q
            target_bootstrap = transition.reward + config.gamma * (1 - transition.done) * next_q
            delta = lambda_returns - next_q
            lambda_returns = target_bootstrap + config.gamma * config.lambda_ * delta
            lambda_returns = (1 - transition.done) * lambda_returns + transition.done * transition.reward
            next_q = jnp.max(transition.q_val, axis=-1)
            return (lambda_returns, next_q), lambda_returns

        last_q = last_q * (1 - transitions.done[-1])
        lambda_returns = transitions.reward[-1] + config.gamma * last_q
        _, targets = jax.lax.scan(
            _get_target,
            (lambda_returns, last_q),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

        # metrics container (kept minimal)
        metrics = {}

        # NETWORKS UPDATE
        def _learn_epoch(carry, _):
            train_state, rng = carry

            def _learn_phase(carry, minibatch_and_target):
                train_state, rng = carry
                minibatch, target = minibatch_and_target

                def _loss_fn(params):
                    q_vals, updates = train_state.apply_fn(
                        {"params": params, "batch_stats": train_state.batch_stats},
                        minibatch.obs,
                        train=True,
                        mutable=["batch_stats"],
                    )
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals, jnp.expand_dims(minibatch.action, axis=-1), axis=-1
                    ).squeeze(axis=-1)
                    loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                    return loss, (updates, chosen_action_qvals)

                (loss, (updates, qvals)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                    batch_stats=updates["batch_stats"],
                )
                return (train_state, rng), (loss, qvals)

            def preprocess_transition(x, rng):
                x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs, ...
                x = jax.random.permutation(rng, x)  # shuffle
                x = x.reshape(config.num_minibatches, -1, *x.shape[1:])
                return x

            rng, _rng = jax.random.split(rng)
            minibatches = jax.tree_util.tree_map(lambda x: preprocess_transition(x, _rng), transitions)
            targets = jax.tree_util.tree_map(lambda x: preprocess_transition(x, _rng), lambda_targets)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_phase, (train_state, rng), (minibatches, targets)
            )
            return (train_state, rng), (loss, qvals)

        rng, _rng = jax.random.split(rng)
        (train_state, rng), (loss, qvals) = jax.lax.scan(
            _learn_epoch, (train_state, rng), None, config.num_epochs
        )

        train_state = train_state.replace(n_updates=train_state.n_updates + 1)
        metrics.update({
            "loss": loss.mean(),
            "qvals": qvals.mean(),
        })

        runner_state = (train_state, env_state, rng)
        return runner_state, metrics
    
    return _update_step


@jax.jit
def evaluate(train_state, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, config.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)
    R = jnp.zeros_like(state.rewards)

    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, R, rng_key = tup
        q_values = train_state.apply_fn(
            {"params": train_state.params, "batch_stats": train_state.batch_stats},
            state.observation,
            train=False,
        )
        action = q_values.argmax(axis=-1)
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = step_fn(state, action, keys)
        return state, R + state.rewards, rng_key
    state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
    return R.mean()


def train(rng):
    tt = 0
    st = time.time()

    lr_scheduler = optax.linear_schedule(
        init_value=config.lr,
        end_value=1e-20,
        transition_steps=config.num_updates_decay * config.num_minibatches * config.num_epochs,
    )
    lr = lr_scheduler if config.lr_linear_decay else config.lr

    # INIT NETWORK AND OPTIMIZER
    network = QNetwork(
        action_dim=env.num_actions,
        norm_type=config.norm_type,
        norm_input=config.norm_input,
    )

    def create_agent(rng):
        init_x = jnp.zeros((1, *env.observation_shape))
        network_variables = network.init(rng, init_x, train=False)
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.radam(learning_rate=lr),
        )

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_variables["params"],
            batch_stats=network_variables.get("batch_stats"),
            tx=tx,
        )
        return train_state

    rng, _rng = jax.random.split(rng)
    train_state = create_agent(rng)

    _update_step = make_update_fn()
    jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, rng)

    # warm up
    _, _ = jitted_update_step(runner_state)

    steps = 0

    # initial evaluation
    st = time.time()
    if config.do_eval:
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate(runner_state[0], _rng)
        log = {f"{config.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        wandb.log(log)

    for i in range(config.num_updates):
        runner_state, metrics = jitted_update_step(runner_state)
        steps += config.num_envs * config.num_steps
        log = {"steps": steps, **metrics}
        wandb.log(log)
        # evaluation
        if config.do_eval and steps % 10 == 0:
            rng, _rng = jax.random.split(rng)
            eval_R = evaluate(runner_state[0], _rng)
            log = {f"{config.env_name}/eval_R": float(eval_R), "steps": steps, **log}
            print(log)
            wandb.log(log)
    et = time.time()
    wandb.log({"train_time": et - st})

    
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate(runner_state[0], _rng)
    log = {f"{config.env_name}/eval_R": float(eval_R), "steps": steps, f"{config.env_name}/final_eval_R": float(eval_R)}
    print(log)
    wandb.log(log)

    return runner_state

if __name__ == "__main__":
    wandb.init(project=config.wandb_project, config=config.dict())
    rng = jax.random.PRNGKey(config.seed)
    out = train(rng)
    if config.save_model:
        with open(f"{config.env_name}-seed={config.seed}.ckpt", "wb") as f:
            pickle.dump(out[0], f)
    

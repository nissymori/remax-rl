import sys
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
from typing import NamedTuple, Literal, Dict, Any
import distrax
import time

import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
import wandb
import copy
import envpool
from atari_wrapper import JaxLogEnvPoolWrapper



class REPPOConfig(BaseModel):
    # env
    env_name: str = "Pong-v5"
    env_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "episodic_life": True,
            "reward_clip": True,
            "repeat_action_probability": 0.1,
            "frame_skip": 4,
            "noop_max": 30,
        }
    )
    sticky_action_prob: float = 0.1

    seed: int = 0
    lr: float = 2.5e-4
    decay_lr: bool = True
    num_envs: int = 128
    num_eval_envs: int = 8
    eval_interval: int = 1000000
    num_steps: int = 128
    total_timesteps: int = 1e7
    update_epochs: int = 4
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.01
    clip_vloss: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_wandb: bool = False # use wandb
    wandb_project: str = "project-name"
    entity: str = "default"
    wandb_mode: str = "online"
    save_model: bool = False
    algo: str = "reppo"
    # reppo
    actor_coef: float = 1.0
    M: float = 1.2
    use_current_probs: bool = True
    use_baseline: bool = True
    replace_q: bool = True
    replace_type: Literal["return", "q_r_m"] = "return"
    normalize_advantage: bool = True

    class Config:
        extra = "forbid"


args = REPPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)


num_updates = int(args.total_timesteps // args.num_envs // args.num_steps)
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size

def make_env(num_envs):
    env = envpool.make(
        args.env_name,
        env_type="gym",
        num_envs=num_envs,
        seed=args.seed,
        **args.env_kwargs,
    )
    env.num_envs = num_envs
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = args.env_name
    env = JaxLogEnvPoolWrapper(env)
    return env

env = make_env(args.num_envs + args.num_eval_envs)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class ActorCritic(nn.Module):
    num_actions: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        features = CNN()(x)
        logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(features)
        q_values = nn.Dense(self.num_actions, kernel_init=orthogonal(1), bias_init=constant(0.0))(features)
        return logits, q_values

network = ActorCritic(env.single_action_space.n)


def linear_schedule(count):
    num_iterations = args.total_timesteps // args.num_envs // args.num_steps
    num_minibatches = args.num_envs * args.num_steps // args.minibatch_size
    frac = 1.0 - (count // (num_minibatches * args.update_epochs)) / num_iterations
    return args.lr * frac


optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.inject_hyperparams(optax.adam)(
        learning_rate=linear_schedule if args.decay_lr else args.lr,
        eps=1e-5,
    )
)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    q_values: jnp.ndarray
    probs: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# ========================
# ReMax utilities
# ========================
def expected_improvement_min(
    R: jnp.ndarray,
    q: jnp.ndarray,
    pi: jnp.ndarray,
    M: float,
    normalize_pi: bool = False
) -> jnp.ndarray:
    """EI_M(R;pi) = E[min_{1..M} (R - q_A)_+],  A~pi i.i.d.
       R:(B,Nr), q:(B,K), pi:(B,K) -> (B,Nr)
    """
    B, K = q.shape
    if pi.shape != (B, K):
        raise ValueError(f"pi must have shape {(B, K)}, got {pi.shape}.")
    if R.shape[0] != B:
        raise ValueError(f"R must have batch dim {B}, got {R.shape[0]}.")

    if normalize_pi:
        pi = pi / jnp.clip(pi.sum(axis=-1, keepdims=True), 1e-12)

    idx = jnp.argsort(-q, axis=-1)
    q_sorted = jnp.take_along_axis(q, idx, axis=-1)
    pi_sorted = jnp.take_along_axis(pi, idx, axis=-1)

    C = jnp.cumsum(pi_sorted, axis=-1)  # (B,K)

    v = jnp.maximum(R[..., None] - q_sorted[:, None, :], 0.0)  # (B,Nr,K)
    v_first = v[..., 0]
    dv = v[..., 1:] - v[..., :-1]
    eps = 1e-8
    w = jnp.power(jnp.clip(1.0 - C[..., :-1], eps, 1.0), M)    # (B,K-1)
    EI = v_first + jnp.sum(dv * w[:, None, :], axis=-1)
    return EI


def calc_remax_advantage(
    R: jnp.ndarray,
    q: jnp.ndarray,
    pi: jnp.ndarray,
    action: jnp.ndarray,
    M: float,
    use_baseline: bool = True,
    replace_q: bool = False,
    replace_type: Literal["return", "q_r_m", "hybrid"] = "return"
):
    """R:(B,) or (B,1) -> (B,)"""
    R = R.reshape((R.shape[0], 1))
    def _set(x, i, v):
        return x.at[i].set(v)

    if replace_q:
        if replace_type == "return":
            q_ref_for_R = jax.vmap(_set)(q, action, R[..., 0])
            q_ref_for_q = jax.vmap(_set)(q, action, R[..., 0])
        elif replace_type == "q_r_m":
            q_ref_for_R = jax.vmap(_set)(q, action, jnp.minimum(R[..., 0], jnp.take_along_axis(q, action[:, None], axis=-1).squeeze(-1)))
            q_ref_for_q = jax.vmap(_set)(q, action, jnp.minimum(R[..., 0], jnp.take_along_axis(q, action[:, None], axis=-1).squeeze(-1)))
        elif replace_type == "hybrid":
            q_ref_for_R = jax.vmap(_set)(q, action, R[..., 0])
            q_ref_for_q = jax.vmap(_set)(q, action, jnp.minimum(R[..., 0], jnp.take_along_axis(q, action[:, None], axis=-1).squeeze(-1)))
        else:
            raise ValueError(f"Invalid replace_type: {replace_type}")
    else:
        q_ref_for_R = q
        q_ref_for_q = q
    
    R_plus = expected_improvement_min(R, q_ref_for_R, pi, M, normalize_pi=False)[..., 0]
    q_plus = expected_improvement_min(q, q_ref_for_q, pi, M, normalize_pi=False)  # (B,K)
    baseline = jnp.sum(pi * jax.lax.stop_gradient(q_plus), axis=-1)
    advantage = jax.lax.cond(use_baseline, lambda: R_plus - baseline, lambda: R_plus)
    return advantage


def make_update_fn(args):
    # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES

        def _env_step(runner_state, unused):
            params, opt_state, env_state, last_obs, rng = runner_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            logits, q_values = network.apply(params, last_obs)
            pi = distrax.Categorical(logits=logits)
            probs = jax.nn.softmax(logits) 
            train_actions = pi.sample(seed=_rng)  # (num_envs, )
            log_prob = pi.log_prob(train_actions)
            eval_actions = logits.argmax(axis=-1)  # (num_eval_envs, )
            action = jnp.concatenate((train_actions[:args.num_envs], eval_actions[args.num_envs:]), axis=0)  # (num_envs + num_eval_envs, )
    
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(env_state, action)
            transition = Transition(
                done=done,
                action=action,
                q_values=q_values,
                probs=probs,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs
            )
            runner_state = (params, opt_state, env_state,
                            obs, rng)
            return runner_state, (transition, info)

        runner_state, (traj_batch, infos) = jax.lax.scan(
            _env_step, runner_state, None, args.num_steps
        )

        traj_batch = jax.tree_util.tree_map(lambda x: x[:, :args.num_envs], traj_batch)  # remove eval envs from training data.

        # CALCULATE ADVANTAGE
        params, opt_state, env_state, last_obs, rng = runner_state
        logits, last_q_values = network.apply(params, last_obs[:args.num_envs])
        # For terminal value, we use the expected Q-value under the policy
        pi = distrax.Categorical(logits=logits)
        probs = jax.nn.softmax(logits)
        last_val = jnp.sum(probs * last_q_values, axis=-1)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, q_values, probs, reward, action, obs = (
                    transition.done,
                    transition.q_values,
                    transition.probs,
                    transition.reward,
                    transition.action,
                    transition.obs,
                )
                value = jnp.sum(probs * q_values, axis=-1)
                delta = reward + args.gamma * next_value * (1 - done) - value
                gae = (
                    delta
                    + args.gamma * args.gae_lambda * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            # For Q-based PPO, targets are Q-values + advantages
            values = jax.vmap(lambda p, q: jnp.sum(p * q, axis=-1))(
                traj_batch.probs, traj_batch.q_values
            )
            return advantages, advantages + values

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets = batch_info
                old_logits, _ = network.apply(params, traj_batch.obs)
                old_probs = jax.nn.softmax(old_logits)

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    logits, q_values = network.apply(params, traj_batch.obs)
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(traj_batch.action)
                    targets = jax.lax.stop_gradient(targets)

                    # CALCULATE Q-VALUE LOSS
                    # Extract Q-values for taken actions
                    q_value = jnp.take_along_axis(q_values, traj_batch.action[:, None], axis=-1).squeeze(-1)
                    old_q_value = jnp.take_along_axis(traj_batch.q_values, traj_batch.action[:, None], axis=-1).squeeze(-1)
                    
                    q_value_pred_clipped = old_q_value + (
                        q_value - old_q_value
                    ).clip(-args.clip_eps, args.clip_eps)
                    value_losses = jnp.square(q_value - targets)
                    value_losses_clipped = jnp.square(
                        q_value_pred_clipped - targets)
                    if args.clip_vloss:
                        value_loss = (
                            0.5 * jnp.maximum(
                                value_losses,
                                value_losses_clipped).mean()
                        )
                    else:
                        value_loss = 0.5 * value_losses.mean()

                    # CALCULATE ACTOR LOSS
                    probs = jax.nn.softmax(logits)
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    if not args.use_current_probs:  # 現在のprobsを使うかどうか
                        probs = traj_batch.probs
                    remax_advantage = calc_remax_advantage(targets, traj_batch.q_values, probs, traj_batch.action, args.M-1, args.use_baseline, args.replace_q, args.replace_type)
                    if args.normalize_advantage:  # アドバンテージを正規化するかどうか
                        advantage = (remax_advantage - remax_advantage.mean()) / (remax_advantage.std() + 1e-8)
                    else:
                        advantage = remax_advantage
                    advantage = jax.lax.stop_gradient(advantage)
        
                    loss_actor1 = ratio * advantage # (B, A)
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - args.clip_eps,
                            1.0 + args.clip_eps,
                        )
                        * advantage
                    )  # (B, A)
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2) # (B, A)
                    loss_actor = loss_actor.mean()
                    loss_actor = args.actor_coef * loss_actor
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + args.vf_coef * value_loss
                        - args.ent_coef * entropy
                    )
                    aux = {
                        "total_loss": total_loss,
                        "value_loss": value_loss,
                        "loss_actor": loss_actor,
                        "q_value": q_value,
                        "remax_advantage": remax_advantage,
                        "remax_advantage_std": remax_advantage.std(),
                        "entropy": entropy,
                    }
                    return total_loss, aux

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, aux), grads = grad_fn(
                    params, traj_batch, advantages, targets)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), aux

            params, opt_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = args.minibatch_size * num_minibatches
            assert (
                batch_size == args.num_steps * args.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            (params, opt_state),  aux = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )

            update_state = (params, opt_state, traj_batch,
                            advantages, targets, rng)  # old param should be the same as new param
            return update_state, aux

        update_state = (params, opt_state, traj_batch,
                        advantages, targets, rng)
        update_state, metrics = jax.lax.scan(
            _update_epoch, update_state, None, args.update_epochs
        )
        params, opt_state, _, _, _, rng = update_state

        metrics = {k: v.mean() for k, v in metrics.items()}

        # update return info
        test_infos = jax.tree_util.tree_map(lambda x: x[:, args.num_envs:], infos)
        infos = jax.tree_util.tree_map(lambda x: x[:, :args.num_envs], infos)
        infos.update({"test/" + k: v for k, v in test_infos.items()})
        metrics.update({"test/" + k: v.mean() for k, v in test_infos.items()})

        runner_state = (params, opt_state, env_state, last_obs, rng)
        return runner_state, metrics
    return _update_step


def train(rng, args):
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *env.single_observation_space.shape))
    network = ActorCritic(env.single_action_space.n)

    params = network.init(_rng, init_x)
    opt_state = optimizer.init(params=params)

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn(args)
    jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    obs, env_state = env.reset()

    rng, _rng = jax.random.split(rng)
    runner_state = (params, opt_state, env_state, obs, _rng)

    # warm up
    _, _ = jitted_update_step(runner_state)

    steps = 0


    st = time.time()
    eval_interval = int(args.eval_interval // args.num_steps // args.num_envs)
    for i in range(num_updates):
        runner_state, metrics = jitted_update_step(runner_state)
        steps += args.num_envs * args.num_steps
        if args.use_wandb:
            wandb.log({"steps": steps, **metrics})
        if i % eval_interval == 0:
            print({"steps": steps, **metrics})

    et = time.time()
    wandb.log({"train_time": et - st})
    return runner_state


if __name__ == "__main__":
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=args.dict())
    rng = jax.random.PRNGKey(args.seed)
    out = train(rng, args)
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as f:
            pickle.dump(out[0], f)
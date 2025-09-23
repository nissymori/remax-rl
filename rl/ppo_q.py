"""This PPO implementation is modified from

- PureJaxRL:https://github.com/luchris429/purejaxrl
- pgx:https://github.com/sotetsuk/pgx/tree/main/examples/minatar-ppo

Please refer to their work if you use this example in your research."""

import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import NamedTuple, Literal
import distrax
import pgx
from pgx.experimental import auto_reset
import time

import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb


class PPOConfig(BaseModel):
    env_name: Literal[
        "minatar-breakout",
        "minatar-freeway",
        "minatar-space_invaders",
        "minatar-asterix",
        "minatar-seaquest",
    ] = "minatar-breakout"
    seed: int = 0
    lr: float = 0.0003
    num_envs: int = 1024
    num_eval_envs: int = 100
    num_steps: int = 128
    total_timesteps: int = 10000000
    update_epochs: int = 3
    minibatch_size: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    wandb_project: str = "pgx-minatar-ppo"
    save_model: bool = False
    algo: str = "ppo_q"

    class Config:
        extra = "forbid"


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)
env = pgx.make(str(args.env_name))


num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size


class ActorCritic(nn.Module):
    num_actions: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.activation == "relu":
            activation = jax.nn.relu
        else:
            activation = jax.nn.tanh
        
        x = nn.Conv(32, kernel_size=(2, 2))(x)
        x = jax.nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(64)(x)
        x = jax.nn.relu(x)
        
        # Actor head
        actor_mean = nn.Dense(64)(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.num_actions)(actor_mean)

        # Q-network instead of V-network
        critic = nn.Dense(64)(x)
        critic = activation(critic)
        critic = nn.Dense(64)(critic)
        critic = activation(critic)
        critic = nn.Dense(self.num_actions)(critic)  # Output Q-values for each action

        return actor_mean, critic


optimizer = optax.chain(optax.clip_by_global_norm(
    args.max_grad_norm), optax.adam(args.lr, eps=1e-5))


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    q_values: jnp.ndarray
    probs: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def make_update_fn():
    # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES
        step_fn = jax.vmap(auto_reset(env.step, env.init))

        def _env_step(runner_state, unused):
            params, opt_state, env_state, last_obs, rng = runner_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            network = ActorCritic(env.num_actions)
            logits, q_values = network.apply(params, last_obs)
            pi = distrax.Categorical(logits=logits)
            probs = jax.nn.softmax(logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, env_state.observation.shape[0])
            env_state = step_fn(env_state, action, keys)
            transition = Transition(
                env_state.terminated,
                action,
                q_values,
                probs,
                jnp.squeeze(env_state.rewards),
                log_prob,
                last_obs
            )
            runner_state = (params, opt_state, env_state,
                            env_state.observation, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, args.num_steps
        )

        # CALCULATE ADVANTAGE
        params, opt_state, env_state, last_obs, rng = runner_state
        network = ActorCritic(env.num_actions)
        logits, last_q_values = network.apply(params, last_obs)
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

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    network = ActorCritic(env.num_actions)
                    logits, q_values = network.apply(params, traj_batch.obs)
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(traj_batch.action)

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
                    value_loss = (
                        0.5 * jnp.maximum(value_losses,
                                          value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - args.clip_eps,
                            1.0 + args.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + args.vf_coef * value_loss
                        - args.ent_coef * entropy
                    )
                    aux = {
                        "value_loss": value_loss,
                        "loss_actor": loss_actor,
                        "entropy": entropy,
                    }
                    return total_loss, aux

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, aux), grads = grad_fn(
                    params, traj_batch, advantages, targets)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (total_loss, aux)

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
            (params, opt_state), (total_loss, aux) = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )
            update_state = (params, opt_state, traj_batch,
                            advantages, targets, rng)
            return update_state, (total_loss, aux)

        update_state = (params, opt_state, traj_batch,
                        advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, args.update_epochs
        )
        params, opt_state, _, _, _, rng = update_state

        runner_state = (params, opt_state, env_state, last_obs, rng)
        return runner_state, loss_info
    return _update_step


@jax.jit
def evaluate(params, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, args.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)
    R = jnp.zeros_like(state.rewards)

    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, R, rng_key = tup
        network = ActorCritic(env.num_actions)
        logits, q_values = network.apply(params, state.observation)
        action = logits.argmax(axis=-1)
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = step_fn(state, action, keys)
        return state, R + state.rewards, rng_key
    state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
    return R.mean()


def train(rng):
    tt = 0
    st = time.time()
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, ) + env.observation_shape)
    network = ActorCritic(env.num_actions)
    params = network.init(_rng, init_x)
    opt_state = optimizer.init(params=params)

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn()
    jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (params, opt_state, env_state, env_state.observation, _rng)

    # warm up
    _, _ = jitted_update_step(runner_state)

    steps = 0

    # initial evaluation
    et = time.time()  # exclude evaluation time
    tt += et - st
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate(runner_state[0], _rng)
    log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log)
    wandb.log(log)
    st = time.time()

    for i in range(num_updates):
        runner_state, loss_info = jitted_update_step(runner_state)
        steps += args.num_envs * args.num_steps
        total_loss, aux = loss_info
        log = {
            "sec": tt,
            "steps": steps,
            "total_loss": total_loss.mean(),
            "value_loss": aux["value_loss"].mean(),
            "loss_actor": aux["loss_actor"].mean(),
            "entropy": aux["entropy"].mean(),
        }
        wandb.log(log)
        # evaluation
        et = time.time()  # exclude evaluation time
        tt += et - st
        if steps % 10 == 0:
            rng, _rng = jax.random.split(rng)
            eval_R = evaluate(runner_state[0], _rng)
            log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
            print(log)
            wandb.log(log)
            st = time.time()

    rng, _rng = jax.random.split(rng)
    eval_R = evaluate(runner_state[0], _rng)
    log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps, f"{args.env_name}/final_eval_R": float(eval_R)}
    print(log)
    wandb.log(log)
    st = time.time()

    return runner_state


if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.dict())
    rng = jax.random.PRNGKey(args.seed)
    out = train(rng)
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as f:
            pickle.dump(out[0], f)
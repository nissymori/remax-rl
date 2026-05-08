"""PPO with Random Network Distillation for MinAtar.

This follows the structure of `ppo/minatar/ppo_v.py` while bringing in the
RND components from the Craftax PPO + RND implementation.
"""

import sys
import time
from typing import Literal, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import pgx
from pgx.experimental import auto_reset
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
    do_eval: bool = True
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
    wandb_project: str = "project-name"
    algo: str = "ppo_v_rnd"
    save_model: bool = False
    # RND
    use_rnd: bool = True
    rnd_layer_size: int = 256
    rnd_output_size: int = 512
    rnd_lr: float = 3e-4
    rnd_reward_coeff: float = 1.0
    rnd_loss_coeff: float = 0.01
    rnd_gae_coeff: float = 0.01
    rnd_is_episodic: bool = False
    exploration_update_epochs: int = 1

    class Config:
        extra = "forbid"


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)
env = pgx.make(str(args.env_name))

num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size


class ActorCriticRND(nn.Module):
    num_actions: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        activation = jax.nn.relu if self.activation == "relu" else jax.nn.tanh

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
        logits = nn.Dense(self.num_actions)(actor_mean)

        # Extrinsic critic
        critic_e = nn.Dense(64)(x)
        critic_e = activation(critic_e)
        critic_e = nn.Dense(64)(critic_e)
        critic_e = activation(critic_e)
        critic_e = nn.Dense(1)(critic_e)

        # Intrinsic critic
        critic_i = nn.Dense(64)(x)
        critic_i = activation(critic_i)
        critic_i = nn.Dense(64)(critic_i)
        critic_i = activation(critic_i)
        critic_i = nn.Dense(1)(critic_i)

        return logits, jnp.squeeze(critic_e, axis=-1), jnp.squeeze(
            critic_i, axis=-1
        )


class RNDNetwork(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        emb = x
        for _ in range(self.num_layers):
            emb = nn.Dense(self.layer_size)(emb)
            emb = nn.relu(emb)

        emb = nn.Dense(self.output_dim)(emb)
        return emb


optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5)
)
rnd_optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.rnd_lr, eps=1e-5)
)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_e: jnp.ndarray
    value_i: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray


def make_update_fn(rnd_random_params, rnd_layer_size, rnd_output_size):
    step_fn = jax.vmap(auto_reset(env.step, env.init))

    def _update_step(runner_state):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            params, opt_state, rnd_params, rnd_opt_state, env_state, last_obs, rng = (
                runner_state
            )
            rng, _rng = jax.random.split(rng)
            network = ActorCriticRND(env.num_actions)
            logits, value_e, value_i = network.apply(params, last_obs)
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, env_state.observation.shape[0])
            env_state = step_fn(env_state, action, keys)

            reward_e = jnp.squeeze(env_state.rewards)
            done = env_state.terminated
            reward_i = jnp.zeros_like(reward_e)

            if args.use_rnd:
                rnd_network = RNDNetwork(
                    num_layers=3,
                    output_dim=rnd_output_size,
                    layer_size=rnd_layer_size,
                )
                obs_flat = env_state.observation.reshape(env_state.observation.shape[0], -1)
                random_pred = rnd_network.apply(rnd_random_params, obs_flat)
                distill_pred = rnd_network.apply(rnd_params, obs_flat)
                error = (random_pred - distill_pred) * (1 - done[:, None])
                mse = jnp.square(error).mean(axis=-1)
                reward_i = mse * args.rnd_reward_coeff

            reward = reward_e + reward_i

            transition = Transition(
                done=done,
                action=action,
                value_e=value_e,
                value_i=value_i,
                reward_e=reward_e,
                reward_i=reward_i,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
                next_obs=env_state.observation,
            )
            runner_state = (
                params,
                opt_state,
                rnd_params,
                rnd_opt_state,
                env_state,
                env_state.observation,
                rng,
            )
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, args.num_steps
        )

        # CALCULATE ADVANTAGE
        params, opt_state, rnd_params, rnd_opt_state, env_state, last_obs, rng = (
            runner_state
        )
        network = ActorCriticRND(env.num_actions)
        _, last_val_e, last_val_i = network.apply(params, last_obs)

        def _calculate_gae(traj_batch, last_val, is_extrinsic):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value, extrinsic = gae_and_next_value
                done = transition.done
                done = jnp.logical_and(
                    done, jnp.logical_or(args.rnd_is_episodic, extrinsic)
                )
                value = jax.lax.select(extrinsic, transition.value_e, transition.value_i)
                reward = jax.lax.select(
                    extrinsic, transition.reward_e, transition.reward_i
                )
                delta = reward + args.gamma * next_value * (1 - done) - value
                gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                return (gae, value, extrinsic), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val, is_extrinsic),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            targets = advantages + jax.lax.select(
                is_extrinsic, traj_batch.value_e, traj_batch.value_i
            )
            return advantages, targets

        advantages_e, targets_e = _calculate_gae(traj_batch, last_val_e, True)
        advantages_i, targets_i = _calculate_gae(traj_batch, last_val_i, False)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                (
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                ) = batch_info

                def _loss_fn(
                    params, traj_batch, gae_e, targets_e, gae_i, targets_i
                ):
                    network = ActorCriticRND(env.num_actions)
                    logits, value_e, value_i = network.apply(params, traj_batch.obs)
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    value_pred_clipped_e = traj_batch.value_e + (
                        value_e - traj_batch.value_e
                    ).clip(-args.clip_eps, args.clip_eps)
                    value_losses_e = jnp.square(value_e - targets_e)
                    value_losses_clipped_e = jnp.square(
                        value_pred_clipped_e - targets_e
                    )
                    value_loss_e = (
                        0.5
                        * jnp.maximum(value_losses_e, value_losses_clipped_e).mean()
                    )

                    value_pred_clipped_i = traj_batch.value_i + (
                        value_i - traj_batch.value_i
                    ).clip(-args.clip_eps, args.clip_eps)
                    value_losses_i = jnp.square(value_i - targets_i)
                    value_losses_clipped_i = jnp.square(
                        value_pred_clipped_i - targets_i
                    )
                    value_loss_i = (
                        0.5
                        * jnp.maximum(value_losses_i, value_losses_clipped_i).mean()
                    )

                    gae = gae_e + gae_i * args.rnd_gae_coeff
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

                    value_loss = value_loss_e + value_loss_i * args.use_rnd

                    total_loss = (
                        loss_actor
                        + args.vf_coef * value_loss
                        - args.ent_coef * entropy
                    )
                    return total_loss, (value_loss_e, value_loss_i, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    params,
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                )
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), total_loss

            (
                params,
                opt_state,
                traj_batch,
                advantages_e,
                targets_e,
                advantages_i,
                targets_i,
                rng,
            ) = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = args.minibatch_size * num_minibatches
            assert (
                batch_size == args.num_steps * args.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (
                traj_batch,
                advantages_e,
                targets_e,
                advantages_i,
                targets_i,
            )
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
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )
            update_state = (
                params,
                opt_state,
                traj_batch,
                advantages_e,
                targets_e,
                advantages_i,
                targets_i,
                rng,
            )
            return update_state, total_loss

        update_state = (
            params,
            opt_state,
            traj_batch,
            advantages_e,
            targets_e,
            advantages_i,
            targets_i,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, args.update_epochs
        )
        params, opt_state, _, _, _, _, _, rng = update_state

        # UPDATE RND DISTILLATION NETWORK
        rnd_loss = 0.0

        if args.use_rnd:

            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(tup, traj_batch):
                    rnd_params, rnd_opt_state = tup

                    def _rnd_loss_fn(rnd_params, traj_batch):
                        rnd_network = RNDNetwork(
                            num_layers=3,
                            output_dim=rnd_output_size,
                            layer_size=rnd_layer_size,
                        )
                        obs_flat = traj_batch.next_obs.reshape(
                            traj_batch.next_obs.shape[0], -1
                        )
                        random_network_out = rnd_network.apply(
                            rnd_random_params, obs_flat
                        )
                        distillation_network_out = rnd_network.apply(
                            rnd_params, obs_flat
                        )
                        error = (random_network_out - distillation_network_out) * (
                            1 - traj_batch.done[:, None]
                        )
                        return jnp.square(error).mean() * args.rnd_loss_coeff

                    rnd_grad_fn = jax.value_and_grad(_rnd_loss_fn)
                    rnd_loss, rnd_grad = rnd_grad_fn(rnd_params, traj_batch)
                    rnd_updates, rnd_opt_state = rnd_optimizer.update(
                        rnd_grad, rnd_opt_state
                    )
                    rnd_params = optax.apply_updates(rnd_params, rnd_updates)
                    return (rnd_params, rnd_opt_state), rnd_loss

                rnd_params, rnd_opt_state, traj_batch, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = args.minibatch_size * num_minibatches
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
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
                (rnd_params, rnd_opt_state), rnd_losses = jax.lax.scan(
                    _update_ex_minbatch, (rnd_params, rnd_opt_state), minibatches
                )
                update_state = (rnd_params, rnd_opt_state, traj_batch, rng)
                return update_state, rnd_losses

            ex_update_state = (rnd_params, rnd_opt_state, traj_batch, rng)
            ex_update_state, ex_loss = jax.lax.scan(
                _update_ex_epoch,
                ex_update_state,
                None,
                args.exploration_update_epochs,
            )
            rnd_params, rnd_opt_state, _, rng = ex_update_state
            rnd_loss = ex_loss.mean()

        runner_state = (
            params,
            opt_state,
            rnd_params,
            rnd_opt_state,
            env_state,
            last_obs,
            rng,
        )
        return runner_state, (loss_info, rnd_loss)

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
        network = ActorCriticRND(env.num_actions)
        logits, _, _ = network.apply(params, state.observation)
        action = logits.argmax(axis=-1)
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = step_fn(state, action, keys)
        return state, R + state.rewards, rng_key

    state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
    return R.mean()


def train(rng):
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + env.observation_shape)
    network = ActorCriticRND(env.num_actions)
    params = network.init(_rng, init_x)
    opt_state = optimizer.init(params=params)

    # RND networks
    obs_dim = int(np.prod(env.observation_shape))
    rnd_network = RNDNetwork(
        num_layers=3, output_dim=args.rnd_output_size, layer_size=args.rnd_layer_size
    )
    rng, _rng = jax.random.split(rng)
    rnd_random_params = rnd_network.init(_rng, jnp.zeros((1, obs_dim)))
    rng, _rng = jax.random.split(rng)
    rnd_params = rnd_network.init(_rng, jnp.zeros((1, obs_dim)))
    rnd_opt_state = rnd_optimizer.init(rnd_params)

    _update_step = make_update_fn(
        rnd_random_params, args.rnd_layer_size, args.rnd_output_size
    )
    jitted_update_step = jax.jit(_update_step)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (
        params,
        opt_state,
        rnd_params,
        rnd_opt_state,
        env_state,
        env_state.observation,
        _rng,
    )

    _, _ = jitted_update_step(runner_state)

    steps = 0

    if args.do_eval:
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate(runner_state[0], _rng)
        log = {f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        wandb.log(log)

    st = time.time()
    for _ in range(num_updates):
        runner_state, loss_info = jitted_update_step(runner_state)
        loss_info, rnd_loss = loss_info
        steps += args.num_envs * args.num_steps
        rng, _rng = jax.random.split(rng)
        if args.do_eval and steps % 10 == 0:
            eval_R = evaluate(runner_state[0], _rng)
            log = {
                f"{args.env_name}/eval_R": float(eval_R),
                "steps": steps,
                f"{args.env_name}/rnd_loss": float(rnd_loss),
            }
            print(log)
            wandb.log(log)
    et = time.time()
    wandb.log({"train_time": et - st})
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate(runner_state[0], _rng)
    log = {
        f"{args.env_name}/eval_R": float(eval_R),
        "steps": steps,
        f"{args.env_name}/final_eval_R": float(eval_R),
    }
    print(log)
    wandb.log(log)
    return runner_state


if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.dict())
    rng = jax.random.PRNGKey(args.seed)
    out = train(rng)
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as f:
            pickle.dump(out[0], f)

"""This PPO implementation is adapted for Atari environments.

It follows the MinAtar PPO implementation in this repository and
borrows the Atari-specific environment handling from the REPPO setup.
"""

import sys
import time
from typing import Any, Dict, NamedTuple

import distrax
import envpool
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from atari_wrapper import JaxLogEnvPoolWrapper


class PPOConfig(BaseModel):
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

    # general training
    seed: int = 0
    lr: float = 2.5e-4
    decay_lr: bool = True
    num_envs: int = 128
    num_eval_envs: int = 8
    eval_interval: int = 1_000_000
    num_steps: int = 128
    total_timesteps: int = int(1e7)
    update_epochs: int = 4
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.01
    clip_vloss: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_wandb: bool = False
    wandb_project: str = "project-name"
    entity: str = "default"
    wandb_mode: str = "online"
    save_model: bool = False
    algo: str = "ppo_v"
    normalize_advantage: bool = True

    class Config:
        extra = "forbid"


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)

num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size


def make_env(num_envs: int):
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
        values = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(features)
        return logits, jnp.squeeze(values, axis=-1)


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
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def make_update_fn(config: PPOConfig):
    def _update_step(runner_state):
        def _env_step(runner_state, _):
            params, opt_state, env_state, last_obs, rng = runner_state
            rng, act_rng = jax.random.split(rng)
            logits, value = network.apply(params, last_obs)
            pi = distrax.Categorical(logits=logits)
            train_actions = pi.sample(seed=act_rng)
            log_prob = pi.log_prob(train_actions)
            greedy_actions = logits.argmax(axis=-1)
            action = jnp.concatenate(
                (train_actions[: config.num_envs], greedy_actions[config.num_envs :]),
                axis=0,
            )

            rng, step_rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(env_state, action)
            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
            )
            runner_state = (params, opt_state, env_state, obs, rng)
            return runner_state, (transition, info)

        runner_state, (traj_batch, infos) = jax.lax.scan(
            _env_step, runner_state, None, config.num_steps
        )

        traj_batch = jax.tree_util.tree_map(
            lambda x: x[:, : config.num_envs], traj_batch
        )

        params, opt_state, env_state, last_obs, rng = runner_state
        _, last_val = network.apply(params, last_obs[: config.num_envs])

        def _calculate_gae(traj_batch, last_val):
            def _step(carry, transition):
                gae, next_value = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config.gamma * next_value * (1.0 - done) - value
                gae = delta + config.gamma * config.gae_lambda * (1.0 - done) * gae
                return (gae, value), gae

            (_, _), advantages = jax.lax.scan(
                _step,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            returns = advantages + traj_batch.value
            return advantages, returns

        advantages, targets = _calculate_gae(traj_batch, last_val)

        if config.normalize_advantage:
            adv_mean = jnp.mean(advantages)
            adv_std = jnp.std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        def _update_epoch(update_state, _):
            def _update_minibatch(state, batch):
                params, opt_state = state
                traj_batch, gae, targets = batch

                def _loss_fn(params, traj_batch, gae, targets):
                    logits, value = network.apply(params, traj_batch.obs)
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    unclipped = ratio * gae
                    clipped = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae
                    loss_actor = -jnp.mean(jnp.minimum(unclipped, clipped))

                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config.clip_eps, config.clip_eps)
                    value_losses = jnp.square(value - targets)
                    if config.clip_vloss:
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
                    else:
                        value_loss = 0.5 * value_losses.mean()

                    entropy = jnp.mean(pi.entropy())
                    approx_kl = jnp.mean(traj_batch.log_prob - log_prob)
                    clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)

                    total_loss = (
                        loss_actor
                        + config.vf_coef * value_loss
                        - config.ent_coef * entropy
                    )
                    metrics = {
                        "total_loss": total_loss,
                        "loss_actor": loss_actor,
                        "value_loss": value_loss,
                        "entropy": entropy,
                        "approx_kl": approx_kl,
                        "clipfrac": clipfrac,
                    }
                    return total_loss, metrics

                gae = jax.lax.stop_gradient(gae)
                targets = jax.lax.stop_gradient(targets)
                (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    params, traj_batch, gae, targets
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), metrics

            params, opt_state, traj_batch, advantages, targets, rng = update_state
            rng, shuffle_rng = jax.random.split(rng)
            batch_size = config.num_steps * config.num_envs
            permutation = jax.random.permutation(shuffle_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            flat_batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), flat_batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]), shuffled
            )

            (params, opt_state), metrics = jax.lax.scan(
                _update_minibatch, (params, opt_state), minibatches
            )
            update_state = (params, opt_state, traj_batch, advantages, targets, rng)
            return update_state, metrics

        update_state = (params, opt_state, traj_batch, advantages, targets, rng)
        update_state, metrics = jax.lax.scan(
            _update_epoch, update_state, None, config.update_epochs
        )
        params, opt_state, _, _, _, rng = update_state

        test_infos = jax.tree_util.tree_map(lambda x: x[:, config.num_envs :], infos)
        infos = jax.tree_util.tree_map(lambda x: x[:, : config.num_envs], infos)
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        metrics = dict(metrics)
        metrics.update({"test/" + k: v.mean() for k, v in test_infos.items()})

        runner_state = (params, opt_state, env_state, last_obs, rng)
        return runner_state, metrics

    return _update_step


def train(rng: jax.Array, config: PPOConfig):
    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *env.single_observation_space.shape))
    params = network.init(init_rng, init_x)
    opt_state = optimizer.init(params)

    update_step = make_update_fn(config)
    jitted_update_step = jax.jit(update_step)

    obs, env_state = env.reset()

    rng, runner_rng = jax.random.split(rng)
    runner_state = (params, opt_state, env_state, obs, runner_rng)

    _, _ = jitted_update_step(runner_state)

    steps = 0
    start_time = time.time()
    eval_interval = int(config.eval_interval // config.num_steps // config.num_envs)

    for i in range(num_updates):
        runner_state, metrics = jitted_update_step(runner_state)
        steps += config.num_envs * config.num_steps
        if config.use_wandb:
            wandb.log(dict(metrics, steps=steps))
        if eval_interval > 0 and i % eval_interval == 0:
            print({"steps": steps, **metrics})

    elapsed = time.time() - start_time
    if config.use_wandb:
        wandb.log({"train_time": elapsed})

    return runner_state


if __name__ == "__main__":
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=args.dict(), mode=args.wandb_mode)
    rng = jax.random.PRNGKey(args.seed)
    result = train(rng, args)
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ppo_v", "wb") as f:
            pickle.dump(result[0], f)

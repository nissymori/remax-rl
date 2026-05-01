from __future__ import annotations

"""
Generalized ReMax bandit experiments for two reward families:
  1) Bernoulli–Beta (conjugate Beta posterior)  
  2) Gaussian–Gaussian (conjugate Normal posterior on means with known noise std)

Algorithms implemented (per user request):
  - Thompson Sampling (family-specific)
  - UCB (family-specific)
  - ReMax-EI (Exact inner objective via Proposition + autodiff over policy)

Key notes
---------
• ReMax-EI uses the exact conditional expected-maximum formula
  J^M_ReMax(π, μ) = μ_(1) - Σ_{i=1}^{K-1} (μ_(i) - μ_(i+1)) (1 - S_i)^M
  where S_i is the policy mass on the top-i indices of μ.
  We Monte Carlo over μ ~ Π_t (posterior) and differentiate through π.

• The same training loop is shared across families via parallel, batched instances.

• CLI examples
  Bernoulli–Beta:
    python generalized_remax_bandits.py --family beta --num_pulls 500 --num_bandit_instances 256 --num_arms 5 \
      --remax_M 2 --remax_pg_iter 20 --prior_alpha 1.0 --prior_beta 1.0

  Gaussian–Gaussian:
    python generalized_remax_bandits.py --family gaussian --num_pulls 500 --num_bandit_instances 256 --num_arms 5 \
      --noise_std 1.0 --prior_mean 0.0 --prior_std 1.0 --remax_M 2 --remax_pg_iter 20
"""

import functools
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse


# =============================================================
# Utilities
# =============================================================

def batch_choice(p: np.ndarray, u: Optional[np.ndarray] = None, sample_per_p: int = 1) -> np.ndarray:
    """Sample one index per row according to row-wise distributions.
    Args:
        p: (B, K) nonnegative rows summing to 1
        u: optional pre-sampled uniforms (B, sample_per_p)
        sample_per_p: not used (always 1 per row), kept for API parity
    Returns:
        choices: (B,) integer indices
    """
    assert p.ndim == 2 and p.shape[1] > 0
    B, K = p.shape
    if not isinstance(u, np.ndarray):
        u = np.random.rand(B, sample_per_p)
    cumsum = np.cumsum(p, axis=1)
    choices = np.argmax(u < cumsum, axis=1)
    return choices


class Policy(nn.Module):
    """Batch-wise categorical policy parameterized by per-(batch,arm) logits."""
    def __init__(self, num_arms: int, batch_size: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(batch_size, num_arms))

    def forward(self):  # (batch_size, num_arms)
        return self.logits


# =============================================================
# Environments
# =============================================================
class BernoulliBandit:
    """Bernoulli bandit with per-arm success probability in (0,1).
    If `arm_probs` is None, draw true probs from Beta(prior_alpha, prior_beta).
    """
    def __init__(
        self,
        num_arms: int,
        num_bandit_instances: int = 1,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self.num_arms = num_arms
        self.num_bandit_instances = num_bandit_instances
        # Sample true arm means:
        self.bandit_means = np.random.beta(
            prior_alpha, prior_beta, size=(num_bandit_instances, num_arms)
        )
        self.best_arm = np.argmax(self.bandit_means, axis=1)
        self.best_reward = np.max(self.bandit_means, axis=1)

    def pull(self, arms: np.ndarray):
        arm_means = self.bandit_means[range(self.num_bandit_instances), arms].squeeze()
        rewards = (np.random.rand(self.num_bandit_instances) < arm_means).astype(np.float64)
        regrets = self.best_reward - arm_means
        return rewards, regrets


class GaussianBandit:
    """Gaussian bandit over means per arm.
    Rewards: R ~ N(mu_arm, noise_std_arm^2), with mu drawn once from N(mean, std^2).
    """
    def __init__(
        self,
        num_arms: int,
        num_bandit_instances: int = 1,
        mean: float = 0.0,
        std: float = 1.0,
        noise_std: float = 1.0,
    ):
        self.num_arms = num_arms
        self.num_bandit_instances = num_bandit_instances
        self.mean = np.ones(num_arms) * mean

        self.std = std * np.ones(num_arms)
        self.noise_std = noise_std * np.ones(num_arms)
        # Sample true arm means:
        self.bandit_means = np.random.normal(self.mean, self.std, (num_bandit_instances, num_arms))
        self.best_arm = np.argmax(self.bandit_means, axis=1)
        self.best_reward = np.max(self.bandit_means, axis=1)

    def pull(self, arms: np.ndarray):
        arm_means = self.bandit_means[range(self.num_bandit_instances), arms].squeeze()
        arm_stds = self.noise_std[arms]
        rewards = np.random.normal(arm_means, arm_stds)
        regrets = self.best_reward - arm_means
        return rewards, regrets


# =============================================================
# Base learners
# =============================================================
class _BaseLearner:
    def __init__(self, batch_size: int, bandit):
        self.batch_size = batch_size
        self.bandit = bandit
        assert (
            self.batch_size == bandit.num_bandit_instances or bandit.num_bandit_instances == 1
        )
        self.num_arms = bandit.num_arms

    def learn(self, num_pulls: int):
        self.regret_history = np.zeros((self.batch_size, num_pulls))
        cum_regrets = np.zeros((self.batch_size))
        self.num_pulls = num_pulls
        self.posterior_mean_history = np.zeros((self.batch_size, num_pulls, self.num_arms))
        self.posterior_std_history = np.zeros((self.batch_size, num_pulls, self.num_arms))
        self.arms_selection_history = np.zeros((self.batch_size, num_pulls), dtype=int)
        for t in tqdm(range(num_pulls)):
            arms = self.select_arms()
            rewards, regrets = self.bandit.pull(arms)
            self.update_posteriors(rewards, arms)
            cum_regrets += regrets
            self.regret_history[:, t] = cum_regrets
            self.posterior_mean_history[:, t] = self.posterior_mean
            self.posterior_std_history[:, t] = self.posterior_std
            self.arms_selection_history[:, t] = arms
        return self.regret_history

    # To implement by subclasses
    def update_posteriors(self, rewards: np.ndarray, arms: np.ndarray):
        raise NotImplementedError

    def select_arms(self) -> np.ndarray:
        raise NotImplementedError


# -------------------- Beta family --------------------
class BernoulliLearner(_BaseLearner):
    def __init__(self, batch_size: int, bandit: BernoulliBandit, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        super().__init__(batch_size, bandit)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.reset_posteriors()


    def reset_posteriors(self):
        B, K = self.batch_size, self.num_arms
        self.posterior_alpha = np.full((B, K), self.prior_alpha, dtype=np.float64)
        self.posterior_beta  = np.full((B, K), self.prior_beta,  dtype=np.float64)
        self._refresh_moments()
        # for UCB1
        self.counts = np.zeros((self.batch_size, self.num_arms))
        self.rew_sum = np.zeros((self.batch_size, self.num_arms))
        self.empirical_means = np.zeros((self.batch_size, self.num_arms))

    def _refresh_moments(self):
        a, b = self.posterior_alpha, self.posterior_beta
        denom = a + b
        self.posterior_mean = a / np.maximum(denom, 1e-12)
        var = (a * b) / (np.maximum(denom, 1e-12) ** 2 * np.maximum(denom + 1.0, 1e-12))
        self.posterior_std = np.sqrt(np.maximum(var, 1e-18))

    def update_posteriors(self, rewards: np.ndarray, arms: np.ndarray):
        rng = np.arange(self.batch_size)
        self.posterior_alpha[rng, arms] += rewards
        self.posterior_beta[rng, arms]  += (1.0 - rewards)
        self._refresh_moments()
        # for UCB1
        self.counts[rng, arms] += 1
        self.rew_sum[rng, arms] += rewards
        self.empirical_means = self.rew_sum / np.maximum(self.counts, 1e-12)

class UCBPosteriorBernoulliLearner(BernoulliLearner):
    def __init__(self, batch_size: int, bandit: BernoulliBandit, c: float = 4.0, **kwargs):
        super().__init__(batch_size, bandit, **kwargs)
        self.c = float(c)

    def select_arms(self):
        upper = self.posterior_mean + self.c * self.posterior_std
        return np.argmax(upper, axis=1)


class UCB1BernoulliLearner(BernoulliLearner):
    def __init__(self, batch_size: int, bandit: BernoulliBandit, c: float = 4.0, **kwargs):
        super().__init__(batch_size, bandit, **kwargs)
        self.c = float(c)

    def select_arms(self):
        B = self.batch_size
        arms = np.empty(B, dtype=int)
        # --- warm start: まだ0回の腕があるバッチは最少カウントの腕を優先（同数タイは乱択で崩す）
        need_init = (self.counts == 0).any(axis=1)
        if need_init.any():
            sub = self.counts[need_init]                    # (B0, K)
            mins = sub.min(axis=1, keepdims=True)           # (B0, 1)
            mask = (sub == mins)                            # 最少カウント候補
            rand = np.random.rand(*mask.shape); rand[~mask] = -1.0
            arms[need_init] = rand.argmax(axis=1)
        # --- 全腕>=1回のバッチにだけUCB1適用（経験平均＋ボーナス）
        if (~need_init).any():
            idx = np.where(~need_init)[0]
            t = self.counts[idx].sum(axis=1)  # 総試行回数
            bonus = self.c * np.sqrt(
                np.log(t).reshape(-1, 1) / (2 * np.maximum(self.counts[idx], 1e-12))
            )
            upper = self.empirical_means[idx] + bonus
            arms[~need_init] = np.argmax(upper, axis=1)
        return arms



class ThompsonBetaLearner(BernoulliLearner):
    def select_arms(self):
        samples = np.random.beta(self.posterior_alpha, self.posterior_beta)
        return np.argmax(samples, axis=1)


# -------------------- Gaussian family --------------------
class GaussianLearner(_BaseLearner):
    def __init__(self, batch_size: int, bandit: GaussianBandit, prior_mean: float = 0.0, prior_std: float = 1.0):
        super().__init__(batch_size, bandit)
        self.reset_posteriors(prior_mean, prior_std)
        self.noise_std = bandit.noise_std
        self.noise_precision = 1.0 / (self.noise_std ** 2)

    def reset_posteriors(self, mean: float, std: float):
        """Initialize independent Normal priors on arm means.
        Accepts scalar or length-K vectors for mean/std; broadcasts to (B,K).
        """
        # Convert to per-arm vectors of length K
        mean_arr = np.asarray(mean, dtype=np.float64)
        if mean_arr.ndim == 0:
            mean_arr = np.full((self.num_arms,), float(mean_arr))
        else:
            assert mean_arr.shape == (self.num_arms,), "prior_mean must be scalar or length-K"

        std_arr = np.asarray(std, dtype=np.float64)
        if std_arr.ndim == 0:
            std_arr = np.full((self.num_arms,), float(std_arr))
        else:
            assert std_arr.shape == (self.num_arms,), "prior_std must be scalar or length-K"

        # Tile across batch: (B,K)
        self.posterior_mean = np.tile(mean_arr, (self.batch_size, 1))
        self.posterior_std  = np.tile(std_arr, (self.batch_size, 1))
        self.precision = 1.0 / (self.posterior_std ** 2)
        self.weighted_mean = self.posterior_mean * self.precision
        # for UCB1
        self.counts = np.zeros((self.batch_size, self.num_arms))
        self.rew_sum = np.zeros((self.batch_size, self.num_arms))
        self.empirical_means = np.zeros((self.batch_size, self.num_arms))

    def update_posteriors(self, rewards: np.ndarray, arms: np.ndarray):
        rng = np.arange(self.batch_size)
        add_prec = self.noise_precision[arms]
        self.precision[rng, arms] += add_prec
        self.weighted_mean[rng, arms] += rewards * add_prec
        self.posterior_mean = self.weighted_mean / self.precision
        self.posterior_std = np.sqrt(1.0 / self.precision)
        # for UCB1
        self.counts[rng, arms] += 1
        self.rew_sum[rng, arms] += rewards
        self.empirical_means = self.rew_sum / np.maximum(self.counts, 1e-12)


class UCBPosteriorGaussianLearner(GaussianLearner):
    def __init__(self, batch_size: int, bandit: GaussianBandit, c: float = 4.0, **kwargs):
        super().__init__(batch_size, bandit, **kwargs)
        self.c = float(c)

    def select_arms(self):
        upper = self.posterior_mean + self.c * self.posterior_std
        return np.argmax(upper, axis=1)


class UCB1GaussianLearner(GaussianLearner):
    def __init__(self, batch_size: int, bandit: GaussianBandit, c: float = 4.0, **kwargs):
        super().__init__(batch_size, bandit, **kwargs)
        self.c = float(c)

    def select_arms(self):
        B = self.batch_size
        arms = np.empty(B, dtype=int)

        # まだ1回も引いていない腕があるバッチは、その最少カウントの腕を優先（各腕1巡）
        need_init = (self.counts == 0).any(axis=1)
        if need_init.any():
            arms[need_init] = np.argmin(self.counts[need_init], axis=1)

        # 全腕>=1回のバッチだけUCB1を適用
        if (~need_init).any():
            idx = np.where(~need_init)[0]
            total_counts = self.counts[idx].sum(axis=1)  # (= t)
            bonus = self.c * np.sqrt(
                np.log(np.maximum(total_counts.reshape(-1, 1), 1)) /
                (2 * np.maximum(self.counts[idx], 1e-12))
            ) * self.noise_std
            upper = self.empirical_means[idx] + bonus
            arms[~need_init] = np.argmax(upper, axis=1)

        return arms



class ThompsonGaussianLearner(GaussianLearner):
    def select_arms(self):
        samples = np.random.normal(self.posterior_mean, self.posterior_std)
        return np.argmax(samples, axis=1)


class SoftmaxGaussianLearner(GaussianLearner):
    def __init__(self, batch_size: int, bandit: GaussianBandit, prior_mean: float = 0.0, prior_std: float = 1.0, temperature: float = 1.0):
        GaussianLearner.__init__(self, batch_size, bandit, prior_mean=prior_mean, prior_std=prior_std)
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_arms(self):
        posterior_mean = torch.as_tensor(self.posterior_mean, dtype=torch.float32, device=self.device) # (B,K)
        softmax_probs = torch.softmax(posterior_mean / self.temperature, dim=1) # (B,K)
        return batch_choice(softmax_probs.detach().cpu().numpy())


class SoftmaxBetaLearner(BernoulliLearner):
    def __init__(self, batch_size: int, bandit: BernoulliBandit, prior_alpha: float = 1.0, prior_beta: float = 1.0, temperature: float = 1.0):
        BernoulliLearner.__init__(self, batch_size, bandit, prior_alpha=prior_alpha, prior_beta=prior_beta)
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_arms(self):
        posterior_alpha = torch.as_tensor(self.posterior_alpha, dtype=torch.float32, device=self.device) # (B,K)
        posterior_beta = torch.as_tensor(self.posterior_beta, dtype=torch.float32, device=self.device) # (B,K)
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        softmax_probs = torch.softmax(posterior_mean / self.temperature, dim=1) # (B,K)
        return batch_choice(softmax_probs.detach().cpu().numpy())


import numpy as np
from scipy.stats import norm


def gaussian_expected_max_single(m, std):
    # m is a vector of means
    # std is the corresponding vector of stds
    # Output: expected maximum matrix:
    # Diagonal elements are the same as m
    # Off-diagonal elements are E[max(x_i, x_j)] where x_i and x_j are
    # independent Gaussian random variables with mean m_i and m_j and
    # standard deviations std_i and std_j
    # E[max(x_i, x_j)] = m_i * CDF((m_i - m_j) / sqrt(std_i^2 + std_j^2)) +
    #                   m_j * CDF((m_j - m_i) / sqrt(std_i^2 + std_j^2)) +
    #                   sqrt(std_i^2 + std_j^2) * PDF((m_i - m_j) / sqrt(std_i^2 + std_j^2))
    # where CDF and PDF are the CDF and PDF of the standard normal distribution.

    # Compute the difference between all pairs of means
    m = m.reshape(1, -1)
    std = std.reshape(1, -1)
    m_diff = m - m.T
    std_sum = np.sqrt(std**2 + std.T**2)
    theta = std_sum.copy()  # for the last term in the expectation
    # Set diagonal to 0, so that the diagonal elements are the same as m
    # Note that norm.cdf(0) = 0.5, for m_diff=0
    np.fill_diagonal(theta, 0)
    m_diff /= std_sum
    exp_max = m * norm.cdf(m_diff) + m.T * norm.cdf(-m_diff) + theta * norm.pdf(m_diff)

    return exp_max


def gaussian_expected_max(m, std):
    # m is a vector of means
    # std is the corresponding vector of stds
    # Output: expected maximum matrix:
    # Diagonal elements are the same as m
    # Off-diagonal elements are E[max(x_i, x_j)] where x_i and x_j are
    # independent Gaussian random variables with mean m_i and m_j and
    # standard deviations std_i and std_j
    # E[max(x_i, x_j)] = m_i * CDF((m_i - m_j) / sqrt(std_i^2 + std_j^2)) +
    #                   m_j * CDF((m_j - m_i) / sqrt(std_i^2 + std_j^2)) +
    #                   sqrt(std_i^2 + std_j^2) * PDF((m_i - m_j) / sqrt(std_i^2 + std_j^2))
    # where CDF and PDF are the CDF and PDF of the standard normal distribution.

    # Compute the difference between all pairs of means
    if len(m.shape) == 1 or m.shape[0] == 1:
        m = m.reshape(1, -1)
        std = std.reshape(1, -1)
        m_diff = m - m.T
        std_sum = np.sqrt(std**2 + std.T**2)
        theta = std_sum.copy()  # for the last term in the expectation
        # Set diagonal to 0, so that the diagonal elements are the same as m
        # Note that norm.cdf(0) = 0.5, for m_diff=0
        np.fill_diagonal(theta, 0)
        m_diff /= std_sum
        exp_max = (
            m * norm.cdf(m_diff) + m.T * norm.cdf(-m_diff) + theta * norm.pdf(m_diff)
        )
    elif len(m.shape) == 2:
        m = np.expand_dims(m, axis=1)
        std = np.expand_dims(std, axis=1)
        m_diff = m - m.transpose((0, 2, 1))
        std_sum = np.sqrt(std**2 + std.transpose((0, 2, 1)) ** 2)
        theta = std_sum.copy()  # see comment in the previous case
        for i in range(m.shape[0]):
            np.fill_diagonal(theta[i, :, :], 0)
        m_diff /= std_sum
        exp_max = (
            m * norm.cdf(m_diff)
            + m.transpose((0, 2, 1)) * norm.cdf(-m_diff)
            + theta * norm.pdf(m_diff)
        )

    return exp_max


def batch_choice(p, u=None, sample_per_p=1):
    # p is a matrix of probabilities [batch_size, num_arms]
    # Output: a vector of indices [batch_size], chosen according to the
    # probabilities in p
    batch_size = p.shape[0]

    if not isinstance(u, np.ndarray):
        u = np.random.rand(batch_size, sample_per_p)

    cumsum = np.cumsum(p, axis=1)

    # Selects the index of first cumsum to be greater than u
    choices = np.argmax(u < cumsum, axis=1)
    return choices

class ReMaxGaussianK2Learner(GaussianLearner):

    def __init__(
        self, batch_size: int, bandit: Type[Bandit], negative_fixing: bool = False
    ):
        super().__init__(batch_size, bandit)
        self.negative_fixing = negative_fixing
        self.probs = []

    def select_arms(self):
        # Selects the arms to pull
        # Returns a vector of arms
        # Compute the expected maximum
        exp_max = gaussian_expected_max(self.posterior_mean, self.posterior_std)
        # Compute the ReMax policy probabilities
        p = self.compute_remax_policy(exp_max)

        # Remove negative probabilities, and normalize to 1
        # TODO: this should be checked, and seen whether a better solution
        # exists, and whether the current approach for computing ReMax is correct
        p = np.maximum(p, 0)

        p = p / np.sum(p, axis=1, keepdims=True)

        # Sample the arms from the ReMax policy
        arms = batch_choice(p)
        self.probs.append(p)
        return arms

    def compute_remax_policy(self, exp_max):
        # Computes the ReMax policy probabilities by
        # solving an equation of the form a @ p = b
        b = exp_max[:, 0:1, 0] - exp_max[:, 1:, 0]  # was 0:1

        a = (
            exp_max[:, 1:, 1:]
            + exp_max[:, 0:1, 0:1]
            - exp_max[:, 1:, 0:1]
            - exp_max[:, 0:1, 1:]
        )

        p = np.linalg.solve(a, b)

        p = np.concatenate((1 - np.sum(p, axis=1, keepdims=True), p), axis=1)

        # Loop through p, and fix negative probabilities
        if self.negative_fixing:
            for i in range(p.shape[0]):
                counter = 0
                if np.all(p[i, :] >= 0):
                    continue
                curr_p = p[i].copy()
                idx = np.where(curr_p >= 0)[0]
                prev_p = curr_p[idx]
                while np.any(curr_p < 0):  # negative, so fix it
                    counter += 1
                    # indexes of non-negative probabilities
                    idx = idx[np.where(prev_p >= 0)[0]]
                    # select non-negative elements
                    local_exp_max = exp_max[i, idx, :][:, idx]
                    b_local = local_exp_max[0:1, 0] - local_exp_max[1:, 0]
                    a_local = (
                        local_exp_max[1:, 1:]
                        + local_exp_max[0:1, 0:1]
                        - local_exp_max[1:, 0:1]
                        - local_exp_max[0:1, 1:]
                    )
                    p_local = np.linalg.solve(a_local, b_local)
                    curr_p = np.concatenate(((1 - np.sum(p_local),), p_local))
                    prev_p = curr_p
                p[i, :] = 0
                p[i, idx] = curr_p

        return p

# =============================================================
# ReMax-EI (Exact inner objective + autodiff) — family-specific wrappers
# =============================================================
class _ReMaxEIBase(_BaseLearner):
    """Shared implementation of ReMax-EI over families.
    Subclasses must provide `_sample_mu_from_posterior(n_mu) -> torch.Tensor`
    returning shape (B, n_mu, K) of mean rewards μ samples.
    """
    def __init__(self, batch_size: int, bandit, M: int = 2, pg_iter: int = 20):
        super().__init__(batch_size, bandit)
        self.M = int(M)
        self.pg_iter = int(pg_iter)
        self.policy = Policy(num_arms=self.num_arms, batch_size=self.batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.probs = []

    @staticmethod
    def _expected_max_conditional(
        mu: torch.Tensor,  # (B,S,K) or (B,K)
        pi: torch.Tensor,  # (B,K)
        M: int
    ) -> torch.Tensor:
        """Compute J^M_ReMax(π, μ) exactly for each μ (Prop. formula).
        Returns: (B,S) if mu is (B,S,K); (B,) if mu is (B,K).
        """
        squeeze = False
        if mu.dim() == 2:
            mu = mu.unsqueeze(1)  # (B,1,K)
            squeeze = True
        B, S, K = mu.shape
        idx = torch.argsort(mu, dim=-1, descending=True)
        mu_s = torch.gather(mu, 2, idx)                  # (B,S,K)
        pi_exp = pi[:, None, :].expand(B, S, K)
        pi_s = torch.gather(pi_exp, 2, idx)              # (B,S,K)
        S_prefix = torch.cumsum(pi_s, dim=-1)[..., :-1].clamp(0.0, 1.0)
        dmu = mu_s[..., :-1] - mu_s[..., 1:]             # (B,S,K-1)
        term = dmu * torch.pow(1.0 - S_prefix, M)
        J = mu_s[..., 0] - term.sum(dim=-1)              # (B,S)
        return J.squeeze(1) if squeeze else J

    def _sample_mu_from_posterior(self, n_mu: int) -> torch.Tensor:
        """To be implemented by subclasses; must return (B, n_mu, K)."""
        raise NotImplementedError

    def _compute_pg_remax_policy(self, n_mu: int):
        policy = self.policy
        optimizer = optim.Adam(policy.parameters(), lr=0.05)
        logits = None
        for _ in range(self.pg_iter):
            optimizer.zero_grad()
            mu_samples = self._sample_mu_from_posterior(n_mu)      # (B,S,K)
            logits = policy()                                      # (B,K)
            pi = torch.softmax(logits, dim=1)                      # (B,K)
            J_per_mu = self._expected_max_conditional(mu_samples, pi, self.M)  # (B,S)
            J = J_per_mu.mean(dim=1)                               # (B,)
            loss = -J.sum()
            loss.backward()
            optimizer.step()
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy() # (B,K)
        return probs

    # Learn-loop hooks
    def select_arms(self):
        # Use moderate outer MC (n_mu) for efficiency; can tune via ctor in subclasses
        p = self._compute_pg_remax_policy(self.n_mu)
        arms = batch_choice(p)
        self.probs.append(p)
        return arms


class ReMaxEIBetaLearner(_ReMaxEIBase, BernoulliLearner):
    def __init__(self, batch_size: int, bandit: BernoulliBandit, M: int = 2, pg_iter: int = 20,
                 n_mu: int = 16, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        BernoulliLearner.__init__(self, batch_size, bandit, prior_alpha=prior_alpha, prior_beta=prior_beta)
        _ReMaxEIBase.__init__(self, batch_size, bandit, M=M, pg_iter=pg_iter)
        self.n_mu = int(n_mu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _sample_mu_from_posterior(self, n_mu: int) -> torch.Tensor:
        alpha = torch.as_tensor(self.posterior_alpha, dtype=torch.float32, device=self.device)
        beta  = torch.as_tensor(self.posterior_beta,  dtype=torch.float32, device=self.device)
        dist = torch.distributions.Beta(alpha, beta)
        samples = dist.sample((n_mu,))                  # (S,B,K)
        return samples.permute(1, 0, 2).contiguous()    # (B,S,K)


class ReMaxEIGaussianLearner(_ReMaxEIBase, GaussianLearner):
    def __init__(self, batch_size: int, bandit: GaussianBandit, M: int = 2, pg_iter: int = 20,
                 n_mu: int = 16, prior_mean: float = 0.0, prior_std: float = 1.0):
        GaussianLearner.__init__(self, batch_size, bandit, prior_mean=prior_mean, prior_std=prior_std)
        _ReMaxEIBase.__init__(self, batch_size, bandit, M=M, pg_iter=pg_iter)
        self.n_mu = int(n_mu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _sample_mu_from_posterior(self, n_mu: int) -> torch.Tensor:
        mean = torch.as_tensor(self.posterior_mean, dtype=torch.float32, device=self.device)
        std  = torch.as_tensor(self.posterior_std,  dtype=torch.float32, device=self.device)
        dist = torch.distributions.Normal(mean, std)
        samples = dist.sample((n_mu,))                  # (S,B,K)
        return samples.permute(1, 0, 2).contiguous()    # (B,S,K)


# =============================================================
# Experiment driver
# =============================================================

# ---- Plot style ----
line_width: float = 1.8
line_alpha: float = 0.85
fill_alpha: float = 0.18
grid_alpha: float = 0.25
marker_interval_ratio: float = 0.1

# LaTeX-consistent fonts (Computer Modern via mathtext; no system TeX required)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# tab10 palette assignment per method (consistent across plots)
# tab10 indices: 0 blue, 1 orange, 2 green, 3 red, 4 purple,
#                5 brown, 6 pink, 7 gray, 8 olive, 9 cyan
_TAB_PALETTE = plt.get_cmap("tab10").colors
METHOD_STYLE = {
    "Softmax": {"color": _TAB_PALETTE[7], "marker": "*", "label": r"$\mathrm{Softmax}$"},
    "TS":      {"color": _TAB_PALETTE[0], "marker": "o", "label": r"$\mathrm{TS}$"},
    "UCB":     {"color": _TAB_PALETTE[1], "marker": "s", "label": r"$\mathrm{UCB}$"},
    "M=2":     {"color": _TAB_PALETTE[2], "marker": "^", "label": r"$M=2$"},
    "M=3":     {"color": _TAB_PALETTE[3], "marker": "D", "label": r"$M=3$"},
    "M=4":     {"color": _TAB_PALETTE[4], "marker": "v", "label": r"$M=4$"},
    "M=5":     {"color": _TAB_PALETTE[5], "marker": "P", "label": r"$M=5$"},
    "M=6":     {"color": _TAB_PALETTE[6], "marker": "X", "label": r"$M=6$"},
    "M=7":     {"color": _TAB_PALETTE[8], "marker": "h", "label": r"$M=7$"},
    "M=8":     {"color": _TAB_PALETTE[9], "marker": "p", "label": r"$M=8$"},
    "M=9":     {"color": "tab:olive",      "marker": "d", "label": r"$M=9$"},
}


def _plot(regret_history: np.ndarray, label: str):
    mean = np.mean(regret_history, axis=0)
    steps = np.arange(mean.shape[0])
    stderr = np.std(regret_history, axis=0) / np.sqrt(regret_history.shape[0])

    style = METHOD_STYLE.get(label, {"color": None, "marker": None, "label": label})
    color = style["color"]
    marker = style["marker"]
    display_label = style.get("label", label)
    markevery = max(int(mean.shape[0] * marker_interval_ratio), 1)

    plt.plot(
        steps,
        mean,
        label=display_label,
        color=color,
        marker=marker,
        markevery=markevery,
        linewidth=line_width,
        alpha=line_alpha,
    )
    plt.fill_between(steps, mean - stderr, mean + stderr, color=color, alpha=fill_alpha, linewidth=0)


def run_experiment(
    family: str,
    num_pulls: int = 500,
    num_bandit_instances: int = 256,
    num_arms: int = 10,
    ucb_c: float = 1.0,
    softmax_temperature: float = 1.0,
    # Beta priors
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    # Gaussian priors/noise
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    noise_std: float = 1.0,
    # ReMax-EI hyperparams
    remax_Ms: list[int] = [2, 3],
    remax_pg_iter: int = 5,
    remax_n_mu: int = 50,
):

    if family == "beta":
        bandit = BernoulliBandit(
            num_arms=num_arms,
            num_bandit_instances=num_bandit_instances,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )
        learners = {
            "Softmax": SoftmaxBetaLearner(
                num_bandit_instances, bandit, prior_alpha=prior_alpha, prior_beta=prior_beta,
                temperature=softmax_temperature,
            ),
            "TS": ThompsonBetaLearner(
                num_bandit_instances, bandit, prior_alpha=prior_alpha, prior_beta=prior_beta
            ),
            "UCB": UCB1BernoulliLearner(
                num_bandit_instances, bandit, c=ucb_c, prior_alpha=prior_alpha, prior_beta=prior_beta
            ),
            **{f"M={remax_M}": ReMaxEIBetaLearner(
                num_bandit_instances, bandit, M=remax_M, pg_iter=remax_pg_iter,
                n_mu=remax_n_mu, prior_alpha=prior_alpha, prior_beta=prior_beta
            ) for remax_M in remax_Ms},
        }
        title = r"(A) Beta--Bernoulli"
        outfile = f"fig/regret_beta_instances_{num_bandit_instances}_K_{num_arms}_a{prior_alpha}_b{prior_beta}.pdf"

    elif family == "gaussian":
        bandit = GaussianBandit(
            num_arms=num_arms,
            num_bandit_instances=num_bandit_instances,
            mean=prior_mean,
            std=prior_std,
            noise_std=noise_std,
        )
        learners = {
            "Softmax": SoftmaxGaussianLearner(
                num_bandit_instances, bandit, prior_mean=prior_mean, prior_std=prior_std,
                temperature=softmax_temperature,
            ),
            "TS": ThompsonGaussianLearner(
                num_bandit_instances, bandit, prior_mean=prior_mean, prior_std=prior_std
            ),
            "UCB": UCB1GaussianLearner(
                num_bandit_instances, bandit, c=ucb_c, prior_mean=prior_mean, prior_std=prior_std
            ),
            **{f"M={remax_M}": ReMaxEIGaussianLearner(
                num_bandit_instances, bandit, M=remax_M, pg_iter=remax_pg_iter,
                n_mu=remax_n_mu, prior_mean=prior_mean, prior_std=prior_std
            ) for remax_M in remax_Ms},
        }
        title = r"(B) Gaussian--Gaussian"
        outfile = (
            f"fig/regret_gaussian_instances_{num_bandit_instances}_K_{num_arms}_pm{prior_mean}_ps{prior_std}_"
            f"noise{noise_std}.pdf"
        )
    else:
        raise ValueError("family must be one of {'beta','gaussian'}")

    os.makedirs("fig", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    last_regrets = []
    for i, (name, learner) in enumerate(learners.items()):
        learner.learn(num_pulls)
        print(f"Learner {name} done with last regret {learner.regret_history[:, -1].mean()}")
        last_regrets.append(learner.regret_history[:, -1].mean())
        _plot(learner.regret_history, name)


    # last_regretのmaxに一番近い10の倍数
    max_last_regret = max(last_regrets)
    max_y_tick = np.ceil(max_last_regret / 10) * 10
    if (max_y_tick // 10) % 2 != 0:
        max_y_tick += 10
    yticks = [0, max_y_tick/2, max_y_tick]
    plt.legend(fontsize=15)

    plt.xticks([0, num_pulls/2, num_pulls], fontsize=20)
    plt.yticks(yticks, fontsize=20)
    plt.xlabel(r"Round $t$", fontsize=22)
    plt.ylabel(r"Cumulative Regret", fontsize=22)
    plt.title(title, fontsize=22)
    plt.grid(True, alpha=grid_alpha)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", format="pdf")
    print("Saved:", outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="beta", choices=["beta", "gaussian"])
    parser.add_argument("--num_pulls", type=int, default=1000)
    parser.add_argument("--num_bandit_instances", type=int, default=256)
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--ucb_c", type=float, default=1.0)
    parser.add_argument("--softmax_temperature", type=float, default=0.1)

    # Beta
    parser.add_argument("--prior_alpha", type=float, default=1.0)
    parser.add_argument("--prior_beta", type=float, default=1.0)

    # Gaussian
    parser.add_argument("--prior_mean", type=float, default=0.0)
    parser.add_argument("--prior_std", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=1.0)

    # ReMax-EI
    parser.add_argument("--remax_Ms", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--remax_pg_iter", type=int, default=50)
    parser.add_argument("--remax_n_mu", type=int, default=16)

    args = parser.parse_args()

    run_experiment(
        family=args.family,
        num_pulls=args.num_pulls,
        num_bandit_instances=args.num_bandit_instances,
        num_arms=args.num_arms,
        ucb_c=args.ucb_c,
        softmax_temperature=args.softmax_temperature,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        prior_mean=args.prior_mean,
        prior_std=args.prior_std,
        noise_std=args.noise_std,
        remax_Ms=args.remax_Ms,
        remax_pg_iter=args.remax_pg_iter,
        remax_n_mu=args.remax_n_mu,
    )


if __name__ == "__main__":
    main()

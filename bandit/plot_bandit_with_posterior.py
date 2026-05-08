from __future__ import annotations

"""
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
"""

from typing import Optional

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


def row_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = x / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


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
# Learners
# =============================================================
class _BaseLearner:
    def __init__(self, batch_size: int, bandit):
        self.batch_size = batch_size
        self.bandit = bandit
        assert (
            self.batch_size == bandit.num_bandit_instances or bandit.num_bandit_instances == 1
        )
        self.num_arms = bandit.num_arms
        # for UCB1
        self.counts = np.zeros((self.batch_size, self.num_arms))
        self.rew_sum = np.zeros((self.batch_size, self.num_arms))
        self.empirical_means = np.zeros((self.batch_size, self.num_arms))

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
        rng = np.arange(self.batch_size)
        self._update_distribution_posteriors(rewards, arms)
        # for UCB1
        self.counts[rng, arms] += 1
        self.rew_sum[rng, arms] += rewards
        self.empirical_means = self.rew_sum / np.maximum(self.counts, 1e-12)

    def _update_distribution_posteriors(self, rewards: np.ndarray, arms: np.ndarray):
        raise NotImplementedError

    def sample_from_posterior(self, n_samples: int = 1):
        raise NotImplementedError

    @property
    def ucb_bonus_scale(self):
        return 1.0

    @staticmethod
    def _format_samples(samples, n_samples: int):
        if n_samples == 1:
            return samples
        return np.transpose(samples, (1, 0, 2))

    def select_arms(self) -> np.ndarray:
        raise NotImplementedError


# -------------------- Posterior families --------------------
class BetaPosteriorLearner(_BaseLearner):
    def __init__(
        self,
        batch_size: int,
        bandit: BernoulliBandit,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        **kwargs,
    ):
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        super().__init__(batch_size, bandit, **kwargs)
        self.reset_posteriors()

    def reset_posteriors(self):
        B, K = self.batch_size, self.num_arms
        self.posterior_alpha = np.full((B, K), self.prior_alpha, dtype=np.float64)
        self.posterior_beta = np.full((B, K), self.prior_beta, dtype=np.float64)
        self._refresh_moments()

    def _refresh_moments(self):
        a, b = self.posterior_alpha, self.posterior_beta
        denom = a + b
        self.posterior_mean = a / np.maximum(denom, 1e-12)
        var = (a * b) / (np.maximum(denom, 1e-12) ** 2 * np.maximum(denom + 1.0, 1e-12))
        self.posterior_std = np.sqrt(np.maximum(var, 1e-18))

    def _update_distribution_posteriors(self, rewards: np.ndarray, arms: np.ndarray):
        rng = np.arange(self.batch_size)
        self.posterior_alpha[rng, arms] += rewards
        self.posterior_beta[rng, arms] += (1.0 - rewards)
        self._refresh_moments()

    def sample_from_posterior(self, n_samples: int = 1):
        if n_samples == 1:
            return np.random.beta(self.posterior_alpha, self.posterior_beta)
        samples = np.random.beta(
            self.posterior_alpha[None, :, :],
            self.posterior_beta[None, :, :],
            size=(n_samples, self.batch_size, self.num_arms),
        )
        return self._format_samples(samples, n_samples)


class GaussianPosteriorLearner(_BaseLearner):
    def __init__(
        self,
        batch_size: int,
        bandit: GaussianBandit,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        **kwargs,
    ):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.noise_std = bandit.noise_std
        self.noise_precision = 1.0 / (self.noise_std ** 2)
        super().__init__(batch_size, bandit, **kwargs)
        self.reset_posteriors()

    def _as_arm_array(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full((self.num_arms,), float(arr))
        assert arr.shape == (self.num_arms,), f"{name} must be scalar or length-K"
        return arr

    def reset_posteriors(self):
        """Initialize independent Normal priors on arm means.
        Accepts scalar or length-K vectors for mean/std; broadcasts to (B,K).
        """
        # Convert to per-arm vectors of length K
        mean_arr = self._as_arm_array(self.prior_mean, "prior_mean")
        std_arr = self._as_arm_array(self.prior_std, "prior_std")

        # Tile across batch: (B,K)
        self.posterior_mean = np.tile(mean_arr, (self.batch_size, 1))
        self.posterior_std = np.tile(std_arr, (self.batch_size, 1))
        self.precision = 1.0 / (self.posterior_std ** 2)
        self.weighted_mean = self.posterior_mean * self.precision

    def _update_distribution_posteriors(self, rewards: np.ndarray, arms: np.ndarray):
        rng = np.arange(self.batch_size)
        add_prec = self.noise_precision[arms]
        self.precision[rng, arms] += add_prec
        self.weighted_mean[rng, arms] += rewards * add_prec
        self.posterior_mean = self.weighted_mean / self.precision
        self.posterior_std = np.sqrt(1.0 / self.precision)

    def sample_from_posterior(self, n_samples: int = 1):
        if n_samples == 1:
            return np.random.normal(self.posterior_mean, self.posterior_std)
        samples = np.random.normal(
            loc=self.posterior_mean[None, :, :],
            scale=self.posterior_std[None, :, :],
            size=(n_samples, self.batch_size, self.num_arms),
        )
        return self._format_samples(samples, n_samples)

    @property
    def ucb_bonus_scale(self):
        return self.noise_std


# -------------------- Algorithm learners --------------------
class UCB1Learner:
    def __init__(self, batch_size: int, bandit, c: float = 4.0, **kwargs):
        self.c = float(c)
        super().__init__(batch_size, bandit, **kwargs)

    def select_arms(self):
        B = self.batch_size
        arms = np.empty(B, dtype=int)
        # --- warm start: warm start: if there are arms with 0 pulls, choose the arm with the least pulls (break ties randomly)
        need_init = (self.counts == 0).any(axis=1)
        if need_init.any():
            sub = self.counts[need_init]                    # (B0, K)
            mins = sub.min(axis=1, keepdims=True)           # (B0, 1)
            mask = (sub == mins)                            # least count candidates
            rand = np.random.rand(*mask.shape)
            rand[~mask] = -1.0
            arms[need_init] = rand.argmax(axis=1)
        # --- apply UCB1 only to batches with all arms>=1 pull (empirical mean + bonus)
        if (~need_init).any():
            idx = np.where(~need_init)[0]
            t = self.counts[idx].sum(axis=1)  # total number of pulls
            bonus = self.c * np.sqrt(
                np.log(np.maximum(t, 1.0)).reshape(-1, 1) / (2 * np.maximum(self.counts[idx], 1e-12))
            )
            upper = self.empirical_means[idx] + bonus * self.ucb_bonus_scale
            arms[~need_init] = np.argmax(upper, axis=1)
        return arms


class ThompsonSamplingLearner:
    def select_arms(self):
        samples = self.sample_from_posterior()
        return np.argmax(samples, axis=1)


class SoftmaxLearner:
    def __init__(self, batch_size: int, bandit, temperature: float = 1.0, **kwargs):
        self.temperature = float(temperature)
        super().__init__(batch_size, bandit, **kwargs)

    def select_arms(self):
        return batch_choice(row_softmax(self.posterior_mean, self.temperature))


class ReMaxGradLearner:
    """Shared implementation of ReMax-Grad over families.
    Subclasses must provide `sample_from_posterior(n_mu, as_tensor=True)`
    returning shape (B, n_mu, K) of mean rewards μ samples.
    """
    def __init__(self, batch_size: int, bandit, M: int = 2, pg_iter: int = 20, n_mu: int = 16, **kwargs):
        self.M = int(M)
        self.pg_iter = int(pg_iter)
        self.n_mu = int(n_mu)
        super().__init__(batch_size, bandit, **kwargs)
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

    def _compute_pg_remax_policy(self, n_mu: int):
        policy = self.policy
        optimizer = optim.Adam(policy.parameters(), lr=0.05)
        logits = policy()
        for _ in range(self.pg_iter):
            optimizer.zero_grad()
            mu_samples = torch.as_tensor(
                self.sample_from_posterior(n_mu), dtype=torch.float32, device=self.device
            )  # (B,S,K)
            logits = policy()                                                                  # (B,K)
            pi = torch.softmax(logits, dim=1)                                                  # (B,K)
            J_per_mu = self._expected_max_conditional(mu_samples, pi, self.M)                  # (B,S)
            J = J_per_mu.mean(dim=1)                                                           # (B,)
            loss = -J.sum()
            loss.backward()
            optimizer.step()
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()  # (B,K)
        return probs

    # Learn-loop hooks
    def select_arms(self):
        # Use moderate outer MC (n_mu) for efficiency; can tune via ctor in subclasses
        p = self._compute_pg_remax_policy(self.n_mu)
        arms = batch_choice(p)
        self.probs.append(p)
        return arms


# -------------------- Family x algorithm composition --------------------
def _compose_learner(name: str, algorithm_cls, posterior_cls):
    return type(name, (algorithm_cls, posterior_cls), {})


def _family_spec(bandit_cls, posterior_cls, title: str, outfile_fn):
    return {
        "bandit_cls": bandit_cls,
        "title": title,
        "outfile_fn": outfile_fn,
        "learner_classes": {
            "Softmax": _compose_learner(f"Softmax{posterior_cls.__name__}", SoftmaxLearner, posterior_cls),
            "TS": _compose_learner(f"Thompson{posterior_cls.__name__}", ThompsonSamplingLearner, posterior_cls),
            "UCB": _compose_learner(f"UCB1{posterior_cls.__name__}", UCB1Learner, posterior_cls),
            "ReMax": _compose_learner(f"ReMaxGrad{posterior_cls.__name__}", ReMaxGradLearner, posterior_cls),
        },
    }


FAMILY_SPECS = {
    "beta": _family_spec(
        BernoulliBandit,
        BetaPosteriorLearner,
        r"(A) Beta--Bernoulli",
        lambda n, k, a, b, pm, ps, ns: f"fig/regret_beta_instances_{n}_K_{k}_a{a}_b{b}.pdf",
    ),
    "gaussian": _family_spec(
        GaussianBandit,
        GaussianPosteriorLearner,
        r"(B) Gaussian--Gaussian",
        lambda n, k, a, b, pm, ps, ns: f"fig/regret_gaussian_instances_{n}_K_{k}_pm{pm}_ps{ps}_noise{ns}.pdf",
    ),
}


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

    if args.family not in FAMILY_SPECS:
        raise ValueError("family must be one of {'beta','gaussian'}")
    spec = FAMILY_SPECS[args.family]

    if args.family == "beta":
        bandit = spec["bandit_cls"](args.num_arms, args.num_bandit_instances, args.prior_alpha, args.prior_beta)
        learner_kwargs = {"prior_alpha": args.prior_alpha, "prior_beta": args.prior_beta}
    else:
        bandit = spec["bandit_cls"](args.num_arms, args.num_bandit_instances, args.prior_mean, args.prior_std, args.noise_std)
        learner_kwargs = {"prior_mean": args.prior_mean, "prior_std": args.prior_std}

    title = spec["title"]
    outfile = spec["outfile_fn"](
        args.num_bandit_instances, args.num_arms, args.prior_alpha, args.prior_beta, args.prior_mean, args.prior_std, args.noise_std
    )
    learner_classes = spec["learner_classes"]

    learners = {
        "Softmax": learner_classes["Softmax"](
            args.num_bandit_instances, bandit, temperature=args.softmax_temperature, **learner_kwargs
        ),
        "TS": learner_classes["TS"](args.num_bandit_instances, bandit, **learner_kwargs),
        "UCB": learner_classes["UCB"](args.num_bandit_instances, bandit, c=args.ucb_c, **learner_kwargs),
        **{
            f"M={remax_M}": learner_classes["ReMax"](
                args.num_bandit_instances,
                bandit,
                M=remax_M,
                pg_iter=args.remax_pg_iter,
                n_mu=args.remax_n_mu,
                **learner_kwargs,
            )
            for remax_M in args.remax_Ms
        },
    }

    os.makedirs("fig", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    last_regrets = []
    for name, learner in learners.items():
        learner.learn(args.num_pulls)
        print(f"Learner {name} done with last regret {learner.regret_history[:, -1].mean()}")
        last_regrets.append(learner.regret_history[:, -1].mean())
        _plot(learner.regret_history, name)


    max_last_regret = max(last_regrets)
    max_y_tick = np.ceil(max_last_regret / 10) * 10
    if (max_y_tick // 10) % 2 != 0:
        max_y_tick += 10
    yticks = [0, max_y_tick/2, max_y_tick]
    plt.legend(fontsize=15)

    plt.xticks([0, args.num_pulls/2, args.num_pulls], fontsize=20)
    plt.yticks(yticks, fontsize=20)
    plt.xlabel(r"Round $t$", fontsize=22)
    plt.ylabel(r"Cumulative Regret", fontsize=22)
    plt.title(title, fontsize=22)
    plt.grid(True, alpha=grid_alpha)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight", format="pdf")
    print("Saved:", outfile)

if __name__ == "__main__":
    main()

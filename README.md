# Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying".

## Bandit Experiments
In `bandit/`, we implement the bandit experiments in the paper.

Binary bandit plot:
```bash
python bandit/plot_binary_bandit.py
```

Bernoulli bandit plot:
```bash
python bandit/plot_bernoulli_bandit_curve.py
```

Fixed binary bandit plot:
```bash
python bandit/plot_fixed_binary_bandit.py
```

Regret plot:
```bash
python bandit/plot_bandit_with_posterior.py --family beta  # for Beta-Bernoulli
python bandit/plot_bandit_with_posterior.py --family gaussian # for Gaussian-Gaussian
```

## RL Experiments
In `rl/`, we implement the RL experiments in the paper.

The main experiments:
```bash
python rl/run.sh
```


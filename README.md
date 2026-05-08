# Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying".



## ReMax objective



## Setup
Please make sure you have installed proper GPU compatible JAX in your environment.

```bash
uv sync
```

For Atari, for the compatibility to the envpool, we recommend to build the docker image with [Dockerfile](./agents/atari/Dockerfile).


## Reproduce the results in the paper

### Bandit Experiments
In `bandit/`, we implement the bandit experiments in the paper.


```bash
python bandit/plot_binary_bandit.py  # Binary bandit plot (Figure 1 (left))
python bandit/plot_scaled_bernoulli_bandit.py  # Bernoulli bandit plot (Figure 1 (center))
python bandit/plot_fixed_binary_bandit.py  # Fixed binary bandit plot (Figure 1 (right))

python bandit/plot_bandit_with_posterior.py --family beta  # for Beta-Bernoulli regret plot (Figure 2 (left))
python bandit/plot_bandit_with_posterior.py --family gaussian # for Gaussian-Gaussian regret plot (Figure 2 (right))
```


### RL Experiments
In [`agents/`](./agents/), we implement the RL experiments in the paper.

The main experiments:
```bash
python rl/run.sh
```


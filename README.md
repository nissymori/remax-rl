# Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying (ReMax)



## ReMax objective

Exploration is the behavior of trying actions that we believe may be promising, in expectation of higher returns.
From this perspective, we argue that exploration matters because we are $\color{#C44E52}{\textsf{uncertain}}$ about the return and are allowed to $\color{#4678C8}{\textsf{retry}}$.

If the returns were known perfectly, the problem would reduce to pure optimization.
Likewise, if no retry were allowed, the only rational choice would be the action currently believed to be best (e.g., what would you choose for your last supper?).
We instantiate this intuition as an objective for RL, which we call **ReMax**.


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
You can run the experiments by running files in [`sh/`](./sh/).

At `sh/`
```bash
./run_minatar.sh  # for MinAtar
./run_atari.sh  # for Atari
./run_craftax.sh  # for CraftAX
```


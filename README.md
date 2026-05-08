# Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying (ReMax)


<p align="center">
  <img src="assets/math.svg" alt="ReMax objective" />
</p>


We argue that exploration matters because we are $\color{#C44E52}{\text{uncertain}}$ about the return and are allowed to $\color{#4678C8}{\textbf{retry}}$.

- If no uncertainty, the problem would reduce to pure optimization.
- If no retry, we should choose the current best action.

We instantiate this intuition as an objective for RL, which we call **ReMax**, where we assume $\color{#C44E52}{\text{distribution over the return}}$ and measure the $\color{#4678C8}{\textbf{best of M retries}}$.

ReMax adaptively explores to the uncertainty and its intensity can be controlled by the number of retries $M$.

## Setup
Please make sure you have installed proper GPU compatible JAX in your environment.

```bash
uv sync
```

For Atari, for the compatibility to the envpool, we recommend to build the docker image with [agents/atari/Dockerfile](./agents/atari/Dockerfile).


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
./run_craftax.sh  # for Craftax
```


# Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying (ReMax)



## ReMax objective

Exploration is a behavior of taking actions that we think is the most promising, expecting the higher return.
In this perspective, we argue that exploration matters because we are $\textcolor{softred}{uncertain}$ about the return and are allowed to $\textcolor{softblue}{retry}$.

If we are perfectly knowledgeable about the return, it is just an pure optimization problem.
Also, if we are not allowed to retry, only the rational choice is the best action we think so far (e.g. what should be you last supper?).

We instanciate this intuition into the objective function of RL, we call **ReMax**.


$$
\begin{aligned}
J_{\mathrm{RL}}(\pi)
&=
\mathbb{E}_{A\sim\pi}[\mu_A]
\\[1.2em]
J_{\mathrm{ReMax}}^{M}(\pi)
&=
\underbrace{\mathbb{E}_{\color{softred}{\mu\sim\Pi}}}_{\text{uncertainty}}
\left[
\underbrace{
\mathbb{E}_{\color{softblue}{A_1,\dots,A_M\sim\pi}}
\left[
\textcolor{softblue}{\max_{m\in[M]} \mu_{A_m}}
\,\middle|\, \mu
\right]
}_{\text{best of $M$ retries}}
\right]
\end{aligned}
$$

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


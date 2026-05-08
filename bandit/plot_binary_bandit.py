import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d


class BanditConfig(BaseModel):
    # General parameters
    project: str = "remax-bandit"
    seed: int = 0
    bias_prob: float = 0.75  # Probability of the bandit giving a reward
    K: int = 2


conf_dict = OmegaConf.from_cli()
config = BanditConfig(**conf_dict)

def exact_remax_return(prob, bias_prob, K):
    """
    Exact ReMax return J(p; K) without Monte Carlo
    prob: policy probability p of choosing arm=1
    bias_prob: environment probability of correct arm being 1
    K: number of samples
    """
    pi = bias_prob
    return pi * (1 - (1 - prob)**K) + (1 - pi) * (1 - prob**K)


def opt_remax_bias_exact(bias_prob, K, num_points=201):
    prob = np.linspace(0.0, 1.0, num_points)
    max_rewards = exact_remax_return(prob, bias_prob, K)
    return prob[np.argmax(max_rewards)], max_rewards


os.makedirs("fig", exist_ok=True)

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

plt.figure(figsize=(8,4))
for i, K in enumerate([1, 2, 3, 4, 5]):
    prob = np.linspace(0.0, 1.0, 101)
    opt_prob, max_rewards = opt_remax_bias_exact(config.bias_prob, K, num_points=101)
    print("K", opt_prob, max(max_rewards))


    if K == 1:
        plt.plot(prob, max_rewards, label=f"M={K}", alpha=0.75, linewidth=8)
    else:
        plt.plot(prob, max_rewards, label=f"{K}", alpha=0.75, linewidth=8)

    # optimal point
    plt.scatter(opt_prob, max(max_rewards), edgecolors="black",
                s=300, zorder=3)


plt.legend(
    fontsize=32,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.05),  # slightly above
    ncol=5,
    frameon=False,
    columnspacing=0.5,
    handlelength=0.6
)

ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.yticks([0.3, 0.7, 1.0], fontsize=25)
plt.xticks([0.0, 0.5, 1.0], fontsize=25)
plt.xlabel("P(a=1)", fontsize=40)
plt.ylabel(r"$J^M_{\mathrm{ReMax}}$", fontsize=40)
plt.savefig("fig/remax_return_for_different_M.pdf", dpi=300, bbox_inches="tight")
plt.close()

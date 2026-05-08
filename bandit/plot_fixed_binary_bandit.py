from matplotlib import pyplot as plt
import numpy as np
import argparse

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

def main():
    fig, ax = plt.subplots(figsize=(8, 4))
    p = np.linspace(0, 1, 200)
    remax0 = 1 - (1 - p) ** 0.5
    remax1 = 1 - (1 - p) ** 0.75
    remax2 = 1 - (1 - p) ** 1
    remax3 = 1 - (1 - p) ** 1.5
    remax4 = 1 - (1 - p) ** 2

    plt.plot(p, remax0, label='m=0.5', linewidth=5)
    plt.plot(p, remax1, label='0.75', linewidth=5)
    plt.plot(p, remax2, label='1', linewidth=5)
    plt.plot(p, remax3, label='1.5', linewidth=5)
    plt.plot(p, remax4, label='2', linewidth=5)
    plt.xticks([0, 0.5, 1], fontsize=25)
    plt.yticks([0, 0.5, 1], fontsize=25)
    plt.xlabel("P(a=1)", fontsize=40)
    plt.ylabel(r"$J^m_{\mathrm{ReMax}}$", fontsize=45)
    #plt.title("ReMax Return for different m for fixed bias")
    plt.legend(
        fontsize=30,
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
    plt.savefig("fig/remax_return_for_different_M_fixed_bias.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
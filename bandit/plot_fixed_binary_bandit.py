from matplotlib import pyplot as plt
import numpy as np
import argparse

def main():
    fig, ax = plt.subplots(figsize=(7, 4))
    p = np.linspace(0, 1, 200)
    remax0 = 1 - (1 - p) ** 0.5
    remax1 = 1 - (1 - p) ** 0.75
    remax2 = 1 - (1 - p) ** 1
    remax3 = 1 - (1 - p) ** 1.5
    remax4 = 1 - (1 - p) ** 2

    plt.plot(p, remax0, label='m=0.5', linewidth=2.5)
    plt.plot(p, remax1, label='m=0.75', linewidth=2.5)
    plt.plot(p, remax2, label='m=1', linewidth=2.5)
    plt.plot(p, remax3, label='m=1.5', linewidth=2.5)
    plt.plot(p, remax4, label='m=2', linewidth=2.5)
    plt.xticks([0, 0.5, 1], fontsize=25)
    plt.yticks([0, 0.5, 1], fontsize=25)
    plt.xlabel("P(a=1)", fontsize=25)
    plt.ylabel("ReMax Return", fontsize=25)
    plt.title("ReMax Return for different m for fixed bias")
    plt.legend(fontsize=13)
    plt.savefig("fig/remax_return_for_different_M_fixed_bias.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
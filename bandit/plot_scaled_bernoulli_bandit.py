import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("fig", exist_ok=True)

def remax_exact_K2(probs: np.ndarray, scales: np.ndarray):
    """
    - 2-arm only, exact calculation of optimal probabilities for ReMax (K=2)
    - r \sim s * Bernoulli(p)

    Args:
        probs: np.ndarray shape (2,) success probabilities of each arm
        scales: np.ndarray shape (2,) scales of each arm

    Returns:
        probs_out: np.ndarray shape (2,) optimal policy probabilities
        best_objective: float, maximum expected value
    """
    assert probs.shape == (2,) and scales.shape == (2,), "2-arm only"
    p_0 = probs[0]
    p_1 = probs[1]
    s_0 = scales[0]  # scale of arm 0
    s_1 = scales[1]  # scale of arm 1

    # 4 events: E1: r=(s_0, 0), E2:r=(0, s_1), E3:r=(s_0, s_1), E4: r=(0, 0)

    p_e_1 = p_0 * (1-p_1)
    p_e_2 = (1-p_0) * p_1
    p_e_3 = p_0 * p_1
    p_e_4 = (1-p_0) * (1-p_1)

    r_e_1 = np.array([s_0, 0])
    r_e_2 = np.array([0, s_1])
    r_e_3 = np.array([s_0, s_1])
    r_e_4 = np.array([0, 0])

    def remax_return(r, pi):
      '''
      pi = p(a=0)
      J^2_ReMax(pi; s_0, s_1) = pi^2 * r_0 + 2*pi*(1-pi)*r_max + (1-pi)^2 * r_1
      where r_max = max(r_0, r_1)
      '''
      return pi*pi*r[0] + 2*pi*(1-pi)*r.max() + (1-pi)*(1-pi)*r[1]

    def expected_remax(pi):
      return p_e_1 * remax_return(r_e_1, pi) + p_e_2 * remax_return(r_e_2, pi) + p_e_3 * remax_return(r_e_3, pi) + p_e_4 * remax_return(r_e_4, pi)


    candidates = np.linspace(0, 1, 5000)
    p_opt_idx = np.argmax(np.array([expected_remax(p) for p in candidates]))
    p_opt = candidates[p_opt_idx]
    return np.array([p_opt, 1.0 - p_opt])


arm_0_probs = 1
scales_list = np.linspace(1, 10, 1000)
arm_0_scale = 2
arm_1_avg = 1

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

fig, ax = plt.subplots(figsize=(8, 4))
opt_remax_probs_list = []
softmax_probs_list = []
for scale_1 in scales_list:
    p_1 = arm_1_avg / scale_1
    probs = np.array([arm_0_probs, p_1])
    scales = np.array([arm_0_scale, scale_1])
    opt_remax_probs = remax_exact_K2(probs, scales)
    softmax_probs = np.exp(np.array([arm_0_scale * arm_0_probs, scale_1 * p_1]))
    softmax_probs = softmax_probs / np.sum(softmax_probs)
    opt_remax_probs_list.append(opt_remax_probs[1])
    softmax_probs_list.append(softmax_probs[1])


plt.plot(scales_list, softmax_probs_list, label="SoftMax", linewidth=8)
plt.plot(scales_list, opt_remax_probs_list, label="ReMax(M=2)", linewidth=8)
plt.xticks(fontsize=25)
plt.yticks([0, 0.2, 0.4], fontsize=25)
plt.xticks(fontsize=20)
plt.xlabel(r"Scale($\alpha_1$)", fontsize=40)
plt.ylabel(r"$\pi^*(a=1)$", fontsize=40)
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.legend(fontsize=20, loc="lower right")
plt.savefig("fig/bernoulli_bandit_curve.pdf", dpi=300, bbox_inches="tight")
plt.close()






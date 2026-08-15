from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
DIMENSIONS = tuple(range(16, 257, 16))

METHODS = (
    ("KL", "kl", "forward KL", "#1F77B4", "o"),
    ("KL+X_mu", "klx", r"KL+$\mathrm{X}_\pi$", "#2CA02C", "s"),
    (
        "KL+X_mu+X_hat_mu",
        "klxx_hat_mu",
        r"KL+$\mathrm{X}_\pi$+$\mathrm{X}_{\hat\pi}$",
        "#D62728",
        "^",
    ),
    (
        "KL+X_mu+X_mix",
        "klxx_mix",
        r"KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$",
        "#9467BD",
        "D",
    ),
)


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    }
)

def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


train_steps = 2000
window = train_steps // 50


def plot_dimension(ax):
    for _, slug, label, color, marker in METHODS:
        ess = [
            float(np.load(ARTIFACTS / f"d{d}" / f"{slug}.npz")["final_ess"])
            for d in DIMENSIONS
        ]
        ax.plot(
            DIMENSIONS,
            ess,
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.5,
            label=label,
        )

    ax.set_title("(a) Validation ESS", loc="left")
    ax.set_xlabel(r"dimension $d$")
    ax.set_ylabel("validation ESS")
    ax.set_xticks(DIMENSIONS)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlim(12, 260)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="lower left", framealpha=0.9)


def plot_training(ax):
    for _, slug, label, color, _ in METHODS:
        ess = np.asarray(
            np.load(ARTIFACTS / "d256" / f"{slug}.npz")["batch_ess_history"],
            dtype=float,
        )
        steps = np.arange(len(ess))
        ax.plot(steps, ess, color=color, linewidth=0.3, alpha=0.15)
        average = moving_average(ess, window)
        ax.plot(
            steps[window - 1 : window - 1 + len(average)],
            average,
            color=color,
            linewidth=1.1,
            label=label,
        )

    ax.set_title(r"(b) Batch ESS at $d=256$", loc="left")
    ax.set_xlabel("step")
    ax.set_ylabel("batch ESS")
    ax.set_xlim(0, train_steps)
    ax.set_ylim(0, 0.7)
    ax.legend(loc="lower right")


RESULTS.mkdir(exist_ok=True)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.0, 3.4),
    gridspec_kw={"width_ratios": (5.4, 4.2)},
)
plot_dimension(axes[0])
plot_training(axes[1])
fig.tight_layout()
fig.savefig(RESULTS / "ess.png", dpi=400, bbox_inches="tight")
plt.close(fig)

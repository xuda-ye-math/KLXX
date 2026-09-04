"""Batch ESS curves of every data-driven training in this folder, one graph.

    python plot_ess.py

Reads artifacts/<run>/batch_ess_hist.npy for each method below that has been
trained (the per-step batch ESS of the detached pushforward), draws the raw
curve faintly and its moving average over TRAIN_STEPS / 50 steps, in the
default colour cycle, on the layout of ../HD_Product/plot_ess.py, and writes
results/ess.png. A
method without a stored history is skipped and listed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import parameters as P

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

# (run directory, label); the colours are the default cycle, in this order among the trained methods
METHODS = (
    ("kl_data_driven", "forward KL"),
    ("klx_data_driven", r"KL+$\mathrm{X}_\pi$"),
    ("kll1_data_driven", r"KL+$L^1$"),
    ("klxm_data_driven", r"KL+$\mathrm{X}_{(\pi+\bar\nu)/2}$"),
    ("klxxt_data_driven", r"KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\pi+\bar\nu)/2}$"),
    ("klxx_data_driven", "KLXX"),
)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "mathtext.fontset": "cm", "font.family": "serif",
})


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    window = P.TRAIN_STEPS // 50
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    missing, plotted = [], 0
    for run, label in METHODS:
        path = ARTIFACTS / run / "batch_ess_hist.npy"
        if not path.exists():
            missing.append(run)
            continue
        color = f"C{plotted}"
        plotted += 1
        ess = np.asarray(np.load(path), dtype=float)
        steps = np.arange(1, len(ess) + 1)
        ax.plot(steps, ess, color=color, linewidth=0.3, alpha=0.15)
        average = moving_average(ess, window)
        ax.plot(steps[window - 1:window - 1 + len(average)], average, color=color, linewidth=1.1, label=label)
        print(f"{run}: {len(ess)} steps, final batch ESS {ess[np.isfinite(ess)][-1]:.4f}")
    ax.set_xlabel("step")
    ax.set_ylabel("batch ESS")
    ax.set_xlim(0, P.TRAIN_STEPS)
    ax.set_ylim(0, 0.85)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "ess.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"written {RESULTS / 'ess.png'}" + (f"; not yet trained: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()

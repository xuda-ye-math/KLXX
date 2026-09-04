"""Render samples.png and ess.png for the Two-Moon benchmark from data.npz.

Reads ``artifacts/data.npz`` (target energy grid, source samples, and the
pushforward samples, training ESS history, final ESS and coverage of each
method, FAB included, written by train.py) and re-renders both figures below
``results/``. FAB is drawn leftmost.
No training, no flow evaluation, no GPU: the figures are reproduced from the
stored arrays alone.

Run from the repo root:
    python Codes_New/2D_Benchmark/Two-Moon/result.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
DATA = ARTIFACTS / "data.npz"

LEVELS = np.linspace(0.0, 15.0, 50).tolist()   # target energy contour levels
PRIOR_ALPHA = 0.15     # opacity of the Gaussian source samples
SAMPLE_ALPHA = 0.35    # opacity of the pushforward samples
DOT_SIZE = 0.04        # scatter marker area

METHOD_LABEL = {
    "FAB":              "FAB",
    "KL":               "forward KL",
    "KL+X_pi":          r"forward KL+$\mathrm{X}_\pi$",
    "KL+X_pi+X_hat_pi": r"forward KL+$\mathrm{X}_\pi$+$\mathrm{X}_{\hat\pi}$",
    "KL+X_pi+X_mix":    r"forward KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$",
}
METHOD_COLOR = {
    "FAB":              "#FF7F0EA0",   # tab:orange
    "KL":               "#1F77B4A0",   # tab:blue
    "KL+X_pi":          "#2CA02CA0",   # tab:green
    "KL+X_pi+X_hat_pi": "#D62728A0",   # tab:red
    "KL+X_pi+X_mix":    "#9467BDA0",   # tab:purple
}

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})
CMAP = LinearSegmentedColormap.from_list("light_yellow_red", ["#fffefa", "#ffc4c4"])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(DATA))
    methods = [name for name in METHOD_LABEL if f"samples_{name}" in data]
    if not methods:
        raise SystemExit(f"{DATA} holds no method samples; run train.py first")
    print(f"loaded {DATA}: {', '.join(methods)}", flush=True)

    X1, X2, U_grid = data["X1"], data["X2"], data["U_grid"]
    prior = data["prior"]
    lim = float(X1.max())

    # ── samples.png: panels of pushforward samples over the target energy ──
    fig, axes = plt.subplots(1, len(methods), figsize=(2.2 * len(methods), 3),
                             sharey=True)
    for ax, name in zip(np.atleast_1d(axes), methods):
        ax.contourf(X1, X2, U_grid, levels=LEVELS, cmap=CMAP.reversed(), extend="both")
        ax.contour(X1, X2, U_grid, levels=LEVELS, colors="gray", linewidths=0.2, alpha=0.2)
        ax.scatter(prior[:, 0], prior[:, 1], s=DOT_SIZE, alpha=PRIOR_ALPHA,
                   color="gray", zorder=5)
        s = data[f"samples_{name}"]
        ax.scatter(s[:, 0], s[:, 1], s=DOT_SIZE, alpha=SAMPLE_ALPHA,
                   color=METHOD_COLOR[name], zorder=10)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        if name == methods[0]:
            ax.set_ylabel(r"$x_2$")
        title = (f"{METHOD_LABEL[name]}\n"
                 f"ESS = $\\mathbf{{{float(data[f'final_ess_{name}']):.2f}}}$")
        if f"coverage_{name}" in data:
            title += f", cvrg = $\\mathbf{{{float(data[f'coverage_{name}']):.2f}}}$"
        ax.set_title(title)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    samples_out = RESULTS / "samples.png"
    fig.savefig(samples_out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {samples_out}", flush=True)

    # ── ess.png: per-step training ESS histories ──
    fig, ax_ess = plt.subplots(1, 1, figsize=(5, 4))
    steps = 0
    for name in methods:
        hist = data[f"ess_hist_{name}"]
        steps = max(steps, len(hist))
        ax_ess.plot(hist, color=METHOD_COLOR[name],
                    label=METHOD_LABEL[name], linewidth=0.6)
    ax_ess.set_xlabel("step"); ax_ess.set_ylabel("ESS")
    ax_ess.set_xlim(0, steps); ax_ess.set_ylim(0, 1)
    ax_ess.legend(loc="lower right")
    plt.tight_layout()
    ess_out = RESULTS / "ess.png"
    fig.savefig(ess_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ess_out}", flush=True)


if __name__ == "__main__":
    main()

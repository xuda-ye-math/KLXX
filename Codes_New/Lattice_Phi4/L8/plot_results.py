"""Replot fig_methods.png from data.npz — no retraining, no GPU.

Reads ``artifacts/data.npz`` (per-run magnetizations and normalized
importance weights `w_{seed}_{method}`, written by train.py) and
pt_reference.npz, recomputes the final ESS and reweighted p(m > 0) from
the stored arrays, and re-renders the seed-FIG_SEED methods figure.

Run from the repo root:
    python Codes_New/Lattice_Phi4/L8/plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"

FIG_SEED = 0           # the seed drawn in fig_methods
PLT_LIM = 1.6          # half-width of the magnetization histogram window
HIST_BINS = 81         # magnetization histogram bins on (-PLT_LIM, PLT_LIM)

METHODS = (
    "FAB",
    "KL",
    "KL+X_pi",
    "KL+X_pi+X_hat_pi",
    "KL+X_pi+X_mix",
)
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
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 7.5,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})


def hist(m, weights=None):
    h, e = np.histogram(m, bins=HIST_BINS, range=(-PLT_LIM, PLT_LIM),
                        density=True, weights=weights)
    return h, 0.5 * (e[:-1] + e[1:])


def main() -> None:
    data = np.load(ARTIFACTS / "data.npz")
    ref = np.load(ARTIFACTS / "pt_reference.npz")
    h0, c0 = hist(ref["m_trace"].ravel())

    n = len(METHODS)
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2.2), squeeze=False, sharey=True)
    for j, name in enumerate(METHODS):
        mag = data[f"mag_{FIG_SEED}_{name}"]
        wn = data[f"w_{FIG_SEED}_{name}"].astype(np.float64)
        wn = wn / wn.sum()
        ess = 1.0 / (wn.size * np.sum(wn**2))       # ESS of normalized weights
        p_plus = float(wn[mag > 0].sum())
        ax = axes[0][j]
        hp, cp = hist(mag)
        ax.semilogy(cp, hp + 1e-12, color="0.6", lw=1.0, label="pushforward")
        hw, cw = hist(mag, weights=wn)
        ax.semilogy(cw, hw + 1e-12, color=METHOD_COLOR[name], lw=1.6, label="reweighted")
        ax.semilogy(c0, h0 + 1e-12, color="black", ls=":", lw=1.2,
                    label="PT reference")
        ax.set_title(f"{METHOD_LABEL[name]}\nESS={ess:.2f}, $p_+$={p_plus:.2f}")
        ax.set_xlabel(r"$m$")
        ax.set_ylim(1e-4, 30)
        if j == 0:
            ax.set_ylabel("density")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.10))
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "fig_methods.png", dpi=400,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {RESULTS / 'fig_methods.png'}")


if __name__ == "__main__":
    main()

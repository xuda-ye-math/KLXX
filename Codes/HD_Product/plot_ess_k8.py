"""Per-step training-ESS figure for one dimension of the product multi-well
sweep — one curve per objective, in the 2D-benchmark ess.png style. Reads the
data_k{k}.npz written by train.py; no retraining, no GPU.

Run from the repo root:
    ~/.envs/jax/bin/python Codes/HD_Product/plot_ess_k8.py [k]
Writes ess_k{k}.png next to this file (default k = 8, d = 256).
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})

HERE = Path(__file__).resolve().parent

METHODS = (
    "KL",
    "KL+X_mu",
    "KL+X_mu+X_hat_mu",
    "KL+X_mu+X_mix",
)
METHOD_LABEL = {
    "KL":               "forward KL",
    "KL+X_mu":          r"KL+$\mathrm{X}_\mu$",
    "KL+X_mu+X_hat_mu": r"KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$",
    "KL+X_mu+X_mix":    r"KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$",
}
METHOD_COLOR = {
    "KL":               "#1F77B4",   # tab:blue
    "KL+X_mu":          "#2CA02C",   # tab:green
    "KL+X_mu+X_hat_mu": "#D62728",   # tab:red
    "KL+X_mu+X_mix":    "#9467BD",   # tab:purple
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def moving_average(a, w):
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="valid")


def main(k: int = 8) -> None:
    path = HERE / f"data_k{k}.npz"
    if not path.exists():
        log(f"{path.name} not found — run train.py first")
        return
    log(f"START plot ess_k{k} | reading {path.name}")
    data = np.load(path)
    d, steps = int(data["d"]), int(data["steps"])

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4))
    win = max(1, steps // 50)   # moving-average window (40 steps at 2000)
    for name in METHODS:
        ess = np.asarray(data[f"ess_history_{name}"], dtype=float)
        xs = np.arange(len(ess))
        ax.plot(xs, ess, color=METHOD_COLOR[name], linewidth=0.3, alpha=0.15)
        ma = moving_average(ess, win)
        ax.plot(xs[win - 1:win - 1 + len(ma)], ma, color=METHOD_COLOR[name],
                label=METHOD_LABEL[name], linewidth=1.1)
    ax.set_xlabel("step"); ax.set_ylabel("ESS")
    ax.set_xlim(0, steps); ax.set_ylim(0, 0.7)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = HERE / f"ess_k{k}.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    log(f"DONE — saved {out}  (d={d}, steps={steps})")


if __name__ == "__main__":
    kk = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(kk)

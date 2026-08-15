"""Render density.png for the 1D quench and temper illustration from data.npz.

Reads ``artifacts/data.npz`` (grid, 1D Rastrigin potential, normalized target
density, written by samples.py) and draws the target distribution pi(x) below
``results/``. No sampling, no flow evaluation, no GPU: the figure is reproduced
from the stored arrays alone.

Run from the repo root:
    python Codes/1D_QT/result.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
DATA = ARTIFACTS / "data.npz"

TARGET_COLOR = "#1F77B4"   # tab:blue
SOURCE_COLOR = "#7F7F7F"   # tab:gray
QT_COLOR = "#D62728"       # tab:red
FILL_ALPHA = 0.20          # opacity of the shaded area under each density
XLIM = 5.0                 # half-width of the plotted range

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found; run samples.py first")
    data = np.load(DATA)
    x, pi, pi0 = data["x"], data["pi"], data["pi0"]
    qt_x, qt_density = data["qt_x"], data["qt_density"]
    lim = float(data["plt_lim"])
    print(f"loaded {DATA}: {x.size} grid points on [{-lim}, {lim}], "
          f"{qt_x.size} QT histogram bins", flush=True)

    inside = np.abs(x) <= XLIM
    qt_inside = np.abs(qt_x) <= XLIM
    ymax = 1.05 * float(max(pi[inside].max(), pi0[inside].max(),
                            qt_density[qt_inside].max()))

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 4.0), sharex=True)
    for ax, (curve_x, curve_y, color, label) in zip(
        axes,
        ((x, pi0, SOURCE_COLOR, r"source $\pi_0$"),
         (qt_x, qt_density, QT_COLOR, r"QT $\hat\pi$")),
    ):
        ax.plot(x, pi, color=TARGET_COLOR, lw=1.4, zorder=10, label=r"target $\pi$")
        ax.fill_between(x, pi, color=TARGET_COLOR, alpha=FILL_ALPHA, lw=0, zorder=5)
        ax.plot(curve_x, curve_y, color=color, lw=1.4, zorder=10, label=label)
        ax.fill_between(curve_x, curve_y, color=color, alpha=FILL_ALPHA, lw=0, zorder=5)
        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel("density")
        ax.legend(frameon=False)
    axes[-1].set_xlabel(r"$x$")
    fig.tight_layout()
    out = RESULTS / "density.png"
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

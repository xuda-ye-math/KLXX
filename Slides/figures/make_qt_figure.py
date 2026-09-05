"""Render the slide copy of the 1D quench-and-temper figure.

Reads the same ``Codes/1D_QT/artifacts/data.npz`` as ``Codes/1D_QT/result.py``
and draws the two panels side by side at a slide aspect ratio, with fonts sized
so the legends stay legible when projected. Writes only into
``Slides/figures/``; the paper's figure is untouched.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("/data/projects/KLXX/Codes/1D_QT/artifacts/data.npz")
OUT = Path("/data/projects/KLXX/Slides/figures/1d_qt_density.png")

TARGET_COLOR = "#1F77B4"
SOURCE_COLOR = "#7F7F7F"
QT_COLOR = "#D62728"
FILL_ALPHA = 0.20
XLIM = 5.0

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "mathtext.fontset": "cm", "font.family": "serif",
})

data = np.load(DATA)
x, pi, pi0 = data["x"], data["pi"], data["pi0"]
qt_x, qt_density = data["qt_x"], data["qt_density"]

inside = np.abs(x) <= XLIM
qt_inside = np.abs(qt_x) <= XLIM
ymax = 1.05 * float(max(pi[inside].max(), pi0[inside].max(),
                        qt_density[qt_inside].max()))

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.3), sharey=True)
for ax, (curve_x, curve_y, color, label, title) in zip(
    axes,
    ((x, pi0, SOURCE_COLOR, r"source $\pi_0$", "Source miss outer wells"),
     (qt_x, qt_density, QT_COLOR, r"QT $\hat\pi$", "QT find wells but distribute wrongly")),
):
    ax.plot(x, pi, color=TARGET_COLOR, lw=2.2, zorder=10, label=r"target $\pi$")
    ax.fill_between(x, pi, color=TARGET_COLOR, alpha=FILL_ALPHA, lw=0, zorder=5)
    ax.plot(curve_x, curve_y, color=color, lw=2.2, zorder=10, label=label)
    ax.fill_between(curve_x, curve_y, color=color, alpha=FILL_ALPHA, lw=0, zorder=5)
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel(r"$x$")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
axes[0].set_ylabel("density")
fig.tight_layout()
fig.savefig(OUT, dpi=250, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT}")

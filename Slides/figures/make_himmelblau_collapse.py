"""Render himmelblau_collapse.png: three panels of the same Himmelblau target.

Left:   four Metropolis-adjusted Langevin chains started at the origin.
Middle: pushforward samples of a flow trained with reverse KL, with its ESS.
Right:  pushforward samples of a flow trained with forward KL, with its ESS.

Everything is read from ``artifacts/himmelblau_collapse.npz``, written by
``train_himmelblau_collapse.py``; this script retrains nothing. The contour
style, colours and marker sizes are those of
``Codes/2D_Benchmark/Himmelblau/result.py``, with the fonts enlarged for
projection.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
DATA = HERE / "artifacts" / "himmelblau_collapse.npz"
OUT = HERE / "himmelblau_collapse.png"
OUT_MARKED = HERE / "himmelblau_collapse_fake_ess.png"

plt.rcParams.update({
    # the benchmark style, fonts enlarged for projection
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "mathtext.fontset": "cm", "font.family": "serif",
})

CMAP = LinearSegmentedColormap.from_list("light_yellow_red", ["#fffefa", "#ffc4c4"])
LEVELS = np.linspace(0.0, 30.0, 50).tolist()
DOT_SIZE = 0.04
SAMPLE_ALPHA = 0.35
PRIOR_ALPHA = 0.35   # the source cloud, raised from the benchmark's 0.15 for projection
LIM = 5.5
CHAIN_COLORS = ["#1F77B4", "#2CA02C", "#D62728", "#9467BD"]
REVERSE_COLOR = "#FF7F0EA0"   # tab:orange
FORWARD_COLOR = "#1F77B4A0"   # tab:blue, the benchmark's forward KL colour

data = np.load(DATA)
X1, X2, U_grid = data["X1"], data["X2"], data["U_grid"]
prior = data["prior"]

fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.4), sharey=True)
for ax in axes:
    ax.contourf(X1, X2, U_grid, levels=LEVELS, cmap=CMAP.reversed(), extend="both")
    ax.contour(X1, X2, U_grid, levels=LEVELS, colors="gray", linewidths=0.2, alpha=0.2)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([t for t in ax.get_xticks() if abs(t) < LIM])
    ax.set_aspect("equal")
    ax.scatter(prior[:, 0], prior[:, 1], s=DOT_SIZE, alpha=PRIOR_ALPHA,
               color="gray", zorder=5)
    ax.set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")

for k, color in enumerate(CHAIN_COLORS):
    chain = data[f"chain_{k}"]
    axes[0].plot(chain[:, 0], chain[:, 1], lw=0.4, color=color, alpha=0.9, zorder=10)
axes[0].plot(0.0, 0.0, "o", color="black", ms=6, zorder=12)
axes[0].set_title("MALA trajectories")

for ax, key, color, label in (
    (axes[1], "reverse_KL", REVERSE_COLOR, "reverse KL"),
    (axes[2], "forward_KL", FORWARD_COLOR, "forward KL"),
):
    s = data[f"samples_{key}"]
    ax.scatter(s[:, 0], s[:, 1], s=DOT_SIZE, alpha=SAMPLE_ALPHA, color=color, zorder=10)
    ess = float(data[f"final_ess_{key}"])
    ax.set_title(f"{label}, ESS = $\\mathbf{{{ess:.2f}}}$")

plt.tight_layout()
fig.subplots_adjust(wspace=0.05)
fig.savefig(OUT, dpi=400, bbox_inches="tight")
print(f"wrote {OUT}")

# second copy for the frame that names the phenomenon: the forward KL panel
# holds a high ESS on two wells of four, so it is marked at the source centre
axes[2].text(0.0, 0.0, "Fake ESS", color="#D62728",
             fontsize=15, fontweight="bold", style="italic",
             ha="center", va="center", zorder=20,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#D62728", linewidth=0.9, alpha=0.88))
fig.savefig(OUT_MARKED, dpi=400, bbox_inches="tight")
print(f"wrote {OUT_MARKED}")
for key in ("reverse_KL", "forward_KL"):
    print(f"  {key}: ESS = {float(data[f'final_ess_{key}']):.4f}")

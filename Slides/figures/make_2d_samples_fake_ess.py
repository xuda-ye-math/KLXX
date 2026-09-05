"""Slide copies of the 2D benchmark panels, with the fake-ESS ones marked.

A copy of ``Codes/2D_Benchmark/<target>/result.py``, unchanged in style,
colours, contour levels, panel order and labels, with one addition: a panel
whose coverage falls below ``FAKE_COVERAGE`` carries the words "Fake ESS" in
red at the source Gaussian's centre, the origin.

Reads each target's own ``artifacts/data.npz`` and writes only into
``Slides/figures/``. Nothing under ``Codes/`` or ``Paper/`` is touched.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
BENCH = Path("/data/projects/KLXX/Codes/2D_Benchmark")

# target, output name, and that target's own contour levels
# marked copies; the unmarked originals are the paper's own files
# name, output, contour levels, methods to leave out, whether to mark fake ESS
TARGETS = [
    ("Three-Well", "2d_three_well_fake_ess.png", (-1.0, 8.0, 20), (), True),
    ("Himmelblau", "2d_himmelblau_fake_ess.png", (0.0, 30.0, 50), (), True),
    ("Sparse", "2d_sparse_fake_ess.png", (-2.0, 20.0, 50), (), True),
    # the mixture frame compares the forward KL family alone, unmarked
    ("Sparse", "2d_sparse_4methods.png", (-2.0, 20.0, 50), ("FAB",), False),
]

FAKE_COVERAGE = 0.90   # below this the ESS is earned on part of the target only

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


def render(target: str, out_name: str, levels: tuple,
           exclude: tuple = (), mark: bool = True) -> None:
    data = dict(np.load(BENCH / target / "artifacts" / "data.npz"))
    methods = [name for name in METHOD_LABEL
               if f"samples_{name}" in data and name not in exclude]
    if not methods:
        raise SystemExit(f"{target}: no method samples in its data.npz")

    LEVELS = np.linspace(*levels[:2], int(levels[2])).tolist()
    X1, X2, U_grid = data["X1"], data["X2"], data["U_grid"]
    prior = data["prior"]
    lim = float(X1.max())
    marked = []

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
        # drop any x tick sitting on the axis limit: with panels this close it
        # would collide with the neighbour's first label
        ax.set_xticks([t for t in ax.get_xticks() if abs(t) < lim])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        if name == methods[0]:
            ax.set_ylabel(r"$x_2$")
        title = (f"{METHOD_LABEL[name]}\n"
                 f"ESS = $\\mathbf{{{float(data[f'final_ess_{name}']):.2f}}}$")
        coverage = None
        if f"coverage_{name}" in data:
            coverage = float(data[f"coverage_{name}"])
            title += f", cvrg = $\\mathbf{{{coverage:.2f}}}$"
        ax.set_title(title)

        # the ESS is high but earned on the wells the flow reached: say so, at
        # the centre of the source Gaussian
        if mark and coverage is not None and coverage < FAKE_COVERAGE:
            ax.text(0.0, 0.0, "Fake ESS", color="#D62728",
                    fontsize=13, fontweight="bold", style="italic",
                    ha="center", va="center", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor="#D62728", linewidth=0.9, alpha=0.88))
            marked.append(name)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    out = HERE / out_name
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"{target}: {len(methods)} panels -> {out.name}; "
          f"marked Fake ESS on {', '.join(marked) if marked else 'none'}",
          flush=True)


if __name__ == "__main__":
    for target, out_name, levels, exclude, mark in TARGETS:
        render(target, out_name, levels, exclude, mark)

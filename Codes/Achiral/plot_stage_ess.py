#!/usr/bin/env python
"""Per-stage sample ESS of the two losses on an achiral molecular target.

Usage
    python plot_stage_ess.py {nma,glycerol,diethanolamine,all}

Each accepted stage is drawn as an arc over the interval it spans, KL+X_pi
above the axis and KLXX below, labelled by the sample ESS of the map the
stage selected. The regularization now follows the diagonal path, so there is
no separate sharpening ESS to report. The figure is written to
<folder>/results/stage_ess.png.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BLUE = "#1F77B4"
RED = "#D62728"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "mathtext.fontset": "cm",
})

METHODS = (
    ("klx", r"$\mathrm{KL}$+$\mathrm{X}_{\pi}$", BLUE, +1),
    ("klxx", r"$\mathrm{KL}$+$\mathrm{X}_{\pi}$+$\mathrm{X}_{(\hat{\pi}+\bar{\nu})/2}$", RED, -1),
)


@dataclass(frozen=True)
class Molecule:
    folder: str
    title: str
    dimension: int


@dataclass(frozen=True)
class Stage:
    t_start: float
    t_end: float
    ess: float


MOLECULES = {
    "nma": Molecule("NMA_30D", "NMA", 30),
    "glycerol": Molecule("Glycerol_36D", "glycerol", 36),
    "diethanolamine": Molecule("Diethanolamine_48D", "neutral diethanolamine", 48),
}


def load_stages(molecule_dir: Path, method: str) -> tuple[list[Stage], bool]:
    """Accepted stages of a run, and whether the run has reached t = 1."""
    run_dir = molecule_dir / "artifacts" / method
    manifest = json.loads((run_dir / "run.json").read_text())
    complete = manifest.get("status") == "complete"
    stages = []
    for entry in manifest["stages"]:
        record = json.loads((run_dir / entry["path"] / "stage.json").read_text())
        ess = float(record["valid_selected_ess"])
        if not 0.0 < ess <= 1.0:
            raise ValueError(f"invalid stage ESS {ess} in {run_dir / entry['path']}")
        stages.append(Stage(float(record["t_start"]), float(record["t"]), ess))
    return stages, complete


def draw_arcs(ax, stages, color, sign):
    theta = np.linspace(0.0, np.pi, 100)
    for stage in stages:
        radius = 0.5 * (stage.t_end - stage.t_start)
        midpoint = 0.5 * (stage.t_start + stage.t_end)
        ax.plot(midpoint + radius * np.cos(theta), sign * 0.20 * np.sin(theta),
                color=color, linewidth=2.2, zorder=3)
        ax.text(midpoint, sign * 0.25, f"{stage.ess:.2f}", color=color, ha="center",
                va="bottom" if sign > 0 else "top", fontsize=11.0, zorder=4)
    nodes = np.asarray([stages[0].t_start, *(s.t_end for s in stages)])
    ax.plot(nodes, np.zeros_like(nodes), "o", color="0.2", ms=4, zorder=5)


def plot(molecule: Molecule) -> Path:
    molecule_dir = HERE / molecule.folder
    series, complete = {}, {}
    for method, _, _, _ in METHODS:
        if not (molecule_dir / "artifacts" / method / "run.json").exists():
            continue
        series[method], complete[method] = load_stages(molecule_dir, method)
    if not series:
        raise ValueError(f"no run to plot in {molecule_dir}")
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    ax.axhline(0.0, color="0.3", linewidth=1.0, zorder=1)
    for method, _, color, sign in METHODS:
        if method in series:
            draw_arcs(ax, series[method], color, sign)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.55, 0.55), yticks=[],
           xticks=np.linspace(0.0, 1.0, 6), xlabel=r"interpolation parameter $t$")
    running = [m for m in series if not complete[m]]
    suffix = "; " + ", ".join(running) + " still running" if running else ""
    ax.set_title(rf"{molecule.title} ($d={molecule.dimension}$): per-stage sample ESS{suffix}", pad=10)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2.5, label=l) for m, l, c, _ in METHODS if m in series],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.015), fontsize=12.5)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    output = molecule_dir / "results" / "stage_ess.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    for method, label, _, _ in METHODS:
        if method in series:
            mark = "" if complete[method] else "  (in progress)"
            print(f"  {label}: " + ", ".join(f"{s.ess:.2f}" for s in series[method]) + mark)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", choices=(*MOLECULES, "all"))
    args = parser.parse_args()
    keys = tuple(MOLECULES) if args.molecule == "all" else (args.molecule,)
    for key in keys:
        plot(MOLECULES[key])


if __name__ == "__main__":
    main()

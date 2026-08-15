#!/usr/bin/env python
"""Plot per-stage flow and sharpening ESS for an achiral molecular target.

Usage
-----
python plot_stage_ess.py {nma,glycerol,diethanolamine}

The figure is written to ``<molecule>_<dimension>d/results/stage_ess.png``.
"""

from __future__ import annotations

import argparse
import json
import math
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
METHODS = (
    (
        "klx",
        "forward_klx",
        r"$\mathrm{KL}$+$\mathrm{X}_{\pi}$",
        BLUE,
        +1,
    ),
    (
        "klxx",
        "forward_klxx",
        r"$\mathrm{KL}$+$\mathrm{X}_{\pi}$+$\mathrm{X}_{(\hat{\pi}+\bar{\nu})/2}$",
        RED,
        -1,
    ),
)


@dataclass(frozen=True)
class Molecule:
    folder: str
    title: str
    dimension: int
    label_fontsize: float = 11.0


@dataclass(frozen=True)
class Stage:
    index: int
    t_start: float
    t_end: float
    flow_ess: float
    sharpen_ess: float


MOLECULES = {
    "nma": Molecule("nma_30d", "NMA", 30),
    "glycerol": Molecule("glycerol_36d", "glycerol", 36),
    "diethanolamine": Molecule(
        "diethanolamine_48d",
        "neutral diethanolamine",
        48,
        label_fontsize=10.0,
    ),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "mathtext.fontset": "cm",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot KLX and KLXX per-stage flow/sharpening ESS."
    )
    parser.add_argument(
        "molecule",
        choices=tuple(MOLECULES),
        help="achiral molecular target",
    )
    return parser.parse_args()


def load_stage_ess(
    molecule_dir: Path, method: str, expected_objective: str
) -> list[Stage]:
    artifact_dir = molecule_dir / "artifacts" / method
    run_path = artifact_dir / "run.json"
    with run_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("status") != "complete":
        raise ValueError(f"run is not complete: {run_path}")

    stage_entries = manifest.get("stages", [])
    if not stage_entries:
        raise ValueError(f"run has no accepted stages: {run_path}")

    stages: list[Stage] = []
    previous_t = 0.0
    for expected_stage, entry in enumerate(stage_entries, start=1):
        if entry.get("stage") != expected_stage:
            raise ValueError(
                f"nonconsecutive stage manifest in {run_path}: {entry!r}"
            )

        stage_path = artifact_dir / entry["path"] / "stage.json"
        with stage_path.open(encoding="utf-8") as stream:
            stage = json.load(stream)

        if stage.get("stage") != expected_stage:
            raise ValueError(f"stage number mismatch: {stage_path}")
        if stage.get("objective") != expected_objective:
            raise ValueError(
                f"unexpected objective in {stage_path}: "
                f"{stage.get('objective')!r}"
            )

        t_start = float(stage["t_start"])
        t_end = float(stage["t"])
        if not math.isclose(t_start, previous_t, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"noncontiguous stage schedule: {stage_path}")
        if not math.isclose(
            t_end, float(entry["t"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"stage endpoint mismatch: {stage_path}")

        flow_ess = float(stage["valid_selected_ess"])
        if not math.isfinite(flow_ess) or not 0.0 < flow_ess <= 1.0:
            raise ValueError(f"invalid flow ESS in {stage_path}: {flow_ess}")

        sharpen_ess = float(stage["sharpen_ess"])
        if not math.isfinite(sharpen_ess) or not 0.0 < sharpen_ess <= 1.0:
            raise ValueError(
                f"invalid sharpening ESS in {stage_path}: {sharpen_ess}"
            )
        stages.append(
            Stage(
                index=expected_stage,
                t_start=t_start,
                t_end=t_end,
                flow_ess=flow_ess,
                sharpen_ess=sharpen_ess,
            )
        )
        previous_t = t_end

    if not math.isclose(previous_t, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"run does not reach t=1: {run_path}")
    return stages


def draw_arcs(
    ax: plt.Axes,
    stages: list[Stage],
    color: str,
    sign: int,
    label_fontsize: float,
) -> None:
    theta = np.linspace(0.0, np.pi, 100)
    arc_height = 0.20
    label_height = 0.25
    for stage in stages:
        radius = 0.5 * (stage.t_end - stage.t_start)
        midpoint = 0.5 * (stage.t_start + stage.t_end)
        x = midpoint + radius * np.cos(theta)
        y = sign * arc_height * np.sin(theta)
        ax.plot(x, y, color=color, linewidth=2.2, zorder=3)

        ax.text(
            midpoint,
            sign * label_height,
            f"{stage.flow_ess:.2f}\n{stage.sharpen_ess:.2f}",
            color=color,
            ha="center",
            va="bottom" if sign > 0 else "top",
            fontsize=label_fontsize,
            linespacing=0.95,
            zorder=4,
        )

    stage_nodes = np.asarray([stages[0].t_start, *(stage.t_end for stage in stages)])
    ax.plot(stage_nodes, np.zeros_like(stage_nodes), "o", color="0.2", ms=4, zorder=5)


def plot_stage_ess(molecule: Molecule) -> Path:
    molecule_dir = HERE / molecule.folder
    series = {
        method: load_stage_ess(molecule_dir, method, objective)
        for method, objective, _, _, _ in METHODS
    }
    fig, ax = plt.subplots(figsize=(10.0, 3.7))
    ax.axhline(0.0, color="0.3", linewidth=1.0, zorder=1)
    for method, _, _, color, sign in METHODS:
        draw_arcs(
            ax,
            series[method],
            color,
            sign,
            molecule.label_fontsize,
        )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.62, 0.62)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks([])
    ax.set_xlabel(r"interpolation parameter $t$")
    ax.set_title(
        rf"{molecule.title} ($d={molecule.dimension}$) — "
        "per-stage flow/sharpening ESS",
        pad=10,
    )
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], color=color, lw=2.5, label=label)
        for _, _, label, color, _ in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))

    output = molecule_dir / "results" / "stage_ess.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")
    for method, _, label, _, _ in METHODS:
        values = ", ".join(
            f"{stage.flow_ess:.2f} / {stage.sharpen_ess:.2f}"
            for stage in series[method]
        )
        print(f"  {label}: {values}")
    return output


def main() -> None:
    args = parse_args()
    plot_stage_ess(MOLECULES[args.molecule])


if __name__ == "__main__":
    main()

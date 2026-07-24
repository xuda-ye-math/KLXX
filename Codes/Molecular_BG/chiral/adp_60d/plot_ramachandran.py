#!/usr/bin/env python
"""Plot a 2-by-2 ADP Ramachandran summary from persisted inference samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import matplotlib


matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from jflows_md import Molecular_Potential
from plot_ramachandran_full import (
    BUNDLE,
    FREE_ENERGY_CMAP,
    INFERENCE_DIR,
    backbone_histogram,
    free_energy_surface,
    load_population,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "ramachandran.png"
SELECTED_STAGES = (4, 6, 8, 10)


def stage_title(stage: dict) -> str:
    regularization = stage["rg_end"]
    return (
        f"stage {int(stage['stage'])}: "
        rf"$t={float(stage['t']):.3f}$, "
        rf"$\rho=({float(regularization[0]):.3g},"
        rf"{float(regularization[1]):.3g})$"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    if args.bins <= 0 or args.smoothing < 0.0 or args.chunk_size <= 0:
        parser.error("bins and chunk size must be positive; smoothing nonnegative")

    root = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "adp-frozen-flow-inference-10m-1":
        raise ValueError("input is not an ADP frozen-flow inference run")
    if manifest.get("status") != "complete":
        raise ValueError("input inference run is not complete")
    if manifest.get("inference_only") is not True:
        raise ValueError("input does not declare inference-only provenance")
    if int(manifest.get("training_updates", -1)) != 0:
        raise ValueError("input reports nonzero training updates")

    sample_count = int(manifest["sample_count"])
    dimension = int(manifest["dimension"])
    stages = {int(stage["stage"]): stage for stage in manifest["stages"]}
    if len(stages) != len(manifest["stages"]):
        raise ValueError("manifest contains duplicate stage numbers")
    missing = sorted(set(SELECTED_STAGES) - stages.keys())
    if missing:
        raise ValueError(f"manifest is missing requested stages: {missing}")

    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)
    if target.dimension != dimension or dimension != 60:
        raise ValueError(
            f"unexpected ADP dimension: manifest={dimension}, "
            f"target={target.dimension}"
        )

    surfaces: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for stage_number in SELECTED_STAGES:
        stage = stages[stage_number]
        if int(stage["sample_count"]) != sample_count:
            raise ValueError(f"stage {stage_number} has inconsistent sample count")
        samples = load_population(
            root,
            stage["samples_path"],
            sample_count,
            dimension,
        )
        edges, counts = backbone_histogram(
            samples,
            target,
            args.bins,
            args.chunk_size,
        )
        surfaces.append(
            (stage, edges, free_energy_surface(counts, args.smoothing))
        )
        print(
            f"stage={stage_number} samples={sample_count:,} "
            f"histogram={int(counts.sum()):,}",
            flush=True,
        )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 9.0),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for panel, (axis, (stage, edges, free_energy)) in enumerate(
        zip(axes.flat, surfaces)
    ):
        row, column = divmod(panel, 2)
        image = axis.pcolormesh(
            edges,
            edges,
            free_energy.T,
            cmap=FREE_ENERGY_CMAP,
            vmin=0.0,
            vmax=11.0,
            shading="flat",
            rasterized=True,
        )
        axis.set(
            xlim=(-np.pi, np.pi),
            ylim=(-np.pi, np.pi),
            title=stage_title(stage),
        )
        if row == 1:
            axis.set_xlabel(r"$\phi$ (rad)")
        if column == 0:
            axis.set_ylabel(r"$\psi$ (rad)")
        axis.set_aspect("equal")
        axis.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
        axis.set_yticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    if image is None:
        raise RuntimeError("no Ramachandran surfaces were generated")
    colorbar = figure.colorbar(
        image,
        ax=axes,
        pad=0.025,
        fraction=0.04,
        shrink=0.94,
    )
    colorbar.set_ticks(np.arange(0.0, 11.0, 2.0))
    colorbar.set_label(r"free energy / $k_{\mathrm{B}}T$")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(
        f"backend={jax.default_backend()} stages={SELECTED_STAGES} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

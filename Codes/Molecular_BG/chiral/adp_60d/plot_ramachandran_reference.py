#!/usr/bin/env python
"""Compare the final KLXX ADP Ramachandran surface with published MD data.

The reference trajectory is the alanine dipeptide implicit-solvent data set at
300 K released with the FAB study (Zenodo record 6993124, DOI
10.5281/zenodo.6993124).  It is stored as an MDTraj HDF5 trajectory holding
Cartesian frames and its own topology, so this driver reads positions directly
and reuses the surface and figure conventions of
``plot_ramachandran_full.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import jax
import matplotlib


matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from jflows_md import Molecular_Potential

from plot_ramachandran_full import (
    BUNDLE,
    DEFAULT_OUTPUT_DIR,
    FREE_ENERGY_CMAP,
    INFERENCE_DIR,
    PHI_ATOMS,
    PSI_ATOMS,
    backbone_histogram,
    dihedral,
    free_energy_surface,
    load_population,
)


HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"
DEFAULT_REFERENCE = REFERENCE_DIR / "test.h5"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "ramachandran_reference.png"
SELECTED_STAGE = 10
REFERENCE_TITLE = "MD reference (FAB)"


def bundle_atom_names() -> tuple[str, ...]:
    """Return the ordered atom names of the frozen ADP bundle."""

    lines = (BUNDLE / "reference.pdb").read_text(encoding="utf-8").splitlines()
    return tuple(
        line[12:16].strip()
        for line in lines
        if line.startswith(("ATOM", "HETATM"))
    )


def trajectory_atom_names(handle: h5py.File) -> tuple[str, ...]:
    """Return the ordered atom names of the embedded MDTraj topology."""

    topology = json.loads(handle["topology"][0].decode("utf-8"))
    return tuple(
        atom["name"]
        for chain in topology["chains"]
        for residue in chain["residues"]
        for atom in residue["atoms"]
    )


def reference_histogram(
    coordinates: h5py.Dataset,
    bins: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram the backbone torsions of a Cartesian reference trajectory."""

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    counts = np.zeros((bins, bins), dtype=np.int64)
    frames = coordinates.shape[0]
    for start in range(0, frames, chunk_size):
        stop = min(start + chunk_size, frames)
        positions = np.asarray(coordinates[start:stop], dtype=np.float64)
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"nonfinite reference frames in rows {start}:{stop}")
        # Match ``plot_ramachandran_full``: wrap in float64 so a dihedral that
        # lands on +pi is not dropped by the histogram edge.
        phi = np.remainder(
            dihedral(positions, PHI_ATOMS) + np.pi, 2.0 * np.pi
        ) - np.pi
        psi = np.remainder(
            dihedral(positions, PSI_ATOMS) + np.pi, 2.0 * np.pi
        ) - np.pi
        chunk_counts, _, _ = np.histogram2d(phi, psi, bins=(edges, edges))
        counts += chunk_counts.astype(np.int64)
    if int(counts.sum()) != frames:
        raise ValueError(f"histogram contains {int(counts.sum()):,} of {frames:,} frames")
    return edges, counts


def klxx_surface(
    root: Path,
    bins: int,
    smoothing: float,
    chunk_size: int,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Return the title, edges, and surface of the selected KLXX stage."""

    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "adp-frozen-flow-inference-10m-1":
        raise ValueError("input is not an ADP frozen-flow inference run")
    if manifest.get("status") != "complete":
        raise ValueError("input inference run is not complete")
    if manifest.get("inference_only") is not True:
        raise ValueError("input does not declare inference-only provenance")
    if int(manifest.get("training_updates", -1)) != 0:
        raise ValueError("input reports nonzero training updates")

    stages = {int(stage["stage"]): stage for stage in manifest["stages"]}
    if SELECTED_STAGE not in stages:
        raise ValueError(f"manifest is missing stage {SELECTED_STAGE}")
    stage = stages[SELECTED_STAGE]

    sample_count = int(manifest["sample_count"])
    dimension = int(manifest["dimension"])
    if int(stage["sample_count"]) != sample_count:
        raise ValueError(f"stage {SELECTED_STAGE} has inconsistent sample count")

    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)
    if target.dimension != dimension or dimension != 60:
        raise ValueError(
            f"unexpected ADP dimension: manifest={dimension}, target={target.dimension}"
        )

    samples = load_population(root, stage["samples_path"], sample_count, dimension)
    edges, counts = backbone_histogram(samples, target, bins, chunk_size)
    regularization = stage["rg_end"]
    title = (
        rf"KLXX stage {SELECTED_STAGE}: "
        rf"$\rho=({float(regularization[0]):.3g},"
        rf"{float(regularization[1]):.3g})$"
    )
    print(
        f"klxx stage={SELECTED_STAGE} samples={sample_count:,} "
        f"histogram={int(counts.sum()):,}",
        flush=True,
    )
    return title, edges, free_energy_surface(counts, smoothing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--reference-chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    if args.bins <= 0 or args.smoothing < 0.0 or args.chunk_size <= 0:
        parser.error("bins and chunk size must be positive; smoothing nonnegative")
    if args.reference_chunk_size <= 0:
        parser.error("reference chunk size must be positive")

    root = args.input_dir.expanduser().resolve()
    source = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()

    klxx_title, klxx_edges, klxx_free_energy = klxx_surface(
        root,
        args.bins,
        args.smoothing,
        args.chunk_size,
    )

    with h5py.File(source, "r") as handle:
        expected = bundle_atom_names()
        found = trajectory_atom_names(handle)
        if found != expected:
            raise ValueError(
                "reference topology does not match the frozen ADP bundle ordering"
            )
        coordinates = handle["coordinates"]
        if coordinates.ndim != 3 or coordinates.shape[1:] != (len(expected), 3):
            raise ValueError(
                f"expected Cartesian frames [F, {len(expected)}, 3], "
                f"found {coordinates.shape}"
            )
        frames = int(coordinates.shape[0])
        reference_edges, reference_counts = reference_histogram(
            coordinates,
            args.bins,
            args.reference_chunk_size,
        )
    reference_free_energy = free_energy_surface(reference_counts, args.smoothing)
    print(
        f"reference frames={frames:,} histogram={int(reference_counts.sum()):,}",
        flush=True,
    )

    panels = (
        (klxx_title, klxx_edges, klxx_free_energy),
        (REFERENCE_TITLE, reference_edges, reference_free_energy),
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
        1,
        2,
        figsize=(10.0, 4.6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for column, (axis, (title, edges, free_energy)) in enumerate(
        zip(axes.flat, panels)
    ):
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
            xlabel=r"$\phi$ (rad)",
            title=title,
        )
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
        f"backend={jax.default_backend()} stage={SELECTED_STAGE} "
        f"reference_frames={frames:,} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

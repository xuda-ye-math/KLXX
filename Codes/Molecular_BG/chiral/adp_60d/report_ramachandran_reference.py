#!/usr/bin/env python
"""Tabulate the ADP Ramachandran comparison against the published MD reference.

Companion to ``plot_ramachandran_reference.py``: it rebuilds the same two
surfaces with the same bins, smoothing, and free-energy convention, then writes
the numerical comparison behind the figure.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from jflows_md import Molecular_Potential

from plot_ramachandran_full import (
    BUNDLE,
    INFERENCE_DIR,
    backbone_histogram,
    free_energy_surface,
    load_population,
)
from plot_ramachandran_reference import (
    DEFAULT_REFERENCE,
    SELECTED_STAGE,
    bundle_atom_names,
    reference_histogram,
    trajectory_atom_names,
)

import h5py


HERE = Path(__file__).resolve().parent
DEFAULT_REPORT = HERE / "results" / "ramachandran_reference_comparison.md"

# Regions are classified by the MD reference surface, so both sets are scored
# on the same bins.
LOW_FREE_ENERGY = 2.0
HIGH_FREE_ENERGY = 3.0


def basin_populations(counts: np.ndarray, edges: np.ndarray) -> tuple[float, ...]:
    """Return alpha-L, beta/PPII, and alpha-R fractions of the phi/psi histogram."""

    centers = 0.5 * (edges[:-1] + edges[1:])
    phi = centers[:, None] * np.ones_like(centers)[None, :]
    psi = np.ones_like(centers)[:, None] * centers[None, :]
    total = counts.sum()
    alpha_l = counts[phi > 0.0].sum() / total
    beta = counts[(phi <= 0.0) & (psi > 0.0)].sum() / total
    alpha_r = counts[(phi <= 0.0) & (psi <= 0.0)].sum() / total
    return float(alpha_l), float(beta), float(alpha_r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--reference-chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    if args.bins <= 0 or args.smoothing < 0.0 or args.chunk_size <= 0:
        parser.error("bins and chunk size must be positive; smoothing nonnegative")

    root = args.input_dir.expanduser().resolve()
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "adp-frozen-flow-inference-10m-1":
        raise ValueError("input is not an ADP frozen-flow inference run")
    if manifest.get("status") != "complete":
        raise ValueError("input inference run is not complete")
    stages = {int(stage["stage"]): stage for stage in manifest["stages"]}
    if SELECTED_STAGE not in stages:
        raise ValueError(f"manifest is missing stage {SELECTED_STAGE}")
    stage = stages[SELECTED_STAGE]
    sample_count = int(manifest["sample_count"])
    dimension = int(manifest["dimension"])

    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)
    samples = load_population(root, stage["samples_path"], sample_count, dimension)
    edges, klxx_counts = backbone_histogram(
        samples, target, args.bins, args.chunk_size
    )
    print(f"klxx stage={SELECTED_STAGE} samples={sample_count:,}", flush=True)

    with h5py.File(args.reference.expanduser().resolve(), "r") as handle:
        if trajectory_atom_names(handle) != bundle_atom_names():
            raise ValueError(
                "reference topology does not match the frozen ADP bundle ordering"
            )
        coordinates = handle["coordinates"]
        frames = int(coordinates.shape[0])
        reference_edges, reference_counts = reference_histogram(
            coordinates, args.bins, args.reference_chunk_size
        )
    if not np.allclose(edges, reference_edges):
        raise ValueError("KLXX and reference histograms use different edges")
    print(f"reference frames={frames:,}", flush=True)

    surfaces = {
        "KLXX": free_energy_surface(klxx_counts, args.smoothing),
        "MD reference (FAB)": free_energy_surface(reference_counts, args.smoothing),
    }
    counts = {"KLXX": klxx_counts, "MD reference (FAB)": reference_counts}
    names = ("KLXX", "MD reference (FAB)")

    reference = surfaces["MD reference (FAB)"]
    # ``free_energy_surface`` smooths an integer count array, so a bin holding a
    # single sample can round to zero density and diverge.  Those bins are
    # excluded here rather than being reported as an infinite free energy.
    both = np.isfinite(surfaces["KLXX"]) & np.isfinite(reference)
    low = both & (reference < LOW_FREE_ENERGY)
    high = both & (reference > HIGH_FREE_ENERGY)

    lines = [
        "# ADP Ramachandran comparison",
        "",
        "The final KLXX stage against the published alanine dipeptide reference",
        "data at 300 K (Zenodo record 6993124, DOI 10.5281/zenodo.6993124).",
        "Both sets use the same 100-bin phi/psi grid, the same wrapped",
        "smoothing, and each surface is normalized to its own most populated",
        "bin. The reference topology matches the frozen bundle atom for atom, so",
        "the same phi/psi atom indices apply to both.",
        "",
        f"- KLXX stage {SELECTED_STAGE}: `{sample_count:,}` samples, "
        f"t = `{float(stage['t']):.3f}`, rho = "
        f"`({float(stage['rg_end'][0]):.3g}, {float(stage['rg_end'][1]):.3g})`",
        f"- MD reference: `{frames:,}` frames",
        "",
        "## Surface extent",
        "",
        "Occupied bins counts every bin holding at least one sample. The",
        "maximum is taken over finite bins only.",
        "",
        "| sample set | occupied bins | max free energy / kBT |",
        "|---|---:|---:|",
    ]
    for name in names:
        surface = surfaces[name]
        finite = surface[np.isfinite(surface)]
        lines.append(
            f"| {name} | {int((counts[name] > 0).sum())} | "
            f"{float(finite.max()):.2f} |"
        )
    lines += [
        "",
        "## Free energy by region",
        "",
        "Bins are classified by the MD reference: low means below "
        f"{LOW_FREE_ENERGY:.0f} kBT, high means above {HIGH_FREE_ENERGY:.0f} kBT.",
        "Entries are the mean free energy of each set over those bins, in kBT.",
        "Only bins finite in both surfaces are used, so the rows are directly",
        "comparable.",
        "",
        "| region | bins | " + " | ".join(names) + " |",
        "|---|---:|---:|---:|",
    ]
    regions = (
        (f"low F (< {LOW_FREE_ENERGY:.0f} kBT)", low),
        (f"high F (> {HIGH_FREE_ENERGY:.0f} kBT)", high),
        ("all bins", both),
    )
    for label, mask in regions:
        values = " | ".join(f"{surfaces[name][mask].mean():.3f}" for name in names)
        lines.append(f"| {label} | {int(mask.sum())} | {values} |")

    lines += [
        "",
        "## Backbone basin populations",
        "",
        "Fractions of the phi/psi histogram; alpha-L is phi > 0, and the",
        "phi < 0 half is split by the sign of psi.",
        "",
        "| sample set | alpha-L (phi > 0) | beta/PPII (psi > 0) | alpha-R (psi < 0) |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        alpha_l, beta, alpha_r = basin_populations(counts[name], edges)
        lines.append(f"| {name} | {alpha_l:.4f} | {beta:.4f} | {alpha_r:.4f} |")

    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"backend={jax.default_backend()} report={report}", flush=True)


if __name__ == "__main__":
    main()

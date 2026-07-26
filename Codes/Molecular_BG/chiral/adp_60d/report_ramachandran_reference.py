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

    names = (
        "KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub>",
        "MD reference (FAB)",
    )
    surfaces = {
        names[0]: free_energy_surface(klxx_counts, args.smoothing),
        names[1]: free_energy_surface(reference_counts, args.smoothing),
    }
    counts = {names[0]: klxx_counts, names[1]: reference_counts}

    reference = surfaces[names[1]]
    # ``free_energy_surface`` smooths an integer count array, so a bin holding a
    # single sample can round to zero density and diverge.  Those bins are
    # excluded here rather than being reported as an infinite free energy.
    both = np.isfinite(surfaces[names[0]]) & np.isfinite(reference)
    low = both & (reference < LOW_FREE_ENERGY)
    high = both & (reference > HIGH_FREE_ENERGY)

    def table(header_rows, body_rows):
        """Wrap rows in the centered HTML table markup used by the reports."""

        return [
            '<div align="center">',
            "",
            "<table>",
            "<thead>",
            *header_rows,
            "</thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "",
            "</div>",
            "",
        ]

    lines = [
        "# ADP Ramachandran comparison",
        "",
        "The KLXX generator and the published alanine dipeptide reference data",
        "at 300 K (Zenodo record 6993124, DOI 10.5281/zenodo.6993124) on the",
        "same 100-bin phi/psi grid with the same wrapped smoothing. Each",
        "surface is normalized to its own most populated bin, so only raw",
        "per-method values are reported; a difference between two such surfaces",
        "would carry an arbitrary additive offset.",
        "",
        "The reference topology matches the frozen bundle atom for atom, so the",
        "same phi/psi atom indices apply to both.",
        "",
        "### Sample sets",
        "",
    ]
    lines += table(
        ["<tr><th>method</th><th>samples</th></tr>"],
        [
            f"<tr><td>{names[0]}</td><td>{sample_count:,}</td></tr>",
            f"<tr><td>{names[1]}</td><td>{frames:,}</td></tr>",
        ],
    )

    lines += ["### Maximum free energy", ""]
    cells = ""
    for name in names:
        finite = surfaces[name][np.isfinite(surfaces[name])]
        cells += f"<td>{float(finite.max()):.2f}</td>"
    lines += table(
        ["<tr>" + "".join(f"<th>{name}</th>" for name in names) + "</tr>"],
        [f"<tr>{cells}</tr>"],
    )
    lines += [
        "Maximum free energy over finite bins, in units of $k_{\\mathrm B}T$.",
        "",
        "### Mean free energy by region",
        "",
    ]

    body = []
    for label, mask in (
        (f"low (&lt; {LOW_FREE_ENERGY:.0f})", low),
        (f"high (&gt; {HIGH_FREE_ENERGY:.0f})", high),
        ("all", both),
    ):
        values = "".join(
            f"<td>{surfaces[name][mask].mean():.3f}</td>" for name in names
        )
        body.append(f"<tr><td>{label}</td>{values}</tr>")
    lines += table(
        [
            "<tr><th>region</th>"
            + "".join(f"<th>{name}</th>" for name in names)
            + "</tr>"
        ],
        body,
    )
    lines += [
        "Regions are classified by the reference surface and restricted to bins",
        "finite in both surfaces, so the rows are directly comparable.",
        "",
        "### Backbone basin populations",
        "",
    ]

    body = []
    for name in names:
        alpha_l, beta, alpha_r = basin_populations(counts[name], edges)
        body.append(
            f"<tr><td>{name}</td><td>{alpha_l:.4f}</td>"
            f"<td>{beta:.4f}</td><td>{alpha_r:.4f}</td></tr>"
        )
    lines += table(
        [
            "<tr><th>method</th><th>alpha-L (phi &gt; 0)</th>"
            "<th>beta/PPII (psi &gt; 0)</th><th>alpha-R (psi &lt; 0)</th></tr>"
        ],
        body,
    )

    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"backend={jax.default_backend()} report={report}", flush=True)


if __name__ == "__main__":
    main()

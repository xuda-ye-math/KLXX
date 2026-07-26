#!/usr/bin/env python
"""Tabulate the Ac-Pro-NHMe landscape for KLXX and the OpenMM reference.

Raw per-method values only; no differenced quantities are reported, because the
surfaces are each normalized to their own most populated bin and their
difference would carry an arbitrary additive offset.

Table markup follows ``Codes/Molecular_BG/results.md``: centered HTML tables
with the manuscript's method names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from jflows_md import Molecular_Potential
from jflows_md.boltzmann.load import load_validation_samples, validate_run

from plot_conformational_landscape import (
    BUNDLE,
    CHI1_ATOMS,
    OMEGA_ATOMS,
    PHI_ATOMS,
    PSI_ATOMS,
    dihedral,
    free_energy,
    histograms,
)


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "pt_reference_frames.npz"
DEFAULT_REPORT = HERE / "results" / "landscape_reference_comparison.md"

METHODS = (
    ("klxx", "KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub>"),
)
REFERENCE_LABEL = "OpenMM"

LOW_FREE_ENERGY = 2.0
HIGH_FREE_ENERGY = 3.0
DIAGNOSTICS = ("backbone", "peptide/ring")


def final_stage_samples(run_dir: Path) -> tuple[dict, np.ndarray]:
    """Load the persisted final stage at t=1 of a completed run."""

    record = validate_run(run_dir)
    stages = record.get("stages", [])
    if record.get("status") != "complete" or not stages:
        raise ValueError(f"run is not complete: {run_dir}")
    stage_ref = stages[-1]
    if not math.isclose(float(stage_ref["t"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"final stage has t={stage_ref['t']}, not t=1")
    stage_root = (run_dir / stage_ref["path"]).resolve()
    metadata = json.loads((stage_root / "stage.json").read_text(encoding="utf-8"))
    samples = load_validation_samples(run_dir, int(stage_ref["stage"]), mmap_mode="r")
    return metadata, samples


def reference_histograms(positions: np.ndarray, bins: int):
    """Histogram the reference torsions on the KLXX grid."""

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    backbone = np.histogram2d(
        dihedral(positions, PHI_ATOMS),
        dihedral(positions, PSI_ATOMS),
        bins=(edges, edges),
    )[0].astype(np.int64)
    peptide_ring = np.histogram2d(
        dihedral(positions, OMEGA_ATOMS),
        dihedral(positions, CHI1_ATOMS),
        bins=(edges, edges),
    )[0].astype(np.int64)
    if int(backbone.sum()) != positions.shape[0]:
        raise ValueError("a reference histogram dropped frames")
    return edges, backbone, peptide_ring


def cis_trans(counts: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """Return trans/cis fractions of the omega marginal of a 2D histogram."""

    marginal = counts.sum(axis=1).astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    trans = np.abs(centers) > np.pi / 2.0
    total = marginal.sum()
    return float(marginal[trans].sum() / total), float(marginal[~trans].sum() / total)


def table(header_rows: list[str], body_rows: list[str]) -> list[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.2)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()

    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)

    labels: list[str] = []
    counts: dict[str, dict] = {name: {} for name in DIAGNOSTICS}
    sizes: dict[str, int] = {}
    edges = None

    for method, label in METHODS:
        _, samples = final_stage_samples(HERE / "artifacts" / method)
        method_edges, backbone, peptide_ring, _ = histograms(
            samples, target, args.bins, args.chunk_size
        )
        edges = method_edges if edges is None else edges
        if not np.allclose(edges, method_edges):
            raise ValueError("methods use different histogram edges")
        labels.append(label)
        counts["backbone"][label] = backbone
        counts["peptide/ring"][label] = peptide_ring
        sizes[label] = int(samples.shape[0])
        print(f"{method} samples={samples.shape[0]:,}", flush=True)

    with np.load(args.reference.expanduser().resolve()) as handle:
        positions = np.asarray(handle["positions_nm"], dtype=np.float64)
    reference_edges, backbone, peptide_ring = reference_histograms(
        positions, args.bins
    )
    if not np.allclose(edges, reference_edges):
        raise ValueError("reference uses different histogram edges")
    labels.append(REFERENCE_LABEL)
    counts["backbone"][REFERENCE_LABEL] = backbone
    counts["peptide/ring"][REFERENCE_LABEL] = peptide_ring
    sizes[REFERENCE_LABEL] = int(positions.shape[0])
    print(f"reference frames={positions.shape[0]:,}", flush=True)

    surfaces = {
        diagnostic: {
            label: free_energy(counts[diagnostic][label], args.smoothing)
            for label in labels
        }
        for diagnostic in DIAGNOSTICS
    }

    lines = [
        "# Ac-Pro-NHMe landscape comparison",
        "",
        "The KLXX generator and an independent OpenMM run on the same",
        "100-bin torsion grid with the same wrapped smoothing. Each",
        "surface is normalized to its own most populated bin, so only raw",
        "per-method values are reported; a difference between two such surfaces",
        "would carry an arbitrary additive offset.",
        "",
        "The OpenMM column is native parallel tempering on the",
        "unregularized physical potential at 300 K.",
        "",
        "### Sample sets",
        "",
    ]
    lines += table(
        ["<tr><th>method</th><th>samples</th></tr>"],
        [f"<tr><td>{label}</td><td>{sizes[label]:,}</td></tr>" for label in labels],
    )

    lines += ["### Maximum free energy", ""]
    header = [
        "<tr><th>diagnostic</th>"
        + "".join(f"<th>{label}</th>" for label in labels)
        + "</tr>"
    ]
    body = []
    for diagnostic in DIAGNOSTICS:
        cells = ""
        for label in labels:
            finite = surfaces[diagnostic][label]
            finite = finite[np.isfinite(finite)]
            cells += f"<td>{float(finite.max()):.2f}</td>"
        body.append(f"<tr><td>{diagnostic}</td>{cells}</tr>")
    lines += table(header, body)
    lines += [
        "Maximum free energy over finite bins, in units of $k_{\\mathrm B}T$.",
        "",
        "### Mean free energy by region",
        "",
    ]

    header = [
        "<tr><th>diagnostic</th><th>region</th>"
        + "".join(f"<th>{label}</th>" for label in labels)
        + "</tr>"
    ]
    body = []
    for diagnostic in DIAGNOSTICS:
        here = [surfaces[diagnostic][label] for label in labels]
        finite = np.logical_and.reduce([np.isfinite(s) for s in here])
        reference = surfaces[diagnostic][REFERENCE_LABEL]
        regions = (
            (f"low (&lt; {LOW_FREE_ENERGY:.0f})", finite & (reference < LOW_FREE_ENERGY)),
            (f"high (&gt; {HIGH_FREE_ENERGY:.0f})", finite & (reference > HIGH_FREE_ENERGY)),
            ("all", finite),
        )
        # The diagnostic name is spanned rather than repeated on every region.
        for index, (label_region, mask) in enumerate(regions):
            cells = "".join(
                f"<td>{surfaces[diagnostic][label][mask].mean():.3f}</td>"
                for label in labels
            )
            span = (
                f'<td rowspan="{len(regions)}">{diagnostic}</td>' if index == 0 else ""
            )
            body.append(f"<tr>{span}<td>{label_region}</td>{cells}</tr>")
    lines += table(header, body)
    lines += [
        "Regions are classified by the OpenMM surface and restricted to bins",
        "finite in both surfaces, so the rows are directly comparable.",
        "",
        "### ACE-PRO peptide torsion populations",
        "",
    ]

    body = []
    for label in labels:
        trans, cis = cis_trans(counts["peptide/ring"][label], edges)
        body.append(f"<tr><td>{label}</td><td>{trans:.4f}</td><td>{cis:.4f}</td></tr>")
    lines += table(
        ["<tr><th>method</th><th>trans</th><th>cis</th></tr>"],
        body,
    )

    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"backend={jax.default_backend()} report={report}", flush=True)


if __name__ == "__main__":
    main()

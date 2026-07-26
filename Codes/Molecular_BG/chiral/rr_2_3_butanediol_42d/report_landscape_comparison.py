#!/usr/bin/env python
"""Tabulate the butanediol landscape for KLXX and the OpenMM reference.

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

from plot_conformational_landscape import (
    BUNDLE,
    CCCC_ATOMS,
    OCCO_ATOMS,
    OH1_ATOMS,
    OH2_ATOMS,
    dihedral,
    free_energy,
    histograms,
    selected_inference_stages,
)


HERE = Path(__file__).resolve().parent
KLXX_RUN = HERE / "artifacts" / "inference_10M"
REFERENCE = HERE / "reference" / "pt_reference_frames_100000.npz"
DEFAULT_REPORT = HERE / "results" / "landscape_reference_comparison.md"

METHODS = (
    ("klxx", "KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub>"),
)
REFERENCE_LABEL = "OpenMM"

LOW_FREE_ENERGY = 2.0
HIGH_FREE_ENERGY = 3.0


def inference_final_stage(run_dir: Path) -> tuple[dict, np.ndarray]:
    """Load the final stage at t=1 of a completed inference-only run.

    ``selected_inference_stages`` already refuses a run that is incomplete, that
    does not declare ``inference_only``, that reports a nonzero training-update
    count, or whose final stage is not at t=1.
    """

    metadata, samples = selected_inference_stages(run_dir)[-1]
    if not math.isclose(float(metadata["t"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"final stage has t={metadata['t']}, not t=1")
    return metadata, samples


def reference_histograms(positions: np.ndarray, bins: int):
    """Histogram the reference torsions on the KLXX grid."""

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    heavy = np.histogram2d(
        dihedral(positions, CCCC_ATOMS),
        dihedral(positions, OCCO_ATOMS),
        bins=(edges, edges),
    )[0].astype(np.int64)
    hydroxyl = np.histogram2d(
        dihedral(positions, OH1_ATOMS),
        dihedral(positions, OH2_ATOMS),
        bins=(edges, edges),
    )[0].astype(np.int64)
    if int(heavy.sum()) != positions.shape[0]:
        raise ValueError("a reference histogram dropped frames")
    return edges, heavy, hydroxyl


def rotamers(counts: np.ndarray, edges: np.ndarray) -> tuple[float, float, float]:
    """Return gauche-minus/gauche-plus/trans fractions of the C-C-C-C marginal."""

    marginal = counts.sum(axis=1).astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    trans = np.abs(centers) >= 2.0 * np.pi / 3.0
    total = marginal.sum()
    return (
        float(marginal[(centers < 0.0) & ~trans].sum() / total),
        float(marginal[(centers >= 0.0) & ~trans].sum() / total),
        float(marginal[trans].sum() / total),
    )


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
    parser.add_argument("--klxx-run", type=Path, default=KLXX_RUN)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.2)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()

    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)

    labels: list[str] = []
    counts: dict[str, dict] = {"heavy-atom": {}, "hydroxyl": {}}
    sizes: dict[str, int] = {}
    arrays: dict[str, np.ndarray] = {}
    edges = None

    for method, label in METHODS:
        metadata, samples = inference_final_stage(args.klxx_run.expanduser().resolve())
        method_edges, heavy, hydroxyl, _ = histograms(
            samples, target, args.bins, args.chunk_size
        )
        edges = method_edges if edges is None else edges
        if not np.allclose(edges, method_edges):
            raise ValueError("methods use different histogram edges")
        labels.append(label)
        counts["heavy-atom"][label] = heavy
        counts["hydroxyl"][label] = hydroxyl
        sizes[label] = int(samples.shape[0])
        arrays[label] = samples
        print(f"{method} samples={samples.shape[0]:,}", flush=True)

    with np.load(args.reference.expanduser().resolve()) as handle:
        positions = np.asarray(handle["positions_nm"], dtype=np.float64)
    reference_edges, heavy, hydroxyl = reference_histograms(positions, args.bins)
    if not np.allclose(edges, reference_edges):
        raise ValueError("reference uses different histogram edges")
    labels.append(REFERENCE_LABEL)
    counts["heavy-atom"][REFERENCE_LABEL] = heavy
    counts["hydroxyl"][REFERENCE_LABEL] = hydroxyl
    sizes[REFERENCE_LABEL] = int(positions.shape[0])
    print(f"reference frames={positions.shape[0]:,}", flush=True)

    # A surface normalized to its own most populated bin cannot resolve deeper
    # than about log(N): the deepest occupied bin is the one holding a single
    # sample.  The raw maxima are therefore not comparable across sets of
    # different size, so each generated set is also histogrammed after being
    # thinned to the reference frame count.  This is a measurement at matched
    # size, not a shift applied to the smaller set -- a log(N) shift is only
    # valid where the surface is count-limited at *both* sizes, which holds for
    # the sparse heavy-atom torsions and fails for the hydroxyl pair.
    matched = {"heavy-atom": {}, "hydroxyl": {}}
    reference_size = sizes[REFERENCE_LABEL]
    for label, samples in arrays.items():
        stride = max(1, samples.shape[0] // reference_size)
        subset = samples[::stride][:reference_size]
        matched_edges, heavy, hydroxyl, _ = histograms(
            subset, target, args.bins, args.chunk_size
        )
        if not np.allclose(edges, matched_edges):
            raise ValueError("thinned set uses different histogram edges")
        matched["heavy-atom"][label] = heavy
        matched["hydroxyl"][label] = hydroxyl
        print(
            f"thinned {label}: stride {stride} -> {subset.shape[0]:,} samples",
            flush=True,
        )
    for diagnostic in matched:
        matched[diagnostic][REFERENCE_LABEL] = counts[diagnostic][REFERENCE_LABEL]

    surfaces = {
        diagnostic: {
            label: free_energy(counts[diagnostic][label], args.smoothing)
            for label in labels
        }
        for diagnostic in counts
    }
    matched_surfaces = {
        diagnostic: {
            label: free_energy(matched[diagnostic][label], args.smoothing)
            for label in labels
        }
        for diagnostic in matched
    }

    lines = [
        "# (2R,3R)-2,3-Butanediol landscape comparison",
        "",
        "The KLXX generator and an independent OpenMM run on the same",
        "100-bin torsion grid with the same wrapped smoothing. Each",
        "surface is normalized to its own most populated bin, so only raw",
        "per-method values are reported; a difference between two such surfaces",
        "would carry an arbitrary additive offset.",
        "",
        "The KLXX column is the final stage at $t=1$ of an inference-only run",
        "with no training update. The OpenMM column is native parallel",
        "tempering on the unregularized physical potential at 300 K: one",
        "continuous chain over six replicas on a geometric 300--800 K grid,",
        "50.0 ns retained after a 0.5 ns discarded equilibration.",
        "",
        "The two sample counts differ by two orders of magnitude, as the",
        "sample-set table records. Generating flow samples is cheap and",
        "parallel; advancing a single tempered trajectory is neither.",
        "",
        "### Sample sets",
        "",
    ]
    lines += table(
        ["<tr><th>method</th><th>samples</th></tr>"],
        [f"<tr><td>{label}</td><td>{sizes[label]:,}</td></tr>" for label in labels],
    )

    lines += ["### Maximum free energy", ""]
    header = (
        "<tr><th>diagnostic</th><th>sample count</th>"
        + "".join(f"<th>{label}</th>" for label in labels)
        + "</tr>",
    )
    body = []
    rows = (
        ("as sampled", surfaces),
        (f"thinned to {reference_size:,}", matched_surfaces),
    )
    for diagnostic in ("heavy-atom", "hydroxyl"):
        # The diagnostic name is spanned rather than repeated on every row.
        for index, (row_label, source) in enumerate(rows):
            cells = ""
            for label in labels:
                surface = source[diagnostic][label]
                finite = surface[np.isfinite(surface)]
                cells += f"<td>{float(finite.max()):.2f}</td>"
            span = f'<td rowspan="{len(rows)}">{diagnostic}</td>' if index == 0 else ""
            body.append(f"<tr>{span}<td>{row_label}</td>{cells}</tr>")
    lines += table(list(header), body)
    lines += [
        "Maximum free energy over finite bins, in units of $k_{\\mathrm B}T$.",
        "",
        "Each surface is normalized to its own most populated bin, so the",
        "deepest value it can resolve is set by the bin holding a single",
        "sample and grows as $\\log N$. The `as sampled` row therefore compares",
        "two different resolutions rather than two free-energy surfaces. The",
        "`thinned` row removes that by histogramming the generated set at the",
        "reference frame count, which is a measurement at matched size rather",
        "than a correction applied to either column.",
        "",
        "### Mean free energy by region",
        "",
    ]

    header = (
        "<tr><th>diagnostic</th><th>region</th>"
        + "".join(f"<th>{label}</th>" for label in labels)
        + "</tr>",
    )
    body = []
    for diagnostic in ("heavy-atom", "hydroxyl"):
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
    lines += table(list(header), body)
    lines += [
        "Regions are classified by the OpenMM surface and restricted to bins",
        "finite in both surfaces, so the rows are directly comparable.",
        "",
        "### Central C-C-C-C rotamer populations",
        "",
    ]

    body = []
    for label in labels:
        minus, plus, trans = rotamers(counts["heavy-atom"][label], edges)
        body.append(
            f"<tr><td>{label}</td><td>{minus:.4f}</td>"
            f"<td>{plus:.4f}</td><td>{trans:.4f}</td></tr>"
        )
    lines += table(
        ["<tr><th>method</th><th>gauche&minus;</th><th>gauche+</th><th>trans</th></tr>"],
        body,
    )

    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"backend={jax.default_backend()} report={report}", flush=True)


if __name__ == "__main__":
    main()

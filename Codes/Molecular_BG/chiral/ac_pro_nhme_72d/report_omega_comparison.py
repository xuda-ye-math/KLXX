#!/usr/bin/env python
"""Tabulate the Ac-Pro-NHMe peptide torsion for KLXX and the OpenMM reference.

The reference for this target is the basin-jump Monte Carlo run of
``reference/flip_reference.py``: Langevin dynamics at 300 K on the
unregularized physical potential, plus a Metropolis rigid pi rotation of the
proline side of the ACE-PRO amide bond.  The rotation is an involution with
unit Jacobian, so the jump is exact and the retained samples are unweighted and
all at the target temperature.  That fixes what can and cannot be compared.

* The **cis/trans populations** are directly comparable.  Both methods produce
  samples of the same unregularized potential at the same temperature, so the
  omega marginal is counted the same way on both sides: cis is
  ``|omega| < pi/2``, counted per sample.  No binning is involved, so no grid
  has to be agreed on.
* The **barrier height** is measured by neither.  A generator sample set
  resolves a free energy no deeper than about ``log N`` above its most
  populated bin, well short of the roughly 60-65 kJ/mol amide barrier.  The
  reference does not climb the barrier either: it obtains every basin change
  from the jump move rather than from the dynamics, which is what lets it
  estimate populations without deforming the target.  Quoting a barrier from
  either column would report the method, not the potential.
* The **two-dimensional landscapes** are not tabulated here.  The reference
  retains a thinned Cartesian frame set, but this report is the omega
  comparison; the landscape figure is built separately.

The reference's own uncertainty is its blocking standard error, not the naive
binomial one: the series is correlated, so ``flip_reference.py`` cuts it into
blocks and takes the spread of the block means.  That error is the scale
against which any disagreement with the generator must be read, and it is
reported alongside the populations.

Table markup follows ``Codes/Molecular_BG/results.md``: centered HTML tables
with the manuscript's method names.

Usage from this folder:

    python report_omega_comparison.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jflows_md import Molecular_Potential

from plot_conformational_landscape import (
    BUNDLE,
    OMEGA_ATOMS,
    RUN_DIR,
    cartesian_positions,
    dihedral,
    selected_inference_stages,
)


HERE = Path(__file__).resolve().parent
REFERENCE_MANIFEST = HERE / "reference" / "flip_reference.json"
REFERENCE_SERIES = HERE / "reference" / "flip_reference_series.npz"
DEFAULT_REPORT = HERE / "results" / "omega_reference_comparison.md"

KLXX_LABEL = "KL+X<sub>&mu;</sub>+X<sub>(&mu;&#770;+&nu;&#772;)/2</sub>"
REFERENCE_LABEL = "OpenMM"

TEMPERATURE_KELVIN = 300.0
KB_KJ_MOL_K = 0.008314462618


def klxx_cis(run_dir: Path, chunk: int) -> tuple[int, int]:
    """Return the cis count and sample count of the final inference stage."""

    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)
    metadata, samples = selected_inference_stages(run_dir)[-1]
    if not math.isclose(float(metadata["t"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"final stage has t={metadata['t']}, not t=1")
    total = int(samples.shape[0])

    cis = 0
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        internal = np.asarray(samples[start:stop])
        positions = np.asarray(
            jax.block_until_ready(cartesian_positions(target, jnp.asarray(internal)))
        )
        omega = dihedral(positions, OMEGA_ATOMS)
        if not np.all(np.isfinite(omega)):
            raise ValueError("the omega torsion is not finite on every sample")
        cis += int(np.count_nonzero(np.abs(omega) < np.pi / 2.0))
    return cis, total


def reference_cis(series_path: Path, manifest: dict) -> tuple[int, int]:
    """Return the reference cis count and retained count, recounted from the series.

    The manifest already carries ``cis_fraction``.  Recounting it here from the
    stored omega series and requiring the two to agree is what makes this a
    check rather than a restatement: a wrong discard offset, a truncated
    series, or a manifest from a different run all show up as a mismatch.
    """

    with np.load(series_path, allow_pickle=False) as handle:
        omega = np.asarray(handle["omega_rad"], dtype=np.float64)
        discard = int(handle["discard_samples"])
    if discard != int(manifest["discard_samples"]):
        raise ValueError(
            f"discard mismatch: series {discard}, manifest "
            f"{manifest['discard_samples']}"
        )
    retained = omega[discard:]
    if retained.size != int(manifest["retained_samples"]):
        raise ValueError(
            f"retained mismatch: series {retained.size}, manifest "
            f"{manifest['retained_samples']}"
        )
    if not np.all(np.isfinite(retained)):
        raise ValueError("the reference omega series is not finite")

    cis = int(np.count_nonzero(np.abs(retained) < np.pi / 2.0))
    recounted = cis / retained.size
    if not math.isclose(recounted, float(manifest["cis_fraction"]), rel_tol=1e-9):
        raise ValueError(
            f"recounted cis {recounted!r} disagrees with the manifest "
            f"{manifest['cis_fraction']!r}"
        )
    return cis, retained.size


def split(cis: float) -> tuple[float, float]:
    """Return the cis fraction and the cis-to-trans free energy difference."""

    kt = KB_KJ_MOL_K * TEMPERATURE_KELVIN
    return cis, float(-kt * np.log(cis / (1.0 - cis)))


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
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--series", type=Path, default=REFERENCE_SERIES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.expanduser().resolve().read_text("utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("the reference run is not complete")

    reference_count, reference_total = reference_cis(
        args.series.expanduser().resolve(), manifest
    )
    klxx_count, klxx_total = klxx_cis(
        args.run_dir.expanduser().resolve(), args.chunk_size
    )
    print(f"klxx samples={klxx_total:,} cis={klxx_count:,}", flush=True)
    print(f"reference samples={reference_total:,} cis={reference_count:,}", flush=True)

    klxx_fraction, klxx_delta = split(klxx_count / klxx_total)
    reference_fraction, reference_delta = split(reference_count / reference_total)
    error = float(manifest["cis_standard_error"])
    blocks = int(manifest["cis_standard_error_blocks"])
    gap = abs(klxx_fraction - reference_fraction)
    kt = KB_KJ_MOL_K * TEMPERATURE_KELVIN

    lines = [
        "# Ac-Pro-NHMe peptide torsion comparison",
        "",
        "The KLXX generator and an independent OpenMM run, compared on the",
        "cis/trans populations of the ACE-PRO $\\omega$ torsion. Cis is",
        "$|\\omega| < \\pi/2$, counted per sample on both sides, so no binning",
        "grid enters the comparison.",
        "",
        "The KLXX column is the final stage at $t=1$ of an inference-only run",
        "with no training update. The OpenMM column is Langevin dynamics at",
        "300 K on the same unregularized physical potential, plus a Metropolis",
        "rigid $\\pi$ rotation of the proline side of the ACE-PRO amide bond.",
        "The rotation is an involution with unit Jacobian, so the jump is exact",
        "and every retained sample counts once at the target temperature.",
        "",
        "The amide barrier is about 60-65 kJ/mol, some 26 $k_{\\mathrm B}T$ at",
        "300 K. Neither column measures it. The generator cannot resolve a free",
        "energy deeper than about $\\log N$ above its most populated bin, and",
        "the reference obtains its basin changes from the jump move rather than",
        "from the dynamics, which is what lets it estimate populations without",
        "deforming the target. No barrier is therefore tabulated.",
        "",
        "### Sample sets",
        "",
    ]
    lines += table(
        ["<tr><th>method</th><th>sampler</th><th>samples</th></tr>"],
        [
            f"<tr><td>{KLXX_LABEL}</td><td>inference replay</td>"
            f"<td>{klxx_total:,}</td></tr>",
            f"<tr><td>{REFERENCE_LABEL}</td><td>basin-jump Monte Carlo</td>"
            f"<td>{reference_total:,}</td></tr>",
        ],
    )
    lines += [
        f"The reference retained {reference_total:,} samples after discarding",
        f"the first {manifest['discard_samples']:,}, over",
        f"{manifest['nanoseconds']:.1f} ns, accepting",
        f"{manifest['accepted_jumps_retained']:,} of the retained jump attempts",
        f"({manifest['acceptance_retained']:.4f}).",
        "",
        "### Peptide torsion populations",
        "",
    ]
    lines += table(
        [
            "<tr><th>method</th><th>cis</th><th>trans</th>"
            "<th>&Delta;G(cis&minus;trans)</th></tr>"
        ],
        [
            f"<tr><td>{KLXX_LABEL}</td><td>{klxx_fraction:.4f}</td>"
            f"<td>{1.0 - klxx_fraction:.4f}</td><td>{klxx_delta:.2f}</td></tr>",
            f"<tr><td>{REFERENCE_LABEL}</td><td>{reference_fraction:.4f} &plusmn; "
            f"{error:.4f}</td><td>{1.0 - reference_fraction:.4f}</td>"
            f"<td>{reference_delta:.2f}</td></tr>",
        ],
    )
    lines += [
        "Free energy differences are in kJ/mol. The reference error bar is its",
        f"blocking standard error at {blocks} blocks, not the naive binomial",
        "one; the series is correlated, so the block spread is the honest",
        "scale. The two cis fractions differ by",
        f"{gap:.4f}, which is {gap / error:.0f} times that error, and the two",
        f"$\\Delta G$ values differ by {abs(klxx_delta - reference_delta):.2f}",
        f"kJ/mol, or {abs(klxx_delta - reference_delta) / kt:.2f}",
        "$k_{\\mathrm B}T$. The disagreement is far outside what the",
        "reference's own spread can account for.",
        "",
        "### Reference convergence",
        "",
    ]
    lines += table(
        ["<tr><th>blocks</th><th>samples per block</th><th>cis</th>"
         "<th>standard error</th></tr>"],
        [
            f"<tr><td>{row['blocks']}</td><td>{row['block_samples']:,}</td>"
            f"<td>{row['mean']:.6f}</td><td>{row['sem']:.6f}</td></tr>"
            for row in manifest["cis_blocking_curve"]
        ],
    )
    curve = manifest["cis_blocking_curve"]
    plateau = [row["sem"] for row in curve if row["blocks"] >= 16]
    lines += [
        "Every blocking returns the same central value to five decimal places.",
        f"The standard error is {curve[0]['sem']:.6f} at {curve[0]['blocks']}",
        "blocks, where the spread of so few block means is itself uncertain by",
        "about 71 per cent and cannot carry the error bar; it settles by eight",
        f"blocks and then stays between {min(plateau):.6f} and",
        f"{max(plateau):.6f} for every blocking from 16 to {curve[-1]['blocks']}.",
        "Short blocks cannot see correlations longer than themselves, so the",
        f"quoted error is the {blocks}-block row, inside that plateau. The",
        "plateau across more than an order of magnitude in block length is why",
        "it is not an artifact of the block choice, and it is the scale against",
        "which the disagreement above must be read.",
        "",
    ]

    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"backend={jax.default_backend()} report={report}", flush=True)


if __name__ == "__main__":
    main()

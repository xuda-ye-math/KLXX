#!/usr/bin/env python
"""Three-row butanediol landscape: KLXX, KLXX reweighted to raw, reference.

Row 1 repeats the final KLXX stage as in ``plot_landscape_reference.py``.
Row 2 applies the endpoint-to-physical correction: the same samples are
reweighted from the regularized target to the raw physical potential by the
regularization ESS (RESS) weights, resampled, and rejuvenated with mixed-domain
MALA under the raw potential.  Row 3 is the OpenMM reference.

The comparison isolates whether that correction step changes the landscape.
The log-weight convention follows ``Codes/Molecular_BG/alkane_family/
regularization_ess.py``: ``log w = U_regularized(q) - U_raw(q)``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib


matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from jflows.utils import compute_ESS_log
from jflows_md import Molecular_Potential
from jflows_md.boltzmann.load import load_validation_samples, validate_run
from jflows_md.utils import mixed_mala

from parameters import CHUNKS, MC_DT, MC_IMAGE_RADIUS, MC_STEPS, RG_PARAM_1
from plot_conformational_landscape import (
    BUNDLE,
    CCCC_ATOMS,
    COLUMN_FREE_ENERGY_LIMITS,
    FREE_ENERGY_CMAP,
    OCCO_ATOMS,
    OH1_ATOMS,
    OH2_ATOMS,
    RUN_DIR,
    dihedral,
    free_energy,
    histograms,
)


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "pt_reference_frames.npz"
DEFAULT_OUTPUT = HERE / "results" / "conformational_landscape_reference_v2.png"
DEFAULT_REPORT = HERE / "results" / "landscape_reference_comparison.md"
SEED = 0


def final_stage(run_dir: Path) -> tuple[dict, np.ndarray]:
    """Load the persisted final KLXX stage at t=1."""

    record = validate_run(run_dir)
    stages = record.get("stages", [])
    if record.get("status") != "complete" or not stages:
        raise ValueError("KLXX run is not complete")
    stage_ref = stages[-1]
    if not math.isclose(float(stage_ref["t"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"final stage has t={stage_ref['t']}, not t=1")
    stage_number = int(stage_ref["stage"])
    stage_root = (run_dir / stage_ref["path"]).resolve()
    try:
        stage_root.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError(f"stage path escapes run directory: {stage_root}") from error
    metadata = json.loads((stage_root / "stage.json").read_text(encoding="utf-8"))
    samples = load_validation_samples(run_dir, stage_number, mmap_mode="r")
    if samples.ndim != 2 or samples.dtype != np.float32:
        raise ValueError(
            f"expected a float32 sample matrix, found {samples.shape} {samples.dtype}"
        )
    return metadata, samples


def reference_histograms(
    positions: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram the reference torsions on the same grid as the KLXX samples."""

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    cccc = dihedral(positions, CCCC_ATOMS)
    occo = dihedral(positions, OCCO_ATOMS)
    oh1 = dihedral(positions, OH1_ATOMS)
    oh2 = dihedral(positions, OH2_ATOMS)
    heavy = np.histogram2d(cccc, occo, bins=(edges, edges))[0].astype(np.int64)
    hydroxyl = np.histogram2d(oh1, oh2, bins=(edges, edges))[0].astype(np.int64)
    frames = positions.shape[0]
    if int(heavy.sum()) != frames or int(hydroxyl.sum()) != frames:
        raise ValueError("a reference histogram dropped frames")
    return edges, heavy, hydroxyl

# Regions are classified by the OpenMM reference surface, so every method is
# scored on the same bins.
LOW_FREE_ENERGY = 2.0
HIGH_FREE_ENERGY = 3.0


def rotamer_occupancy(counts: np.ndarray, edges: np.ndarray) -> tuple[float, ...]:
    """Return gauche-minus/gauche-plus/trans fractions of the C-C-C-C marginal."""

    marginal = counts.sum(axis=1).astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    trans = np.abs(centers) >= 2.0 * np.pi / 3.0
    gauche_plus = (centers >= 0.0) & ~trans
    gauche_minus = (centers < 0.0) & ~trans
    total = marginal.sum()
    return (
        float(marginal[gauche_minus].sum() / total),
        float(marginal[gauche_plus].sum() / total),
        float(marginal[trans].sum() / total),
    )


def write_report(
    path: Path,
    edges: np.ndarray,
    panels: dict,
    counts: dict,
    ress: float,
    acceptance: float,
) -> None:
    """Write the three-way free-energy comparison table."""

    names = ("KLXX", "KLXX resampled + rejuvenated", "OpenMM reference")
    lines = [
        "# Butanediol landscape comparison",
        "",
        "Three sample sets on the same 100-bin torsion grid with the same",
        "wrapped smoothing, each surface normalized to its own most populated",
        "bin. `KLXX resampled + rejuvenated` is the KLXX set reweighted by the",
        "regularization ESS weights, resampled, then rejuvenated with",
        "mixed-domain MALA under the raw potential.",
        "",
        f"- Regularization ESS (RESS): `{ress:.6f}`",
        f"- Mixed-domain MALA acceptance during rejuvenation: `{acceptance:.4f}`",
        "",
        "## Surface extent",
        "",
        "| diagnostic | sample set | occupied bins | max free energy / kBT |",
        "|---|---|---:|---:|",
    ]
    for diagnostic in ("heavy-atom", "hydroxyl"):
        for name in names:
            surface = panels[diagnostic][name]
            occupied = int(np.isfinite(surface).sum())
            lines.append(
                f"| {diagnostic} | {name} | {occupied} | "
                f"{float(np.nanmax(surface)):.2f} |"
            )
    lines += [
        "",
        "## Free energy by region",
        "",
        "Bins are classified by the OpenMM reference: low means below "
        f"{LOW_FREE_ENERGY:.0f} kBT, high means above {HIGH_FREE_ENERGY:.0f} kBT.",
        "Entries are the mean free energy of each set over those bins, in kBT.",
        "Only bins finite in all three surfaces are used, so the rows are",
        "directly comparable.",
        "",
        "| diagnostic | region | bins | " + " | ".join(names) + " |",
        "|---|---|---:|---:|---:|---:|",
    ]
    masks = {}
    for diagnostic in ("heavy-atom", "hydroxyl"):
        surfaces_here = [panels[diagnostic][name] for name in names]
        finite = np.logical_and.reduce([np.isfinite(s) for s in surfaces_here])
        reference = panels[diagnostic]["OpenMM reference"]
        regions = (
            (f"low F (< {LOW_FREE_ENERGY:.0f} kBT)", finite & (reference < LOW_FREE_ENERGY)),
            (f"high F (> {HIGH_FREE_ENERGY:.0f} kBT)", finite & (reference > HIGH_FREE_ENERGY)),
            ("all bins", finite),
        )
        masks[diagnostic] = regions
        for label, mask in regions:
            values = " | ".join(
                f"{panels[diagnostic][name][mask].mean():.3f}" for name in names
            )
            lines.append(f"| {diagnostic} | {label} | {int(mask.sum())} | {values} |")

    lines += [
        "",
        "## Central C-C-C-C rotamer populations",
        "",
        "| sample set | gauche-minus | gauche-plus | trans |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        minus, plus, trans = rotamer_occupancy(counts[name], edges)
        lines.append(f"| {name} | {minus:.4f} | {plus:.4f} | {trans:.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@eqx.filter_jit
def log_weight(samples, regularized, exact):
    return regularized(samples) - exact(samples)


def reweighted_samples(
    samples: np.ndarray,
    target: Molecular_Potential,
    chunks: int,
) -> tuple[np.ndarray, float, float]:
    """Resample by RESS weights and rejuvenate under the raw potential."""

    regularized = target.regularized(RG_PARAM_1)
    parts = []
    for part in np.array_split(np.asarray(samples), chunks, axis=0):
        parts.append(
            np.asarray(
                jax.block_until_ready(
                    log_weight(jnp.asarray(part), regularized, target)
                )
            )
        )
    weights = np.concatenate(parts)
    ess = float(jax.block_until_ready(compute_ESS_log(jnp.asarray(weights))))

    shifted = weights - weights.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    generator = np.random.default_rng(SEED)
    index = generator.choice(
        weights.size, size=weights.size, replace=True, p=probabilities
    )
    resampled = jnp.asarray(np.asarray(samples)[np.sort(index)])

    rejuvenated, acceptance = mixed_mala(
        jax.random.key(SEED),
        resampled,
        target,
        target.domain,
        dt=MC_DT,
        steps=MC_STEPS,
        image_radius=MC_IMAGE_RADIUS,
        chunks=chunks,
    )
    # ``mixed_mala`` returns the per-step acceptance history, one entry per
    # MALA step; report its mean over the chain.
    return np.asarray(rejuvenated), ess, float(np.asarray(acceptance).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.2)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    if args.bins <= 0 or args.smoothing < 0.0 or args.chunk_size <= 0:
        parser.error("bins and chunk size must be positive; smoothing nonnegative")

    output = args.output.expanduser().resolve()
    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)

    with np.load(args.reference.expanduser().resolve()) as handle:
        positions = np.asarray(handle["positions_nm"], dtype=np.float64)
    atoms = (target.dimension + 6) // 3
    if positions.ndim != 3 or positions.shape[1:] != (atoms, 3):
        raise ValueError(
            f"expected reference frames [F, {atoms}, 3], found {positions.shape}"
        )
    edges, reference_heavy, reference_hydroxyl = reference_histograms(
        positions, args.bins
    )
    print(f"reference frames={positions.shape[0]:,}", flush=True)

    metadata, samples = final_stage(args.run_dir.expanduser().resolve())
    if target.dimension != samples.shape[1] or samples.shape[1] != 42:
        raise ValueError(
            f"unexpected dimension: target={target.dimension}, samples={samples.shape}"
        )
    klxx_edges, klxx_heavy, klxx_hydroxyl, _ = histograms(
        samples, target, args.bins, args.chunk_size
    )
    if not np.allclose(edges, klxx_edges):
        raise ValueError("KLXX and reference histograms use different edges")
    print(
        f"klxx stage={int(metadata['stage'])} samples={samples.shape[0]:,}",
        flush=True,
    )

    corrected, ress, acceptance = reweighted_samples(samples, target, CHUNKS)
    corrected_edges, corrected_heavy, corrected_hydroxyl, _ = histograms(
        corrected, target, args.bins, args.chunk_size
    )
    if not np.allclose(edges, corrected_edges):
        raise ValueError("corrected histogram uses different edges")
    print(
        f"corrected samples={corrected.shape[0]:,} RESS={ress:.6g} "
        f"mala_acceptance={acceptance:.4f}",
        flush=True,
    )

    rows = (
        (
            (
                rf"KLXX stage {int(metadata['stage'])}",
                rf"$t={float(metadata['t']):.3f}$, "
                rf"$\rho=({float(metadata['rg_end'][0]):.3g},"
                rf"{float(metadata['rg_end'][1]):.3g})$",
            ),
            (klxx_heavy, klxx_hydroxyl),
        ),
        (
            (
                "KLXX resampled + rejuvenated",
                rf"unregularized, RESS $={ress:.4f}$",
            ),
            (corrected_heavy, corrected_hydroxyl),
        ),
        (
            ("OpenMM reference", r"unregularized, $300$ K"),
            (reference_heavy, reference_hydroxyl),
        ),
    )
    surfaces = [
        (labels, tuple(free_energy(counts, args.smoothing) for counts in pair))
        for labels, pair in rows
    ]

    vlimits = []
    for column in range(2):
        finite = np.concatenate(
            [pair[column][np.isfinite(pair[column])] for _, pair in surfaces]
        )
        if finite.size == 0:
            raise ValueError(f"column {column} has no finite free-energy values")
        fixed = COLUMN_FREE_ENERGY_LIMITS[column]
        if fixed is not None:
            vlimits.append(float(fixed))
        else:
            vlimits.append(max(float(math.ceil(float(finite.max()))), 1.0))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(9.8, 13.2), layout="constrained")
    # Tighten the column gap; the default constrained spacing leaves a wide
    # channel around the column-0 colorbar.
    figure.get_layout_engine().set(wspace=0.0, w_pad=0.01)
    diagnostics = (
        (r"C-C-C-C dihedral (rad)", r"O-C-C-O dihedral (rad)"),
        (r"H-O1-C-C dihedral (rad)", r"H-O2-C-C dihedral (rad)"),
    )
    panel_letters = ("a", "b", "c", "d", "e", "f")
    images = [None, None]
    for row, ((heading, subheading), pair) in enumerate(surfaces):
        for column, (xlabel, ylabel) in enumerate(diagnostics):
            axis = axes[row, column]
            letter = panel_letters[2 * row + column]
            images[column] = axis.pcolormesh(
                edges,
                edges,
                pair[column].T,
                cmap=FREE_ENERGY_CMAP,
                vmin=0.0,
                vmax=vlimits[column],
                shading="flat",
                rasterized=True,
            )
            axis.set(
                xlim=(-np.pi, np.pi),
                ylim=(-np.pi, np.pi),
                xlabel=xlabel,
                ylabel=ylabel,
                title=f"({letter}) {heading}\n{subheading}",
            )
            axis.set_aspect("equal")
            axis.set_box_aspect(1)
            axis.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
            axis.set_yticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
            for spine in axis.spines.values():
                spine.set_linewidth(1.0)

    for column, image in enumerate(images):
        colorbar = figure.colorbar(
            image, ax=axes[:, column], pad=0.008, fraction=0.032, shrink=0.95
        )
        tick_step = 2.0 if vlimits[column] > 6.0 else 1.0
        colorbar.set_ticks(np.arange(0.0, vlimits[column] + 0.1, tick_step))
        colorbar.set_label(r"free energy / $k_{\mathrm{B}}T$")

    figure.suptitle("(2R,3R)-2,3-butanediol", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

    names = ("KLXX", "KLXX resampled + rejuvenated", "OpenMM reference")
    panels = {
        "heavy-atom": {
            name: surfaces[row][1][0] for row, name in enumerate(names)
        },
        "hydroxyl": {
            name: surfaces[row][1][1] for row, name in enumerate(names)
        },
    }
    heavy_counts = (klxx_heavy, corrected_heavy, reference_heavy)
    write_report(
        args.report.expanduser().resolve(),
        edges,
        panels,
        dict(zip(names, heavy_counts)),
        ress,
        acceptance,
    )
    print(
        f"backend={jax.default_backend()} output={output} "
        f"report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()

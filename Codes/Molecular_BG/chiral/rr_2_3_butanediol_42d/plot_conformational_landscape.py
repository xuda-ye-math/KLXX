#!/usr/bin/env python
"""Plot intermediate- and final-stage butanediol conformational landscapes."""

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
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter

from jflows_md import Molecular_Potential
from jflows_md.boltzmann.load import load_validation_samples, validate_run


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "artifacts" / "inference_10M"
BUNDLE = HERE / "bundle"
DEFAULT_OUTPUT = HERE / "results" / "conformational_landscape.png"

# Zero-based PDB atom indices, matching the audited regularization observables.
CCCC_ATOMS = (4, 2, 3, 5)
OCCO_ATOMS = (0, 2, 3, 1)
OH1_ATOMS = (14, 0, 2, 3)
OH2_ATOMS = (15, 1, 3, 2)

# Per-column free-energy colorbar limits; ``None`` rounds up to the observed
# maximum.  Both columns are pinned so the landscape figures share one scale.
COLUMN_FREE_ENERGY_LIMITS = (11.0, 2.5)

FREE_ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "reference_free_energy",
    (
        "#1b3569",
        "#286e86",
        "#369893",
        "#4aac8e",
        "#73c077",
        "#a6d656",
        "#dcdc47",
        "#ffd443",
        "#ffb450",
        "#ef894b",
        "#cc5338",
    ),
)
FREE_ENERGY_CMAP.set_bad("white")

# The hydroxyl column uses a pinned range; anything above it is shown as
# white rather than saturating at the top colour, so the modes stay legible.
FREE_ENERGY_CMAP_CLIPPED = FREE_ENERGY_CMAP.copy()
FREE_ENERGY_CMAP_CLIPPED.set_over("white")
COLUMN_CMAPS = (FREE_ENERGY_CMAP, FREE_ENERGY_CMAP_CLIPPED)


@eqx.filter_jit
def cartesian_positions(
    target: Molecular_Potential, internal: jax.Array
) -> jax.Array:
    return target.cartesian(internal)


def dihedral(positions: np.ndarray, atoms: tuple[int, int, int, int]) -> np.ndarray:
    """Return wrapped right-handed a-b-c-d dihedrals in radians."""

    a, b, c, d = (positions[:, index] for index in atoms)
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    norm = np.linalg.norm(b1, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError(f"zero-length central bond for atom tuple {atoms}")
    b1 = b1 / norm
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    angle = np.arctan2(
        np.sum(np.cross(b1, v) * w, axis=1),
        np.sum(v * w, axis=1),
    )
    if not np.all(np.isfinite(angle)):
        raise ValueError(f"nonfinite dihedral for atom tuple {atoms}")
    return np.remainder(angle.astype(np.float64) + np.pi, 2.0 * np.pi) - np.pi


def selected_inference_stages(run_dir: Path) -> list[tuple[dict, np.memmap]]:
    """Load the inference stage nearest t=0.5 and the final stage."""

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("stages"):
        raise ValueError(f"inference run is not complete: {run_dir}")
    if manifest.get("inference_only") is not True:
        raise ValueError("input does not declare inference-only provenance")
    if int(manifest.get("training_updates", -1)) != 0:
        raise ValueError("input reports nonzero training updates")
    stages = manifest["stages"]
    intermediate = min(stages, key=lambda item: abs(float(item["t"]) - 0.5))
    final = stages[-1]
    if intermediate["stage"] == final["stage"]:
        raise ValueError("the intermediate and final stage selections coincide")
    if not math.isclose(float(final["t"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"final stage has t={final['t']}, not t=1")

    loaded = []
    for stage_ref in (intermediate, final):
        path = (run_dir / stage_ref["samples_path"]).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as error:
            raise ValueError(f"sample path escapes run directory: {path}") from error
        samples = np.load(path, mmap_mode="r", allow_pickle=False)
        if samples.ndim != 2 or samples.dtype != np.float32:
            raise ValueError(
                f"expected a float32 sample matrix, found {samples.shape} "
                f"{samples.dtype}"
            )
        loaded.append((stage_ref, samples))
    return loaded


def selected_stages(run_dir: Path) -> list[tuple[dict, np.memmap]]:
    """Load the persisted stage nearest t=0.5 and the final stage."""

    record = validate_run(run_dir)
    stages = record.get("stages", [])
    if record.get("status") != "complete" or not stages:
        raise ValueError("KLXX run is not complete")
    intermediate = min(stages, key=lambda item: abs(float(item["t"]) - 0.5))
    final = stages[-1]
    if intermediate["stage"] == final["stage"]:
        raise ValueError("the intermediate and final stage selections coincide")
    if not math.isclose(float(final["t"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"final stage has t={final['t']}, not t=1")

    loaded = []
    for stage_ref in (intermediate, final):
        stage_number = int(stage_ref["stage"])
        stage_root = (run_dir / stage_ref["path"]).resolve()
        try:
            stage_root.relative_to(run_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"stage path escapes run directory: {stage_root}"
            ) from error
        metadata = json.loads(
            (stage_root / "stage.json").read_text(encoding="utf-8")
        )
        samples = load_validation_samples(run_dir, stage_number, mmap_mode="r")
        if samples.ndim != 2 or samples.dtype != np.float32:
            raise ValueError(
                f"expected a float32 sample matrix, found {samples.shape} "
                f"{samples.dtype}"
            )
        loaded.append((metadata, samples))
    return loaded


def histograms(
    samples: np.ndarray,
    target: Molecular_Potential,
    bins: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    heavy = np.zeros((bins, bins), dtype=np.int64)
    hydroxyl = np.zeros((bins, bins), dtype=np.int64)
    rotamer_counts = np.zeros(3, dtype=np.int64)
    contact_1 = 0
    contact_2 = 0
    either_contact = 0

    for start in range(0, samples.shape[0], chunk_size):
        stop = min(start + chunk_size, samples.shape[0])
        internal = np.asarray(samples[start:stop])
        if not np.all(np.isfinite(internal)):
            raise ValueError(f"nonfinite internal samples in rows {start}:{stop}")
        positions = np.asarray(
            jax.block_until_ready(
                cartesian_positions(target, jnp.asarray(internal))
            )
        )
        cccc = dihedral(positions, CCCC_ATOMS)
        occo = dihedral(positions, OCCO_ATOMS)
        oh1 = dihedral(positions, OH1_ATOMS)
        oh2 = dihedral(positions, OH2_ATOMS)
        heavy += np.histogram2d(cccc, occo, bins=(edges, edges))[0].astype(
            np.int64
        )
        hydroxyl += np.histogram2d(oh1, oh2, bins=(edges, edges))[0].astype(
            np.int64
        )

        trans = np.abs(cccc) >= 2.0 * np.pi / 3.0
        gauche_plus = (cccc >= 0.0) & ~trans
        gauche_minus = (cccc < 0.0) & ~trans
        rotamer_counts += np.array(
            [gauche_minus.sum(), gauche_plus.sum(), trans.sum()], dtype=np.int64
        )

        short_1 = np.linalg.norm(positions[:, 0] - positions[:, 15], axis=1) < 0.25
        short_2 = np.linalg.norm(positions[:, 1] - positions[:, 14], axis=1) < 0.25
        contact_1 += int(short_1.sum())
        contact_2 += int(short_2.sum())
        either_contact += int(np.count_nonzero(short_1 | short_2))

    expected = samples.shape[0]
    if int(heavy.sum()) != expected or int(hydroxyl.sum()) != expected:
        raise ValueError("a histogram dropped validation samples")
    if int(rotamer_counts.sum()) != expected:
        raise ValueError("central-rotamer classification dropped samples")
    rotamers = rotamer_counts / expected
    metrics = {
        "gauche_minus_occupancy": float(rotamers[0]),
        "gauche_plus_occupancy": float(rotamers[1]),
        "trans_occupancy": float(rotamers[2]),
        "O1_accepts_H10_fraction": contact_1 / expected,
        "O2_accepts_H9_fraction": contact_2 / expected,
        "either_short_OH_contact_fraction": either_contact / expected,
    }
    return edges, heavy, hydroxyl, metrics


def free_energy(counts: np.ndarray, smoothing: float) -> np.ndarray:
    density = gaussian_filter(counts, sigma=smoothing, mode="wrap")
    support = gaussian_filter(
        (counts > 0).astype(np.float64), sigma=smoothing, mode="wrap"
    )
    if not np.isfinite(density).all() or density.max() <= 0.0:
        raise ValueError("invalid smoothed density")
    surface = np.full_like(density, np.nan, dtype=np.float64)
    positive = density > 0.0
    surface[positive] = -np.log(density[positive] / density.max())
    surface[support < 0.05] = np.nan
    return surface


def plot_landscapes(
    landscapes: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict, int]],
    smoothing: float,
    output: Path,
) -> None:
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
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9.5, 8.6),
        layout="constrained",
    )
    stage_surfaces = []
    for edges, heavy, hydroxyl, metadata, sample_count in landscapes:
        stage_surfaces.append(
            (
                edges,
                (free_energy(heavy, smoothing), free_energy(hydroxyl, smoothing)),
                metadata,
                sample_count,
            )
        )

    vlimits = []
    for column in range(2):
        finite = np.concatenate(
            [
                surfaces[column][np.isfinite(surfaces[column])]
                for _, surfaces, _, _ in stage_surfaces
            ]
        )
        if finite.size == 0:
            raise ValueError(f"column {column} has no finite free-energy values")
        fixed = COLUMN_FREE_ENERGY_LIMITS[column]
        if fixed is not None:
            vlimits.append(float(fixed))
        else:
            vlimits.append(max(float(math.ceil(float(finite.max()))), 1.0))

    diagnostics = (
        ("heavy-atom torsions", r"C-C-C-C dihedral (rad)", r"O-C-C-O dihedral (rad)"),
        ("hydroxyl torsions", r"H-O1-C-C dihedral (rad)", r"H-O2-C-C dihedral (rad)"),
    )
    panel_letters = ("a", "b", "c", "d")
    images = [None, None]
    for row, (edges, surfaces, metadata, _) in enumerate(stage_surfaces):
        for column, (diagnostic, xlabel, ylabel) in enumerate(diagnostics):
            axis = axes[row, column]
            letter = panel_letters[2 * row + column]
            image = axis.pcolormesh(
                edges,
                edges,
                surfaces[column].T,
                cmap=COLUMN_CMAPS[column],
                vmin=0.0,
                vmax=vlimits[column],
                shading="flat",
                rasterized=True,
            )
            images[column] = image
            regularization = metadata["rg_end"]
            axis.set(
                xlim=(-np.pi, np.pi),
                ylim=(-np.pi, np.pi),
                xlabel=xlabel,
                ylabel=ylabel,
                title=(
                    rf"({letter}) stage {int(metadata['stage'])}: {diagnostic}"
                    "\n"
                    rf"$t={float(metadata['t']):.3f}$, "
                    rf"$\rho=({float(regularization[0]):.3g},"
                    rf"{float(regularization[1]):.3g})$"
                ),
            )
            axis.set_aspect("equal")
            axis.set_box_aspect(1)
            axis.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
            axis.set_yticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
            for spine in axis.spines.values():
                spine.set_linewidth(1.0)

    for column, image in enumerate(images):
        colorbar = figure.colorbar(
            image,
            ax=axes[:, column],
            pad=0.025,
            fraction=0.05,
            shrink=0.95,
        )
        tick_step = 2.0 if vlimits[column] > 6.0 else 1.0
        colorbar.set_ticks(np.arange(0.0, vlimits[column] + 0.1, tick_step))
        colorbar.set_label(r"free energy / $k_{\mathrm{B}}T$")

    figure.suptitle("(2R,3R)-2,3-butanediol", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    stages = [int(item[2]["stage"]) for item in stage_surfaces]
    counts = [int(item[3]) for item in stage_surfaces]
    print(f"stages={stages} samples={counts} output={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.2)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bins <= 0 or args.smoothing < 0.0 or args.chunk_size <= 0:
        parser.error("bins and chunk size must be positive; smoothing nonnegative")

    run_dir = args.run_dir.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing figure: {output}")

    target = Molecular_Potential.from_bundle(bundle, temperature_kelvin=300.0)
    landscapes = []
    all_metrics = {}
    for metadata, samples in selected_inference_stages(run_dir):
        if target.dimension != samples.shape[1] or samples.shape[1] != 42:
            raise ValueError(
                f"unexpected dimension: target={target.dimension}, "
                f"samples={samples.shape}"
            )
        edges, heavy, hydroxyl, metrics = histograms(
            samples,
            target,
            args.bins,
            args.chunk_size,
        )
        landscapes.append((edges, heavy, hydroxyl, metadata, samples.shape[0]))
        all_metrics[f"stage_{int(metadata['stage']):06d}"] = metrics
    plot_landscapes(
        landscapes,
        args.smoothing,
        output,
    )
    print(f"backend={jax.default_backend()}", flush=True)
    print(json.dumps(all_metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

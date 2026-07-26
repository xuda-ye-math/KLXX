#!/usr/bin/env python
"""Plot intermediate- and final-stage Ac-Pro-NHMe landscapes."""

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
RUN_DIR = HERE / "artifacts" / "inference_15M"
BUNDLE = HERE / "bundle"
DEFAULT_OUTPUT = HERE / "results" / "conformational_landscape.png"

# Zero-based PDB atom indices.
# phi = C_ACE-N_PRO-CA_PRO-C_PRO
# psi = N_PRO-CA_PRO-C_PRO-N_NME
# omega = CH3_ACE-C_ACE-N_PRO-CA_PRO
# chi1 = N_PRO-CA_PRO-CB_PRO-CG_PRO
PHI_ATOMS = (4, 6, 16, 18)
PSI_ATOMS = (6, 16, 18, 20)
OMEGA_ATOMS = (1, 4, 6, 16)
CHI1_ATOMS = (6, 16, 13, 10)

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


@eqx.filter_jit
def cartesian_positions(
    target: Molecular_Potential, internal: jax.Array
) -> jax.Array:
    return target.cartesian(internal)


def dihedral(positions: np.ndarray, atoms: tuple[int, int, int, int]) -> np.ndarray:
    """Return right-handed a-b-c-d dihedrals in radians."""

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
    backbone = np.zeros((bins, bins), dtype=np.int64)
    peptide_ring = np.zeros((bins, bins), dtype=np.int64)
    state_counts = np.zeros(4, dtype=np.int64)

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
        phi = dihedral(positions, PHI_ATOMS)
        psi = dihedral(positions, PSI_ATOMS)
        omega = dihedral(positions, OMEGA_ATOMS)
        chi1 = dihedral(positions, CHI1_ATOMS)
        backbone += np.histogram2d(phi, psi, bins=(edges, edges))[0].astype(
            np.int64
        )
        peptide_ring += np.histogram2d(
            omega, chi1, bins=(edges, edges)
        )[0].astype(np.int64)
        cis = np.abs(omega) < np.pi / 2.0
        pucker_plus = chi1 >= 0.0
        labels = 2 * cis.astype(np.int64) + pucker_plus.astype(np.int64)
        state_counts += np.bincount(labels, minlength=4)

    expected = samples.shape[0]
    if int(backbone.sum()) != expected or int(peptide_ring.sum()) != expected:
        raise ValueError("a histogram dropped validation samples")
    occupancy = state_counts / state_counts.sum()
    metrics = {
        "trans_pucker_minus": float(occupancy[0]),
        "trans_pucker_plus": float(occupancy[1]),
        "cis_pucker_minus": float(occupancy[2]),
        "cis_pucker_plus": float(occupancy[3]),
    }
    return edges, backbone, peptide_ring, metrics


def free_energy(counts: np.ndarray, smoothing: float) -> np.ndarray:
    density = gaussian_filter(counts, sigma=smoothing, mode="wrap")
    if not np.isfinite(density).all() or density.max() <= 0.0:
        raise ValueError("invalid smoothed density")
    with np.errstate(divide="ignore"):
        surface = -np.log(density / density.max())
    surface[counts == 0] = np.nan
    return surface


def occupied_y_limits(
    counts: np.ndarray, edges: np.ndarray
) -> tuple[float, float]:
    """Return padded limits from the occupied y bins of a 2D histogram."""

    occupied = np.flatnonzero(np.sum(counts, axis=0) > 0)
    if occupied.size == 0:
        raise ValueError("cannot determine y limits from an empty histogram")
    lower = float(edges[int(occupied[0])])
    upper = float(edges[int(occupied[-1]) + 1])
    span = upper - lower
    padding = max(0.08 * span, 2.0 * float(edges[1] - edges[0]))
    return max(-np.pi, lower - padding), min(np.pi, upper + padding)


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
    vmax = 10.0
    panel_letters = ("a", "b", "c", "d")
    pucker_limits = []
    image = None
    for row, (edges, backbone, peptide_ring, metadata, _) in enumerate(landscapes):
        surfaces = (
            free_energy(backbone, smoothing),
            free_energy(peptide_ring, smoothing),
        )
        pucker_ylim = occupied_y_limits(peptide_ring, edges)
        pucker_limits.append(pucker_ylim)
        regularization = metadata["rg_end"]

        left = axes[row, 0]
        image = left.pcolormesh(
            edges,
            edges,
            surfaces[0].T,
            cmap=FREE_ENERGY_CMAP,
            vmin=0.0,
            vmax=vmax,
            shading="flat",
            rasterized=True,
        )
        left.set(
            xlim=(-np.pi, np.pi),
            ylim=(-np.pi, np.pi),
            xlabel=r"proline $\phi$ (rad)",
            ylabel=r"proline $\psi$ (rad)",
            title=(
                rf"({panel_letters[2 * row]}) stage "
                rf"{int(metadata['stage'])}: backbone $(\phi,\psi)$"
                "\n"
                rf"$t={float(metadata['t']):.3f}$, "
                rf"$\rho=({float(regularization[0]):.3g},"
                rf"{float(regularization[1]):.3g})$"
            ),
        )
        left.set_aspect("equal")
        left.set_box_aspect(1)
        left.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
        left.set_yticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))

        right = axes[row, 1]
        right.pcolormesh(
            edges,
            edges,
            surfaces[1].T,
            cmap=FREE_ENERGY_CMAP,
            vmin=0.0,
            vmax=vmax,
            shading="flat",
            rasterized=True,
        )
        right.set(
            xlim=(-np.pi, np.pi),
            ylim=pucker_ylim,
            xlabel=r"ACE-PRO $\omega$ (rad)",
            ylabel=r"proline-ring $\chi_1$ (rad)",
            title=(
                rf"({panel_letters[2 * row + 1]}) stage "
                rf"{int(metadata['stage'])}: torsion / pucker"
                "\n"
                rf"$t={float(metadata['t']):.3f}$, "
                rf"$\rho=({float(regularization[0]):.3g},"
                rf"{float(regularization[1]):.3g})$"
            ),
        )
        right.set_xticks(
            (-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi),
            (r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"),
        )
        right.set_box_aspect(1)
        right.axvline(-np.pi / 2.0, color="white", lw=0.9, ls="--")
        right.axvline(np.pi / 2.0, color="white", lw=0.9, ls="--")
        right.axhline(0.0, color="black", lw=0.9, ls=":")
        label_y = pucker_ylim[1] - 0.04 * (pucker_ylim[1] - pucker_ylim[0])
        right.text(0.0, label_y, "cis", ha="center", va="top", color="#222222")
        right.text(-2.45, label_y, "trans", ha="center", va="top", color="#222222")
        right.text(2.45, label_y, "trans", ha="center", va="top", color="#222222")

        for axis in axes[row]:
            for spine in axis.spines.values():
                spine.set_linewidth(1.0)

    colorbar = figure.colorbar(
        image,
        ax=axes,
        pad=0.025,
        fraction=0.04,
        shrink=0.95,
    )
    colorbar.set_ticks(np.arange(0.0, vmax + 0.1, 2.0))
    colorbar.set_label(r"free energy / $k_{\mathrm{B}}T$")

    figure.suptitle("N-acetyl-L-proline N-methylamide", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    stages = [int(item[3]["stage"]) for item in landscapes]
    counts = [int(item[4]) for item in landscapes]
    print(
        f"stages={stages} samples={counts} pucker_ylim={pucker_limits} "
        f"output={output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.0)
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
        if target.dimension != samples.shape[1] or samples.shape[1] != 72:
            raise ValueError(
                f"unexpected dimension: target={target.dimension}, "
                f"samples={samples.shape}"
            )
        edges, backbone, peptide_ring, metrics = histograms(
            samples,
            target,
            args.bins,
            args.chunk_size,
        )
        landscapes.append(
            (edges, backbone, peptide_ring, metadata, samples.shape[0])
        )
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

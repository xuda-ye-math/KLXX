#!/usr/bin/env python
"""Plot ADP Ramachandran surfaces from persisted inference populations."""

from __future__ import annotations

import argparse
import json
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


HERE = Path(__file__).resolve().parent
INFERENCE_DIR = HERE / "artifacts" / "inference_10M"
BUNDLE = HERE / "bundle"
DEFAULT_OUTPUT_DIR = HERE / "results"

# Zero-based PDB atom indices for the conventional alanine backbone angles:
# phi = C_(i-1)-N-C_alpha-C_i and psi = N-C_alpha-C_i-N_(i+1).
PHI_ATOMS = (4, 6, 8, 14)
PSI_ATOMS = (6, 8, 14, 16)

# Color stops sampled from the free-energy colorbar in arXiv:2602.03729v2.
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


def dihedral(positions: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
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
    return angle


def population_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"population path escapes inference root: {path}") from error
    return path


def load_population(
    root: Path,
    relative_path: str,
    sample_count: int,
    dimension: int,
) -> np.memmap:
    path = population_path(root, relative_path)
    samples = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = (sample_count, dimension)
    if samples.shape != expected or samples.dtype != np.float32:
        raise ValueError(
            f"expected float32 population {expected}, found "
            f"{samples.shape} {samples.dtype} at {path}"
        )
    return samples


def backbone_histogram(
    samples: np.ndarray,
    target: Molecular_Potential,
    bins: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    counts = np.zeros((bins, bins), dtype=np.int64)
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
        # ``arctan2`` can return the float32 approximation of +pi, which is
        # slightly larger than the float64 histogram edge.  Wrap in float64
        # so the periodic boundary is represented without dropping samples.
        phi = np.remainder(
            dihedral(positions, PHI_ATOMS).astype(np.float64) + np.pi,
            2.0 * np.pi,
        ) - np.pi
        psi = np.remainder(
            dihedral(positions, PSI_ATOMS).astype(np.float64) + np.pi,
            2.0 * np.pi,
        ) - np.pi
        chunk_counts, _, _ = np.histogram2d(phi, psi, bins=(edges, edges))
        counts += chunk_counts.astype(np.int64)
    if int(counts.sum()) != samples.shape[0]:
        raise ValueError(
            f"histogram contains {int(counts.sum()):,} of "
            f"{samples.shape[0]:,} samples"
        )
    return edges, counts


def free_energy_surface(counts: np.ndarray, smoothing: float) -> np.ndarray:
    density = gaussian_filter(counts, sigma=smoothing, mode="wrap")
    if not np.isfinite(density).all() or density.max() <= 0.0:
        raise ValueError("invalid Ramachandran density")
    with np.errstate(divide="ignore"):
        free_energy = -np.log(density / density.max())
    # Preserve white unsampled bins while smoothing colors within the support.
    free_energy[counts == 0] = np.nan
    return free_energy


def plot_surface(
    edges: np.ndarray,
    free_energy: np.ndarray,
    title: str,
    output: Path,
) -> None:
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
    figure, axis = plt.subplots(figsize=(5.7, 4.8), layout="constrained")
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
        ylabel=r"$\psi$ (rad)",
        title=title,
    )
    axis.set_aspect("equal")
    axis.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
    axis.set_yticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.06)
    colorbar.set_ticks(np.arange(0.0, 11.0, 2.0))
    colorbar.set_label(r"free energy / $k_{\mathrm{B}}T$")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def stage_title(stage: dict, sample_count: int) -> str:
    regularization = stage["rg_end"]
    return (
        f"ADP stage {int(stage['stage'])}: "
        rf"$t={float(stage['t']):.3f}$, "
        rf"$\rho=({float(regularization[0]):.3g},"
        rf"{float(regularization[1]):.3g})$ "
        rf"$({sample_count / 1e6:g}\times10^6$ samples)"
    )


def plot_population(
    root: Path,
    relative_path: str,
    sample_count: int,
    dimension: int,
    target: Molecular_Potential,
    bins: int,
    smoothing: float,
    chunk_size: int,
    title: str,
    output: Path,
) -> None:
    samples = load_population(root, relative_path, sample_count, dimension)
    edges, counts = backbone_histogram(samples, target, bins, chunk_size)
    free_energy = free_energy_surface(counts, smoothing)
    plot_surface(edges, free_energy, title, output)
    print(
        f"samples={samples.shape[0]:,} histogram={int(counts.sum()):,} "
        f"output={output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    if args.bins <= 0 or args.smoothing < 0.0 or args.chunk_size <= 0:
        parser.error("bins and chunk size must be positive; smoothing nonnegative")

    root = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "adp-frozen-flow-inference-10m-1":
        raise ValueError("input is not an ADP frozen-flow inference run")
    if manifest.get("inference_only") is not True:
        raise ValueError("input does not declare inference-only provenance")
    if int(manifest.get("training_updates", -1)) != 0:
        raise ValueError("input reports nonzero training updates")
    sample_count = int(manifest["sample_count"])
    dimension = int(manifest["dimension"])
    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)
    if target.dimension != dimension or dimension != 60:
        raise ValueError(
            f"unexpected ADP dimension: manifest={dimension}, target={target.dimension}"
        )

    written: list[Path] = []
    for stage in manifest["stages"]:
        stage_number = int(stage["stage"])
        output = output_dir / f"ramachandran_stage_{stage_number}.png"
        plot_population(
            root,
            stage["samples_path"],
            sample_count,
            dimension,
            target,
            args.bins,
            args.smoothing,
            args.chunk_size,
            stage_title(stage, sample_count),
            output,
        )
        written.append(output)

    raw = manifest.get("raw")
    if raw is not None:
        output = output_dir / "ramachandran_raw.png"
        plot_population(
            root,
            raw["samples_path"],
            sample_count,
            dimension,
            target,
            args.bins,
            args.smoothing,
            args.chunk_size,
            rf"ADP raw potential at 300 K "
            rf"$({sample_count / 1e6:g}\times10^6$ samples)",
            output,
        )
        written.append(output)

    expected = len(manifest["stages"]) + int(raw is not None)
    if len(written) != expected or not written:
        raise ValueError(
            f"expected {expected} Ramachandran figures, wrote {len(written)}"
        )
    print(
        f"backend={jax.default_backend()} status={manifest['status']} "
        f"figures={len(written)} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

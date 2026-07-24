#!/usr/bin/env python
"""Render representative conformations from the three completed chiral runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
import numpy as np
from scipy.ndimage import gaussian_filter

from jflows_md import Molecular_Bundle, Molecular_Potential
from jflows_md.boltzmann.load import load_validation_samples, validate_run


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "conformations.png"

CPK_COLORS = {
    1: np.asarray((0.95, 0.95, 0.95)),
    6: np.asarray((0.28, 0.28, 0.28)),
    7: np.asarray((0.12, 0.26, 0.78)),
    8: np.asarray((0.90, 0.02, 0.02)),
}
COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
LIGHT = np.asarray((-0.42, 0.50, 0.76))
LIGHT = LIGHT / np.linalg.norm(LIGHT)
GOLD = "#E8990C"


@dataclass(frozen=True)
class Molecule:
    folder: str
    display_name: str
    dimension: int
    dihedrals: tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ]
    stereocenters: tuple[int, ...]
    configurations: tuple[str, ...]
    view_elevation: float
    view_azimuth: float


MOLECULES = (
    Molecule(
        folder="rr_2_3_butanediol_42d",
        display_name="(2R,3R)-2,3-butanediol",
        dimension=42,
        dihedrals=((4, 2, 3, 5), (0, 2, 3, 1)),
        stereocenters=(2, 3),
        configurations=("R", "R"),
        view_elevation=14.0,
        view_azimuth=72.0,
    ),
    Molecule(
        folder="adp_60d",
        display_name="alanine dipeptide",
        dimension=60,
        dihedrals=((4, 6, 8, 14), (6, 8, 14, 16)),
        stereocenters=(8,),
        configurations=("S",),
        view_elevation=18.0,
        view_azimuth=64.0,
    ),
    Molecule(
        folder="ac_pro_nhme_72d",
        display_name="Ac-Pro-NHMe",
        dimension=72,
        dihedrals=((4, 6, 16, 18), (6, 16, 18, 20)),
        stereocenters=(16,),
        configurations=("S",),
        view_elevation=18.0,
        view_azimuth=64.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot modal final-stage conformations for the chiral targets."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--smoothing", type=float, default=1.2)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dpi <= 0 or args.bins <= 0 or args.chunk_size <= 0:
        parser.error("dpi, bins, and chunk size must be positive")
    if args.smoothing < 0.0:
        parser.error("smoothing must be nonnegative")
    return args


def dihedral(
    positions: np.ndarray, atoms: tuple[int, int, int, int]
) -> np.ndarray:
    """Return wrapped right-handed A-B-C-D dihedrals in radians."""

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


def load_final_population(molecule: Molecule) -> tuple[dict, np.memmap]:
    run_dir = HERE / molecule.folder / "artifacts" / "klxx"
    record = validate_run(run_dir)
    stages = record.get("stages", [])
    if record.get("status") != "complete" or not stages:
        raise ValueError(f"KLXX run is not complete: {run_dir}")
    if record.get("config", {}).get("objective") != "forward_klxx":
        raise ValueError(f"unexpected objective in {run_dir}")
    final = stages[-1]
    if not math.isclose(float(final["t"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"final stage does not reach t=1: {run_dir}")
    samples = load_validation_samples(
        run_dir,
        int(final["stage"]),
        mmap_mode="r",
    )
    expected = (samples.shape[0], molecule.dimension)
    if samples.shape != expected or samples.dtype != np.float32:
        raise ValueError(
            f"expected float32 samples with shape {expected}, found "
            f"{samples.shape} {samples.dtype}"
        )
    return final, samples


def validate_stereocenters(molecule: Molecule, bundle: Molecular_Bundle) -> None:
    fixed = bundle.coordinates.get("fixed_stereocenters")
    if fixed:
        centers = tuple(int(record["atoms"][0]) for record in fixed)
        configurations = tuple(str(record["configuration"]) for record in fixed)
        if configurations != molecule.configurations:
            raise ValueError(f"configuration mismatch for {molecule.folder}")
    else:
        legacy_atoms = bundle.coordinates.get("chirality_atoms")
        if not legacy_atoms:
            raise ValueError(f"bundle has no fixed stereocenter for {molecule.folder}")
        centers = (int(legacy_atoms[0]),)
    if centers != molecule.stereocenters:
        raise ValueError(f"stereocenter mismatch for {molecule.folder}")


def modal_conformation(
    molecule: Molecule,
    target: Molecular_Potential,
    samples: np.ndarray,
    bins: int,
    smoothing: float,
    chunk_size: int,
) -> tuple[np.ndarray, int, tuple[float, float], int]:
    converter = eqx.filter_jit(target.cartesian)
    angles = np.empty((samples.shape[0], 2), dtype=np.float32)
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(start + chunk_size, samples.shape[0])
        internal = np.asarray(samples[start:stop])
        if not np.all(np.isfinite(internal)):
            raise ValueError(
                f"nonfinite validation samples in {molecule.folder} "
                f"rows {start}:{stop}"
            )
        positions = np.asarray(
            jax.block_until_ready(converter(jnp.asarray(internal)))
        )
        angles[start:stop, 0] = dihedral(positions, molecule.dihedrals[0])
        angles[start:stop, 1] = dihedral(positions, molecule.dihedrals[1])

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    histogram = np.histogram2d(
        angles[:, 0],
        angles[:, 1],
        bins=(edges, edges),
    )[0]
    if int(histogram.sum()) != samples.shape[0]:
        raise ValueError(f"modal histogram dropped samples for {molecule.folder}")
    density = gaussian_filter(histogram, sigma=smoothing, mode="wrap")
    peak = np.unravel_index(int(np.argmax(density)), density.shape)
    center = np.asarray(
        (
            0.5 * (edges[peak[0]] + edges[peak[0] + 1]),
            0.5 * (edges[peak[1]] + edges[peak[1] + 1]),
        )
    )
    displacement = np.remainder(angles - center + np.pi, 2.0 * np.pi) - np.pi
    index = int(np.argmin(np.sum(displacement**2, axis=1)))
    selected = np.asarray(
        jax.block_until_ready(
            converter(jnp.asarray(np.asarray(samples[index : index + 1])))
        )
    )[0]
    if selected.shape != (target.reference_positions_nm.shape[0], 3):
        raise ValueError(f"unexpected Cartesian shape for {molecule.folder}")
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"nonfinite selected conformation for {molecule.folder}")
    modal_bin_count = int(histogram[peak])
    return selected, index, (float(center[0]), float(center[1])), modal_bin_count


def align_to_dihedral_bond(
    positions: np.ndarray, atoms: tuple[int, int, int, int]
) -> np.ndarray:
    a, b, c, _ = atoms
    positions = positions - positions[b]
    x_axis = positions[c] / np.linalg.norm(positions[c])
    a_direction = positions[a] - np.dot(positions[a], x_axis) * x_axis
    y_axis = a_direction / np.linalg.norm(a_direction)
    z_axis = np.cross(x_axis, y_axis)
    return positions @ np.stack((x_axis, y_axis, z_axis)).T


def conformer_view_matrix(elevation: float, azimuth: float) -> np.ndarray:
    elevation = np.deg2rad(elevation)
    azimuth = np.deg2rad(azimuth)
    rotate_y = np.asarray(
        (
            (np.cos(azimuth), 0.0, np.sin(azimuth)),
            (0.0, 1.0, 0.0),
            (-np.sin(azimuth), 0.0, np.cos(azimuth)),
        )
    )
    rotate_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(elevation), -np.sin(elevation)),
            (0.0, np.sin(elevation), np.cos(elevation)),
        )
    )
    return rotate_x @ rotate_y


def shaded_sphere(color: np.ndarray, depth: float, pixels: int = 144) -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, pixels)
    xx, yy = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(xx, yy)
    normal_z = np.sqrt(np.clip(1.0 - xx**2 - yy**2, 0.0, 1.0))
    diffuse = np.clip(
        xx * LIGHT[0] + yy * LIGHT[1] + normal_z * LIGHT[2],
        0.0,
        1.0,
    )
    shade = (0.40 + 0.60 * diffuse) * depth
    rgb = np.clip(
        color[None, None, :] * shade[..., None]
        + 0.75 * (diffuse**16)[..., None],
        0.0,
        1.0,
    )
    alpha = np.clip((1.0 - radius) / 0.035, 0.0, 1.0)
    return np.dstack((rgb, alpha))


def draw_conformation(
    axis: plt.Axes,
    positions_nm: np.ndarray,
    atomic_numbers: np.ndarray,
    bonds: np.ndarray,
    molecule: Molecule,
) -> None:
    positions = align_to_dihedral_bond(
        10.0 * positions_nm,
        molecule.dihedrals[0],
    )
    view = conformer_view_matrix(
        molecule.view_elevation,
        molecule.view_azimuth,
    )
    projected = positions @ view.T
    x, y, depth = projected.T
    radii = np.asarray(
        [0.18 + 0.30 * COVALENT_RADII_ANGSTROM[int(z)] for z in atomic_numbers]
    )
    depth_low = float(depth.min())
    depth_span = float(np.ptp(depth)) or 1.0
    radii *= 0.90 + 0.14 * (depth - depth_low) / depth_span

    def depth_factor(value: float) -> float:
        return 0.50 + 0.50 * (value - depth_low) / depth_span

    for first, second in bonds:
        first = int(first)
        second = int(second)
        midpoint_x = 0.5 * (x[first] + x[second])
        midpoint_y = 0.5 * (y[first] + y[second])
        bond_depth = 0.5 * (depth[first] + depth[second])
        bond_zorder = 2.0 + 2.0 * (bond_depth - depth_low) / depth_span
        axis.plot(
            [x[first], x[second]],
            [y[first], y[second]],
            color="0.20",
            linewidth=6.0,
            alpha=0.72,
            solid_capstyle="round",
            zorder=bond_zorder,
        )
        for index in (first, second):
            color = np.clip(
                CPK_COLORS[int(atomic_numbers[index])] * depth_factor(depth[index]),
                0.0,
                1.0,
            )
            axis.plot(
                [x[index], midpoint_x],
                [y[index], midpoint_y],
                color=color,
                linewidth=4.2,
                solid_capstyle="round",
                zorder=bond_zorder + 0.1,
            )
    for index in np.argsort(depth):
        radius = radii[index]
        atom_zorder = 6.0 + 4.0 * (depth[index] - depth_low) / depth_span
        axis.add_patch(
            Circle(
                (x[index] + 0.055, y[index] - 0.065),
                radius * 1.03,
                facecolor="black",
                edgecolor="none",
                alpha=0.13,
                zorder=atom_zorder - 0.1,
            )
        )
        sphere = shaded_sphere(
            CPK_COLORS[int(atomic_numbers[index])],
            depth_factor(depth[index]),
        )
        axis.imshow(
            sphere,
            extent=(
                x[index] - radius,
                x[index] + radius,
                y[index] - radius,
                y[index] + radius,
            ),
            origin="lower",
            interpolation="bilinear",
            zorder=atom_zorder,
        )

    if len(molecule.stereocenters) != len(molecule.configurations):
        raise ValueError(f"stereocenter metadata mismatch for {molecule.folder}")
    for center_index, configuration in zip(
        molecule.stereocenters,
        molecule.configurations,
        strict=True,
    ):
        marker_radius = 1.25 * radii[center_index]
        axis.add_patch(
            Circle(
                (x[center_index], y[center_index]),
                marker_radius,
                facecolor="none",
                edgecolor="white",
                linewidth=5.2,
                zorder=19,
            )
        )
        axis.add_patch(
            Circle(
                (x[center_index], y[center_index]),
                marker_radius,
                facecolor="none",
                edgecolor=GOLD,
                linewidth=2.8,
                zorder=20,
            )
        )
        label = axis.annotate(
            configuration,
            xy=(x[center_index], y[center_index]),
            xytext=(8, 9),
            textcoords="offset points",
            color=GOLD,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=21,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.4, foreground="white")]
        )

    center = projected[:, :2].mean(axis=0)
    half_range = float(np.max(np.abs(projected[:, :2] - center)))
    half_range += float(radii.max()) + 0.26
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_aspect("equal")
    axis.set_box_aspect(1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing figure: {output}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 14.5,
            "axes.titlesize": 16,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.3),
        layout="constrained",
    )
    print(f"JAX backend: {jax.default_backend()}", flush=True)

    for panel, (axis, molecule) in enumerate(
        zip(axes, MOLECULES, strict=True),
        start=1,
    ):
        final, samples = load_final_population(molecule)
        bundle = Molecular_Bundle.load(HERE / molecule.folder / "bundle")
        validate_stereocenters(molecule, bundle)
        target = Molecular_Potential(
            bundle,
            temperature_kelvin=300.0,
        )
        if target.dimension != molecule.dimension:
            raise ValueError(f"bundle dimension mismatch for {molecule.folder}")
        positions, index, center, modal_bin_count = modal_conformation(
            molecule,
            target,
            samples,
            args.bins,
            args.smoothing,
            args.chunk_size,
        )
        atomic_numbers = np.asarray(bundle.system["atomic_numbers"], dtype=np.int16)
        bonds = np.asarray(bundle.system["bonds"], dtype=np.int16)
        if positions.shape[0] != atomic_numbers.size:
            raise ValueError(f"topology mismatch for {molecule.folder}")
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            raise ValueError(f"invalid bond topology for {molecule.folder}")
        if not set(map(int, atomic_numbers)).issubset(CPK_COLORS):
            raise ValueError(f"unsupported element for {molecule.folder}")
        draw_conformation(axis, positions, atomic_numbers, bonds, molecule)
        axis.set_title(
            rf"({chr(96 + panel)}) {molecule.display_name} ($d={molecule.dimension}$)",
            pad=4,
        )
        print(
            f"{molecule.display_name}: stage={final['stage']} "
            f"samples={samples.shape[0]:,} selected_index={index} "
            f"modal_center=({center[0]:.4f}, {center[1]:.4f}) "
            f"modal_bin_count={modal_bin_count}",
            flush=True,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=10,
            markerfacecolor=CPK_COLORS[6],
            markeredgecolor="0.25",
            linestyle="none",
            label="C",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=10,
            markerfacecolor=CPK_COLORS[7],
            markeredgecolor="0.25",
            linestyle="none",
            label="N",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=10,
            markerfacecolor=CPK_COLORS[8],
            markeredgecolor="0.25",
            linestyle="none",
            label="O",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=10,
            markerfacecolor=CPK_COLORS[1],
            markeredgecolor="0.35",
            linestyle="none",
            label="H",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor=GOLD,
            markeredgewidth=2.2,
            linestyle="none",
            label="fixed chiral carbon",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=5,
        frameon=False,
        columnspacing=2.0,
        handletextpad=0.45,
        fontsize=14.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"saved {output} at {args.dpi} DPI", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot one crucial dihedral for each completed achiral molecular target.

The OpenMM reference trajectories are read from the saved regularization
artifacts.  The final KLX and KLXX sharpening samples are converted from
mixed coordinates with their molecular bundles.  Extracted angles and
representative trans conformers are cached, so ``--replot`` requires neither
OpenMM nor JAX reconstruction.

Usage
-----
/home/xuda/.envs/jflows/bin/python plot_dihedrals.py
/home/xuda/.envs/jflows/bin/python plot_dihedrals.py --replot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d


HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "regularization" / "data" / "production"
RESULTS_DIR = HERE / "results"
CACHE_PATH = RESULTS_DIR / "dihedrals_data.npz"
FIGURE_PATH = RESULTS_DIR / "dihedrals.png"
BLUE = "#1F77B4"
RED = "#D62728"
GOLD = "#E8990C"
CPK_COLORS = {
    1: np.asarray((0.95, 0.95, 0.95)),
    6: np.asarray((0.28, 0.28, 0.28)),
    7: np.asarray((0.12, 0.26, 0.78)),
    8: np.asarray((0.90, 0.02, 0.02)),
}
COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
LIGHT = np.asarray((-0.42, 0.50, 0.76))
LIGHT = LIGHT / np.linalg.norm(LIGHT)


@dataclass(frozen=True)
class Molecule:
    key: str
    folder: str
    display_name: str
    dimension: int
    bundle: str | Path
    atoms: tuple[int, int, int, int]
    dihedral_label: str
    reference_seeds: tuple[int, ...]
    reference_selection: str


MOLECULES = (
    Molecule(
        key="nma",
        folder="nma_30d",
        display_name="NMA",
        dimension=30,
        bundle=HERE / "nma_30d" / "bundle",
        atoms=(1, 4, 6, 8),
        dihedral_label="C-C(O)-N-C",
        reference_seeds=(3402,),
        reference_selection="trans-initialized raw trajectory",
    ),
    Molecule(
        key="glycerol",
        folder="glycerol_36d",
        display_name="glycerol",
        dimension=36,
        bundle=HERE / "glycerol_36d" / "bundle",
        atoms=(1, 2, 4, 5),
        dihedral_label="C-C-C-O",
        reference_seeds=(3401, 3402),
        reference_selection="pooled raw trajectories",
    ),
    Molecule(
        key="diethanolamine",
        folder="diethanolamine_48d",
        display_name="neutral diethanolamine",
        dimension=48,
        bundle=HERE / "diethanolamine_48d" / "bundle",
        atoms=(1, 2, 3, 4),
        dihedral_label="C-C-N-C",
        reference_seeds=(3401, 3402),
        reference_selection="pooled raw trajectories",
    ),
)
METHODS = {
    "klx": "forward_klx",
    "klxx": "forward_klxx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one KLX/KLXX dihedral marginal for each achiral molecule."
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="plot only from the processed angle cache",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20_000,
        help="mixed-coordinate samples converted per GPU batch (default: 20000)",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(HERE))
    except ValueError:
        return str(path.resolve())


def dihedral(positions: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    """Return right-handed a-b-c-d dihedrals in radians."""
    a, b, c, d = (positions[:, index] for index in atoms)
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    norms = np.linalg.norm(b1, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"zero-length central bond for atom tuple {atoms}")
    b1 = b1 / norms
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    angles = np.arctan2(
        np.sum(np.cross(b1, v) * w, axis=1),
        np.sum(v * w, axis=1),
    )
    if not np.all(np.isfinite(angles)):
        raise ValueError(f"nonfinite dihedral for atom tuple {atoms}")
    return angles.astype(np.float32, copy=False)


def load_reference(
    molecule: Molecule,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, dict]:
    paths = [
        REFERENCE_DIR / f"{molecule.key}__raw__seed{seed}.npz"
        for seed in molecule.reference_seeds
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing selected raw OpenMM trajectories: {missing}"
        )

    angles = []
    provenance = []
    expected_bundle_hashes = None
    best_conformer = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            positions = np.asarray(data["positions_nm"])
            energies = np.asarray(data["energies_kj_mol"])

        if metadata.get("molecule") != molecule.key:
            raise ValueError(f"molecule mismatch in {path}")
        if metadata.get("phase") != "production" or metadata.get("rg_param") is not None:
            raise ValueError(f"reference is not a raw production trajectory: {path}")
        if int(metadata.get("dimension", -1)) != molecule.dimension:
            raise ValueError(f"dimension mismatch in {path}")
        temperatures = metadata.get("temperatures_kelvin", [])
        if not temperatures or not math.isclose(float(temperatures[0]), 300.0):
            raise ValueError(f"replica zero is not 300 K in {path}")
        burnin = int(metadata["burnin_rounds"])
        if (
            positions.ndim != 4
            or energies.shape != positions.shape[:2]
            or not 0 <= burnin < positions.shape[0]
        ):
            raise ValueError(f"invalid saved trajectory shape or burn-in in {path}")

        bundle_hashes = metadata.get("bundle_sha256")
        if expected_bundle_hashes is None:
            expected_bundle_hashes = bundle_hashes
        elif bundle_hashes != expected_bundle_hashes:
            raise ValueError(f"bundle provenance differs across reference seeds: {path}")

        selected = positions[burnin:, 0]
        selected_energies = energies[burnin:, 0]
        selected_angles = dihedral(selected, molecule.atoms)
        angles.append(selected_angles)
        trans_distance = np.abs(
            (selected_angles - np.pi + np.pi) % (2.0 * np.pi) - np.pi
        )
        eligible = np.flatnonzero(trans_distance <= np.deg2rad(2.0))
        if eligible.size == 0:
            raise ValueError(f"no trans conformer within two degrees in {path}")
        local_index = int(eligible[np.argmin(selected_energies[eligible])])
        candidate = (
            float(selected_energies[local_index]),
            np.asarray(selected[local_index], dtype=np.float32),
            {
                "path": relative_path(path),
                "seed": int(metadata["seed"]),
                "saved_round_index": burnin + local_index,
                "replica_index": 0,
                "temperature_kelvin": float(temperatures[0]),
                "energy_kj_mol": float(selected_energies[local_index]),
                "dihedral_radians": float(selected_angles[local_index]),
                "selection": "lowest-energy frame within two degrees of trans",
            },
        )
        if best_conformer is None or candidate[0] < best_conformer[0]:
            best_conformer = candidate
        provenance.append(
            {
                "path": relative_path(path),
                "sha256": sha256(path),
                "seed": int(metadata["seed"]),
                "saved_rounds": int(positions.shape[0]),
                "discarded_burnin_rounds": burnin,
                "replica_index": 0,
                "temperature_kelvin": float(temperatures[0]),
            }
        )

    if best_conformer is None or expected_bundle_hashes is None:
        raise RuntimeError(f"failed to select a trans conformer for {molecule.key}")
    return (
        np.concatenate(angles),
        best_conformer[1],
        provenance,
        expected_bundle_hashes,
        best_conformer[2],
    )


def load_pdb_topology(path: Path) -> tuple[np.ndarray, np.ndarray]:
    atomic_numbers = []
    serial_to_index = {}
    connection_records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            serial = int(line[6:11])
            element = line[76:78].strip().title()
            if element not in {"H", "C", "N", "O"}:
                raise ValueError(f"unsupported or missing element in {path}: {line}")
            serial_to_index[serial] = len(atomic_numbers)
            atomic_numbers.append({"H": 1, "C": 6, "N": 7, "O": 8}[element])
        elif line.startswith("CONECT"):
            connection_records.append([int(value) for value in line[6:].split()])

    bonds = set()
    for record in connection_records:
        origin = serial_to_index[record[0]]
        for serial in record[1:]:
            target = serial_to_index[serial]
            if origin != target:
                bonds.add(tuple(sorted((origin, target))))
    if not atomic_numbers or not bonds:
        raise ValueError(f"incomplete atom or bond topology in {path}")
    return (
        np.asarray(atomic_numbers, dtype=np.int16),
        np.asarray(sorted(bonds), dtype=np.int16),
    )


def final_population(
    molecule: Molecule, method: str
) -> tuple[Path, dict, dict]:
    artifact_dir = HERE / molecule.folder / "artifacts" / method
    run_path = artifact_dir / "run.json"
    with run_path.open(encoding="utf-8") as stream:
        run = json.load(stream)
    if run.get("status") != "complete":
        raise ValueError(f"Boltzmann generator run is not complete: {run_path}")
    config = run.get("config", {})
    if config.get("objective") != METHODS[method]:
        raise ValueError(f"unexpected objective in {run_path}")
    stages = run.get("stages", [])
    if not stages or not math.isclose(float(stages[-1]["t"]), 1.0):
        raise ValueError(f"Boltzmann generator run does not reach t=1: {run_path}")
    sample_path = artifact_dir / stages[-1]["path"] / "samples.npy"
    return sample_path, config, stages[-1]


def verify_bundle(bundle_path: Path, expected_hashes: dict) -> None:
    for name, expected in expected_hashes.items():
        path = bundle_path / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"bundle does not match raw reference provenance: {path}")


def convert_population(
    sample_path: Path,
    molecule: Molecule,
    target,
    converter,
    chunk_size: int,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    samples = np.load(sample_path, mmap_mode="r")
    if samples.ndim != 2 or samples.shape[1] != molecule.dimension:
        raise ValueError(f"unexpected mixed-coordinate samples: {sample_path}")
    angles = np.empty(samples.shape[0], dtype=np.float32)
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(start + chunk_size, samples.shape[0])
        chunk = np.asarray(samples[start:stop])
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"nonfinite mixed-coordinate samples: {sample_path}")
        cartesian = converter(jnp.asarray(chunk))
        cartesian = np.asarray(jax.block_until_ready(cartesian))
        angles[start:stop] = dihedral(cartesian, molecule.atoms)
    return angles


def build_cache(chunk_size: int) -> dict[str, np.ndarray]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import equinox as eqx
    import jax
    from jflows_md import Molecular_Potential

    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "schema_version": 2,
        "description": (
            "one trans conformer and one crucial dihedral per achiral molecular target"
        ),
        "angle_units": "radians",
        "reference": "selected raw OpenMM 300 K replica(s) after saved burn-in",
        "bg": "final regularized-potential sharpening samples at t=1",
        "density_processing": "achiral phi/-phi symmetrization and periodic smoothing",
        "molecules": {},
    }
    print(f"JAX backend: {jax.default_backend()}", flush=True)

    for molecule in MOLECULES:
        (
            reference,
            conformer,
            reference_provenance,
            bundle_hashes,
            conformer_provenance,
        ) = load_reference(molecule)
        target = Molecular_Potential.from_bundle(
            molecule.bundle, temperature_kelvin=300.0
        )
        if int(target.domain.dimension) != molecule.dimension:
            raise ValueError(f"bundle dimension mismatch for {molecule.key}")
        bundle_path = Path(target.bundle_path)
        verify_bundle(bundle_path, bundle_hashes)
        converter = eqx.filter_jit(target.cartesian)
        atomic_numbers, bonds = load_pdb_topology(bundle_path / "reference.pdb")
        if conformer.shape != (atomic_numbers.size, 3):
            raise ValueError(f"conformer/topology atom mismatch for {molecule.key}")

        arrays[f"{molecule.key}_reference"] = reference
        arrays[f"{molecule.key}_conformer_nm"] = conformer
        arrays[f"{molecule.key}_atomic_numbers"] = atomic_numbers
        arrays[f"{molecule.key}_bonds"] = bonds
        molecule_metadata = {
            "display_name": molecule.display_name,
            "dimension": molecule.dimension,
            "atom_indices": list(molecule.atoms),
            "dihedral_label": molecule.dihedral_label,
            "bundle_path": relative_path(bundle_path),
            "reference_selection": molecule.reference_selection,
            "reference_seeds": list(molecule.reference_seeds),
            "reference_sources": reference_provenance,
            "conformer_source": conformer_provenance,
            "bg_sources": {},
        }
        print(
            f"{molecule.display_name}: OpenMM reference "
            f"{reference.size} angles from "
            f"{len(reference_provenance)} selected seed(s); "
            f"trans conformer seed {conformer_provenance['seed']} "
            f"round {conformer_provenance['saved_round_index']}",
            flush=True,
        )

        for method in METHODS:
            sample_path, config, stage = final_population(molecule, method)
            arrays[f"{molecule.key}_{method}"] = convert_population(
                sample_path,
                molecule,
                target,
                converter,
                chunk_size,
            )
            molecule_metadata["bg_sources"][method] = {
                "path": relative_path(sample_path),
                "sha256": sha256(sample_path),
                "objective": config["objective"],
                "rg_param": config["rg_param_1"],
                "stage": int(stage["stage"]),
                "t": float(stage["t"]),
                "sample_count": int(arrays[f"{molecule.key}_{method}"].size),
            }
            print(
                f"  {method.upper()}: "
                f"{arrays[f'{molecule.key}_{method}'].size} final angles",
                flush=True,
            )

        metadata["molecules"][molecule.key] = molecule_metadata

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_name(f".{CACHE_PATH.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
            **arrays,
        )
    temporary.replace(CACHE_PATH)
    print(f"saved processed angle cache: {CACHE_PATH}", flush=True)
    return arrays


def load_cache() -> tuple[dict[str, np.ndarray], dict]:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"processed cache not found: {CACHE_PATH}; run without --replot first"
        )
    arrays = {}
    with np.load(CACHE_PATH, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("schema_version") != 2:
            raise ValueError(f"unsupported cache schema in {CACHE_PATH}")
        for molecule in MOLECULES:
            for series in ("reference", *METHODS):
                key = f"{molecule.key}_{series}"
                if key not in data:
                    raise ValueError(f"missing {key} in {CACHE_PATH}")
                values = np.asarray(data[key])
                if (
                    values.ndim != 1
                    or values.size == 0
                    or not np.all(np.isfinite(values))
                    or np.any(np.abs(values) > np.pi + 1e-6)
                ):
                    raise ValueError(f"invalid cached angles for {key}")
                arrays[key] = values
            conformer = np.asarray(data[f"{molecule.key}_conformer_nm"])
            atomic_numbers = np.asarray(data[f"{molecule.key}_atomic_numbers"])
            bonds = np.asarray(data[f"{molecule.key}_bonds"])
            if (
                conformer.ndim != 2
                or conformer.shape[1] != 3
                or not np.all(np.isfinite(conformer))
                or atomic_numbers.shape != (conformer.shape[0],)
                or not set(map(int, atomic_numbers)).issubset(CPK_COLORS)
                or bonds.ndim != 2
                or bonds.shape[1] != 2
                or np.any(bonds < 0)
                or np.any(bonds >= conformer.shape[0])
            ):
                raise ValueError(f"invalid cached conformer for {molecule.key}")
            arrays[f"{molecule.key}_conformer_nm"] = conformer
            arrays[f"{molecule.key}_atomic_numbers"] = atomic_numbers
            arrays[f"{molecule.key}_bonds"] = bonds
    return arrays, metadata


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


def conformer_view_matrix(elevation: float = 14.0, azimuth: float = 72.0) -> np.ndarray:
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


def shaded_sphere(color: np.ndarray, depth: float, pixels: int = 72) -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, pixels)
    xx, yy = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(xx, yy)
    normal_z = np.sqrt(np.clip(1.0 - xx**2 - yy**2, 0.0, 1.0))
    diffuse = np.clip(
        xx * LIGHT[0] + yy * LIGHT[1] + normal_z * LIGHT[2], 0.0, 1.0
    )
    shade = (0.40 + 0.60 * diffuse) * depth
    rgb = np.clip(
        color[None, None, :] * shade[..., None]
        + 0.75 * (diffuse**16)[..., None],
        0.0,
        1.0,
    )
    alpha = np.clip((1.0 - radius) / 0.045, 0.0, 1.0)
    return np.dstack((rgb, alpha))


def draw_dihedral_marker(
    axis: plt.Axes,
    positions: np.ndarray,
    atoms: tuple[int, int, int, int],
    view: np.ndarray,
    compact: bool = False,
) -> None:
    a, b, c, d = atoms
    point_a, point_b, point_c, point_d = positions[[a, b, c, d]]
    bond_axis = point_c - point_b
    bond_axis = bond_axis / np.linalg.norm(bond_axis)
    midpoint = 0.5 * (point_b + point_c)
    first = point_a - point_b
    first -= np.dot(first, bond_axis) * bond_axis
    first = first / np.linalg.norm(first)
    second = point_d - point_c
    second -= np.dot(second, bond_axis) * bond_axis
    second = second / np.linalg.norm(second)
    perpendicular = np.cross(bond_axis, first)
    angle = np.arctan2(
        np.dot(np.cross(first, second), bond_axis), np.dot(first, second)
    )
    if abs(abs(angle) - np.pi) <= np.deg2rad(2.0):
        angle = np.pi

    marker_radius = 0.92
    parameter = np.linspace(0.0, angle, 64)
    arc = midpoint + marker_radius * (
        np.cos(parameter)[:, None] * first
        + np.sin(parameter)[:, None] * perpendicular
    )
    arc = arc @ view.T
    marker_linewidth = 1.5 if compact else 2.7
    ray_linewidth = 1.0 if compact else 1.7
    halo = [
        path_effects.Stroke(
            linewidth=2.7 if compact else 4.5,
            foreground="white",
        ),
        path_effects.Normal(),
    ]
    axis.plot(
        arc[:, 0],
        arc[:, 1],
        color=GOLD,
        linewidth=marker_linewidth,
        solid_capstyle="round",
        path_effects=halo,
        zorder=60,
    )
    for direction in (first, second):
        ray = np.vstack((midpoint, midpoint + marker_radius * direction)) @ view.T
        axis.plot(
            ray[:, 0],
            ray[:, 1],
            color=GOLD,
            linewidth=ray_linewidth,
            linestyle=(0, (3, 2)),
            path_effects=halo,
            zorder=60,
        )
    label_position = midpoint + 1.42 * marker_radius * (
        np.cos(0.5 * angle) * first
        + np.sin(0.5 * angle) * perpendicular
    )
    label_position = label_position @ view.T
    axis.text(
        label_position[0],
        label_position[1],
        r"trans, $\phi\approx\pi$",
        color=GOLD,
        fontsize=7.5 if compact else 11.5,
        fontweight="bold",
        ha="center",
        va="center",
        path_effects=[
            path_effects.withStroke(
                linewidth=1.8 if compact else 3.0,
                foreground="white",
            )
        ],
        zorder=61,
    )


def draw_conformer(
    axis: plt.Axes,
    positions_nm: np.ndarray,
    atomic_numbers: np.ndarray,
    bonds: np.ndarray,
    atoms: tuple[int, int, int, int],
    compact: bool = False,
) -> None:
    positions = align_to_dihedral_bond(10.0 * positions_nm, atoms)
    view = conformer_view_matrix()
    projected = positions @ view.T
    x, y, depth = projected.T
    radii = np.asarray(
        [0.18 + 0.30 * COVALENT_RADII_ANGSTROM[int(z)] for z in atomic_numbers]
    )
    depth_low = float(depth.min())
    depth_span = float(np.ptp(depth)) or 1.0

    def depth_factor(value: float) -> float:
        return 0.50 + 0.50 * (value - depth_low) / depth_span

    for first, second in bonds:
        first = int(first)
        second = int(second)
        midpoint_x = 0.5 * (x[first] + x[second])
        midpoint_y = 0.5 * (y[first] + y[second])
        bond_depth = 0.5 * (depth[first] + depth[second])
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
                linewidth=3.0 if compact else 5.0,
                solid_capstyle="round",
                zorder=2.0 + 2.0 * (bond_depth - depth_low) / depth_span,
            )
    for index in np.argsort(depth):
        radius = radii[index]
        sphere = shaded_sphere(
            CPK_COLORS[int(atomic_numbers[index])], depth_factor(depth[index])
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
            zorder=6.0 + 4.0 * (depth[index] - depth_low) / depth_span,
        )
    draw_dihedral_marker(axis, positions, atoms, view, compact=compact)
    center = projected[:, :2].mean(axis=0)
    half_range = float(np.max(np.abs(projected[:, :2] - center)))
    half_range += float(radii.max()) + 0.12
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def periodic_density(
    values: np.ndarray, bins: int = 240, smoothing_bins: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate((values, -values))
    values = values % (2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    bin_width = edges[1] - edges[0]
    density = counts.astype(float) / (counts.sum() * bin_width)
    density = gaussian_filter1d(density, smoothing_bins, mode="wrap")
    centers = 0.5 * (edges[:-1] + edges[1:])
    boundary = 0.5 * (density[0] + density[-1])
    grid = np.concatenate(([0.0], centers, [2.0 * np.pi]))
    density = np.concatenate(([boundary], density, [boundary]))
    return grid, density


def annotate_well(
    axis: plt.Axes,
    grid: np.ndarray,
    envelope: np.ndarray,
    center: float,
    label: str,
    scale: float,
) -> None:
    window = np.abs(grid - center) <= 0.55
    local_indices = np.flatnonzero(window)
    peak = local_indices[np.argmax(envelope[window])]
    axis.text(
        grid[peak],
        envelope[peak] + 0.035 * scale,
        label,
        ha="center",
        va="bottom",
        fontsize=12,
    )


def plot(arrays: dict[str, np.ndarray]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "mathtext.fontset": "cm",
        }
    )
    fig, marginal_axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
    for index, (axis, molecule) in enumerate(zip(marginal_axes, MOLECULES)):
        grid, reference = periodic_density(
            arrays[f"{molecule.key}_reference"]
        )
        _, klx = periodic_density(arrays[f"{molecule.key}_klx"])
        _, klxx = periodic_density(arrays[f"{molecule.key}_klxx"])

        axis.fill_between(
            grid,
            reference,
            color="0.65",
            alpha=0.55,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(grid, klxx, color=RED, linewidth=2.3, zorder=3)
        axis.plot(
            grid,
            klx,
            color=BLUE,
            linewidth=2.2,
            linestyle=(0, (6, 3)),
            zorder=4,
        )
        envelope = np.maximum.reduce((reference, klx, klxx))
        scale = float(envelope.max())
        well_labels = [(np.pi, "trans")]
        if molecule.key != "nma":
            well_labels = [
                (np.pi / 3.0, r"gauche $+$"),
                *well_labels,
                (5.0 * np.pi / 3.0, r"gauche $-$"),
            ]
        for center, label in well_labels:
            annotate_well(axis, grid, envelope, center, label, scale)
        axis.set_xlim(0.0, 2.0 * np.pi)
        axis.set_ylim(0.0, 1.15 * scale)
        axis.set_xticks([0.0, np.pi, 2.0 * np.pi])
        axis.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
        axis.set_yticks([])
        axis.set_xlabel("dihedral (rad)")
        axis.set_title(
            rf"{molecule.display_name} ($d={molecule.dimension}$) — "
            f"{molecule.dihedral_label}",
            pad=8,
        )
        if index == 0:
            axis.set_ylabel("density")

        conformer_axis = axis.inset_axes(
            [0.012, 0.625, 0.32, 0.35],
            transform=axis.transAxes,
            zorder=10,
        )
        conformer_axis.set_facecolor((1.0, 1.0, 1.0, 0.90))
        draw_conformer(
            conformer_axis,
            arrays[f"{molecule.key}_conformer_nm"],
            arrays[f"{molecule.key}_atomic_numbers"],
            arrays[f"{molecule.key}_bonds"],
            molecule.atoms,
            compact=True,
        )

    handles = [
        Patch(color="0.65", alpha=0.55, label="OpenMM reference"),
        Line2D(
            [0],
            [0],
            color=BLUE,
            linewidth=2.2,
            linestyle=(0, (6, 3)),
            label=r"regularized Boltzmann generator: $\mathrm{KL}$+$\mathrm{X}_{\mu}$",
        ),
        Line2D(
            [0],
            [0],
            color=RED,
            linewidth=2.3,
            label=(
                r"regularized Boltzmann generator: $\mathrm{KL}$+$\mathrm{X}_{\mu}$+"
                r"$\mathrm{X}_{(\hat{\mu}+\bar{\nu})/2}$"
            ),
        ),
    ]
    fig.suptitle(
        "Dihedral marginals: regularized Boltzmann generator samples vs OpenMM reference",
        fontsize=17,
        y=0.98,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.25,
        top=0.84,
        wspace=0.13,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_name(f".{FIGURE_PATH.name}.tmp")
    fig.savefig(temporary, format="png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(FIGURE_PATH)
    print(f"saved combined 1x3 figure: {FIGURE_PATH}", flush=True)


def main() -> None:
    args = parse_args()
    if args.replot:
        arrays, _ = load_cache()
        print(f"loaded processed angle cache: {CACHE_PATH}", flush=True)
    else:
        build_cache(args.chunk_size)
        arrays, _ = load_cache()
    plot(arrays)


if __name__ == "__main__":
    main()

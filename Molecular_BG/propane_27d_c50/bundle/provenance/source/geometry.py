"""Deterministic explicit-H n-alkane seed geometries and MOL2 rendering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


CC_ANGSTROM = 1.54
CH_ANGSTROM = 1.09
TETRAHEDRAL_COSINE = -1.0 / 3.0


@dataclass(frozen=True)
class Alkane_Geometry:
    carbon_count: int
    atom_names: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    positions_angstrom: np.ndarray
    bonds: tuple[tuple[int, int], ...]
    hydrogen_parent: tuple[int, ...]

    @property
    def atom_count(self) -> int:
        return len(self.atom_names)

    @property
    def formula(self) -> str:
        return f"C{self.carbon_count if self.carbon_count > 1 else ''}H{2 * self.carbon_count + 2}"

    @property
    def canonical_smiles(self) -> str:
        return "C" * self.carbon_count


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalize a zero or nonfinite vector")
    return np.asarray(value, dtype=float) / norm


def _terminal_hydrogen_directions(carbon_neighbor: np.ndarray) -> tuple[np.ndarray, ...]:
    """Complete one C-C direction to an exact tetrahedral frame."""

    axis = _unit(carbon_neighbor)
    trial = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(axis, trial))) > 0.9:
        trial = np.asarray((0.0, 1.0, 0.0))
    first = _unit(np.cross(axis, trial))
    second = _unit(np.cross(axis, first))
    transverse = math.sqrt(8.0 / 9.0)
    directions = []
    for index in range(3):
        phase = 2.0 * math.pi * index / 3.0
        direction = (
            TETRAHEDRAL_COSINE * axis
            + transverse * (math.cos(phase) * first + math.sin(phase) * second)
        )
        directions.append(_unit(direction))
    return tuple(directions)


def _internal_hydrogen_directions(
    first_neighbor: np.ndarray, second_neighbor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Complete two tetrahedral C-C directions with the two hydrogen directions."""

    first = _unit(first_neighbor)
    second = _unit(second_neighbor)
    if abs(float(np.dot(first, second)) - TETRAHEDRAL_COSINE) > 1e-12:
        raise ValueError("carbon backbone is not tetrahedral")
    normal = _unit(np.cross(first, second))
    center = -0.5 * (first + second)
    offset = math.sqrt(2.0 / 3.0) * normal
    return _unit(center + offset), _unit(center - offset)


def build_alkane_geometry(carbon_count: int) -> Alkane_Geometry:
    """Return a heavy-backbone-first ideal all-trans labeled n-alkane."""

    if carbon_count not in (1, 2, 3, 4):
        raise ValueError("the frozen diagnostic series contains one through four carbons")

    if carbon_count == 1:
        carbons = np.zeros((1, 3), dtype=float)
        tetrahedron = np.asarray(
            ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0),
             (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)),
            dtype=float,
        ) / math.sqrt(3.0)
        hydrogen_positions = [CH_ANGSTROM * direction for direction in tetrahedron]
        parents = [0, 0, 0, 0]
    else:
        angle = 0.5 * math.acos(1.0 / 3.0)
        directions = [
            np.asarray((math.cos(angle), (-1.0) ** index * math.sin(angle), 0.0))
            for index in range(carbon_count - 1)
        ]
        carbon_positions = [np.zeros(3, dtype=float)]
        for direction in directions:
            carbon_positions.append(carbon_positions[-1] + CC_ANGSTROM * direction)
        carbons = np.asarray(carbon_positions)
        hydrogen_positions, parents = [], []
        for carbon in range(carbon_count):
            neighbors = []
            if carbon > 0:
                neighbors.append(carbons[carbon - 1] - carbons[carbon])
            if carbon + 1 < carbon_count:
                neighbors.append(carbons[carbon + 1] - carbons[carbon])
            if len(neighbors) == 1:
                local = _terminal_hydrogen_directions(neighbors[0])
            else:
                local = _internal_hydrogen_directions(neighbors[0], neighbors[1])
            for direction in local:
                hydrogen_positions.append(carbons[carbon] + CH_ANGSTROM * direction)
                parents.append(carbon)

    positions = np.concatenate((carbons, np.asarray(hydrogen_positions)), axis=0)
    expected_hydrogens = 2 * carbon_count + 2
    if len(parents) != expected_hydrogens:
        raise AssertionError("internal hydrogen-count error")
    carbon_names = tuple(f"C{index + 1}" for index in range(carbon_count))
    hydrogen_names = tuple(f"H{index + 1}" for index in range(expected_hydrogens))
    bonds = [(index, index + 1) for index in range(carbon_count - 1)]
    bonds.extend((parent, carbon_count + index) for index, parent in enumerate(parents))
    geometry = Alkane_Geometry(
        carbon_count=carbon_count,
        atom_names=carbon_names + hydrogen_names,
        atomic_numbers=(6,) * carbon_count + (1,) * expected_hydrogens,
        positions_angstrom=positions,
        bonds=tuple(bonds),
        hydrogen_parent=tuple(parents),
    )
    validate_geometry(geometry)
    return geometry


def validate_geometry(geometry: Alkane_Geometry) -> None:
    """Check formula, distances, heavy-first order, and tetrahedral angles."""

    expected_atoms = 3 * geometry.carbon_count + 2
    if geometry.positions_angstrom.shape != (expected_atoms, 3):
        raise ValueError("alkane coordinate array has the wrong shape")
    if not np.isfinite(geometry.positions_angstrom).all():
        raise ValueError("alkane coordinates must be finite")
    if geometry.atomic_numbers != (6,) * geometry.carbon_count + (1,) * (
        2 * geometry.carbon_count + 2
    ):
        raise ValueError("atom order is not heavy-backbone first")

    for first, second in geometry.bonds:
        distance = float(
            np.linalg.norm(
                geometry.positions_angstrom[first] - geometry.positions_angstrom[second]
            )
        )
        expected = CC_ANGSTROM if first < geometry.carbon_count and second < geometry.carbon_count else CH_ANGSTROM
        if abs(distance - expected) > 1e-10:
            raise ValueError(f"unexpected seed bond length {distance} != {expected}")

    adjacency = [set() for _ in range(geometry.atom_count)]
    for first, second in geometry.bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    for carbon in range(geometry.carbon_count):
        vectors = [
            _unit(geometry.positions_angstrom[neighbor] - geometry.positions_angstrom[carbon])
            for neighbor in sorted(adjacency[carbon])
        ]
        if len(vectors) != 4:
            raise ValueError("every alkane carbon must have four neighbors")
        for first in range(4):
            for second in range(first):
                if abs(float(np.dot(vectors[first], vectors[second])) + 1.0 / 3.0) > 1e-10:
                    raise ValueError("seed carbon environment is not tetrahedral")


def render_mol2(geometry: Alkane_Geometry) -> str:
    """Render a deterministic zero-charge seed MOL2 for AmberTools."""

    lines = [
        "@<TRIPOS>MOLECULE",
        geometry.formula,
        f"{geometry.atom_count} {len(geometry.bonds)} 1 0 0",
        "SMALL",
        "NO_CHARGES",
        "@<TRIPOS>ATOM",
    ]
    for index, (name, atomic_number, xyz) in enumerate(
        zip(
            geometry.atom_names,
            geometry.atomic_numbers,
            geometry.positions_angstrom,
            strict=True,
        ),
        start=1,
    ):
        atom_type = "C.3" if atomic_number == 6 else "H"
        lines.append(
            f"{index:7d} {name:<8s} {xyz[0]:12.7f} {xyz[1]:12.7f} "
            f"{xyz[2]:12.7f} {atom_type:<6s} 1 MOL 0.000000"
        )
    lines.append("@<TRIPOS>BOND")
    for index, (first, second) in enumerate(geometry.bonds, start=1):
        lines.append(f"{index:7d} {first + 1:7d} {second + 1:7d} 1")
    lines.extend(("@<TRIPOS>SUBSTRUCTURE", "      1 MOL             1 RESIDUE    0 **** ROOT      0"))
    return "\n".join(lines) + "\n"


def geometry_sha256(geometry: Alkane_Geometry) -> str:
    return hashlib.sha256(render_mol2(geometry).encode("utf-8")).hexdigest()


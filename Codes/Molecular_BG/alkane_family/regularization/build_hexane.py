#!/usr/bin/env python
"""Build a six-file n-hexane bundle matching the existing alkane family."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import openmm as mm
from openmm import app, unit
import parmed

from jflows_md.bundle_build.builder import (
    KB_KJ_MOL_K,
    _formula,
    _minimize,
    _raw_internal,
    build_obc1_system,
    build_validation_spec,
    extract_system_spec,
)
from jflows_md.bundle_build.zmatrix import build_zmatrix, validate_zmatrix
from jflows_md.system import Molecular_Bundle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "bundle" / "hexane_54d"
PROVENANCE = ROOT / "provenance" / "hexane_54d"
CARBONS = 6
CC_ANGSTROM = 1.54
CH_ANGSTROM = 1.09
TETRAHEDRAL_COSINE = -1.0 / 3.0


def unit_vector(value):
    return np.asarray(value, dtype=float) / np.linalg.norm(value)


def terminal_directions(neighbor):
    axis = unit_vector(neighbor)
    trial = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(axis, trial)) > 0.9:
        trial = np.array((0.0, 1.0, 0.0))
    first = unit_vector(np.cross(axis, trial))
    second = unit_vector(np.cross(axis, first))
    transverse = math.sqrt(8.0 / 9.0)
    return tuple(
        unit_vector(
            TETRAHEDRAL_COSINE * axis
            + transverse
            * (math.cos(2 * math.pi * i / 3) * first + math.sin(2 * math.pi * i / 3) * second)
        )
        for i in range(3)
    )


def internal_directions(first_neighbor, second_neighbor):
    first = unit_vector(first_neighbor)
    second = unit_vector(second_neighbor)
    normal = unit_vector(np.cross(first, second))
    center = -0.5 * (first + second)
    offset = math.sqrt(2.0 / 3.0) * normal
    return unit_vector(center + offset), unit_vector(center - offset)


def geometry():
    angle = 0.5 * math.acos(1.0 / 3.0)
    directions = [
        np.array((math.cos(angle), (-1.0) ** i * math.sin(angle), 0.0))
        for i in range(CARBONS - 1)
    ]
    carbon_positions = [np.zeros(3)]
    for direction in directions:
        carbon_positions.append(carbon_positions[-1] + CC_ANGSTROM * direction)
    carbons = np.asarray(carbon_positions)
    hydrogens = []
    parents = []
    for carbon in range(CARBONS):
        neighbors = []
        if carbon:
            neighbors.append(carbons[carbon - 1] - carbons[carbon])
        if carbon + 1 < CARBONS:
            neighbors.append(carbons[carbon + 1] - carbons[carbon])
        directions = (
            terminal_directions(neighbors[0])
            if len(neighbors) == 1
            else internal_directions(*neighbors)
        )
        for direction in directions:
            hydrogens.append(carbons[carbon] + CH_ANGSTROM * direction)
            parents.append(carbon)
    positions = np.concatenate((carbons, np.asarray(hydrogens)))
    names = tuple(f"C{i + 1}" for i in range(CARBONS)) + tuple(
        f"H{i + 1}" for i in range(len(hydrogens))
    )
    atomic_numbers = (6,) * CARBONS + (1,) * len(hydrogens)
    bonds = [(i, i + 1) for i in range(CARBONS - 1)]
    bonds.extend((parent, CARBONS + i) for i, parent in enumerate(parents))
    return names, atomic_numbers, positions, tuple(bonds)


def mol2_text(names, atomic_numbers, positions, bonds):
    lines = [
        "@<TRIPOS>MOLECULE",
        "C6H14",
        f"{len(names)} {len(bonds)} 1 0 0",
        "SMALL",
        "NO_CHARGES",
        "@<TRIPOS>ATOM",
    ]
    for i, (name, z, xyz) in enumerate(zip(names, atomic_numbers, positions, strict=True), 1):
        atom_type = "C.3" if z == 6 else "H"
        lines.append(
            f"{i:7d} {name:<8s} {xyz[0]:12.7f} {xyz[1]:12.7f} "
            f"{xyz[2]:12.7f} {atom_type:<6s} 1 MOL 0.000000"
        )
    lines.append("@<TRIPOS>BOND")
    for i, (first, second) in enumerate(bonds, 1):
        lines.append(f"{i:7d} {first + 1:7d} {second + 1:7d} 1")
    lines.extend(("@<TRIPOS>SUBSTRUCTURE", "      1 MOL             1 RESIDUE    0 **** ROOT      0"))
    return "\n".join(lines) + "\n"


def mol2_charges(path):
    charges = []
    section = None
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line[9:]
        elif section == "ATOM" and line.strip():
            charges.append(float(line.split()[-1]))
    return np.asarray(charges)


def project_charges(path, atomic_numbers, bonds):
    adjacency = [set() for _ in atomic_numbers]
    for first, second in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    labels = list(atomic_numbers)
    while True:
        keys = [
            (labels[i], tuple(sorted(labels[j] for j in adjacency[i])))
            for i in range(len(labels))
        ]
        values = {key: value for value, key in enumerate(sorted(set(keys)))}
        updated = [values[key] for key in keys]
        if updated == labels:
            break
        labels = updated
    groups = [
        [i for i, value in enumerate(labels) if value == label]
        for label in sorted(set(labels))
    ]
    raw = mol2_charges(path)
    projected = raw.copy()
    for group in groups:
        projected[group] = np.mean(raw[group])
    projected -= np.mean(projected)

    output = []
    section = None
    atom = 0
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line[9:]
            output.append(line)
        elif section == "ATOM" and line.strip():
            fields = line.split()
            fields[-1] = f"{projected[atom]:.10f}"
            output.append(" ".join(fields))
            atom += 1
        else:
            output.append(line)
    path.write_text("\n".join(output) + "\n")
    return raw, mol2_charges(path), groups


def run(command, cwd, log):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    (cwd / log).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))} failed; see {cwd / log}")


def coordinate_spec(system_spec, positions_nm, bonds):
    n_atoms = len(positions_nm)
    order, refs = build_zmatrix(
        bonds, n_atoms, positions=positions_nm, root=0, prefix=tuple(range(CARBONS))
    )
    validate_zmatrix(order, refs, bonds)
    raw_bonds, raw_angles, raw_torsions = _raw_internal(positions_nm, order, refs)
    kbt = KB_KJ_MOL_K * 300.0
    bond_parameters = {
        frozenset(pair): k
        for pair, k in zip(system_spec["bond_idx"], system_spec["bond_k_kj_mol_nm2"], strict=True)
    }
    bond_scales = [
        float(np.clip(math.sqrt(kbt / bond_parameters[frozenset((order[i], refs[i][0]))]) / distance, 0.01, 0.15))
        for i, distance in enumerate(raw_bonds, 1)
    ]
    angle_parameters = {}
    for triple, k in zip(system_spec["angle_idx"], system_spec["angle_k_kj_mol_rad2"], strict=True):
        first, center, third = triple
        angle_parameters[(center, frozenset((first, third)))] = k
    angle_scales = []
    for i, theta in enumerate(raw_angles, 2):
        atom = order[i]
        first, second, _ = refs[i]
        k = angle_parameters.get((first, frozenset((atom, second))))
        sigma = 0.08 if k is None else math.sqrt(kbt / k)
        fraction = theta / math.pi
        angle_scales.append(float(np.clip(sigma / (math.pi * fraction * (1 - fraction)), 0.02, 0.5)))
    backbone = []
    for i in range(3, n_atoms):
        if all(atom < CARBONS for atom in (order[i], *refs[i])):
            backbone.append(i - 3)
    return {
        "schema_version": 2,
        "chart": "log-bond_logit-angle_quotient_BAT_v2",
        "jacobian_measure": "rigid_motion_quotient_v1",
        "dimension": 3 * n_atoms - 6,
        "euclidean_dim": 2 * n_atoms - 3,
        "periodic_dim": n_atoms - 3,
        "order": list(order),
        "refs": [list(row) for row in refs],
        "bond_log_offset": np.log(raw_bonds).tolist(),
        "bond_log_scale": bond_scales,
        "angle_logit_offset": (np.log(raw_angles / math.pi) - np.log1p(-raw_angles / math.pi)).tolist(),
        "angle_logit_scale": angle_scales,
        "reference_torsions_rad": raw_torsions.tolist(),
        "chiral_torsion_index": -1,
        "chiral_torsion_sign": 0,
        "chirality_atoms": [-1, -1, -1, -1],
        "chirality_sign": 0,
        "diagnostic_chirality_atoms": [0, 1, CARBONS, CARBONS + 1],
        "source_mean": np.zeros(2 * n_atoms - 3).tolist(),
        "source_variance": np.ones(2 * n_atoms - 3).tolist(),
        "backbone_torsion_indices": backbone,
        "atom_permutation_groups": [],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    if OUTPUT.exists() or PROVENANCE.exists():
        raise FileExistsError("hexane bundle or provenance already exists")
    names, atomic_numbers, positions, bonds = geometry()
    executable = Path(sys.prefix) / "bin"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".hexane_build_") as temporary:
        work = Path(temporary)
        (work / "input.mol2").write_text(mol2_text(names, atomic_numbers, positions, bonds))
        run(
            [
                executable / "antechamber", "-i", "input.mol2", "-fi", "mol2",
                "-o", "gaff2.mol2", "-fo", "mol2", "-at", "gaff2", "-c", "bcc",
                "-nc", "0", "-m", "1", "-eq", "0", "-seq", "n", "-an", "n",
                "-s", "2", "-pf", "n",
            ],
            work,
            "antechamber.log",
        )
        raw_charges, final_charges, groups = project_charges(work / "gaff2.mol2", atomic_numbers, bonds)
        run(
            [executable / "parmchk2", "-i", "gaff2.mol2", "-f", "mol2", "-o", "gaff2.frcmod", "-s", "gaff2"],
            work,
            "parmchk2.log",
        )
        (work / "leap.in").write_text(
            "source leaprc.gaff2\nset default PBRadii mbondi2\n"
            "loadamberparams gaff2.frcmod\nMOL = loadmol2 gaff2.mol2\n"
            "check MOL\nsaveamberparm MOL molecule.prmtop molecule.rst7\nquit\n"
        )
        run([executable / "tleap", "-f", "leap.in"], work, "tleap.log")

        topology, system = build_obc1_system(work / "molecule.prmtop")
        amber_positions = app.AmberInpcrdFile(str(work / "molecule.rst7")).positions
        minimized = _minimize(topology.topology, system, amber_positions)
        positions_nm = np.asarray(minimized.value_in_unit(unit.nanometer))
        structure = parmed.load_file(str(work / "molecule.prmtop"), xyz=str(work / "molecule.rst7"))
        system_spec = extract_system_spec(system)
        system_spec.update(
            {
                "atomic_numbers": [int(atom.atomic_number) for atom in structure.atoms],
                "atom_names": [atom.name for atom in structure.atoms],
                "residue_names": [atom.residue.name for atom in structure.atoms],
                "bonds": [[bond.atom1.idx, bond.atom2.idx] for bond in structure.bonds],
                "formula": _formula([atom.atomic_number for atom in structure.atoms]),
                "net_charge_e": float(sum(atom.charge for atom in structure.atoms)),
            }
        )
        coordinates = coordinate_spec(system_spec, positions_nm, system_spec["bonds"])
        validation = build_validation_spec(system, minimized)
        candidate = work / "bundle"
        candidate.mkdir()
        (candidate / "system.xml").write_text(mm.XmlSerializer.serialize(system))
        with (candidate / "reference.pdb").open("w") as handle:
            app.PDBFile.writeFile(topology.topology, minimized, handle, keepIds=True)
        write_json(candidate / "system.json", system_spec)
        write_json(candidate / "coordinates.json", coordinates)
        write_json(candidate / "validation.json", validation)
        manifest = {
            "schema_version": 1,
            "name": "n_hexane_gaff2_am1bcc_obc1",
            "target": "n_hexane",
            "temperature_kelvin": 300.0,
            "canonical_smiles": "CCCCCC",
            "formula": "C6H14",
            "formal_charge": 0,
            "model": {
                "force_field": "GAFF2",
                "charges": "AM1-BCC with graph-symmetry/neutrality projection",
                "implicit_solvent": "OBC1 (igb=2)",
                "radii": "mbondi2",
                "sasa": "ACE",
                "solute_dielectric": 1.0,
                "solvent_dielectric": 78.5,
                "salt_molar": 0.0,
                "nonbonded_method": "NoCutoff",
                "constraints": None,
            },
            "coordinate_measure": "rigid_motion_quotient_v1",
            "openmm_version": mm.__version__,
            "system_spec": "system.json",
            "coordinate_spec": "coordinates.json",
            "validation_spec": "validation.json",
        }
        write_json(candidate / "manifest.json", manifest)
        bundle = Molecular_Bundle.load(candidate, verify=True)
        if bundle.n_atoms != 20 or bundle.dimension != 54 or bundle.manifest["formula"] != "C6H14":
            raise ValueError("hexane bundle identity mismatch")

        provenance = work / "provenance"
        provenance.mkdir()
        for name in (
            "input.mol2", "gaff2.mol2", "gaff2.frcmod", "leap.in",
            "molecule.prmtop", "molecule.rst7", "antechamber.log",
            "parmchk2.log", "tleap.log",
        ):
            shutil.copy2(work / name, provenance / name)
        record = {
            "python": sys.version.split()[0],
            "openmm": mm.__version__,
            "parmed": parmed.__version__,
            "geometry": "heavy-carbon-first ideal tetrahedral all-trans n-hexane",
            "input_mol2_sha256": hashlib.sha256((work / "input.mol2").read_bytes()).hexdigest(),
            "raw_charges_e": raw_charges.tolist(),
            "projected_charges_e": final_charges.tolist(),
            "equivalence_groups_zero_based": groups,
        }
        write_json(provenance / "build.json", record)
        candidate.rename(OUTPUT)
        provenance.rename(PROVENANCE)
    print(f"bundle ready: {OUTPUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build content-addressed experimental n-alkane OBC1 bundles.

The builder is deliberately project-local.  It does not modify the public
`jflows_md` bundle registry and every runtime consumer must load the resulting
directory explicitly.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import openmm
import parmed

from geometry import build_alkane_geometry, geometry_sha256, render_mol2
import parameters as P

from jflows_md.core.builder import (
    KB_KJ_MOL_K,
    _formula,
    _json_write,
    _minimize,
    _raw_internal,
    build_obc1_system,
    build_validation_spec,
    extract_system_spec,
    normalize_amber_transcript,
)
from jflows_md.core.zmatrix import build_zmatrix, validate_zmatrix
from jflows_md import package_source_sha256
import jflows_md
from jflows_md.system import Molecular_Bundle, sha256_file


ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = ROOT / "candidates"


def run(command: list[str], cwd: Path, log_name: str) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    (cwd / log_name).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}; "
            f"see {cwd / log_name}"
        )


def ambertools_prefix() -> Path:
    executables = {
        name: shutil.which(name) for name in ("antechamber", "parmchk2", "tleap")
    }
    if any(path is None for path in executables.values()):
        raise RuntimeError("AmberTools executables are not available in PATH")
    prefixes = {
        Path(path).resolve().parent.parent
        for path in executables.values()
        if path is not None
    }
    if len(prefixes) != 1:
        raise RuntimeError(f"AmberTools executables resolve to several prefixes: {prefixes}")
    prefix = prefixes.pop()
    if prefix != Path(sys.prefix).resolve():
        raise RuntimeError(
            f"AmberTools must come from the active environment: {prefix} != {sys.prefix}"
        )
    return prefix


def amber_data_hashes(prefix: Path) -> dict[str, str]:
    files = {
        "gaff2.dat": prefix / "dat/leap/parm/gaff2.dat",
        "BCCPARM.DAT": prefix / "dat/antechamber/BCCPARM.DAT",
        "ATOMTYPE_GFF2.DEF": prefix / "dat/antechamber/ATOMTYPE_GFF2.DEF",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot identify source revision at {root}")
    return result.stdout.strip()


def _read_mol2(path: Path) -> tuple[list[dict], set[tuple[int, int]]]:
    atoms, bonds = [], set()
    section = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>")
            continue
        if not line.strip():
            continue
        fields = line.split()
        if section == "ATOM":
            atoms.append(
                {
                    "index": int(fields[0]) - 1,
                    "name": fields[1],
                    "type": fields[5],
                    "charge": float(fields[-1]),
                }
            )
        elif section == "BOND":
            bonds.add(tuple(sorted((int(fields[1]) - 1, int(fields[2]) - 1))))
    return atoms, bonds


def _graph_equivalence_groups(geometry) -> tuple[tuple[int, ...], ...]:
    """Return exact automorphism classes for these acyclic alkane graphs."""

    adjacency = [set() for _ in range(geometry.atom_count)]
    for first, second in geometry.bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    labels = list(geometry.atomic_numbers)
    while True:
        keys = [
            (labels[index], tuple(sorted(labels[neighbor] for neighbor in adjacency[index])))
            for index in range(geometry.atom_count)
        ]
        unique = {key: value for value, key in enumerate(sorted(set(keys)))}
        updated = [unique[key] for key in keys]
        if updated == labels:
            break
        labels = updated
    groups = []
    for label in sorted(set(labels)):
        groups.append(tuple(index for index, value in enumerate(labels) if value == label))
    return tuple(groups)


def _project_mol2_charges(path: Path, geometry) -> dict:
    """Project rounded AM1-BCC charges onto graph symmetry and exact neutrality.

    AmberTools 26 emits methane charges summing to -0.002 e even with its
    built-in equivalence switches.  The nearest least-squares projection first
    averages every exact graph-automorphism class, then applies the unique
    uniform shift enforcing the requested integer net charge.  This rule is
    fixed for the complete alkane series and is recorded as part of the model.
    """

    atoms, _ = _read_mol2(path)
    raw = np.asarray([atom["charge"] for atom in atoms], dtype=float)
    projected = raw.copy()
    groups = _graph_equivalence_groups(geometry)
    raw_spread = {}
    for group in groups:
        values = raw[list(group)]
        projected[list(group)] = np.mean(values)
        raw_spread[",".join(map(str, group))] = float(np.ptp(values))
    projected -= float(np.sum(projected)) / projected.size

    lines, output = path.read_text(encoding="utf-8").splitlines(), []
    section, atom_index = None, 0
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>")
            output.append(line)
            continue
        if section == "ATOM" and line.strip():
            fields = line.split()
            fields[-1] = f"{projected[atom_index]:.10f}"
            output.append(" ".join(fields))
            atom_index += 1
        else:
            output.append(line)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    written, _ = _read_mol2(path)
    written_charges = np.asarray([atom["charge"] for atom in written])
    if abs(float(written_charges.sum())) > 1e-9:
        raise ValueError(
            f"charge projection did not survive MOL2 serialization: {written_charges.sum()}"
        )
    return {
        "method": "graph-class mean plus uniform least-squares neutrality shift",
        "equivalence_groups_zero_based": [list(group) for group in groups],
        "raw_net_charge_e": float(raw.sum()),
        "raw_charges_e": raw.tolist(),
        "projected_net_charge_e": float(written_charges.sum()),
        "projected_charges_e": written_charges.tolist(),
        "maximum_absolute_correction_e": float(np.max(np.abs(written_charges - raw))),
        "raw_within_group_ranges_e": raw_spread,
    }


def _validate_antechamber_output(path: Path, geometry) -> dict:
    atoms, bonds = _read_mol2(path)
    if len(atoms) != geometry.atom_count:
        raise ValueError("Antechamber changed the atom count")
    if [atom["index"] for atom in atoms] != list(range(geometry.atom_count)):
        raise ValueError("Antechamber output atom indices are not sequential")
    if tuple(atom["name"] for atom in atoms) != geometry.atom_names:
        raise ValueError(
            "Antechamber changed the heavy-first atom order/names: "
            f"{tuple(atom['name'] for atom in atoms)} != {geometry.atom_names}"
        )
    expected_bonds = {tuple(sorted(pair)) for pair in geometry.bonds}
    if bonds != expected_bonds:
        raise ValueError(f"Antechamber changed connectivity: {bonds} != {expected_bonds}")
    carbon_count = geometry.carbon_count
    carbon_types = {atom["type"] for atom in atoms[:carbon_count]}
    hydrogen_types = {atom["type"] for atom in atoms[carbon_count:]}
    if len(carbon_types) != 1 or len(hydrogen_types) != 1:
        raise ValueError(
            f"equivalent alkane atom types differ: C={carbon_types}, H={hydrogen_types}"
        )
    charges = np.asarray([atom["charge"] for atom in atoms])
    charge_sum = float(charges.sum())
    if abs(charge_sum) > 1e-5:
        raise ValueError(
            f"Antechamber charges are not neutral: {charge_sum}; charges={charges.tolist()}"
        )
    if geometry.carbon_count == 1 and float(np.ptp(charges[1:])) > 1e-10:
        raise ValueError(f"methane hydrogen charges differ: {charges[1:]}")
    return {
        "net_charge_e": charge_sum,
        "carbon_types": sorted(carbon_types),
        "hydrogen_types": sorted(hydrogen_types),
        "methane_hydrogen_charge_range_e": float(np.ptp(charges[1:]))
        if geometry.carbon_count == 1
        else None,
    }


def _signed_volume(positions: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    center, first, second, third = atoms
    return float(
        np.dot(
            positions[first] - positions[center],
            np.cross(
                positions[second] - positions[center],
                positions[third] - positions[center],
            ),
        )
    )


def _build_coordinate_spec(
    system_spec: dict,
    positions_nm: np.ndarray,
    bonds: list[list[int]],
    *,
    carbon_count: int,
) -> dict:
    """Build a full-support heavy-backbone-first schema-2 BAT chart."""

    n_atoms = int(system_spec["n_atoms"])
    order, refs = build_zmatrix(
        bonds,
        n_atoms,
        positions=positions_nm,
        root=0,
        prefix=tuple(range(carbon_count)),
    )
    validate_zmatrix(order, refs, bonds)
    if tuple(order[:carbon_count]) != tuple(range(carbon_count)):
        raise ValueError("Z-matrix did not preserve the heavy-backbone prefix")
    raw_bonds, raw_angles, raw_torsions = _raw_internal(positions_nm, order, refs)

    kbt = KB_KJ_MOL_K * 300.0
    bond_parameters = {
        frozenset(pair): (length, k)
        for pair, length, k in zip(
            system_spec["bond_idx"],
            system_spec["bond_length_nm"],
            system_spec["bond_k_kj_mol_nm2"],
            strict=True,
        )
    }
    bond_scales = []
    for placement, distance in enumerate(raw_bonds, start=1):
        pair = frozenset((order[placement], refs[placement][0]))
        _, force_constant = bond_parameters[pair]
        bond_scales.append(
            float(
                np.clip(
                    math.sqrt(kbt / force_constant) / distance,
                    0.01,
                    0.15,
                )
            )
        )

    angle_parameters = {}
    for triple, force_constant in zip(
        system_spec["angle_idx"],
        system_spec["angle_k_kj_mol_rad2"],
        strict=True,
    ):
        first, center, third = map(int, triple)
        angle_parameters[(center, frozenset((first, third)))] = float(force_constant)
    angle_scales = []
    for placement, theta in enumerate(raw_angles, start=2):
        atom = order[placement]
        first, second, _ = refs[placement]
        force_constant = angle_parameters.get(
            (first, frozenset((atom, second)))
        )
        sigma = 0.08 if force_constant is None else math.sqrt(kbt / force_constant)
        fraction = theta / math.pi
        scale = sigma / (math.pi * fraction * (1.0 - fraction))
        angle_scales.append(float(np.clip(scale, 0.02, 0.5)))

    bond_offsets = np.log(raw_bonds)
    angle_offsets = np.log(raw_angles / math.pi) - np.log1p(-raw_angles / math.pi)
    if carbon_count == 1:
        # Equivalent labeled H atoms must not acquire different chart scales
        # from harmless minimizer/serialization roundoff.
        bond_offsets[:] = np.mean(bond_offsets)
        bond_scales[:] = [float(np.mean(bond_scales))] * len(bond_scales)
        angle_offsets[:] = np.mean(angle_offsets)
        angle_scales[:] = [float(np.mean(angle_scales))] * len(angle_scales)

    hydrogens_on_first = [
        atom
        for atom in range(carbon_count, n_atoms)
        if frozenset((0, atom)) in {frozenset(pair) for pair in bonds}
    ]
    if carbon_count == 1:
        diagnostic_atoms = (0, *hydrogens_on_first[:3])
    else:
        diagnostic_atoms = (0, 1, *hydrogens_on_first[:2])
    if len(diagnostic_atoms) != 4 or abs(_signed_volume(positions_nm, diagnostic_atoms)) < 1e-10:
        raise ValueError("diagnostic signed-volume frame is degenerate")

    backbone_torsions = []
    for placement in range(3, len(order)):
        atom = order[placement]
        first, second, third = refs[placement]
        if all(index < carbon_count for index in (atom, first, second, third)):
            backbone_torsions.append(placement - 3)

    euclidean_dim = 2 * n_atoms - 3
    periodic_dim = n_atoms - 3
    return {
        "schema_version": 2,
        "chart": "log-bond_logit-angle_quotient_BAT_v2",
        "jacobian_measure": "rigid_motion_quotient_v1",
        "dimension": 3 * n_atoms - 6,
        "euclidean_dim": euclidean_dim,
        "periodic_dim": periodic_dim,
        "order": list(order),
        "refs": [list(row) for row in refs],
        "bond_log_offset": bond_offsets.tolist(),
        "bond_log_scale": list(map(float, bond_scales)),
        "angle_logit_offset": angle_offsets.tolist(),
        "angle_logit_scale": list(map(float, angle_scales)),
        "reference_torsions_rad": raw_torsions.tolist(),
        "chiral_torsion_index": -1,
        "chiral_torsion_sign": 0,
        "chirality_atoms": [-1, -1, -1, -1],
        "chirality_sign": 0,
        "diagnostic_chirality_atoms": list(diagnostic_atoms),
        "source_mean": np.zeros(euclidean_dim).tolist(),
        "source_variance": np.ones(euclidean_dim).tolist(),
        "backbone_torsion_indices": backbone_torsions,
        "atom_permutation_groups": [list(range(carbon_count, n_atoms))]
        if carbon_count == 1
        else [],
    }


def _write_experimental_bundle(
    output: Path,
    *,
    semantic_name: str,
    molecule_name: str,
    carbon_count: int,
    prmtop_path: Path,
    coordinate_path: Path,
    canonical_smiles: str,
    expected_formula: str,
    model: dict,
    provenance_files: dict[str, Path],
) -> Path:
    import openmm as mm
    from openmm import app, unit
    import parmed as pmd

    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.mkdir(parents=True)
    shutil.copy2(prmtop_path, output / "system.prmtop")
    shutil.copy2(coordinate_path, output / "system.rst7")

    structure = pmd.load_file(str(prmtop_path), xyz=str(coordinate_path))
    topology_file, system = build_obc1_system(prmtop_path)
    positions = app.AmberInpcrdFile(str(coordinate_path)).positions
    positions = _minimize(topology_file.topology, system, positions)
    positions_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)

    (output / "system.xml").write_text(
        mm.XmlSerializer.serialize(system), encoding="utf-8"
    )
    with (output / "reference.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(
            topology_file.topology, positions, handle, keepIds=True
        )

    atomic_numbers = [int(atom.atomic_number) for atom in structure.atoms]
    formula = _formula(atomic_numbers)
    charge = float(sum(atom.charge for atom in structure.atoms))
    if formula != expected_formula:
        raise ValueError(f"formula mismatch: {formula} != {expected_formula}")
    if abs(charge) > 1e-5:
        raise ValueError(f"net charge is not zero: {charge}")

    system_spec = extract_system_spec(system)
    bonds = [[bond.atom1.idx, bond.atom2.idx] for bond in structure.bonds]
    coordinate_spec = _build_coordinate_spec(
        system_spec,
        positions_nm,
        bonds,
        carbon_count=carbon_count,
    )
    validation_spec = build_validation_spec(system, positions, seed=20260711)
    system_spec.update(
        {
            "atomic_numbers": atomic_numbers,
            "atom_names": [atom.name for atom in structure.atoms],
            "residue_names": [atom.residue.name for atom in structure.atoms],
            "bonds": bonds,
            "formula": formula,
            "net_charge_e": charge,
        }
    )
    _json_write(output / "system.json", system_spec)
    _json_write(output / "coordinates.json", coordinate_spec)
    _json_write(output / "validation.json", validation_spec)

    provenance = output / "provenance"
    provenance.mkdir()
    for relative, source in provenance_files.items():
        destination = provenance / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    files = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file())
    }
    manifest = {
        "schema_version": 1,
        "name": semantic_name,
        "target": molecule_name,
        "temperature_kelvin": 300.0,
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "formal_charge": 0,
        "model": model,
        "coordinate_measure": coordinate_spec["jacobian_measure"],
        "openmm_version": mm.__version__,
        "system_spec": "system.json",
        "coordinate_spec": "coordinates.json",
        "validation_spec": "validation.json",
        "files": files,
    }
    _json_write(output / "manifest.json", manifest)
    return Molecular_Bundle.load(output, verify=True).path


def _normalize_transcripts(work: Path, prefix: Path) -> None:
    for filename in (
        "antechamber.stdout",
        "parmchk2.stdout",
        "tleap.stdout",
        "leap.log",
        "sqm.in",
        "sqm.out",
    ):
        path = work / filename
        if path.is_file():
            text = normalize_amber_transcript(path.read_text(errors="replace"), prefix)
            text = re.sub(r"=\s+[0-9]+\.[0-9]+ seconds", "= <timing> seconds", text)
            text = re.sub(r"^log started:.*$", "log started: <normalized>", text, flags=re.MULTILINE)
            path.write_text(text, encoding="utf-8")


def build_bundle(molecule_index: int) -> Path:
    name = P.MOLECULES[molecule_index]
    carbon_count = P.CARBON_COUNTS[molecule_index]
    expected_formula = P.FORMULAS[molecule_index]
    expected_dimension = P.DIMENSIONS[molecule_index]
    geometry = build_alkane_geometry(carbon_count)
    if geometry.formula != expected_formula:
        raise AssertionError(f"geometry formula mismatch: {geometry.formula}")

    prefix = ambertools_prefix()
    amber_version = metadata.version("ambertools-unofficial")
    amber_hashes = amber_data_hashes(prefix)
    semantic_name = f"{name}_gaff2_am1bcc_obc1"
    with tempfile.TemporaryDirectory(prefix=f"jflows_md_diag_{name}_") as temporary:
        work = Path(temporary)
        (work / "input.mol2").write_text(render_mol2(geometry), encoding="utf-8")
        run(
            [
                str(prefix / "bin/antechamber"),
                "-i", "input.mol2", "-fi", "mol2",
                "-o", "gaff2.mol2", "-fo", "mol2",
                "-at", "gaff2", "-c", "bcc", "-nc", "0", "-m", "1",
                "-eq", "0", "-seq", "n", "-an", "n", "-s", "2", "-pf", "n",
            ],
            work,
            "antechamber.stdout",
        )
        shutil.copy2(work / "gaff2.mol2", work / "gaff2_raw.mol2")
        charge_projection = _project_mol2_charges(
            work / "gaff2.mol2", geometry
        )
        antechamber_validation = _validate_antechamber_output(
            work / "gaff2.mol2", geometry
        )
        run(
            [
                str(prefix / "bin/parmchk2"),
                "-i", "gaff2.mol2", "-f", "mol2",
                "-o", "gaff2.frcmod", "-s", "gaff2",
            ],
            work,
            "parmchk2.stdout",
        )
        (work / "leap.in").write_text(
            "source leaprc.gaff2\n"
            "set default PBRadii mbondi2\n"
            "loadamberparams gaff2.frcmod\n"
            "MOL = loadmol2 gaff2.mol2\n"
            "check MOL\n"
            "saveamberparm MOL molecule.prmtop molecule.rst7\n"
            "savepdb MOL molecule.pdb\n"
            "quit\n",
            encoding="utf-8",
        )
        run([str(prefix / "bin/tleap"), "-f", "leap.in"], work, "tleap.stdout")

        # Normalize only nonphysical timestamps and installation paths.
        prmtop = work / "molecule.prmtop"
        prmtop.write_text(
            re.sub(
                r"^(%VERSION\s+VERSION_STAMP\s*=\s*V0001\.000\s+DATE\s*=\s*).*$",
                r"\g<1>01/01/70  00:00:00",
                prmtop.read_text(encoding="utf-8"),
                count=1,
                flags=re.MULTILINE,
            ),
            encoding="utf-8",
        )
        _normalize_transcripts(work, prefix)
        logs = "\n".join(
            (work / filename).read_text(errors="replace")
            for filename in ("antechamber.stdout", "parmchk2.stdout", "tleap.stdout")
        )
        if "Exiting LEaP: Errors = 0" not in logs or "FATAL" in logs:
            raise RuntimeError("AmberTools parameterization did not pass its log gate")

        build_record = {
            "name": name,
            "canonical_smiles": geometry.canonical_smiles,
            "formula": expected_formula,
            "formal_charge": 0,
            "multiplicity": 1,
            "ambertools_version": amber_version,
            "amber_data_sha256": amber_hashes,
            "antechamber_output_validation": antechamber_validation,
            "charge_projection": charge_projection,
            "geometry_generator": "deterministic_heavy_first_ideal_tetrahedral_v1",
            "geometry_seed": P.GEOMETRY_SEED + P.SEED_MOLECULE_STRIDE * molecule_index,
            "input_mol2_sha256": geometry_sha256(geometry),
            "source_sha256": {
                "build_bundles.py": sha256_file(Path(__file__)),
                "geometry.py": sha256_file(ROOT / "geometry.py"),
                "parameters.py": sha256_file(ROOT / "parameters.py"),
            },
            "construction_toolchain": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "openmm": openmm.__version__,
                "parmed": parmed.__version__,
                "openmm_platform": "Reference",
                "jflows_md_git_head": _git_head(
                    Path(jflows_md.__file__).resolve().parents[1]
                ),
                "jflows_md_package_source_sha256": package_source_sha256(jflows_md),
                "jflows_md_core_builder_sha256": sha256_file(
                    Path(jflows_md.__file__).resolve().parent / "core/builder.py"
                ),
                "jflows_md_core_zmatrix_sha256": sha256_file(
                    Path(jflows_md.__file__).resolve().parent / "core/zmatrix.py"
                ),
            },
            "charge_equalization": (
                "graph-class mean plus uniform least-squares neutrality projection"
            ),
            "atom_order": "heavy carbon backbone, then hydrogens grouped by parent carbon",
            "pipeline": [
                "antechamber -at gaff2 -c bcc -nc 0 -m 1 -eq 0 -seq n",
                "project charges onto graph equivalence and exact neutrality",
                "parmchk2 -s gaff2",
                "tleap leaprc.gaff2 with PBRadii mbondi2",
            ],
        }
        (work / "build_record.json").write_text(
            json.dumps(build_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        model = {
            "force_field": "GAFF2",
            "charges": "AM1-BCC with recorded graph-symmetry/neutrality projection",
            "charge_equalization": charge_projection["method"],
            "implicit_solvent": "OBC1 (igb=2)",
            "radii": "mbondi2",
            "sasa": "ACE",
            "solute_dielectric": 1.0,
            "solvent_dielectric": 78.5,
            "salt_molar": 0.0,
            "nonbonded_method": "NoCutoff",
            "constraints": None,
            "ambertools_version": amber_version,
            "amber_data_sha256": amber_hashes,
        }
        provenance = {
            "input.mol2": work / "input.mol2",
            "gaff2.mol2": work / "gaff2.mol2",
            "gaff2_raw.mol2": work / "gaff2_raw.mol2",
            "gaff2.frcmod": work / "gaff2.frcmod",
            "leap.in": work / "leap.in",
            "leap.transcript": work / "leap.log",
            "antechamber.stdout": work / "antechamber.stdout",
            "parmchk2.stdout": work / "parmchk2.stdout",
            "tleap.stdout": work / "tleap.stdout",
            "sqm.in": work / "sqm.in",
            "sqm.transcript": work / "sqm.out",
            "build_record.json": work / "build_record.json",
            "source/build_bundles.py": Path(__file__),
            "source/geometry.py": ROOT / "geometry.py",
            "source/parameters.py": ROOT / "parameters.py",
        }
        missing = [relative for relative, path in provenance.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"required provenance files are missing: {missing}")
        candidate = _write_experimental_bundle(
            work / "candidate",
            semantic_name=semantic_name,
            molecule_name=name,
            carbon_count=carbon_count,
            prmtop_path=work / "molecule.prmtop",
            coordinate_path=work / "molecule.rst7",
            canonical_smiles=geometry.canonical_smiles,
            expected_formula=expected_formula,
            model=model,
            provenance_files=provenance,
        )
        bundle = Molecular_Bundle.load(candidate, verify=True)
        if bundle.dimension != expected_dimension:
            raise ValueError(
                f"dimension mismatch: {bundle.dimension} != {expected_dimension}"
            )
        if bundle.coordinates["chiral_torsion_index"] != -1 or bundle.coordinates["chirality_sign"] != 0:
            raise ValueError("an alkane candidate unexpectedly restricts chirality")
        if carbon_count == 1:
            if (bundle.coordinates["euclidean_dim"], bundle.coordinates["periodic_dim"]) != (7, 2):
                raise ValueError("methane candidate must use R^7 x T^2")
            hydrogen_charges = np.asarray(bundle.system["gb_charge_e"])[1:]
            if float(np.ptp(hydrogen_charges)) > 1e-10:
                raise ValueError(
                    f"methane OBC/nonbonded hydrogen charges differ: {hydrogen_charges}"
                )
        manifest_sha = sha256_file(candidate / "manifest.json")
        destination = CANDIDATE_ROOT / f"{semantic_name}__{manifest_sha[:12]}"
        if destination.exists():
            existing = Molecular_Bundle.load(destination, verify=True)
            if sha256_file(existing.path / "manifest.json") != manifest_sha:
                raise FileExistsError(f"content-address collision: {destination}")
        else:
            CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, destination)
    return Molecular_Bundle.load(destination, verify=True).path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--molecule",
        choices=P.MOLECULES,
        default="methane",
        help="build only this preregistered member (default: methane)",
    )
    args = parser.parse_args()
    index = P.MOLECULES.index(args.molecule)
    path = build_bundle(index)
    print(f"candidate built {path}")
    print(f"manifest sha256 {sha256_file(path / 'manifest.json')}")


if __name__ == "__main__":
    main()

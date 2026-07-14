#!/usr/bin/env python
"""Independent bundle, chart, symmetry, and OpenMM/JAX validation."""

from __future__ import annotations

import argparse
import hashlib
from itertools import permutations, product
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

import parameters as P


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(root: Path) -> dict:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=True,
    ).stdout
    return {
        "head": head,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty": bool(diff),
    }


def _runtime_provenance() -> dict:
    import equinox
    import jflows
    import jflows_md
    import openmm
    import parmed
    from jflows_md import package_source_sha256

    jflows_root = Path(jflows.__file__).resolve().parents[1]
    molecular_root = Path(jflows_md.__file__).resolve().parents[1]
    return {
        "validator_sha256": _sha256(Path(__file__)),
        "parameters_sha256": _sha256(ROOT / "parameters.py"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "jax": jax.__version__,
        "equinox": equinox.__version__,
        "openmm": openmm.__version__,
        "parmed": parmed.__version__,
        "jax_backend": jax.default_backend(),
        "jflows": _git_state(jflows_root),
        "jflows_md": {
            **_git_state(molecular_root),
            "package_source_sha256": package_source_sha256(jflows_md),
        },
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _signed_volume(positions: np.ndarray, atoms: list[int]) -> np.ndarray:
    center, first, second, third = atoms
    return np.einsum(
        "...i,...i->...",
        positions[..., first, :] - positions[..., center, :],
        np.cross(
            positions[..., second, :] - positions[..., center, :],
            positions[..., third, :] - positions[..., center, :],
        ),
    )


def _interaction_maps(system: dict) -> dict[str, dict]:
    def pair_key(pair):
        return tuple(sorted(map(int, pair)))

    bonds = {
        pair_key(pair): (float(length), float(force))
        for pair, length, force in zip(
            system["bond_idx"],
            system["bond_length_nm"],
            system["bond_k_kj_mol_nm2"],
            strict=True,
        )
    }
    angles = {}
    for triple, theta, force in zip(
        system["angle_idx"],
        system["angle_theta_rad"],
        system["angle_k_kj_mol_rad2"],
        strict=True,
    ):
        first, center, third = map(int, triple)
        angles[(center, *sorted((first, third)))] = (float(theta), float(force))
    torsions = {}
    for quad, periodicity, phase, force in zip(
        system["torsion_idx"],
        system["torsion_periodicity"],
        system["torsion_phase_rad"],
        system["torsion_k_kj_mol"],
        strict=True,
    ):
        forward = tuple(map(int, quad))
        key = min(forward, tuple(reversed(forward)))
        torsions.setdefault(key, []).append(
            (int(periodicity), float(phase), float(force))
        )
    for values in torsions.values():
        values.sort()
    pairs = {
        pair_key(pair): (float(charge), float(sigma), float(epsilon))
        for pair, charge, sigma, epsilon in zip(
            system["pair_idx"],
            system["pair_chargeprod_e2"],
            system["pair_sigma_nm"],
            system["pair_epsilon_kj_mol"],
            strict=True,
        )
    }
    exceptions = {
        pair_key(pair): (float(charge), float(sigma), float(epsilon))
        for pair, charge, sigma, epsilon in zip(
            system["exception_idx"],
            system["exception_chargeprod_e2"],
            system["exception_sigma_nm"],
            system["exception_epsilon_kj_mol"],
            strict=True,
        )
    }
    return {
        "bond": bonds,
        "angle": angles,
        "torsion": torsions,
        "pair": pairs,
        "exception": exceptions,
    }


def _map_interactions(interactions: dict[str, dict], permutation: tuple[int, ...]) -> dict:
    def pair(pair):
        return tuple(sorted((permutation[pair[0]], permutation[pair[1]])))

    mapped = {
        "bond": {pair(key): value for key, value in interactions["bond"].items()},
        "pair": {pair(key): value for key, value in interactions["pair"].items()},
        "exception": {
            pair(key): value for key, value in interactions["exception"].items()
        },
    }
    mapped["angle"] = {
        (
            permutation[key[0]],
            *sorted((permutation[key[1]], permutation[key[2]])),
        ): value
        for key, value in interactions["angle"].items()
    }
    mapped_torsions = {}
    for key, value in interactions["torsion"].items():
        transformed = tuple(permutation[index] for index in key)
        transformed = min(transformed, tuple(reversed(transformed)))
        mapped_torsions[transformed] = value
    mapped["torsion"] = mapped_torsions
    return mapped


def _openmm_reference(bundle_path: Path, frames_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    import openmm as mm
    from openmm import unit

    system = mm.XmlSerializer.deserialize((bundle_path / "system.xml").read_text())
    group_for_class = {
        "HarmonicBondForce": 0,
        "HarmonicAngleForce": 1,
        "PeriodicTorsionForce": 2,
        "NonbondedForce": 3,
        "CustomGBForce": 4,
    }
    for force in system.getForces():
        force.setForceGroup(group_for_class[force.__class__.__name__])
    names = {0: "bond", 1: "angle", 2: "torsion", 3: "nonbonded", 4: "gb"}
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    energies, forces = [], []
    terms = {name: [] for name in names.values()}
    for frame in frames_nm:
        context.setPositions(frame * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )
        forces.append(
            np.asarray(
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
            )
        )
        for group, name in names.items():
            energy = context.getState(getEnergy=True, groups=1 << group).getPotentialEnergy()
            terms[name].append(float(energy.value_in_unit(unit.kilojoule_per_mole)))
    del context, integrator
    return np.asarray(energies), np.asarray(forces), {
        name: np.asarray(values) for name, values in terms.items()
    }


def _methane_permutations(n_atoms: int) -> tuple[tuple[int, ...], ...]:
    if n_atoms != 5:
        raise ValueError("the all-permutation gate is defined for methane only")
    return tuple((0, *hydrogens) for hydrogens in permutations((1, 2, 3, 4)))


def _graph_automorphisms(system: dict) -> tuple[tuple[int, ...], ...]:
    """Enumerate exact atom-graph automorphisms for the small alkane series."""

    n_atoms = len(system["atomic_numbers"])
    bonds = {tuple(sorted(map(int, pair))) for pair in system["bonds"]}
    adjacency = [set() for _ in range(n_atoms)]
    for first, second in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    labels = [
        (int(system["atomic_numbers"][index]), len(adjacency[index]))
        for index in range(n_atoms)
    ]
    while True:
        keys = [
            (labels[index], tuple(sorted(labels[neighbor] for neighbor in adjacency[index])))
            for index in range(n_atoms)
        ]
        unique = {key: value for value, key in enumerate(sorted(set(keys)))}
        updated = [unique[key] for key in keys]
        if updated == labels:
            break
        labels = updated
    groups = [
        tuple(index for index, value in enumerate(labels) if value == label)
        for label in sorted(set(labels))
    ]
    result = []
    for choices in product(*(permutations(group) for group in groups)):
        mapping = list(range(n_atoms))
        for group, choice in zip(groups, choices, strict=True):
            for old, new in zip(group, choice, strict=True):
                mapping[old] = new
        mapped_bonds = {
            tuple(sorted((mapping[first], mapping[second])))
            for first, second in bonds
        }
        if mapped_bonds == bonds:
            result.append(tuple(mapping))
    return tuple(result)


def validate_x64(bundle_path: Path) -> dict:
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import parmed as pmd

    from jflows_md import Molecular_Bundle, Molecular_Potential
    from jflows_md.core.validation import compare_stored_openmm
    from jflows_md.system import sha256_file

    bundle = Molecular_Bundle.load(bundle_path, verify=True)
    if bundle.manifest["target"] != "methane":
        raise ValueError("the first bundle gate is methane-only")
    if (bundle.n_atoms, bundle.dimension) != (5, 9):
        raise ValueError("methane must contain five atoms and have dimension nine")
    if (bundle.coordinates["euclidean_dim"], bundle.coordinates["periodic_dim"]) != (7, 2):
        raise ValueError("methane chart must be R^7 x T^2")
    if bundle.coordinates["chiral_torsion_index"] != -1 or bundle.coordinates["chirality_sign"] != 0:
        raise ValueError("methane must use a full-support, nonchiral chart")

    potential = Molecular_Potential.from_bundle(bundle_path, verify=True)
    target = potential.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = potential.source()
    stored = compare_stored_openmm(potential, bundle)

    reference = np.asarray(bundle.validation["frames_nm"][0])
    atom_permutations = _methane_permutations(bundle.n_atoms)
    permutation_frames = np.asarray([reference[list(order)] for order in atom_permutations])
    center = reference[:1]
    mirrored = (reference - center) * np.asarray((-1.0, 1.0, 1.0)) + center
    q_reference = potential.reference_internal()
    torsion_frames = []
    for torsion in range(bundle.coordinates["periodic_dim"]):
        for shift in (-0.6, 0.6):
            q = q_reference.at[bundle.coordinates["euclidean_dim"] + torsion].add(shift)
            torsion_frames.append(np.asarray(potential.cartesian(q[None])[0]))
    source_probe = source.samples(jax.random.key(20260712), 128)
    source_energy = np.asarray(potential.physical_energy(source_probe))
    finite = np.flatnonzero(np.isfinite(source_energy))
    if finite.size < 4:
        raise ValueError("not enough finite high-energy source probes")
    selected = finite[np.argsort(source_energy[finite])[-4:]]
    high_frames = np.asarray(potential.cartesian(source_probe[selected]))
    frames = np.concatenate(
        (
            reference[None],
            permutation_frames,
            mirrored[None],
            np.asarray(torsion_frames),
            high_frames,
        ),
        axis=0,
    )

    openmm_energy, openmm_force, openmm_terms = _openmm_reference(bundle_path, frames)
    jax_frames = jnp.asarray(frames)
    jax_terms = potential.forcefield.energy_terms(jax_frames)
    jax_energy = np.asarray(jax_terms["total"])
    jax_force = np.asarray(
        -jax.vmap(jax.grad(lambda x: potential.forcefield(x[None])[0]))(jax_frames)
    )
    force_delta = jax_force - openmm_force
    energy_error = float(np.max(np.abs(jax_energy - openmm_energy)))
    force_rmse = float(np.sqrt(np.mean(force_delta**2)))
    force_max = float(np.max(np.abs(force_delta)))
    term_errors = {
        name: float(np.max(np.abs(np.asarray(jax_terms[name]) - expected)))
        for name, expected in openmm_terms.items()
    }
    if energy_error > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(f"OpenMM/JAX energy error {energy_error}")
    if force_rmse > P.FORCE_RMSE_KJ_MOL_NM_MAX or force_max > P.FORCE_COMPONENT_KJ_MOL_NM_MAX:
        raise AssertionError(f"OpenMM/JAX force error rmse={force_rmse} max={force_max}")
    if max(term_errors.values()) > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(f"OpenMM/JAX term errors {term_errors}")

    # Parameter/topology invariance under every labeled-H permutation.
    interactions = _interaction_maps(bundle.system)
    structure = pmd.load_file(
        str(bundle_path / "system.prmtop"), xyz=str(bundle_path / "system.rst7")
    )
    hydrogen_types = {structure.atoms[index].type for index in range(1, 5)}
    hydrogen_charges = np.asarray(bundle.system["gb_charge_e"])[1:5]
    hydrogen_radii = np.asarray(bundle.system["gb_offset_radius_nm"])[1:5]
    hydrogen_scaled = np.asarray(bundle.system["gb_scaled_offset_radius_nm"])[1:5]
    if len(hydrogen_types) != 1:
        raise AssertionError(f"methane H atom types differ: {hydrogen_types}")
    for values, label in (
        (hydrogen_charges, "charges"),
        (hydrogen_radii, "OBC radii"),
        (hydrogen_scaled, "OBC scaled radii"),
    ):
        if float(np.ptp(values)) > 1e-12:
            raise AssertionError(f"methane H {label} are not equivalent: {values}")
    for order in atom_permutations:
        if _map_interactions(interactions, order) != interactions:
            raise AssertionError(f"force-field graph is not invariant under {order}")

    q_perm, inverse_logdet = potential.coordinates.to_internal(jnp.asarray(permutation_frames))
    reconstructed, logdet = potential.coordinates.to_cartesian(q_perm)
    roundtrip_q = potential.coordinates.to_internal(reconstructed)[0]
    roundtrip = float(
        jnp.max(jnp.abs(potential.domain.displacement(roundtrip_q, q_perm)))
    )
    if roundtrip > 1e-9 or float(jnp.max(jnp.abs(logdet + inverse_logdet))) > 1e-9:
        raise AssertionError("methane internal-coordinate round trip failed")
    perm_energy = np.asarray(potential.physical_energy(q_perm))
    perm_target = np.asarray(target(q_perm))
    perm_source = np.asarray(source(q_perm))
    perm_logdet = np.asarray(logdet)
    permutation_errors = {
        "physical_energy_kj_mol": float(np.ptp(perm_energy)),
        "regularized_reduced_target": float(np.ptp(perm_target)),
        "source_reduced_energy": float(np.ptp(perm_source)),
        "coordinate_logdet": float(np.ptp(perm_logdet)),
    }
    if permutation_errors["physical_energy_kj_mol"] > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(f"methane permutation energy failure: {permutation_errors}")
    if max(
        permutation_errors[key]
        for key in ("regularized_reduced_target", "coordinate_logdet")
    ) > P.REDUCED_PERMUTATION_ERROR_MAX:
        raise AssertionError(f"methane internal permutation failure: {permutation_errors}")

    # Away from the symmetric reference, scalar equality is not the correct
    # pulled-back-density rule. For q' = T_p(q), configurational volume gives
    # log J(q) = log J(q') + log|det dT_p/dq| and hence
    # U(q') = U(q) + log|det dT_p/dq|.
    covariance_key = jax.random.key(20260716)
    covariance_q = potential.domain.wrap(
        q_reference[None]
        + 0.15 * jax.random.normal(covariance_key, (8, bundle.dimension))
    )

    @jax.jit
    def covariance_batch(q_batch, order):
        def transform(value):
            cartesian = potential.cartesian(value[None])[0]
            return potential.coordinates.to_internal(
                cartesian[order][None]
            )[0][0]

        transformed = jax.vmap(transform)(q_batch)
        jacobian = jax.vmap(jax.jacfwd(transform))(q_batch)
        map_logdet = jnp.linalg.slogdet(jacobian)[1]
        original_logdet = potential.coordinates.to_cartesian(q_batch)[1]
        transformed_logdet = potential.coordinates.to_cartesian(transformed)[1]
        energy_delta = potential.physical_energy(transformed) - potential.physical_energy(q_batch)
        jacobian_residual = original_logdet - transformed_logdet - map_logdet
        target_residual = target(transformed) - target(q_batch) - map_logdet
        return energy_delta, jacobian_residual, target_residual

    covariance_error = {
        "physical_energy_kj_mol": 0.0,
        "jacobian_log_volume": 0.0,
        "reduced_target": 0.0,
    }
    for order in atom_permutations:
        energy_delta, jacobian_residual, target_residual = covariance_batch(
            covariance_q, jnp.asarray(order)
        )
        covariance_error["physical_energy_kj_mol"] = max(
            covariance_error["physical_energy_kj_mol"],
            float(jnp.max(jnp.abs(energy_delta))),
        )
        covariance_error["jacobian_log_volume"] = max(
            covariance_error["jacobian_log_volume"],
            float(jnp.max(jnp.abs(jacobian_residual))),
        )
        covariance_error["reduced_target"] = max(
            covariance_error["reduced_target"],
            float(jnp.max(jnp.abs(target_residual))),
        )
    if covariance_error["physical_energy_kj_mol"] > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(f"methane off-reference energy covariance failed: {covariance_error}")
    if max(
        covariance_error["jacobian_log_volume"],
        covariance_error["reduced_target"],
    ) > P.REDUCED_PERMUTATION_ERROR_MAX:
        raise AssertionError(f"methane off-reference chart covariance failed: {covariance_error}")

    reference_index = 1  # first permutation is identity
    reference_force = jax_force[reference_index]
    force_covariance = 0.0
    for frame_index, order in enumerate(atom_permutations, start=1):
        force_covariance = max(
            force_covariance,
            float(np.max(np.abs(jax_force[frame_index] - reference_force[list(order)]))),
        )
    if force_covariance > P.FORCE_COMPONENT_KJ_MOL_NM_MAX:
        raise AssertionError(f"methane force covariance failure: {force_covariance}")

    symmetry_fields = {
        "bond_offset_range": float(np.ptp(bundle.coordinates["bond_log_offset"])),
        "bond_scale_range": float(np.ptp(bundle.coordinates["bond_log_scale"])),
        "angle_offset_range": float(np.ptp(bundle.coordinates["angle_logit_offset"])),
        "angle_scale_range": float(np.ptp(bundle.coordinates["angle_logit_scale"])),
    }
    if max(symmetry_fields.values()) > 1e-12:
        raise AssertionError(f"methane chart symmetry fields differ: {symmetry_fields}")

    # A fixed BAT chart's factorized source is not S4 invariant away from the
    # perfectly symmetric reference. Quantify this declared proposal confound.
    asymmetry_q = source.samples(jax.random.key(20260715), 256)
    asymmetry_x = potential.cartesian(asymmetry_q)
    permuted_source = []
    for order in atom_permutations:
        q_order = potential.coordinates.to_internal(
            asymmetry_x[:, jnp.asarray(order), :]
        )[0]
        permuted_source.append(source(q_order))
    permuted_source = np.asarray(jnp.stack(permuted_source, axis=0))
    source_ranges = np.ptp(permuted_source, axis=0)
    source_asymmetry = {
        "samples": int(source_ranges.size),
        "mean_permutation_range": float(np.mean(source_ranges)),
        "maximum_permutation_range": float(np.max(source_ranges)),
        "fraction_range_above_1e-6": float(np.mean(source_ranges > 1e-6)),
    }

    parity_q = source.samples(jax.random.key(20260713), 20000)
    parity_x = np.asarray(potential.cartesian(parity_q))
    atoms = bundle.coordinates["diagnostic_chirality_atoms"]
    volume = _signed_volume(parity_x, atoms)
    positive_fraction = float(np.mean(volume > 0))
    if not 0.40 <= positive_fraction <= 0.60 or np.any(volume == 0):
        raise AssertionError(f"methane source does not cover both parity sectors: {positive_fraction}")
    if not bool(np.asarray(potential.support_mask(jnp.asarray(parity_x))).all()):
        raise AssertionError("nonchiral methane support unexpectedly rejects a parity sector")

    return {
        "mode": "x64",
        "bundle": str(bundle_path.resolve()),
        "manifest_sha256": sha256_file(bundle_path / "manifest.json"),
        "formula": bundle.system["formula"],
        "atoms": bundle.n_atoms,
        "dimension": bundle.dimension,
        "domain": {"euclidean": 7, "periodic": 2},
        "stored_openmm": {
            "energy_error_kj_mol": stored.maximum_energy_kj_mol,
            "force_rmse_kj_mol_nm": stored.force_rmse_kj_mol_nm,
            "force_max_kj_mol_nm": stored.maximum_force_component_kj_mol_nm,
        },
        "diversified_frames": int(frames.shape[0]),
        "energy_error_kj_mol": energy_error,
        "force_rmse_kj_mol_nm": force_rmse,
        "force_max_kj_mol_nm": force_max,
        "term_energy_errors_kj_mol": term_errors,
        "hydrogen_charge_range_e": float(np.ptp(hydrogen_charges)),
        "permutation_count": len(atom_permutations),
        "permutation_errors": permutation_errors,
        "off_reference_permutation_covariance_errors": covariance_error,
        "source_permutation_asymmetry": source_asymmetry,
        "force_covariance_max_kj_mol_nm": force_covariance,
        "chart_symmetry_ranges": symmetry_fields,
        "roundtrip_max": roundtrip,
        "positive_parity_fraction": positive_fraction,
    }


def validate_generic_x64(bundle_path: Path) -> dict:
    """Validate a non-methane alkane against independent OpenMM references.

    Methane retains the stronger exhaustive S4 permutation audit above.  For
    larger alkanes this gate checks the transferable requirements: bundle
    integrity, chart dimensions and round trip, stored references, and
    diversified energy/force/term parity with OpenMM's Reference platform.
    """

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jflows_md import Molecular_Bundle, Molecular_Potential
    from jflows_md.core.validation import compare_stored_openmm
    from jflows_md.system import sha256_file

    bundle = Molecular_Bundle.load(bundle_path, verify=True)
    if bundle.manifest["target"] == "methane":
        raise ValueError("generic alkane validation must not replace the methane gate")
    expected_dimension = 3 * bundle.n_atoms - 6
    expected_euclidean = 2 * bundle.n_atoms - 3
    expected_periodic = bundle.n_atoms - 3
    if bundle.dimension != expected_dimension:
        raise ValueError(
            f"dimension mismatch: {bundle.dimension} != {expected_dimension}"
        )
    if (
        bundle.coordinates["euclidean_dim"],
        bundle.coordinates["periodic_dim"],
    ) != (expected_euclidean, expected_periodic):
        raise ValueError(
            "alkane chart has the wrong mixed-domain split: "
            f"{bundle.coordinates['euclidean_dim']},"
            f"{bundle.coordinates['periodic_dim']} != "
            f"{expected_euclidean},{expected_periodic}"
        )
    if (
        bundle.coordinates["chiral_torsion_index"] != -1
        or bundle.coordinates["chirality_sign"] != 0
    ):
        raise ValueError("achiral alkane bundle unexpectedly restricts chirality")

    potential = Molecular_Potential.from_bundle(bundle_path, verify=True)
    target = potential.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = potential.source()
    stored = compare_stored_openmm(potential, bundle)

    q_reference = potential.reference_internal()
    reference = np.asarray(potential.cartesian(q_reference[None])[0])
    atom_permutations = _graph_automorphisms(bundle.system)
    permutation_orders = [
        np.argsort(np.asarray(mapping)) for mapping in atom_permutations
    ]
    permutation_frames = np.asarray(
        [reference[order] for order in permutation_orders]
    )
    torsion_frames = []
    for torsion in range(bundle.coordinates["periodic_dim"]):
        for shift in (-0.6, 0.6):
            q = q_reference.at[expected_euclidean + torsion].add(shift)
            torsion_frames.append(np.asarray(potential.cartesian(q[None])[0]))
    source_probe = source.samples(jax.random.key(20260717), 256)
    source_energy = np.asarray(potential.physical_energy(source_probe))
    finite = np.flatnonzero(np.isfinite(source_energy))
    if finite.size < 8:
        raise ValueError("not enough finite diversified source probes")
    selected = finite[np.argsort(source_energy[finite])[-8:]]
    high_frames = np.asarray(potential.cartesian(source_probe[selected]))
    frames = np.concatenate(
        (permutation_frames, np.asarray(torsion_frames), high_frames), axis=0
    )

    openmm_energy, openmm_force, openmm_terms = _openmm_reference(bundle_path, frames)
    jax_frames = jnp.asarray(frames)
    jax_terms = potential.forcefield.energy_terms(jax_frames)
    jax_energy = np.asarray(jax_terms["total"])
    jax_force = np.asarray(
        -jax.vmap(jax.grad(lambda x: potential.forcefield(x[None])[0]))(jax_frames)
    )
    force_delta = jax_force - openmm_force
    energy_error = float(np.max(np.abs(jax_energy - openmm_energy)))
    force_rmse = float(np.sqrt(np.mean(force_delta**2)))
    force_max = float(np.max(np.abs(force_delta)))
    term_errors = {
        name: float(np.max(np.abs(np.asarray(jax_terms[name]) - expected)))
        for name, expected in openmm_terms.items()
    }
    if energy_error > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(f"OpenMM/JAX energy error {energy_error}")
    if (
        force_rmse > P.FORCE_RMSE_KJ_MOL_NM_MAX
        or force_max > P.FORCE_COMPONENT_KJ_MOL_NM_MAX
    ):
        raise AssertionError(
            f"OpenMM/JAX force error rmse={force_rmse} max={force_max}"
        )
    if max(term_errors.values()) > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(f"OpenMM/JAX term errors {term_errors}")

    permutation_energy_error = float(np.ptp(jax_energy[: len(atom_permutations)]))
    if permutation_energy_error > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(
            f"alkane automorphism energy invariance failed: {permutation_energy_error}"
        )
    identity_index = atom_permutations.index(tuple(range(bundle.n_atoms)))
    identity_force = jax_force[identity_index]
    force_covariance = 0.0
    for frame_index, order in enumerate(permutation_orders):
        force_covariance = max(
            force_covariance,
            float(np.max(np.abs(jax_force[frame_index] - identity_force[order]))),
        )
    if force_covariance > P.FORCE_COMPONENT_KJ_MOL_NM_MAX:
        raise AssertionError(
            f"alkane automorphism force covariance failed: {force_covariance}"
        )

    covariance_q = potential.domain.wrap(
        q_reference[None]
        + 0.12 * jax.random.normal(
            jax.random.key(20260719), (2, bundle.dimension)
        )
    )

    @jax.jit
    def covariance_batch(q_batch, order):
        def transform(value):
            cartesian = potential.cartesian(value[None])[0]
            return potential.coordinates.to_internal(
                cartesian[order][None]
            )[0][0]

        transformed = jax.vmap(transform)(q_batch)
        jacobian = jax.vmap(jax.jacfwd(transform))(q_batch)
        map_logdet = jnp.linalg.slogdet(jacobian)[1]
        original_logdet = potential.coordinates.to_cartesian(q_batch)[1]
        transformed_logdet = potential.coordinates.to_cartesian(transformed)[1]
        energy_delta = (
            potential.physical_energy(transformed)
            - potential.physical_energy(q_batch)
        )
        jacobian_residual = original_logdet - transformed_logdet - map_logdet
        target_residual = target(transformed) - target(q_batch) - map_logdet
        return energy_delta, jacobian_residual, target_residual

    covariance_error = {
        "physical_energy_kj_mol": 0.0,
        "jacobian_log_volume": 0.0,
        "reduced_target": 0.0,
    }
    for order in permutation_orders:
        energy_delta, jacobian_residual, target_residual = covariance_batch(
            covariance_q, jnp.asarray(order)
        )
        covariance_error["physical_energy_kj_mol"] = max(
            covariance_error["physical_energy_kj_mol"],
            float(jnp.max(jnp.abs(energy_delta))),
        )
        covariance_error["jacobian_log_volume"] = max(
            covariance_error["jacobian_log_volume"],
            float(jnp.max(jnp.abs(jacobian_residual))),
        )
        covariance_error["reduced_target"] = max(
            covariance_error["reduced_target"],
            float(jnp.max(jnp.abs(target_residual))),
        )
    if covariance_error["physical_energy_kj_mol"] > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(
            f"alkane off-reference energy covariance failed: {covariance_error}"
        )
    if max(
        covariance_error["jacobian_log_volume"],
        covariance_error["reduced_target"],
    ) > P.REDUCED_PERMUTATION_ERROR_MAX:
        raise AssertionError(
            f"alkane off-reference chart covariance failed: {covariance_error}"
        )

    q_probe = potential.domain.wrap(
        q_reference[None]
        + 0.15 * jax.random.normal(
            jax.random.key(20260718), (64, bundle.dimension)
        )
    )
    cartesian, logdet = potential.coordinates.to_cartesian(q_probe)
    roundtrip_q, inverse_logdet = potential.coordinates.to_internal(cartesian)
    roundtrip = float(
        jnp.max(jnp.abs(potential.domain.displacement(roundtrip_q, q_probe)))
    )
    jacobian_roundtrip = float(jnp.max(jnp.abs(logdet + inverse_logdet)))
    values, gradient = jax.jit(lambda x: (target(x), target.grad(x)))(source_probe)
    finite_target = bool(jnp.isfinite(values).all() & jnp.isfinite(gradient).all())
    if roundtrip > 1e-9 or jacobian_roundtrip > 1e-9:
        raise AssertionError(
            f"internal-coordinate round trip failed: q={roundtrip}, "
            f"logdet={jacobian_roundtrip}"
        )
    if not finite_target:
        raise AssertionError("x64 source target/gradient gate failed")

    return {
        "mode": "x64",
        "bundle": str(bundle_path.resolve()),
        "manifest_sha256": sha256_file(bundle_path / "manifest.json"),
        "target": bundle.manifest["target"],
        "formula": bundle.system["formula"],
        "atoms": bundle.n_atoms,
        "dimension": bundle.dimension,
        "domain": {
            "euclidean": expected_euclidean,
            "periodic": expected_periodic,
        },
        "stored_openmm": {
            "energy_error_kj_mol": stored.maximum_energy_kj_mol,
            "force_rmse_kj_mol_nm": stored.force_rmse_kj_mol_nm,
            "force_max_kj_mol_nm": stored.maximum_force_component_kj_mol_nm,
        },
        "diversified_frames": int(frames.shape[0]),
        "energy_error_kj_mol": energy_error,
        "force_rmse_kj_mol_nm": force_rmse,
        "force_max_kj_mol_nm": force_max,
        "term_energy_errors_kj_mol": term_errors,
        "graph_automorphisms": len(atom_permutations),
        "automorphism_energy_range_kj_mol": permutation_energy_error,
        "automorphism_force_covariance_max_kj_mol_nm": force_covariance,
        "off_reference_automorphism_covariance_errors": covariance_error,
        "roundtrip_max": roundtrip,
        "jacobian_roundtrip_max": jacobian_roundtrip,
        "finite_target_and_gradient": finite_target,
    }


def validate_float32(bundle_path: Path) -> dict:
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    from jflows_md import Molecular_Bundle, Molecular_Potential
    from jflows_md.system import sha256_file

    bundle = Molecular_Bundle.load(bundle_path, verify=True)
    potential = Molecular_Potential.from_bundle(bundle_path, verify=True)
    target = potential.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = potential.source()
    q = source.samples(jax.random.key(20260714), 4096)
    values, gradient, physical = jax.jit(
        lambda x: (target(x), target.grad(x), potential.physical_energy(x))
    )(q)
    jax.block_until_ready((values, gradient, physical))
    finite = bool(
        jnp.isfinite(values).all()
        & jnp.isfinite(gradient).all()
        & jnp.isfinite(physical).all()
    )
    if not finite:
        raise AssertionError("float32 molecular source energy/gradient gate failed")
    if q.dtype != jnp.float32 or values.dtype != jnp.float32 or gradient.dtype != jnp.float32:
        raise AssertionError(f"float32 gate used unexpected dtypes: {q.dtype}, {values.dtype}, {gradient.dtype}")
    return {
        "mode": "float32",
        "bundle": str(bundle_path.resolve()),
        "manifest_sha256": sha256_file(bundle_path / "manifest.json"),
        "target": bundle.manifest["target"],
        "samples": int(q.shape[0]),
        "sample_dtype": str(q.dtype),
        "target_dtype": str(values.dtype),
        "gradient_dtype": str(gradient.dtype),
        "finite": finite,
        "target_min": float(jnp.min(values)),
        "target_max": float(jnp.max(values)),
        "gradient_abs_max": float(jnp.max(jnp.abs(gradient))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mode", choices=("x64", "float32"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle_path = args.bundle.expanduser().resolve()
    if args.mode == "x64":
        from jflows_md import Molecular_Bundle

        bundle = Molecular_Bundle.load(bundle_path, verify=True)
        result = (
            validate_x64(bundle_path)
            if bundle.manifest["target"] == "methane"
            else validate_generic_x64(bundle_path)
        )
    else:
        result = validate_float32(bundle_path)
    result["command"] = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    result["runtime"] = _runtime_provenance()
    _json_write(args.output, result)
    print(f"PASS {result.get('target', 'molecular')} bundle {args.mode}: {args.output}")


if __name__ == "__main__":
    main()

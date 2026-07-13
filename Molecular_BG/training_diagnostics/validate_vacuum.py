#!/usr/bin/env python
"""Validate the experiment-local CH4 vacuum potential against OpenMM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import openmm as mm
from openmm import unit

import parameters as P
from vacuum import Vacuum_Molecular_Potential


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def openmm_vacuum(bundle: Path, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    system = mm.XmlSerializer.deserialize((bundle / "system.xml").read_text())
    for index in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(index), mm.CustomGBForce):
            system.removeForce(index)
    if any(isinstance(force, mm.CustomGBForce) for force in system.getForces()):
        raise AssertionError("CustomGBForce remains in the OpenMM vacuum system")
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    energies, forces = [], []
    for frame in frames:
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
    del context, integrator
    return np.asarray(energies), np.asarray(forces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    jax.config.update("jax_enable_x64", True)
    vacuum = Vacuum_Molecular_Potential.from_bundle(bundle)
    target = vacuum.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = vacuum.source()
    validation = json.loads((bundle / "validation.json").read_text())
    stored = np.asarray(validation["frames_nm"], dtype=np.float64)
    probes = source.samples(jax.random.key(20260717), 32)
    probe_frames = np.asarray(vacuum.cartesian(probes))
    frames = np.concatenate((stored, probe_frames), axis=0)
    expected_energy, expected_force = openmm_vacuum(bundle, frames)
    jax_frames = jnp.asarray(frames)

    def cartesian_energy(x):
        terms = vacuum.base.forcefield.energy_terms(x)
        return terms["bond"] + terms["angle"] + terms["torsion"] + terms["nonbonded"]

    actual_energy = np.asarray(cartesian_energy(jax_frames))
    actual_force = np.asarray(
        -jax.vmap(jax.grad(lambda x: cartesian_energy(x[None])[0]))(jax_frames)
    )
    delta = actual_force - expected_force
    energy_error = float(np.max(np.abs(actual_energy - expected_energy)))
    force_rmse = float(np.sqrt(np.mean(delta * delta)))
    force_max = float(np.max(np.abs(delta)))
    if energy_error > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise AssertionError(f"vacuum energy mismatch: {energy_error}")
    if force_rmse > P.FORCE_RMSE_KJ_MOL_NM_MAX:
        raise AssertionError(f"vacuum force RMSE mismatch: {force_rmse}")
    if force_max > P.FORCE_COMPONENT_KJ_MOL_NM_MAX:
        raise AssertionError(f"vacuum force component mismatch: {force_max}")

    solvent = np.asarray(vacuum.solvent_energy(probes))
    jax.config.update("jax_enable_x64", False)
    vacuum32 = Vacuum_Molecular_Potential.from_bundle(bundle)
    target32 = vacuum32.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source32 = vacuum32.source()
    q = source32.samples(jax.random.key(20260718), 4096)
    value, gradient = jax.jit(lambda x: (target32(x), target32.grad(x)))(q)
    jax.block_until_ready((value, gradient))
    if not bool(jnp.isfinite(value).all() & jnp.isfinite(gradient).all()):
        raise AssertionError("vacuum c50 target is not finite on the source probe")
    result = {
        "schema_version": 1,
        "complete": True,
        "model": "GAFF2/AM1-BCC vacuum (OBC1 and ACE removed)",
        "bundle": str(bundle),
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "frames": int(frames.shape[0]),
        "energy_error_kj_mol": energy_error,
        "force_rmse_kj_mol_nm": force_rmse,
        "force_max_kj_mol_nm": force_max,
        "removed_solvent_energy": {
            "minimum_kj_mol": float(np.min(solvent)),
            "maximum_kj_mol": float(np.max(solvent)),
            "mean_kj_mol": float(np.mean(solvent)),
        },
        "float64_parity": True,
        "float32_training_probe": {
            "samples": int(q.shape[0]),
            "sample_dtype": str(q.dtype),
            "target_dtype": str(value.dtype),
            "gradient_dtype": str(gradient.dtype),
            "gradient_abs_max": float(jnp.max(jnp.abs(gradient))),
        },
        "runtime": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "driver_sha256": sha256(Path(__file__)),
            "vacuum_source_sha256": sha256(Path(__file__).with_name("vacuum.py")),
            "parameters_sha256": sha256(Path(__file__).with_name("parameters.py")),
        },
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"PASS vacuum OpenMM/JAX validation: {output}")


if __name__ == "__main__":
    main()

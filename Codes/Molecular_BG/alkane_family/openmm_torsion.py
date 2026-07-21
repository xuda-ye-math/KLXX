#!/usr/bin/env python
"""Mode-mixed 0.25 fs OpenMM references for the torsional alkanes."""

import json
import time

import numpy as np

from jflows_md.openmm import OpenMM_Potential, parallel_tempering

from openmm_reference import (
    MOLECULES,
    ROOT,
    SEEDS,
    TEMPERATURES_KELVIN,
    bundle,
    observables,
    statistics,
)


OUTPUT = ROOT / "openmm_reference"
SUMMARY = OUTPUT / "torsion_reference.json"
LOG = OUTPUT / "torsion.log"

DIMENSIONS = (36, 45, 54)
ROUNDS = 2000
EQUILIBRATION_ROUNDS = 400
STEPS_PER_ROUND = 800
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
PLATFORM = "CUDA"


def log(message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as stream:
        stream.write(line + "\n")


def run_molecule(dimension, molecule, folder):
    potential = OpenMM_Potential.from_bundle(bundle(folder))
    carbon_indices = np.flatnonzero(
        np.asarray(potential.bundle.system["atomic_numbers"]) == 6
    )
    positions = []
    energies = []
    seed_records = []
    for seed in SEEDS:
        precondition = np.load(OUTPUT / f"{folder}_seed{seed}.npz")["positions_nm"]
        indices = np.linspace(0, len(precondition) - 1, len(TEMPERATURES_KELVIN))
        initial = precondition[np.rint(indices).astype(int)]
        start = time.time()
        log(f"START {molecule} ({dimension}d) seed={seed}")
        trajectory, energy_history, acceptance = parallel_tempering(
            potential,
            TEMPERATURES_KELVIN,
            positions_nm=initial,
            rounds=ROUNDS,
            steps_per_round=STEPS_PER_ROUND,
            timestep_fs=TIMESTEP_FS,
            friction_per_ps=FRICTION_PER_PS,
            seed=seed,
            platform=PLATFORM,
        )
        cold_positions = trajectory[EQUILIBRATION_ROUNDS:, 0]
        cold_energy = energy_history[EQUILIBRATION_ROUNDS:, 0]
        radius, end_to_end, phi = observables(cold_positions, carbon_indices)
        record = statistics(cold_energy, radius, end_to_end, phi)
        record["seed"] = seed
        record["swap_acceptance"] = acceptance.tolist()
        record["elapsed_seconds"] = time.time() - start
        seed_records.append(record)
        positions.append(cold_positions)
        energies.append(cold_energy)
        np.savez_compressed(
            OUTPUT / f"{folder}_torsion_seed{seed}.npz",
            positions_nm=cold_positions.astype(np.float32),
            physical_energy_kj_mol=cold_energy,
        )
        log(
            f"DONE {molecule} seed={seed} frames={len(cold_energy)} "
            f"trans={record['trans_fraction']:.4f} "
            f"swap={acceptance.min():.3f}-{acceptance.max():.3f} "
            f"time={record['elapsed_seconds'] / 60:.1f}min"
        )
    positions = np.concatenate(positions)
    energies = np.concatenate(energies)
    radius, end_to_end, phi = observables(positions, carbon_indices)
    result = statistics(energies, radius, end_to_end, phi)
    result.update({
        "molecule": molecule,
        "dimension": dimension,
        "sample_count": int(len(energies)),
        "temperature_kelvin": TEMPERATURES_KELVIN[0],
        "bundle": str(bundle(folder).relative_to(ROOT)),
        "seeds": seed_records,
    })
    return result


def main():
    LOG.write_text("")
    log(
        f"RAW OPENMM TORSION REFERENCE | temperatures={TEMPERATURES_KELVIN} "
        f"rounds={ROUNDS} equilibration={EQUILIBRATION_ROUNDS} "
        f"steps_per_round={STEPS_PER_ROUND} timestep_fs={TIMESTEP_FS} "
        f"friction_per_ps={FRICTION_PER_PS} seeds={SEEDS} platform={PLATFORM}"
    )
    results = {
        str(dimension): run_molecule(dimension, *MOLECULES[dimension])
        for dimension in DIMENSIONS
    }
    SUMMARY.write_text(json.dumps({
        "configuration": {
            "temperatures_kelvin": TEMPERATURES_KELVIN,
            "rounds": ROUNDS,
            "equilibration_rounds": EQUILIBRATION_ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "timestep_fs": TIMESTEP_FS,
            "friction_per_ps": FRICTION_PER_PS,
            "seeds": SEEDS,
            "platform": PLATFORM,
        },
        "molecules": results,
    }, indent=2) + "\n")
    log(f"COMPLETE {SUMMARY}")


if __name__ == "__main__":
    main()

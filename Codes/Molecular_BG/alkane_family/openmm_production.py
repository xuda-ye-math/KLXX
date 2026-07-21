#!/usr/bin/env python
"""Accurate 300 K OpenMM production from mode-mixed reference states."""

import json
import time
from pathlib import Path

import numpy as np

from jflows_md.openmm import OpenMM_Potential, langevin

from openmm_reference import MOLECULES, ROOT, SEEDS, bundle, observables, statistics


OUTPUT = ROOT / "openmm_reference"
SUMMARY = OUTPUT / "reference.json"
LOG = OUTPUT / "production.log"

STEPS = 2400000
SAMPLE_INTERVAL = 1000
EQUILIBRATION_FRAMES = 400
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
TEMPERATURE_KELVIN = 300.0
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
        precondition = OUTPUT / f"{folder}_seed{seed}.npz"
        initial = np.load(precondition)["positions_nm"][-1]
        start = time.time()
        log(f"START {molecule} ({dimension}d) seed={seed}")
        trajectory, energy = langevin(
            potential,
            positions_nm=initial,
            steps=STEPS,
            sample_interval=SAMPLE_INTERVAL,
            timestep_fs=TIMESTEP_FS,
            friction_per_ps=FRICTION_PER_PS,
            temperature_kelvin=TEMPERATURE_KELVIN,
            seed=seed,
            platform=PLATFORM,
        )
        trajectory = trajectory[EQUILIBRATION_FRAMES:]
        energy = energy[EQUILIBRATION_FRAMES:]
        radius, end_to_end, phi = observables(trajectory, carbon_indices)
        record = statistics(energy, radius, end_to_end, phi)
        record["seed"] = seed
        record["elapsed_seconds"] = time.time() - start
        seed_records.append(record)
        positions.append(trajectory)
        energies.append(energy)
        np.savez_compressed(
            OUTPUT / f"{folder}_production_seed{seed}.npz",
            positions_nm=trajectory.astype(np.float32),
            physical_energy_kj_mol=energy,
        )
        log(
            f"DONE {molecule} seed={seed} frames={len(energy)} "
            f"energy={record['physical_energy_mean_kj_mol']:.4f} "
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
        "temperature_kelvin": TEMPERATURE_KELVIN,
        "bundle": str(bundle(folder).relative_to(ROOT)),
        "seeds": seed_records,
    })
    return result


def main():
    LOG.write_text("")
    log(
        f"RAW OPENMM PRODUCTION | temperature={TEMPERATURE_KELVIN} "
        f"steps={STEPS} sample_interval={SAMPLE_INTERVAL} "
        f"equilibration_frames={EQUILIBRATION_FRAMES} "
        f"timestep_fs={TIMESTEP_FS} friction_per_ps={FRICTION_PER_PS} "
        f"seeds={SEEDS} platform={PLATFORM}"
    )
    results = {
        str(dimension): run_molecule(dimension, *MOLECULES[dimension])
        for dimension in sorted(MOLECULES)
    }
    SUMMARY.write_text(json.dumps({
        "configuration": {
            "temperature_kelvin": TEMPERATURE_KELVIN,
            "steps": STEPS,
            "sample_interval": SAMPLE_INTERVAL,
            "equilibration_frames": EQUILIBRATION_FRAMES,
            "timestep_fs": TIMESTEP_FS,
            "friction_per_ps": FRICTION_PER_PS,
            "seeds": SEEDS,
            "platform": PLATFORM,
            "initialization": "final cold states from mode-mixing replica exchange",
        },
        "molecules": results,
    }, indent=2) + "\n")
    log(f"COMPLETE {SUMMARY}")


if __name__ == "__main__":
    main()

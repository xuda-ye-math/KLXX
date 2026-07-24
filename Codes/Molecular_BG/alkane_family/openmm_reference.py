#!/usr/bin/env python
"""Raw-potential OpenMM reference observables for the alkane family."""

import json
import math
import time
from pathlib import Path

import numpy as np

from jflows_md.openmm import OpenMM_Potential, parallel_tempering


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "openmm_reference"
SUMMARY = OUTPUT / "summary.json"
LOG = OUTPUT / "run.log"

TEMPERATURES_KELVIN = (300.0, 345.0, 397.0, 457.0, 526.0, 605.0, 696.0, 800.0)
ROUNDS = 2400
EQUILIBRATION_ROUNDS = 400
STEPS_PER_ROUND = 250
TIMESTEP_FS = 1.0
FRICTION_PER_PS = 1.0
SEEDS = (1729, 2718)
PLATFORM = "CUDA"

MOLECULES = {
    9: ("Methane", "methane_9d_raw"),
    18: ("Ethane", "ethane_18d_raw"),
    27: ("Propane", "propane_27d_raw"),
    36: ("Butane", "butane_36d"),
    45: ("Pentane", "pentane_45d"),
    54: ("Hexane", "hexane_54d"),
}


def log(message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as stream:
        stream.write(line + "\n")


def bundle(folder):
    molecule = folder.removesuffix("_raw")
    return ROOT / f"{molecule}_raw" / "bundle"


def observables(positions, carbon_indices):
    carbons = positions[:, carbon_indices]
    center = np.mean(carbons, axis=1, keepdims=True)
    radius = np.sqrt(np.mean(np.sum((carbons - center) ** 2, axis=2), axis=1))
    end_to_end = np.linalg.norm(carbons[:, -1] - carbons[:, 0], axis=1)
    torsions = []
    for first in range(carbons.shape[1] - 3):
        a, b, c, d = (carbons[:, first + index] for index in range(4))
        b0 = -(b - a)
        b1 = c - b
        b2 = d - c
        b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
        v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
        w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
        torsions.append(np.arctan2(
            np.sum(np.cross(b1, v) * w, axis=1),
            np.sum(v * w, axis=1),
        ))
    phi = (
        np.stack(torsions, axis=1)
        if torsions else np.zeros((positions.shape[0], 0))
    )
    return radius, end_to_end, phi


def statistics(energy, radius, end_to_end, phi):
    result = {
        "physical_energy_mean_kj_mol": float(np.mean(energy)),
        "physical_energy_sd_kj_mol": float(np.std(energy)),
        "carbon_radius_mean_nm": float(np.mean(radius)),
        "carbon_radius_sd_nm": float(np.std(radius)),
        "carbon_end_to_end_mean_nm": float(np.mean(end_to_end)),
        "carbon_end_to_end_sd_nm": float(np.std(end_to_end)),
        "backbone_torsion_count": int(phi.shape[1]),
    }
    if phi.shape[1]:
        trans = np.abs(phi) >= 2 * math.pi / 3
        result.update({
            "trans_fraction": float(np.mean(trans)),
            "gauche_plus_fraction": float(np.mean((phi >= 0) & ~trans)),
            "gauche_minus_fraction": float(np.mean((phi < 0) & ~trans)),
        })
    else:
        result.update({
            "trans_fraction": None,
            "gauche_plus_fraction": None,
            "gauche_minus_fraction": None,
        })
    return result


def run_molecule(dimension, molecule, folder):
    potential = OpenMM_Potential.from_bundle(bundle(folder))
    carbon_indices = np.flatnonzero(
        np.asarray(potential.bundle.system["atomic_numbers"]) == 6
    )
    positions = []
    energies = []
    seed_records = []
    for seed in SEEDS:
        start = time.time()
        log(f"START {molecule} ({dimension}d) seed={seed}")
        trajectory, energy_history, acceptance = parallel_tempering(
            potential,
            TEMPERATURES_KELVIN,
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
            OUTPUT / f"{folder}_seed{seed}.npz",
            positions_nm=cold_positions.astype(np.float32),
            physical_energy_kj_mol=cold_energy,
        )
        log(
            f"DONE {molecule} seed={seed} frames={len(cold_energy)} "
            f"energy={record['physical_energy_mean_kj_mol']:.4f} "
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
    OUTPUT.mkdir(exist_ok=True)
    LOG.write_text("")
    log(
        f"RAW OPENMM REFERENCE | temperatures={TEMPERATURES_KELVIN} "
        f"rounds={ROUNDS} equilibration={EQUILIBRATION_ROUNDS} "
        f"steps_per_round={STEPS_PER_ROUND} timestep_fs={TIMESTEP_FS} "
        f"friction_per_ps={FRICTION_PER_PS} seeds={SEEDS} platform={PLATFORM}"
    )
    results = {
        str(dimension): run_molecule(dimension, *MOLECULES[dimension])
        for dimension in sorted(MOLECULES)
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

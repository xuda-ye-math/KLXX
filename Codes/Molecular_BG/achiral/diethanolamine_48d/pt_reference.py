#!/usr/bin/env python
"""Generate the 48D neutral diethanolamine parallel-tempering reference.

**One** continuous replica-exchange chain on the *unregularized* physical
potential of the frozen 48D bundle, advanced from a single seed, persisting the
300 K cold-replica frames.  The chain starts from the bundle reference
geometry, whose C-C-N-C torsion sits in a gauche well at -65 degrees.

Six replicas on a geometric 300--800 K grid with the established 0.25 fs
unconstrained-bond timestep, matching the sibling reference drivers.

The run is bounded by wall clock rather than by a frame count: it equilibrates
for a fixed number of rounds and then produces frames until the budget is
reached.  Frames are flushed to disk at every chunk, so an interruption loses
only the work since the last flush, and the log is written from the first line.

Every output stays inside ``artifacts/pt_reference``.

Usage from this folder:

    /home/xuda/.envs/jflows/bin/python pt_reference.py               # 45 min
    /home/xuda/.envs/jflows/bin/python pt_reference.py --minutes 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jflows_md.openmm import OpenMM_Potential, parallel_tempering


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
OUTPUT = HERE / "artifacts" / "pt_reference"
FRAMES = OUTPUT / "pt_reference_frames.npz"
MANIFEST = OUTPUT / "pt_reference.json"
LOG = OUTPUT / "pt_reference.log"

# Geometric 300--800 K, the grid used by the sibling reference drivers.
TEMPERATURES_KELVIN = (300.0, 365.0, 444.0, 540.0, 658.0, 800.0)
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
PLATFORM = "CUDA"
STEPS_PER_ROUND = 2000          # 0.5 ps between saved cold frames
EQUILIBRATION_ROUNDS = 200      # 0.1 ns discarded
CHUNK_ROUNDS = 100
BUDGET_MINUTES = 45.0
SEED = 3401

# Heavy-atom torsion, taken from ``plot_dihedrals.py`` so the diagnostic
# matches the figure.
TORSION_ATOMS = (1, 2, 3, 4)


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def dihedral(positions: np.ndarray, atoms: tuple[int, int, int, int]) -> np.ndarray:
    """Return wrapped right-handed a-b-c-d dihedrals in radians."""

    a, b, c, d = (positions[:, index] for index in atoms)
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    return np.arctan2(
        np.sum(np.cross(b1, v) * w, axis=1),
        np.sum(v * w, axis=1),
    )


def torsion_occupancy(angles: np.ndarray) -> dict:
    """Return gauche-minus/trans/gauche-plus occupancy and transition count."""

    # Three staggered wells centred near -pi/3, +pi (trans), and +pi/3.
    labels = np.full(angles.shape, 1, dtype=int)
    labels[angles > np.pi / 3.0] = 2
    labels[angles < -np.pi / 3.0] = 0
    labels[np.abs(angles) > 2.0 * np.pi / 3.0] = 1
    counts = np.bincount(labels, minlength=3) / labels.size
    return {
        "occupancy": [float(value) for value in counts],
        "visited": int((counts > 0.01).sum()),
        "transitions": int(np.count_nonzero(np.diff(labels))),
    }


def run_chunk(potential, positions, rounds, seed, platform):
    """Advance every replica by one chunk and return its trajectory."""

    trajectory, energy_history, acceptance = parallel_tempering(
        potential,
        TEMPERATURES_KELVIN,
        positions_nm=positions,
        rounds=rounds,
        steps_per_round=STEPS_PER_ROUND,
        timestep_fs=TIMESTEP_FS,
        friction_per_ps=FRICTION_PER_PS,
        seed=seed,
        platform=platform,
    )
    if not np.all(np.isfinite(energy_history)):
        raise ValueError("nonfinite replica energies")
    return trajectory, energy_history, acceptance


def checkpoint(frames, cold_energies) -> None:
    """Write the frames accumulated so far."""

    np.savez_compressed(
        FRAMES,
        positions_nm=np.concatenate(frames),
        cold_energy_kj_mol=np.concatenate(cold_energies),
        temperatures_kelvin=np.asarray(TEMPERATURES_KELVIN),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=BUDGET_MINUTES)
    parser.add_argument("--equilibration-rounds", type=int, default=EQUILIBRATION_ROUNDS)
    parser.add_argument("--chunk-rounds", type=int, default=CHUNK_ROUNDS)
    parser.add_argument("--platform", default=PLATFORM)
    args = parser.parse_args()
    if args.minutes <= 0 or args.chunk_rounds <= 0 or args.equilibration_rounds < 0:
        parser.error("minutes and chunk rounds must be positive")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    round_ps = STEPS_PER_ROUND * TIMESTEP_FS / 1000.0

    started = time.time()
    log(f"START pt_reference budget={args.minutes:.1f} min -> {OUTPUT}")
    potential = OpenMM_Potential.from_bundle(BUNDLE)
    positions = np.asarray(potential.reference_positions_nm)
    start_torsion = float(dihedral(positions[None], TORSION_ATOMS)[0])
    log(
        f"bundle={potential.bundle_name} atoms={potential.n_atoms} "
        f"unregularized target={potential.temperature_kelvin:.0f} K seed={SEED}"
    )
    log(
        f"initialized from the bundle reference geometry: C-C-N-C torsion "
        f"{np.degrees(start_torsion):.4f} deg"
    )
    log(
        f"grid: {len(TEMPERATURES_KELVIN)} replicas "
        f"{'/'.join(f'{v:.0f}' for v in TEMPERATURES_KELVIN)} K | "
        f"timestep={TIMESTEP_FS} fs | {STEPS_PER_ROUND} steps/round "
        f"({round_ps:.2f} ps per frame) | platform={args.platform}"
    )
    log(
        f"plan: {args.equilibration_rounds:,} equilibration rounds, then "
        f"production until the budget is spent | log and checkpoint every "
        f"{args.chunk_rounds:,} rounds -> {FRAMES.name}"
    )

    acceptances = []
    equilibrated = 0
    while equilibrated < args.equilibration_rounds:
        rounds = min(args.chunk_rounds, args.equilibration_rounds - equilibrated)
        chunk_started = time.time()
        trajectory, energies, acceptance = run_chunk(
            potential, positions, rounds, SEED + equilibrated, args.platform
        )
        positions = trajectory[-1]
        acceptances.append(acceptance)
        equilibrated += rounds
        log(
            f"equilibration {equilibrated:,}/{args.equilibration_rounds:,} rounds | "
            f"{rounds / (time.time() - chunk_started):.2f} rounds/s | "
            f"exchange min={acceptance.min():.3f} mean={acceptance.mean():.3f} | "
            f"cold E={energies[:, 0].mean():.2f} kJ/mol | "
            f"{(time.time() - started) / 60:.1f} min elapsed"
        )

    frames: list[np.ndarray] = []
    cold_energies: list[np.ndarray] = []
    production_started = time.time()
    deadline = started + args.minutes * 60.0
    chunk_index = 0
    while time.time() < deadline:
        chunk_started = time.time()
        trajectory, energies, acceptance = run_chunk(
            potential, positions, args.chunk_rounds, SEED + 1 + chunk_index, args.platform
        )
        positions = trajectory[-1]
        frames.append(trajectory[:, 0].astype(np.float32))
        cold_energies.append(energies[:, 0].astype(np.float64))
        acceptances.append(acceptance)
        chunk_index += 1
        collected = sum(len(block) for block in frames)

        checkpoint(frames, cold_energies)

        elapsed = time.time() - production_started
        occupancy = torsion_occupancy(
            dihedral(np.concatenate(frames).astype(np.float64), TORSION_ATOMS)
        )
        log(
            f"chunk {chunk_index}: {collected:,} frames "
            f"({collected * round_ps / 1000.0:.2f} ns) | "
            f"{args.chunk_rounds / (time.time() - chunk_started):.2f} rounds/s now, "
            f"{collected / elapsed:.2f} avg | "
            f"exchange min={acceptance.min():.3f} mean={acceptance.mean():.3f} | "
            f"cold E={energies[:, 0].mean():.2f} kJ/mol | "
            f"wells={occupancy['visited']}/3 transitions={occupancy['transitions']} | "
            f"{(time.time() - started) / 60:.1f}/{args.minutes:.1f} min"
        )

    positions_all = np.concatenate(frames)
    energies_all = np.concatenate(cold_energies)
    angles = dihedral(positions_all.astype(np.float64), TORSION_ATOMS)
    occupancy = torsion_occupancy(angles)
    acceptance_all = np.stack(acceptances)
    manifest = {
        "molecule": "diethanolamine",
        "dimension": 48,
        "bundle": potential.bundle_name,
        "method": "parallel tempering, cold replica retained",
        "initialization": "bundle reference geometry",
        "initial_torsion_deg": float(np.degrees(start_torsion)),
        "temperatures_kelvin": list(TEMPERATURES_KELVIN),
        "timestep_fs": TIMESTEP_FS,
        "friction_per_ps": FRICTION_PER_PS,
        "steps_per_round": STEPS_PER_ROUND,
        "ps_per_frame": round_ps,
        "seed": SEED,
        "equilibration_rounds": args.equilibration_rounds,
        "production_frames": int(positions_all.shape[0]),
        "production_ns": float(positions_all.shape[0] * round_ps / 1000.0),
        "torsion_atoms": list(TORSION_ATOMS),
        "torsion_occupancy": occupancy,
        "cold_energy_mean_kj_mol": float(energies_all.mean()),
        "cold_energy_std_kj_mol": float(energies_all.std()),
        "exchange_acceptance_min": float(acceptance_all.min()),
        "exchange_acceptance_mean": float(acceptance_all.mean()),
        "budget_minutes": args.minutes,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "status": "complete",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(
        f"DONE {positions_all.shape[0]:,} frames "
        f"({manifest['production_ns']:.2f} ns) | "
        f"occupancy={['%.4f' % v for v in occupancy['occupancy']]} "
        f"wells={occupancy['visited']}/3 transitions={occupancy['transitions']} | "
        f"exchange min={manifest['exchange_acceptance_min']:.3f} | "
        f"{manifest['elapsed_minutes']:.1f} min -> {FRAMES.name}, {MANIFEST.name}"
    )


if __name__ == "__main__":
    main()

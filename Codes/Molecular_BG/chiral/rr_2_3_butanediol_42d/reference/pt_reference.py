#!/usr/bin/env python
"""Generate the 10^5-frame (2R,3R)-2,3-butanediol parallel-tempering reference.

**One** continuous replica-exchange chain on the *unregularized* physical
potential of the frozen 42D bundle, advanced from a single seed, persisting the
300 K cold-replica frames.  Output names carry the ``_100000`` frame count, so
nothing already in this folder is touched.

Splitting the budget across concurrently running independent chains was
measured and rejected: one process reaches 3.85 frames/s and five concurrent
processes reach 5 x 0.76 = 3.79 frames/s aggregate.  A 16-atom system is
kernel-launch-latency-bound at about 23 microseconds per integration step, so
the device is never the constraint and dividing the work buys nothing.

Six replicas on a geometric 300--800 K grid, which screened at a 0.500--0.540
per-pair exchange probability, with the established 0.25 fs unconstrained-bond
timestep.

This driver is self-contained: it defines its own grid and diagnostics rather
than importing them, so removing any sibling script cannot disable it.

Progress is reported every ``CHUNK_ROUNDS`` rounds -- 100 chunks over the run
-- with cumulative frames, instantaneous and average rate, projected finish
clock time, exchange acceptance, cold-replica energy, and the running torsion
occupancy, so convergence can be watched as it happens.  Frames are flushed to
disk at every chunk, so an interruption loses only the work since the last
flush.  Every output file stays inside this ``reference`` folder.

Usage from this folder:

    python pt_reference.py                          # 100,000 frames
    python pt_reference.py --production-rounds 5000 # shorter run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jflows_md.openmm import OpenMM_Potential, parallel_tempering


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "bundle"
FRAMES = HERE / "pt_reference_frames_100000.npz"
MANIFEST = HERE / "pt_reference_100000.json"
LOG = HERE / "pt_reference_100000.log"

# ``wide_6`` from the smoke screen: geometric 300--800 K, per-pair exchange
# probability 0.500--0.540 in production.
TEMPERATURES_KELVIN = (300.0, 365.0, 444.0, 540.0, 658.0, 800.0)
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
PLATFORM = "CUDA"
STEPS_PER_ROUND = 2000          # 0.5 ps between saved cold frames
EQUILIBRATION_ROUNDS = 1000     # 0.5 ns discarded
PRODUCTION_ROUNDS = 100_000     # 50 ns retained, one frame per round
CHUNK_ROUNDS = 1000             # 100 chunks in total, one line every ~4.3 min
CHECKPOINT_EVERY_CHUNKS = 1     # flush to disk at every chunk
SEED = 1729

# Established heavy-atom torsion definitions, taken from the sibling
# ``plot_conformational_landscape.py`` so the diagnostic matches the figures.
CCCC_ATOMS = (4, 2, 3, 5)
OCCO_ATOMS = (0, 2, 3, 1)


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


def basin_occupancy(angles: np.ndarray) -> dict:
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
    parser.add_argument("--production-rounds", type=int, default=PRODUCTION_ROUNDS)
    parser.add_argument("--equilibration-rounds", type=int, default=EQUILIBRATION_ROUNDS)
    parser.add_argument("--chunk-rounds", type=int, default=CHUNK_ROUNDS)
    parser.add_argument("--platform", default=PLATFORM)
    args = parser.parse_args()
    if args.production_rounds <= 0 or args.equilibration_rounds < 0:
        parser.error("production rounds must be positive; equilibration nonnegative")
    if args.chunk_rounds <= 0:
        parser.error("chunk rounds must be positive")

    round_ps = STEPS_PER_ROUND * TIMESTEP_FS / 1000.0
    potential = OpenMM_Potential.from_bundle(BUNDLE)
    log(
        f"bundle={potential.bundle_name} atoms={potential.n_atoms} "
        f"unregularized target={potential.temperature_kelvin:.0f} K seed={SEED}"
    )
    log(
        f"grid wide_6: {len(TEMPERATURES_KELVIN)} replicas "
        f"{'/'.join(f'{v:.0f}' for v in TEMPERATURES_KELVIN)} K | "
        f"timestep={TIMESTEP_FS} fs | {STEPS_PER_ROUND} steps/round "
        f"({round_ps:.2f} ps per frame) | platform={args.platform}"
    )
    log(
        f"plan: one continuous chain, {args.equilibration_rounds:,} equilibration "
        f"rounds then {args.production_rounds:,} production frames "
        f"({args.production_rounds * round_ps / 1000.0:.1f} ns retained) | "
        f"log and checkpoint every {args.chunk_rounds:,} rounds "
        f"({args.production_rounds // args.chunk_rounds} chunks) -> {FRAMES.name}"
    )

    started = time.time()
    positions = potential.reference_positions_nm
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
        elapsed = time.time() - started
        remaining = (args.equilibration_rounds - equilibrated) / (equilibrated / elapsed)
        log(
            f"equilibration {equilibrated:,}/{args.equilibration_rounds:,} rounds "
            f"({100.0 * equilibrated / args.equilibration_rounds:.1f}%) | "
            f"{rounds / (time.time() - chunk_started):.2f} rounds/s | "
            f"exchange min={acceptance.min():.3f} mean={acceptance.mean():.3f} | "
            f"cold E={energies[:, 0].mean():.2f} kJ/mol | "
            f"{elapsed / 60:.1f} min elapsed, {remaining / 60:.1f} min to production"
        )

    frames: list[np.ndarray] = []
    cold_energies: list[np.ndarray] = []
    production_started = time.time()
    remaining = args.production_rounds
    chunk_index = 0
    while remaining > 0:
        rounds = min(args.chunk_rounds, remaining)
        chunk_started = time.time()
        trajectory, energies, acceptance = run_chunk(
            potential, positions, rounds, SEED + 1 + chunk_index, args.platform
        )
        positions = trajectory[-1]
        frames.append(trajectory[:, 0].astype(np.float32))
        cold_energies.append(energies[:, 0].astype(np.float64))
        acceptances.append(acceptance)
        remaining -= rounds
        chunk_index += 1
        collected = args.production_rounds - remaining

        flushed = chunk_index % CHECKPOINT_EVERY_CHUNKS == 0 or remaining == 0
        if flushed:
            checkpoint(frames, cold_energies)

        elapsed = time.time() - production_started
        average = collected / elapsed
        eta = remaining / average
        cccc = basin_occupancy(
            dihedral(np.concatenate(frames).astype(np.float64), CCCC_ATOMS)
        )
        log(
            f"chunk {chunk_index}: {collected:,}/{args.production_rounds:,} frames "
            f"({100.0 * collected / args.production_rounds:.2f}%) | "
            f"{rounds / (time.time() - chunk_started):.2f} rounds/s now, "
            f"{average:.2f} avg | exchange min={acceptance.min():.3f} "
            f"mean={acceptance.mean():.3f} | cold E={energies[:, 0].mean():.2f} "
            f"kJ/mol | C-C-C-C {'/'.join(f'{v:.3f}' for v in cccc['occupancy'])} "
            f"({cccc['transitions']:,} transitions) | {elapsed / 3600:.2f} h "
            f"elapsed, eta {eta / 3600:.2f} h -> "
            f"{time.strftime('%H:%M:%S', time.localtime(time.time() + eta))}"
            f"{' | checkpoint' if flushed else ''}"
        )

    positions_nm = np.concatenate(frames)
    cold_energy = np.concatenate(cold_energies)
    acceptance_all = np.concatenate([part.ravel() for part in acceptances])
    coordinates = positions_nm.astype(np.float64)
    cccc = basin_occupancy(dihedral(coordinates, CCCC_ATOMS))
    occo = basin_occupancy(dihedral(coordinates, OCCO_ATOMS))
    elapsed = time.time() - started

    log(
        f"C-C-C-C: visited={cccc['visited']}/3 transitions={cccc['transitions']:,} "
        f"occupancy={['%.4f' % v for v in cccc['occupancy']]}"
    )
    log(
        f"O-C-C-O: visited={occo['visited']}/3 transitions={occo['transitions']:,} "
        f"occupancy={['%.4f' % v for v in occo['occupancy']]}"
    )
    MANIFEST.write_text(
        json.dumps(
            {
                "bundle": potential.bundle_name,
                "atoms": int(potential.n_atoms),
                "potential": "unregularized physical",
                "sampler": "parallel tempering",
                "chains": 1,
                "temperature_kelvin": float(potential.temperature_kelvin),
                "grid": "wide_6",
                "temperatures_kelvin": list(TEMPERATURES_KELVIN),
                "timestep_fs": TIMESTEP_FS,
                "friction_per_ps": FRICTION_PER_PS,
                "platform": args.platform,
                "steps_per_round": STEPS_PER_ROUND,
                "ps_per_frame": round_ps,
                "equilibration_rounds": args.equilibration_rounds,
                "production_frames": int(positions_nm.shape[0]),
                "nanoseconds_retained": float(positions_nm.shape[0] * round_ps / 1000.0),
                "acceptance_mean": float(acceptance_all.mean()),
                "acceptance_min": float(acceptance_all.min()),
                "acceptance_max": float(acceptance_all.max()),
                "cold_energy_mean_kj_mol": float(cold_energy.mean()),
                "cold_energy_std_kj_mol": float(cold_energy.std()),
                "cccc": cccc,
                "occo": occo,
                "seconds": elapsed,
                "seed": SEED,
                "frames_path": FRAMES.name,
                "status": "complete",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(
        f"complete: {positions_nm.shape[0]:,} frames "
        f"({positions_nm.shape[0] * round_ps / 1000.0:.1f} ns) in "
        f"{elapsed / 3600:.2f} h -> {FRAMES.name}"
    )


if __name__ == "__main__":
    main()

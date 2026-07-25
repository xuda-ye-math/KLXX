#!/usr/bin/env python
"""Generate the (2R,3R)-2,3-butanediol OpenMM parallel-tempering reference.

Runs native replica exchange on the *unregularized* physical potential of the
frozen 42D bundle and persists the 300 K cold-replica frames.  Settings follow
the ``pt_smoke.py`` screen: six replicas on a geometric 300--800 K grid, which
measured a 0.500--0.540 per-pair exchange probability, and the established
0.25 fs unconstrained-bond timestep.

The run is executed in chunks so partial results survive an interruption; each
chunk continues from the previous chunk's replica positions.  Every output file
stays inside this ``reference`` folder.

Usage from this folder:

    python pt_reference.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jflows_md.openmm import OpenMM_Potential, parallel_tempering

from pt_smoke import (
    CANDIDATES,
    CCCC_ATOMS,
    FRICTION_PER_PS,
    OCCO_ATOMS,
    PLATFORM,
    TIMESTEP_FS,
    basin_occupancy,
    dihedral,
)


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "bundle"
LOG = HERE / "pt_reference.log"
FRAMES = HERE / "pt_reference_frames.npz"
MANIFEST = HERE / "pt_reference.json"

GRID = "wide_6"
STEPS_PER_ROUND = 2000          # 0.5 ps between saved cold frames
EQUILIBRATION_ROUNDS = 1000     # 0.5 ns discarded
PRODUCTION_ROUNDS = 24000       # 12 ns retained, one frame per round
CHUNK_ROUNDS = 2000
SEED = 1729


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_chunk(potential, temperatures, positions, rounds, seed):
    """Advance every replica by one chunk and return its cold-replica frames."""

    trajectory, energy_history, acceptance = parallel_tempering(
        potential,
        temperatures,
        positions_nm=positions,
        rounds=rounds,
        steps_per_round=STEPS_PER_ROUND,
        timestep_fs=TIMESTEP_FS,
        friction_per_ps=FRICTION_PER_PS,
        seed=seed,
        platform=PLATFORM,
    )
    if not np.all(np.isfinite(energy_history)):
        raise ValueError("nonfinite replica energies")
    return trajectory, energy_history, acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-rounds", type=int, default=PRODUCTION_ROUNDS)
    parser.add_argument("--equilibration-rounds", type=int, default=EQUILIBRATION_ROUNDS)
    args = parser.parse_args()
    if args.production_rounds <= 0 or args.equilibration_rounds < 0:
        parser.error("production rounds must be positive; equilibration nonnegative")

    temperatures = CANDIDATES[GRID]
    round_ps = STEPS_PER_ROUND * TIMESTEP_FS / 1000.0
    potential = OpenMM_Potential.from_bundle(BUNDLE)
    log(
        f"bundle={potential.bundle_name} atoms={potential.n_atoms} "
        f"unregularized target={potential.temperature_kelvin:.0f} K"
    )
    log(
        f"grid={GRID} {len(temperatures)} replicas "
        f"{'/'.join(f'{value:.0f}' for value in temperatures)} K | "
        f"timestep={TIMESTEP_FS} fs | {STEPS_PER_ROUND} steps/round "
        f"({round_ps:.2f} ps per frame) | platform={PLATFORM}"
    )
    log(
        f"plan: {args.equilibration_rounds} equilibration rounds then "
        f"{args.production_rounds} production frames "
        f"({args.production_rounds * round_ps / 1000.0:.1f} ns retained)"
    )

    started = time.time()
    positions = potential.reference_positions_nm
    acceptances = []

    if args.equilibration_rounds:
        trajectory, energies, acceptance = run_chunk(
            potential, temperatures, positions, args.equilibration_rounds, SEED
        )
        positions = trajectory[-1]
        acceptances.append(acceptance)
        log(
            f"equilibration done: acceptance mean={acceptance.mean():.3f} "
            f"cold E={energies[:, 0].mean():.2f} kJ/mol "
            f"[{time.time() - started:.0f} s]"
        )

    frames: list[np.ndarray] = []
    cold_energies: list[np.ndarray] = []
    remaining = args.production_rounds
    chunk_index = 0
    while remaining > 0:
        rounds = min(CHUNK_ROUNDS, remaining)
        trajectory, energies, acceptance = run_chunk(
            potential, temperatures, positions, rounds, SEED + 1 + chunk_index
        )
        positions = trajectory[-1]
        frames.append(trajectory[:, 0].astype(np.float32))
        cold_energies.append(energies[:, 0].astype(np.float64))
        acceptances.append(acceptance)
        remaining -= rounds
        chunk_index += 1
        collected = sum(part.shape[0] for part in frames)
        np.savez_compressed(
            FRAMES,
            positions_nm=np.concatenate(frames),
            cold_energy_kj_mol=np.concatenate(cold_energies),
            temperatures_kelvin=np.asarray(temperatures),
        )
        elapsed = time.time() - started
        rate = collected / elapsed
        log(
            f"chunk {chunk_index}: {collected:,}/{args.production_rounds:,} frames "
            f"acceptance mean={acceptance.mean():.3f} "
            f"cold E={energies[:, 0].mean():.2f} kJ/mol "
            f"[{elapsed:.0f} s, {rate:.1f} frames/s, "
            f"eta {remaining / rate:.0f} s]"
        )

    positions_nm = np.concatenate(frames)
    cold_energy = np.concatenate(cold_energies)
    acceptance_all = np.stack(acceptances)
    cccc = basin_occupancy(dihedral(positions_nm, CCCC_ATOMS))
    occo = basin_occupancy(dihedral(positions_nm, OCCO_ATOMS))
    elapsed = time.time() - started

    log(
        f"C-C-C-C: visited={cccc['visited']}/3 transitions={cccc['transitions']} "
        f"occupancy={['%.3f' % v for v in cccc['occupancy']]}"
    )
    log(
        f"O-C-C-O: visited={occo['visited']}/3 transitions={occo['transitions']} "
        f"occupancy={['%.3f' % v for v in occo['occupancy']]}"
    )

    MANIFEST.write_text(
        json.dumps(
            {
                "bundle": potential.bundle_name,
                "atoms": int(potential.n_atoms),
                "potential": "unregularized physical",
                "temperature_kelvin": float(potential.temperature_kelvin),
                "grid": GRID,
                "temperatures_kelvin": list(temperatures),
                "timestep_fs": TIMESTEP_FS,
                "friction_per_ps": FRICTION_PER_PS,
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
        f"({positions_nm.shape[0] * round_ps / 1000.0:.1f} ns) in {elapsed:.0f} s "
        f"-> {FRAMES.name}"
    )


if __name__ == "__main__":
    main()

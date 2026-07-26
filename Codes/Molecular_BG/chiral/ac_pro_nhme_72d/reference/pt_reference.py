#!/usr/bin/env python
"""Generate the Ac-Pro-NHMe OpenMM parallel-tempering reference.

Runs native replica exchange on the *unregularized* physical potential of the
frozen 72D bundle and persists the 300 K cold-replica frames.  Settings follow
the ``pt_smoke.py`` screen: eight replicas on a geometric 300--800 K grid,
which measured a 0.515 minimum per-pair exchange probability, and the
established 0.25 fs unconstrained-bond timestep.

This driver is self-contained: it defines its own grid and diagnostics rather
than importing them, so removing any sibling script cannot disable it.

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


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "bundle"
LOG = HERE / "pt_reference.log"
FRAMES = HERE / "pt_reference_frames_32000.npz"
MANIFEST = HERE / "pt_reference_32000.json"

# ``wide_8`` from the smoke screen: geometric 300--800 K, minimum per-pair
# exchange probability 0.515.
TEMPERATURES_KELVIN = (300.0, 345.0, 397.0, 457.0, 526.0, 605.0, 696.0, 800.0)
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
PLATFORM = "CUDA"
STEPS_PER_ROUND = 2000          # 0.5 ps between saved cold frames
EQUILIBRATION_ROUNDS = 1000     # 0.5 ns discarded
PRODUCTION_ROUNDS = 32000       # 16 ns retained, one frame per round
CHUNK_ROUNDS = 2000
SEED = 1729

# Established torsion definitions, matching the sibling conformational
# landscape figure.
PHI_ATOMS = (4, 6, 16, 18)
PSI_ATOMS = (6, 16, 18, 20)
OMEGA_ATOMS = (1, 4, 6, 16)
CHI1_ATOMS = (6, 16, 13, 10)


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


def cis_trans(omega: np.ndarray) -> dict:
    """Return cis/trans occupancy and crossing count of the peptide torsion."""

    trans = np.abs(omega) > np.pi / 2.0
    return {
        "trans_fraction": float(trans.mean()),
        "cis_fraction": float(1.0 - trans.mean()),
        "crossings": int(np.count_nonzero(np.diff(trans.astype(int)))),
    }


def run_chunk(potential, positions, rounds, seed):
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

    round_ps = STEPS_PER_ROUND * TIMESTEP_FS / 1000.0
    potential = OpenMM_Potential.from_bundle(BUNDLE)
    log(
        f"bundle={potential.bundle_name} atoms={potential.n_atoms} "
        f"unregularized target={potential.temperature_kelvin:.0f} K"
    )
    log(
        f"grid wide_8: {len(TEMPERATURES_KELVIN)} replicas "
        f"{'/'.join(f'{v:.0f}' for v in TEMPERATURES_KELVIN)} K | "
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
            potential, positions, args.equilibration_rounds, SEED
        )
        positions = trajectory[-1]
        acceptances.append(acceptance)
        log(
            f"equilibration done: acceptance min={acceptance.min():.3f} "
            f"mean={acceptance.mean():.3f} cold E={energies[:, 0].mean():.2f} "
            f"kJ/mol [{time.time() - started:.0f} s]"
        )

    frames: list[np.ndarray] = []
    cold_energies: list[np.ndarray] = []
    remaining = args.production_rounds
    chunk_index = 0
    while remaining > 0:
        rounds = min(CHUNK_ROUNDS, remaining)
        trajectory, energies, acceptance = run_chunk(
            potential, positions, rounds, SEED + 1 + chunk_index
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
            temperatures_kelvin=np.asarray(TEMPERATURES_KELVIN),
        )
        elapsed = time.time() - started
        rate = collected / elapsed
        log(
            f"chunk {chunk_index}: {collected:,}/{args.production_rounds:,} frames "
            f"acceptance min={acceptance.min():.3f} "
            f"cold E={energies[:, 0].mean():.2f} kJ/mol "
            f"[{elapsed:.0f} s, {rate:.2f} frames/s, "
            f"eta {remaining / rate:.0f} s]"
        )

    positions_nm = np.concatenate(frames)
    cold_energy = np.concatenate(cold_energies)
    acceptance_all = np.stack(acceptances)
    omega = cis_trans(dihedral(positions_nm.astype(np.float64), OMEGA_ATOMS))
    elapsed = time.time() - started

    log(
        f"omega: trans={omega['trans_fraction']:.4f} "
        f"cis={omega['cis_fraction']:.4f} crossings={omega['crossings']}"
    )
    MANIFEST.write_text(
        json.dumps(
            {
                "bundle": potential.bundle_name,
                "atoms": int(potential.n_atoms),
                "potential": "unregularized physical",
                "sampler": "parallel tempering",
                "temperature_kelvin": float(potential.temperature_kelvin),
                "grid": "wide_8",
                "temperatures_kelvin": list(TEMPERATURES_KELVIN),
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
                "omega": omega,
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

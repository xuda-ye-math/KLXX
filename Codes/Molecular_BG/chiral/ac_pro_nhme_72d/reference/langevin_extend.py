#!/usr/bin/env python
"""Extend the Ac-Pro-NHMe reference to a target frame count by short Langevin runs.

This driver performs **no** replica exchange.  It runs single-particle Langevin
dynamics at the target distribution: one integrator at 300 K, one context, no
temperature grid.  Parallel tempering enters only through the seed frames.

Each equilibrium frame from ``pt_reference.py`` seeds many independent short
propagations.  A Boltzmann-preserving kernel started from an equilibrium sample
returns an equilibrium sample, so every variant is a valid draw from the same
unregularized physical target at 300 K.

Note the variants are *not* independent samples: hundreds of children of one
parent, separated by tens of femtoseconds, stay strongly correlated.  They fill
histogram bins and extend the resolvable free-energy range, but the effective
sample size remains that of the parent set.

Frames accumulate in a memory-mapped scratch file outside the repository, so an
interruption loses only the work since the last flush, and the final ``.npz``
is written once at the end.

Usage from this folder:

    python langevin_extend.py                      # 10,000,000 frames
    python langevin_extend.py --frames 2000000     # shorter run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

from jflows_md.openmm import OpenMM_Potential


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "bundle"
PARENTS = HERE / "pt_reference_frames_32000.npz"
PARENT_MANIFEST = HERE / "pt_reference_32000.json"
FRAMES = HERE / "pt_reference_frames.npz"
MANIFEST = HERE / "pt_reference.json"
LOG = HERE / "langevin_extend.log"
SCRATCH = Path(
    "/tmp/claude-1000/-data-projects-KLXX/"
    "28ef3bb7-ef8d-47fc-afa6-dcf0f982fa44/scratchpad/extend_72d_frames.npy"
)

TEMPERATURE_KELVIN = 300.0
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
STEPS = 50               # 12.5 fs of very short Langevin per variant
TARGET_FRAMES = 15_000_000
PLATFORM = "CUDA"
SEED = 20260725
LOG_EVERY_PARENTS = 200

OMEGA_ATOMS = (1, 4, 6, 16)


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
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


def rotamers(angles: np.ndarray) -> dict:
    """Return trans/cis fractions of the ACE-PRO peptide torsion."""

    trans = np.abs(angles) > np.pi / 2.0
    total = angles.size
    return {
        "trans": float(np.count_nonzero(trans) / total),
        "cis": float(np.count_nonzero(~trans) / total),
    }


def variant_counts(parents: int, frames: int) -> np.ndarray:
    """Spread ``frames`` over ``parents`` as evenly as possible."""

    base, remainder = divmod(frames, parents)
    counts = np.full(parents, base, dtype=np.int64)
    counts[:remainder] += 1
    if int(counts.sum()) != frames:
        raise ValueError("variant counts do not sum to the requested frames")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=TARGET_FRAMES)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--parents", type=Path, default=PARENTS)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    args = parser.parse_args()
    if args.frames <= 0 or args.steps <= 0:
        parser.error("frames and steps must be positive")

    with np.load(args.parents.expanduser().resolve()) as handle:
        parents = np.asarray(handle["positions_nm"], dtype=np.float64)
    if parents.ndim != 3 or parents.shape[2] != 3:
        raise ValueError(f"expected parent frames [F, A, 3], found {parents.shape}")
    n_parents, n_atoms, _ = parents.shape
    counts = variant_counts(n_parents, args.frames)
    offsets = np.concatenate([[0], np.cumsum(counts)])

    potential = OpenMM_Potential.from_bundle(BUNDLE)
    log(
        f"bundle={potential.bundle_name} atoms={n_atoms} parents={n_parents:,} "
        f"target={args.frames:,} frames | variants/parent "
        f"{counts.min()}-{counts.max()} | steps={args.steps} "
        f"({args.steps * TIMESTEP_FS:.1f} fs each) platform={PLATFORM}"
    )

    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_KELVIN * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        TIMESTEP_FS * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(SEED)
    context = mm.Context(
        potential.create_system(),
        integrator,
        mm.Platform.getPlatformByName(PLATFORM),
    )

    scratch = args.scratch.expanduser().resolve()
    scratch.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        scratch, mode="w+", dtype=np.float32, shape=(args.frames, n_atoms, 3)
    )
    log(f"scratch={scratch}")

    started = time.time()
    for index in range(n_parents):
        parent = parents[index] * unit.nanometer
        base = int(offsets[index])
        for variant in range(int(counts[index])):
            context.setPositions(parent)
            context.setVelocitiesToTemperature(
                TEMPERATURE_KELVIN * unit.kelvin, SEED + base + variant
            )
            integrator.step(args.steps)
            state = context.getState(getPositions=True)
            output[base + variant] = state.getPositions(
                asNumpy=True
            ).value_in_unit(unit.nanometer)
        done = index + 1
        if done % LOG_EVERY_PARENTS == 0 or done == n_parents:
            output.flush()
            produced = int(offsets[done])
            elapsed = time.time() - started
            rate = produced / elapsed
            log(
                f"{done:,}/{n_parents:,} parents ({produced:,}/{args.frames:,} "
                f"frames) [{elapsed / 60:.1f} min, {rate:,.0f} frames/s, "
                f"eta {(args.frames - produced) / rate / 60:.1f} min]"
            )
    del context, integrator
    output.flush()

    if not np.all(np.isfinite(output[-100000:])):
        raise ValueError("nonfinite extended frames")
    parent_rotamers = rotamers(dihedral(parents, OMEGA_ATOMS))
    sample = np.asarray(output[:: max(1, args.frames // 2_000_000)], dtype=np.float64)
    extended_rotamers = rotamers(dihedral(sample, OMEGA_ATOMS))
    elapsed = time.time() - started

    log("writing npz")
    np.savez(FRAMES, positions_nm=np.asarray(output))
    parent_manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    MANIFEST.write_text(
        json.dumps(
            {
                **{
                    key: parent_manifest[key]
                    for key in ("bundle", "atoms", "potential")
                },
                "sampler": "single-particle Langevin at the target distribution",
                "temperature_kelvin": TEMPERATURE_KELVIN,
                "timestep_fs": TIMESTEP_FS,
                "friction_per_ps": FRICTION_PER_PS,
                "parent_sampler": "parallel tempering (seed frames only)",
                "parent_frames": int(n_parents),
                "parent_path": args.parents.name,
                "variants_per_parent_min": int(counts.min()),
                "variants_per_parent_max": int(counts.max()),
                "extension_steps": int(args.steps),
                "extension_femtoseconds": float(args.steps * TIMESTEP_FS),
                "production_frames": int(args.frames),
                "parent_omega": parent_rotamers,
                "omega": extended_rotamers,
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
        f"omega parent trans={parent_rotamers['trans']:.4f} -> "
        f"extended trans={extended_rotamers['trans']:.4f}"
    )
    log(f"complete: {args.frames:,} frames in {elapsed / 3600:.2f} h -> {FRAMES.name}")


if __name__ == "__main__":
    main()

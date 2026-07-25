#!/usr/bin/env python
"""Extend the butanediol reference to 240,000 frames by short Langevin runs.

This driver performs **no** replica exchange.  It runs single-particle Langevin
dynamics at the target distribution: one integrator at 300 K, one context, no
temperature grid.  Parallel tempering enters only through the seed frames.

Each of the 24,000 equilibrium frames from ``pt_reference.py`` seeds ten
independent short propagations, giving ten variants per parent frame.  A
Boltzmann-preserving kernel started from an equilibrium sample returns an
equilibrium sample, so every variant is a valid draw from the same
unregularized physical target at 300 K.

The integrator is ``LangevinMiddleIntegrator`` at 0.25 fs with 1/ps friction,
matching the kernel that produced the seed frames.  The context is built from
the public ``OpenMM_Potential.create_system`` and reused, because creating one
context per propagation costs ~137 ms and would dominate the run.

Usage from this folder:

    python langevin_extend.py
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

from pt_smoke import CCCC_ATOMS, OCCO_ATOMS, basin_occupancy, dihedral


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "bundle"
PARENTS = HERE / "pt_reference_frames_24000.npz"
FRAMES = HERE / "pt_reference_frames.npz"
MANIFEST = HERE / "pt_reference.json"
LOG = HERE / "langevin_extend.log"

TEMPERATURE_KELVIN = 300.0
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
STEPS = 200          # 50 fs of very short Langevin per variant
VARIANTS = 10
PLATFORM = "CUDA"
SEED = 20260725
CHUNK_PARENTS = 2000


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--variants", type=int, default=VARIANTS)
    parser.add_argument("--parents", type=Path, default=PARENTS)
    args = parser.parse_args()
    if args.steps <= 0 or args.variants <= 0:
        parser.error("steps and variants must be positive")

    with np.load(args.parents.expanduser().resolve()) as handle:
        parents = np.asarray(handle["positions_nm"], dtype=np.float64)
    if parents.ndim != 3 or parents.shape[2] != 3:
        raise ValueError(f"expected parent frames [F, A, 3], found {parents.shape}")
    n_parents, n_atoms, _ = parents.shape

    potential = OpenMM_Potential.from_bundle(BUNDLE)
    log(
        f"bundle={potential.bundle_name} atoms={n_atoms} parents={n_parents:,} "
        f"variants={args.variants} steps={args.steps} "
        f"({args.steps * TIMESTEP_FS:.0f} fs each) platform={PLATFORM}"
    )
    log(f"output frames = {n_parents * args.variants:,}")

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

    started = time.time()
    output = np.empty(
        (n_parents, args.variants, n_atoms, 3), dtype=np.float32
    )
    for index in range(n_parents):
        parent = parents[index] * unit.nanometer
        for variant in range(args.variants):
            context.setPositions(parent)
            context.setVelocitiesToTemperature(
                TEMPERATURE_KELVIN * unit.kelvin,
                SEED + index * args.variants + variant,
            )
            integrator.step(args.steps)
            state = context.getState(getPositions=True)
            output[index, variant] = state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer
            )
        if (index + 1) % CHUNK_PARENTS == 0 or index + 1 == n_parents:
            done = index + 1
            elapsed = time.time() - started
            rate = done * args.variants / elapsed
            log(
                f"{done:,}/{n_parents:,} parents "
                f"({done * args.variants:,} frames) "
                f"[{elapsed:.0f} s, {rate:.0f} frames/s, "
                f"eta {(n_parents - done) * args.variants / rate:.0f} s]"
            )
    del context, integrator

    frames = output.reshape(-1, n_atoms, 3)
    if not np.all(np.isfinite(frames)):
        raise ValueError("nonfinite extended frames")
    positions = frames.astype(np.float64)
    cccc = basin_occupancy(dihedral(positions, CCCC_ATOMS))
    occo = basin_occupancy(dihedral(positions, OCCO_ATOMS))
    elapsed = time.time() - started

    np.savez_compressed(FRAMES, positions_nm=frames)
    parent_manifest = json.loads(
        (HERE / "pt_reference_24000.json").read_text(encoding="utf-8")
    )
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
                "variants_per_parent": int(args.variants),
                "extension_steps": int(args.steps),
                "extension_femtoseconds": float(args.steps * TIMESTEP_FS),
                "production_frames": int(frames.shape[0]),
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
        f"C-C-C-C: visited={cccc['visited']}/3 "
        f"occupancy={['%.3f' % v for v in cccc['occupancy']]}"
    )
    log(
        f"O-C-C-O: visited={occo['visited']}/3 "
        f"occupancy={['%.3f' % v for v in occo['occupancy']]}"
    )
    log(f"complete: {frames.shape[0]:,} frames in {elapsed:.0f} s -> {FRAMES.name}")


if __name__ == "__main__":
    main()

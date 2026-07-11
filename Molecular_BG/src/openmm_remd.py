#!/usr/bin/env python
"""Observable, restartable OpenMM REMD for alanine dipeptide.

The sampling schedule follows the FAB paper: 21 temperature slots from 300 K
to 1300 K in 50 K increments, unconstrained dynamics at 1 fs, neighbor
exchanges every 200 steps, and a saved 300 K frame every 1000 steps. This
historical driver deliberately retains ``amber96_obc.xml`` (OBC2), matching the
completed local comparison chains; it is not the exact FAB ff96/OBC1 target.
All quantities needed for analysis/replotting and native OpenMM context
checkpoints are stored in one compressed NPZ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import openmm as mm
from openmm import app, unit


SCHEMA_VERSION = 2
HERE = Path(__file__).resolve().parent
PDB = HERE.parent / "assets" / "system" / "alanine_dipeptide.pdb"
START = HERE.parent / "assets" / "system" / "l_minimum_positions_nm.npy"
PHI = (4, 6, 8, 14)
PSI = (6, 8, 14, 16)
R_GAS = 0.00831446261815324  # kJ mol^-1 K^-1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dihedral(position_nm: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = position_nm[np.asarray(atoms)]
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    b2 /= np.linalg.norm(b2)
    return float(np.arctan2(np.dot(np.cross(n1, n2), b2), np.dot(n1, n2)))


def temperature_ladder(n: int, minimum: float, maximum: float, kind: str) -> np.ndarray:
    if n < 2 or not (0 < minimum < maximum):
        raise ValueError("need at least two replicas and 0 < Tmin < Tmax")
    if kind == "linear":
        return np.linspace(minimum, maximum, n)
    if kind == "geometric":
        return np.geomspace(minimum, maximum, n)
    raise ValueError(f"unknown ladder: {kind}")


def atomic_save(path: Path, **arrays: object) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def make_system(topology: app.Topology, constraints_name: str) -> mm.System:
    forcefield = app.ForceField("amber96.xml", "amber96_obc.xml")
    constraints = None if constraints_name == "none" else app.HBonds
    return forcefield.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=constraints
    )


def make_simulations(
    topology: app.Topology,
    system: mm.System,
    temperatures: np.ndarray,
    timestep_fs: float,
    friction_ps: float,
    platform_name: str,
    seed: int,
) -> list[app.Simulation]:
    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {"Precision": "mixed"} if platform_name == "CUDA" else {}
    simulations = []
    for index, temperature in enumerate(temperatures):
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            friction_ps / unit.picosecond,
            timestep_fs * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(seed + 1009 * (index + 1))
        simulations.append(app.Simulation(topology, system, integrator, platform, properties))
    return simulations


def slot_state(simulations: list[app.Simulation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, velocities, energies = [], [], []
    for simulation in simulations:
        state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        positions.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        velocities.append(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    return np.asarray(positions), np.asarray(velocities), np.asarray(energies)


def context_checkpoints(simulations: list[app.Simulation]) -> np.ndarray:
    raw = [np.frombuffer(simulation.context.createCheckpoint(), dtype=np.uint8) for simulation in simulations]
    lengths = {len(item) for item in raw}
    if len(lengths) != 1:
        raise RuntimeError("OpenMM context checkpoints unexpectedly have unequal lengths")
    return np.stack(raw)


def attempt_exchanges(
    simulations: list[app.Simulation],
    temperatures: np.ndarray,
    rng: np.random.Generator,
    parity: int,
    attempts: np.ndarray,
    accepts: np.ndarray,
    walker_at_slot: np.ndarray,
) -> None:
    positions, _, energies = slot_state(simulations)
    for left in range(parity, len(simulations) - 1, 2):
        right = left + 1
        attempts[left] += 1
        beta_left = 1.0 / (R_GAS * temperatures[left])
        beta_right = 1.0 / (R_GAS * temperatures[right])
        log_acceptance = (beta_left - beta_right) * (energies[left] - energies[right])
        if math.log(rng.random()) < min(0.0, log_acceptance):
            simulations[left].context.setPositions(positions[right] * unit.nanometer)
            simulations[right].context.setPositions(positions[left] * unit.nanometer)
            seed_left = int(rng.integers(1, 2**31 - 1))
            seed_right = int(rng.integers(1, 2**31 - 1))
            simulations[left].context.setVelocitiesToTemperature(
                temperatures[left] * unit.kelvin, seed_left
            )
            simulations[right].context.setVelocitiesToTemperature(
                temperatures[right] * unit.kelvin, seed_right
            )
            walker_at_slot[left], walker_at_slot[right] = (
                walker_at_slot[right], walker_at_slot[left]
            )
            accepts[left] += 1


def write_checkpoint(
    output: Path,
    simulations: list[app.Simulation],
    temperatures: np.ndarray,
    phi: list[float],
    psi: list[float],
    sample_steps: list[int],
    attempts: np.ndarray,
    accepts: np.ndarray,
    completed_steps: int,
    exchange_count: int,
    config_json: str,
    rng: np.random.Generator,
    walker_at_slot: np.ndarray,
    walker_history: list[np.ndarray],
) -> None:
    positions, velocities, energies = slot_state(simulations)
    atomic_save(
        output,
        schema_version=np.asarray(SCHEMA_VERSION),
        angle_units=np.asarray("radians"),
        phi=np.asarray(phi),
        psi=np.asarray(psi),
        sample_steps=np.asarray(sample_steps, dtype=np.int64),
        temperatures_K=temperatures,
        slot_positions_nm=positions,
        slot_velocities_nm_ps=velocities,
        slot_energies_kj_mol=energies,
        context_checkpoints=context_checkpoints(simulations),
        exchange_attempts=attempts,
        exchange_accepts=accepts,
        completed_steps=np.asarray(completed_steps, dtype=np.int64),
        exchange_count=np.asarray(exchange_count, dtype=np.int64),
        walker_at_slot=walker_at_slot,
        walker_history=np.asarray(walker_history, dtype=np.int16),
        rng_state_json=np.asarray(json.dumps(rng.bit_generator.state, sort_keys=True)),
        config_json=np.asarray(config_json),
    )


def validate_args(args: argparse.Namespace) -> None:
    numeric_positive = {
        "ns": args.ns,
        "timestep_fs": args.timestep_fs,
        "friction_ps": args.friction_ps,
        "exchange_steps": args.exchange_steps,
        "sample_steps": args.sample_steps,
        "checkpoint_exchanges": args.checkpoint_exchanges,
    }
    for name, value in numeric_positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent / "data" / "runs" / "adp_remd_v2.npz",
    )
    parser.add_argument("--ns", type=float, default=20.0, help="total time per temperature slot")
    parser.add_argument("--replicas", type=int, default=21)
    parser.add_argument("--tmin", type=float, default=300.0)
    parser.add_argument("--tmax", type=float, default=1300.0)
    parser.add_argument("--ladder", choices=("linear", "geometric"), default="linear")
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction-ps", type=float, default=1.0)
    parser.add_argument("--constraints", choices=("none", "hbonds"), default="none")
    parser.add_argument("--exchange-steps", type=int, default=200)
    parser.add_argument("--sample-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-exchanges", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--platform", choices=("CUDA", "CPU", "Reference"), default="CUDA")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temperatures = temperature_ladder(args.replicas, args.tmin, args.tmax, args.ladder)
    total_steps = round(args.ns * 1_000_000 / args.timestep_fs)
    pdb = app.PDBFile(str(PDB))
    system = make_system(pdb.topology, args.constraints)
    system_xml = mm.XmlSerializer.serialize(system)
    config = {
        "schema_version": SCHEMA_VERSION,
        "replicas": args.replicas,
        "temperatures_K": temperatures.tolist(),
        "ladder": args.ladder,
        "timestep_fs": args.timestep_fs,
        "friction_ps": args.friction_ps,
        "constraints": args.constraints,
        "exchange_steps": args.exchange_steps,
        "sample_steps": args.sample_steps,
        "seed": args.seed,
        "integrator": "LangevinMiddleIntegrator",
        "forcefield": "amber96.xml + amber96_obc.xml; NoCutoff",
        "platform": args.platform,
        "platform_precision": "mixed" if args.platform == "CUDA" else "default",
        "openmm_version": mm.__version__,
        "pdb_sha256": sha256(PDB),
        "start_sha256": sha256(START),
        "system_xml_sha256": hashlib.sha256(system_xml.encode()).hexdigest(),
        "phi_atoms_zero_based": list(PHI),
        "psi_atoms_zero_based": list(PSI),
    }
    config_json = json.dumps(config, sort_keys=True)
    simulations = make_simulations(
        pdb.topology, system, temperatures, args.timestep_fs, args.friction_ps,
        args.platform, args.seed,
    )
    rng = np.random.default_rng(args.seed)
    phi: list[float] = []
    psi: list[float] = []
    sample_steps: list[int] = []
    attempts = np.zeros(args.replicas - 1, dtype=np.int64)
    accepts = np.zeros(args.replicas - 1, dtype=np.int64)
    walker_at_slot = np.arange(args.replicas, dtype=np.int16)
    walker_history: list[np.ndarray] = []
    completed_steps = 0
    exchange_count = 0

    if args.resume and args.output.exists():
        with np.load(args.output) as old:
            if int(old["schema_version"]) != SCHEMA_VERSION:
                raise ValueError("checkpoint schema version mismatch")
            if str(old["config_json"]) != config_json:
                raise ValueError("checkpoint configuration does not match this run")
            checkpoints = old["context_checkpoints"]
            if checkpoints.shape[0] != len(simulations):
                raise ValueError("checkpoint replica count mismatch")
            for simulation, checkpoint in zip(simulations, checkpoints):
                simulation.context.loadCheckpoint(checkpoint.tobytes())
            phi, psi = old["phi"].tolist(), old["psi"].tolist()
            sample_steps = old["sample_steps"].tolist()
            attempts = old["exchange_attempts"].copy()
            accepts = old["exchange_accepts"].copy()
            completed_steps = int(old["completed_steps"])
            exchange_count = int(old["exchange_count"])
            walker_at_slot = old["walker_at_slot"].copy()
            walker_history = [row.copy() for row in old["walker_history"]]
            rng.bit_generator.state = json.loads(str(old["rng_state_json"]))
        print(f"resumed {args.output} at step {completed_steps:,}", flush=True)
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {args.output}")
        start = np.load(START)
        if start.shape != (pdb.topology.getNumAtoms(), 3) or not np.isfinite(start).all():
            raise ValueError("invalid starting coordinates")
        for index, simulation in enumerate(simulations):
            simulation.context.setPositions(start * unit.nanometer)
            if index == 0:
                simulation.minimizeEnergy(maxIterations=500)
                start = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
                start = start.value_in_unit(unit.nanometer)
            else:
                simulation.context.setPositions(start * unit.nanometer)
            simulation.context.setVelocitiesToTemperature(
                temperatures[index] * unit.kelvin, args.seed + index
            )

    if completed_steps > total_steps:
        raise ValueError("requested total duration is shorter than checkpoint duration")
    start_time = time.time()
    last_report = completed_steps
    while completed_steps < total_steps:
        next_exchange = ((completed_steps // args.exchange_steps) + 1) * args.exchange_steps
        next_sample = ((completed_steps // args.sample_steps) + 1) * args.sample_steps
        next_event = min(next_exchange, next_sample, total_steps)
        block = next_event - completed_steps
        for simulation in simulations:
            simulation.step(block)
        completed_steps = next_event

        if completed_steps % args.exchange_steps == 0:
            attempt_exchanges(
                simulations, temperatures, rng, exchange_count % 2,
                attempts, accepts, walker_at_slot,
            )
            exchange_count += 1
            walker_history.append(walker_at_slot.copy())

        if completed_steps % args.sample_steps == 0:
            position = simulations[0].context.getState(getPositions=True).getPositions(asNumpy=True)
            position = position.value_in_unit(unit.nanometer)
            phi.append(dihedral(position, PHI))
            psi.append(dihedral(position, PSI))
            sample_steps.append(completed_steps)

        checkpoint_due = (
            exchange_count > 0
            and exchange_count % args.checkpoint_exchanges == 0
            and completed_steps % args.exchange_steps == 0
        )
        if checkpoint_due or completed_steps == total_steps:
            write_checkpoint(
                args.output, simulations, temperatures, phi, psi, sample_steps,
                attempts, accepts, completed_steps, exchange_count, config_json,
                rng, walker_at_slot, walker_history,
            )
        if checkpoint_due or completed_steps == total_steps or completed_steps - last_report >= 1_000_000:
            elapsed = time.time() - start_time
            rates = np.divide(accepts, attempts, out=np.zeros_like(accepts, dtype=float), where=attempts > 0)
            print(
                f"step {completed_steps:,}/{total_steps:,}; samples={len(phi):,}; "
                f"exchange={rates.min():.3f}..{rates.max():.3f}; elapsed={elapsed:.1f}s",
                flush=True,
            )
            last_report = completed_steps

    print(f"wrote restartable benchmark data to {args.output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Short parallel-tempering smoke tests for the (2R,3R)-2,3-butanediol target.

Screens candidate replica-exchange settings on the *unregularized* physical
potential of the frozen 42D bundle and reports the per-pair exchange
probability for each candidate.  Every run and every output file stays inside
this ``reference`` folder.

Usage from this folder:

    python pt_smoke.py                 # full screen
    python pt_smoke.py --probe         # single short timing probe
    python pt_smoke.py --rounds 100    # override the smoke length
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
LOG = HERE / "pt_smoke.log"
SUMMARY = HERE / "pt_smoke.json"

# Established torsional-reference conventions from the alkane family:
# 0.25 fs unconstrained-bond timestep, 1/ps friction, CUDA platform.
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
PLATFORM = "CUDA"
SEED = 1729


def geometric_grid(low: float, high: float, replicas: int) -> tuple[float, ...]:
    """Return a geometric temperature grid, rounded to whole kelvin."""

    values = low * (high / low) ** (np.arange(replicas) / (replicas - 1))
    return tuple(float(np.round(value)) for value in values)


# Candidate grids.  ``alkane_8`` reproduces the published alkane-family grid
# (300--800 K over eight replicas) and is the reference point for the others.
CANDIDATES = {
    "alkane_8": geometric_grid(300.0, 800.0, 8),
    "wide_10": geometric_grid(300.0, 800.0, 10),
    "wide_6": geometric_grid(300.0, 800.0, 6),
    "wide_5": geometric_grid(300.0, 800.0, 5),
    "wide_4": geometric_grid(300.0, 800.0, 4),
    "narrow_8": geometric_grid(300.0, 600.0, 8),
    "narrow_6": geometric_grid(300.0, 500.0, 6),
}

# Established heavy-atom torsion definitions, taken from the sibling
# ``plot_conformational_landscape.py`` so the diagnostic matches the figures.
CCCC_ATOMS = (4, 2, 3, 5)
OCCO_ATOMS = (0, 2, 3, 1)


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


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_candidate(
    potential: OpenMM_Potential,
    name: str,
    temperatures: tuple[float, ...],
    rounds: int,
    steps_per_round: int,
) -> dict:
    """Run one short replica-exchange smoke test and summarize its exchanges."""

    picoseconds = rounds * steps_per_round * TIMESTEP_FS / 1000.0
    log(
        f"START {name}: {len(temperatures)} replicas "
        f"{temperatures[0]:.0f}-{temperatures[-1]:.0f} K, "
        f"{rounds} rounds x {steps_per_round} steps ({picoseconds:.1f} ps/replica)"
    )
    start = time.time()
    trajectory, energy_history, acceptance = parallel_tempering(
        potential,
        temperatures,
        rounds=rounds,
        steps_per_round=steps_per_round,
        timestep_fs=TIMESTEP_FS,
        friction_per_ps=FRICTION_PER_PS,
        seed=SEED,
        platform=PLATFORM,
    )
    elapsed = time.time() - start

    if not np.all(np.isfinite(energy_history)):
        raise ValueError(f"{name}: nonfinite replica energies")
    cold = energy_history[:, 0]
    cold_positions = trajectory[:, 0]
    cccc = basin_occupancy(dihedral(cold_positions, CCCC_ATOMS))
    occo = basin_occupancy(dihedral(cold_positions, OCCO_ATOMS))
    record = {
        "temperatures_kelvin": list(temperatures),
        "replicas": len(temperatures),
        "rounds": rounds,
        "steps_per_round": steps_per_round,
        "picoseconds_per_replica": picoseconds,
        "acceptance": [float(value) for value in acceptance],
        "acceptance_min": float(acceptance.min()),
        "acceptance_mean": float(acceptance.mean()),
        "acceptance_max": float(acceptance.max()),
        "cold_energy_mean_kj_mol": float(cold.mean()),
        "cold_energy_std_kj_mol": float(cold.std()),
        "seconds": elapsed,
        "frames": int(trajectory.shape[0]),
    }
    pairs = " ".join(f"{value:.3f}" for value in acceptance)
    log(
        f"DONE  {name}: acceptance min={record['acceptance_min']:.3f} "
        f"mean={record['acceptance_mean']:.3f} max={record['acceptance_max']:.3f} "
        f"| pairs {pairs} | cold E={record['cold_energy_mean_kj_mol']:.2f}"
        f"+-{record['cold_energy_std_kj_mol']:.2f} kJ/mol | {elapsed:.1f} s"
    )
    return record


def integrated_time(series: np.ndarray) -> float:
    """Return the integrated autocorrelation time in rounds (initial positive sum)."""

    values = np.asarray(series, dtype=float)
    values = values - values.mean()
    variance = float(values.dot(values) / values.size)
    if variance <= 0.0:
        return float("inf")
    total = 0.0
    for lag in range(1, min(values.size // 4, 2000)):
        rho = float(values[lag:].dot(values[:-lag]) / (values.size * variance))
        if rho <= 0.0:
            break
        total += rho
    return 1.0 + 2.0 * total


def throughput_scan(potential, temperatures, blocks, rounds_budget) -> list[dict]:
    """Time replica-exchange integration at several steps-per-round blocks."""

    records = []
    for steps in blocks:
        rounds = max(4, rounds_budget // steps)
        start = time.time()
        parallel_tempering(
            potential,
            temperatures,
            rounds=rounds,
            steps_per_round=steps,
            timestep_fs=TIMESTEP_FS,
            friction_per_ps=FRICTION_PER_PS,
            seed=SEED,
            platform=PLATFORM,
        )
        elapsed = time.time() - start
        replicas = len(temperatures)
        cold_ps = rounds * steps * TIMESTEP_FS / 1000.0
        records.append({
            "steps_per_round": steps,
            "rounds": rounds,
            "replicas": replicas,
            "seconds": elapsed,
            "steps_per_second": rounds * steps * replicas / elapsed,
            "cold_ps_per_second": cold_ps / elapsed,
        })
        log(
            f"throughput steps/round={steps:5d} replicas={replicas} "
            f"aggregate={records[-1]['steps_per_second']:,.0f} steps/s "
            f"cold={records[-1]['cold_ps_per_second']:.3f} ps/s"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--steps-per-round", type=int, default=800)
    parser.add_argument("--probe", action="store_true", help="one short timing probe")
    parser.add_argument("--throughput", action="store_true", help="time step blocks")
    parser.add_argument("--autocorr", choices=sorted(CANDIDATES))
    parser.add_argument("--candidate", action="append", choices=sorted(CANDIDATES))
    args = parser.parse_args()
    if args.rounds <= 0 or args.steps_per_round <= 0:
        parser.error("rounds and steps per round must be positive")

    potential = OpenMM_Potential.from_bundle(BUNDLE)
    log(
        f"bundle={potential.bundle_name} atoms={potential.n_atoms} "
        f"target={potential.temperature_kelvin:.0f} K unregularized "
        f"timestep={TIMESTEP_FS} fs platform={PLATFORM}"
    )

    if args.probe:
        record = run_candidate(
            potential, "probe", CANDIDATES["alkane_8"], 20, args.steps_per_round
        )
        rate = record["seconds"] / record["rounds"]
        log(f"probe rate={rate:.3f} s/round for 8 replicas")
        return

    if args.throughput:
        for name in ("wide_6", "alkane_8"):
            throughput_scan(
                potential, CANDIDATES[name], (200, 800, 3200, 12800), 400_000
            )
        return

    if args.autocorr:
        temperatures = CANDIDATES[args.autocorr]
        start = time.time()
        trajectory, energy_history, acceptance = parallel_tempering(
            potential,
            temperatures,
            rounds=args.rounds,
            steps_per_round=args.steps_per_round,
            timestep_fs=TIMESTEP_FS,
            friction_per_ps=FRICTION_PER_PS,
            seed=SEED,
            platform=PLATFORM,
        )
        elapsed = time.time() - start
        cold_positions = trajectory[:, 0]
        round_ps = args.steps_per_round * TIMESTEP_FS / 1000.0
        report = {"candidate": args.autocorr, "round_ps": round_ps}
        for label, atoms in (("cccc", CCCC_ATOMS), ("occo", OCCO_ATOMS)):
            angles = dihedral(cold_positions, atoms)
            basins = basin_occupancy(angles)
            tau_cos = integrated_time(np.cos(angles))
            tau_sin = integrated_time(np.sin(angles))
            tau = max(tau_cos, tau_sin)
            report[label] = {
                **basins,
                "tau_rounds": tau,
                "tau_ps": tau * round_ps,
            }
            log(
                f"{label}: visited={basins['visited']}/3 "
                f"transitions={basins['transitions']} "
                f"occupancy={['%.3f' % v for v in basins['occupancy']]} "
                f"tau={tau:.2f} rounds = {tau * round_ps:.3f} ps"
            )
        cold_ps = args.rounds * round_ps
        report["cold_ps"] = cold_ps
        report["cold_ps_per_second"] = cold_ps / elapsed
        report["acceptance_mean"] = float(acceptance.mean())
        report["seconds"] = elapsed
        log(
            f"cold sampled {cold_ps:.1f} ps in {elapsed:.1f} s "
            f"({report['cold_ps_per_second']:.3f} ps/s), "
            f"acceptance mean {report['acceptance_mean']:.3f}"
        )
        (HERE / "pt_autocorr.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return

    selected = args.candidate or sorted(CANDIDATES)
    records = {}
    for name in selected:
        records[name] = run_candidate(
            potential,
            name,
            CANDIDATES[name],
            args.rounds,
            args.steps_per_round,
        )

    SUMMARY.write_text(
        json.dumps(
            {
                "bundle": potential.bundle_name,
                "atoms": potential.n_atoms,
                "potential": "unregularized physical",
                "timestep_fs": TIMESTEP_FS,
                "friction_per_ps": FRICTION_PER_PS,
                "platform": PLATFORM,
                "seed": SEED,
                "candidates": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"summary={SUMMARY}")


if __name__ == "__main__":
    main()

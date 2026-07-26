#!/usr/bin/env python
"""Short parallel-tempering smoke tests for the Ac-Pro-NHMe target.

Screens candidate replica-exchange settings on the *unregularized* physical
potential of the frozen 72D bundle and reports the per-pair exchange
probability for each candidate.  The acceptance target is a minimum pair
probability of at least 0.4.  Every run and every output file stays inside this
``reference`` folder.

At 26 atoms this molecule has a wider energy distribution than the 16-atom
butanediol target, so a grid that mixed well there is expected to need more
replicas here.

Usage from this folder:

    python pt_smoke.py                 # full screen
    python pt_smoke.py --probe         # single short timing probe
    python pt_smoke.py --rounds 200    # override the smoke length
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

# Established torsional-reference conventions: 0.25 fs unconstrained-bond
# timestep, 1/ps friction, CUDA platform.
TIMESTEP_FS = 0.25
FRICTION_PER_PS = 1.0
PLATFORM = "CUDA"
SEED = 1729
ACCEPTANCE_TARGET = 0.4


def geometric_grid(low: float, high: float, replicas: int) -> tuple[float, ...]:
    """Return a geometric temperature grid, rounded to whole kelvin."""

    values = low * (high / low) ** (np.arange(replicas) / (replicas - 1))
    return tuple(float(np.round(value)) for value in values)


# The cis/trans peptide barrier motivates keeping a high top temperature, so
# the candidates vary replica count at a fixed 300--800 K span before trying a
# narrower span.
CANDIDATES = {
    "wide_6": geometric_grid(300.0, 800.0, 6),
    "wide_7": geometric_grid(300.0, 800.0, 7),
    "wide_8": geometric_grid(300.0, 800.0, 8),
    "wide_10": geometric_grid(300.0, 800.0, 10),
    "wide_12": geometric_grid(300.0, 800.0, 12),
    "wide_14": geometric_grid(300.0, 800.0, 14),
    "mid_10": geometric_grid(300.0, 600.0, 10),
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
    replicas = len(temperatures)
    cold_ps = rounds * steps_per_round * TIMESTEP_FS / 1000.0
    record = {
        "temperatures_kelvin": list(temperatures),
        "replicas": replicas,
        "rounds": rounds,
        "steps_per_round": steps_per_round,
        "picoseconds_per_replica": picoseconds,
        "acceptance": [float(value) for value in acceptance],
        "acceptance_min": float(acceptance.min()),
        "acceptance_mean": float(acceptance.mean()),
        "acceptance_max": float(acceptance.max()),
        "meets_target": bool(acceptance.min() >= ACCEPTANCE_TARGET),
        "cold_energy_mean_kj_mol": float(cold.mean()),
        "cold_energy_std_kj_mol": float(cold.std()),
        "seconds": elapsed,
        "steps_per_second": rounds * steps_per_round * replicas / elapsed,
        "cold_ps_per_second": cold_ps / elapsed,
    }
    pairs = " ".join(f"{value:.3f}" for value in acceptance)
    log(
        f"DONE  {name}: min={record['acceptance_min']:.3f} "
        f"mean={record['acceptance_mean']:.3f} "
        f"{'PASS' if record['meets_target'] else 'FAIL'} "
        f"| pairs {pairs} | cold={record['cold_ps_per_second']:.3f} ps/s "
        f"| {elapsed:.1f} s"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--steps-per-round", type=int, default=800)
    parser.add_argument("--probe", action="store_true", help="one short timing probe")
    parser.add_argument("--candidate", action="append", choices=sorted(CANDIDATES))
    args = parser.parse_args()
    if args.rounds <= 0 or args.steps_per_round <= 0:
        parser.error("rounds and steps per round must be positive")

    potential = OpenMM_Potential.from_bundle(BUNDLE)
    log(
        f"bundle={potential.bundle_name} atoms={potential.n_atoms} "
        f"target={potential.temperature_kelvin:.0f} K unregularized "
        f"timestep={TIMESTEP_FS} fs platform={PLATFORM} "
        f"acceptance target min>={ACCEPTANCE_TARGET}"
    )

    if args.probe:
        record = run_candidate(
            potential, "probe", CANDIDATES["wide_8"], 20, args.steps_per_round
        )
        log(f"probe rate={record['seconds'] / record['rounds']:.3f} s/round")
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
                "acceptance_target_min": ACCEPTANCE_TARGET,
                "candidates": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    passing = [name for name, r in records.items() if r["meets_target"]]
    log(f"candidates meeting min>={ACCEPTANCE_TARGET}: {passing or 'none'}")
    log(f"summary={SUMMARY}")


if __name__ == "__main__":
    main()

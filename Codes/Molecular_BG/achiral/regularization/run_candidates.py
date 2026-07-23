#!/usr/bin/env python
"""Run provenance-preserving candidate regularization trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import openmm as mm

from jflows_md.openmm import OpenMM_Potential, parallel_tempering
from jflows_md.system import Molecular_Bundle

from run import (
    CONFIG_PATH,
    ROOT,
    atomic_save,
    bundle_hashes,
    bundle_path,
    condition_label,
    load_config,
    sha256,
    starts,
)


CANDIDATE_CONFIG_PATH = ROOT / "candidate_config.json"


def load_candidate_config() -> dict:
    return json.loads(CANDIDATE_CONFIG_PATH.read_text(encoding="utf-8"))


def settings(candidate_config: dict, phase: str) -> dict:
    return candidate_config[phase]


def candidate_path(phase: str, molecule: str, seed: int, rg_param) -> Path:
    name = f"{molecule}__{condition_label(rg_param)}__seed{seed}.npz"
    return ROOT / "data" / f"candidate_{phase}" / name


def candidate_metadata(base: dict, candidates: dict, phase: str, molecule: str, seed: int, rg_param) -> dict:
    run = settings(candidates, phase)
    return {
        "schema_version": 1,
        "study": "achiral_regularization_candidate",
        "phase": phase,
        "molecule": molecule,
        "dimension": base["molecules"][molecule]["dimension"],
        "seed": seed,
        "rg_param": list(map(float, rg_param)),
        "rounds": run["rounds"],
        "burnin_rounds": run["burnin_rounds"],
        "temperatures_kelvin": base["temperatures_kelvin"],
        "steps_per_round": run["steps_per_round"],
        "timestep_fs": base["timestep_fs"],
        "friction_per_ps": base["friction_per_ps"],
        "platform": base["platform"],
        "precision": base["precision"],
        "base_config_sha256": sha256(CONFIG_PATH),
        "candidate_config_sha256": sha256(CANDIDATE_CONFIG_PATH),
        "bundle_sha256": bundle_hashes(base, molecule),
        "jflows_version": version("jflows"),
        "jflows_md_version": version("jflows_md"),
        "openmm_version": mm.version.full_version,
    }


def complete(path: Path, expected: dict, n_atoms: int) -> bool:
    if not path.is_file():
        return False
    replicas = len(expected["temperatures_kelvin"])
    try:
        with np.load(path, allow_pickle=False) as data:
            return bool(
                json.loads(str(data["metadata"])) == expected
                and data["positions_nm"].shape == (expected["rounds"], replicas, n_atoms, 3)
                and data["energies_kj_mol"].shape == (expected["rounds"], replicas)
                and data["swap_acceptance"].shape == (replicas - 1,)
                and np.isfinite(data["positions_nm"]).all()
                and np.isfinite(data["energies_kj_mol"]).all()
                and np.isfinite(data["swap_acceptance"]).all()
                and np.all((data["swap_acceptance"] >= 0.0) & (data["swap_acceptance"] <= 1.0))
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_one(base: dict, candidates: dict, phase: str, molecule: str, seed: int, rg_param) -> None:
    expected = candidate_metadata(base, candidates, phase, molecule, seed, rg_param)
    destination = candidate_path(phase, molecule, seed, rg_param)
    bundle = Molecular_Bundle.load(bundle_path(base, molecule))
    if complete(destination, expected, bundle.n_atoms):
        print(f"SKIP {destination.relative_to(ROOT)}", flush=True)
        return
    physical = OpenMM_Potential.from_bundle(bundle_path(base, molecule))
    potential = physical.regularized(rg_param)
    initial = starts(base, "production", molecule, seed)
    started = time.time()
    print(
        f"START candidate-{phase} {molecule} {condition_label(rg_param)} seed={seed} "
        f"rounds={expected['rounds']}",
        flush=True,
    )
    positions, energies, acceptance = parallel_tempering(
        potential,
        base["temperatures_kelvin"],
        initial,
        rounds=expected["rounds"],
        steps_per_round=expected["steps_per_round"],
        timestep_fs=base["timestep_fs"],
        friction_per_ps=base["friction_per_ps"],
        seed=seed,
        platform=base["platform"],
    )
    if not (np.isfinite(positions).all() and np.isfinite(energies).all()):
        raise FloatingPointError(f"nonfinite candidate output for {molecule} {rg_param}")
    atomic_save(
        destination,
        metadata=json.dumps(expected, sort_keys=True),
        positions_nm=positions,
        energies_kj_mol=energies,
        swap_acceptance=acceptance,
    )
    print(
        f"DONE  {destination.relative_to(ROOT)} seconds={time.time() - started:.1f} "
        f"acceptance={np.array2string(acceptance, precision=3)}",
        flush=True,
    )
    del positions, energies, acceptance, potential, physical, initial
    gc.collect()


def requested_pairs(candidates: dict, phase: str, molecule: str, e, r):
    available = [tuple(map(float, pair)) for pair in candidates["candidates"][molecule]]
    if e is None and r is None:
        if phase == "verification":
            raise ValueError("verification requires --e and --r")
        return available
    if e is None or r is None:
        raise ValueError("--e and --r must be supplied together")
    selected = (float(e), float(r))
    if selected not in available:
        raise ValueError(f"{selected} is not a frozen candidate for {molecule}")
    return [selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pilot", "verification"))
    parser.add_argument("--molecule", choices=("glycerol", "diethanolamine"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--e", type=float)
    parser.add_argument("--r", type=float)
    args = parser.parse_args()
    base = load_config()
    candidates = load_candidate_config()
    if sha256(CONFIG_PATH) != candidates["base_config_sha256"]:
        parser.error("base config hash no longer matches candidate_config.json")
    run = settings(candidates, args.phase)
    if args.seed is not None and args.seed not in run["seeds"]:
        parser.error(f"seed must be one of {run['seeds']}")
    platform = mm.Platform.getPlatformByName(base["platform"])
    platform.setPropertyDefaultValue("Precision", base["precision"])
    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"OpenMM={mm.version.full_version} platform={base['platform']} "
        f"precision={platform.getPropertyDefaultValue('Precision')}",
        flush=True,
    )
    molecules = list(candidates["candidates"]) if args.molecule is None else [args.molecule]
    seeds = run["seeds"] if args.seed is None else [args.seed]
    try:
        for molecule in molecules:
            for rg_param in requested_pairs(candidates, args.phase, molecule, args.e, args.r):
                for seed in seeds:
                    run_one(base, candidates, args.phase, molecule, seed, rg_param)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run resumable native-OpenMM sampling for the alkane regularization study."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import openmm as mm

from jflows_md.openmm import OpenMM_Potential, langevin, parallel_tempering
from jflows_md.system import Molecular_Bundle


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def bundle_path(config, molecule):
    return (ROOT / config["molecules"][molecule]["bundle"]).resolve()


def bundle_hashes(config, molecule):
    bundle = bundle_path(config, molecule)
    return {
        name: sha256(bundle / name)
        for name in (
            "coordinates.json",
            "manifest.json",
            "reference.pdb",
            "system.json",
            "system.xml",
            "validation.json",
        )
    }


def label(rg_param):
    if rg_param is None:
        return "raw"
    e, r = rg_param
    return f"e{e:g}_r{r:g}".replace(".", "p")


def metadata(config, phase, molecule, seed, rg_param, rounds, steps_per_round, burnin):
    bundle = bundle_path(config, molecule)
    return {
        "schema_version": 1,
        "phase": phase,
        "molecule": molecule,
        "dimension": config["molecules"][molecule]["dimension"],
        "carbon_count": config["molecules"][molecule]["carbon_count"],
        "seed": seed,
        "rg_param": None if rg_param is None else list(rg_param),
        "rounds": rounds,
        "burnin_rounds": burnin,
        "temperatures_kelvin": config["temperatures_kelvin"],
        "steps_per_round": steps_per_round,
        "timestep_fs": config["timestep_fs"],
        "friction_per_ps": config["friction_per_ps"],
        "platform": config["platform"],
        "precision": config["precision"],
        "config_sha256": sha256(CONFIG_PATH),
        "bundle_sha256": bundle_hashes(config, molecule),
        "openmm_version": mm.version.full_version,
    }


def complete(path, expected, n_atoms, replicas):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            actual = json.loads(str(data["metadata"]))
            positions = data["positions_nm"]
            energies = data["energies_kj_mol"]
            acceptance = data["swap_acceptance"]
            return (
                actual == expected
                and positions.shape == (expected["rounds"], replicas, n_atoms, 3)
                and energies.shape == (expected["rounds"], replicas)
                and acceptance.shape == (replicas - 1,)
                and np.isfinite(positions).all()
                and np.isfinite(energies).all()
                and np.isfinite(acceptance).all()
                and np.all((acceptance >= 0) & (acceptance <= 1))
            )
    except Exception:
        return False


def save(path, expected, positions, energies, acceptance):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=json.dumps(expected, sort_keys=True),
            positions_nm=positions,
            energies_kj_mol=energies,
            swap_acceptance=acceptance,
        )
    temporary.replace(path)


def start_path(phase, molecule, seed):
    return ROOT / "data" / phase / "starts" / f"{molecule}__seed{seed}.npz"


def starts(config, phase, molecule, seed, *, sanity=False):
    path = start_path(phase, molecule, seed)
    bundle = bundle_path(config, molecule)
    expected = {
        "schema_version": 1,
        "molecule": molecule,
        "seed": seed,
        "config_sha256": sha256(CONFIG_PATH),
        "bundle_sha256": bundle_hashes(config, molecule),
        "temperature_kelvin": config["seed_temperature_kelvin"],
        "steps": 800 if sanity else config["seed_steps"],
        "sample_interval": 100 if sanity else config["seed_sample_interval"],
    }
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                positions = data["positions_nm"]
                energies = data["energies_kj_mol"]
                replicas = len(config["temperatures_kelvin"])
                n_atoms = Molecular_Bundle.load(bundle).n_atoms
                if (
                    json.loads(str(data["metadata"])) == expected
                    and positions.shape == (replicas, n_atoms, 3)
                    and energies.shape == (replicas,)
                    and np.isfinite(positions).all()
                    and np.isfinite(energies).all()
                ):
                    return np.asarray(positions)
        except Exception:
            pass
    physical = OpenMM_Potential.from_bundle(bundle)
    trajectory, energies = langevin(
        physical,
        steps=expected["steps"],
        sample_interval=expected["sample_interval"],
        timestep_fs=config["timestep_fs"],
        friction_per_ps=config["friction_per_ps"],
        temperature_kelvin=config["seed_temperature_kelvin"],
        seed=seed + 100000,
        platform=config["platform"],
    )
    if len(trajectory) != len(config["temperatures_kelvin"]):
        raise ValueError("seeding trajectory did not produce one start per replica")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=json.dumps(expected, sort_keys=True),
            positions_nm=trajectory,
            energies_kj_mol=energies,
        )
    temporary.replace(path)
    return trajectory


def run_one(
    config,
    phase,
    molecule,
    seed,
    rg_param,
    *,
    sanity=False,
    rounds_override=None,
    start_phase=None,
):
    rounds = 20 if sanity else (rounds_override or config["rounds"])
    burnin = 0 if sanity else config["burnin_rounds"]
    steps_per_round = 10 if sanity else config["steps_per_round"]
    expected = metadata(
        config, phase, molecule, seed, rg_param, rounds, steps_per_round, burnin
    )
    destination = ROOT / "data" / phase / f"{molecule}__{label(rg_param)}__seed{seed}.npz"
    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    if complete(destination, expected, bundle.n_atoms, len(config["temperatures_kelvin"])):
        print(f"SKIP {destination.relative_to(ROOT)}", flush=True)
        return

    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    potential = physical if rg_param is None else physical.regularized(rg_param)
    initial = starts(config, start_phase or phase, molecule, seed, sanity=sanity)
    started = time.time()
    print(
        f"START {phase} {molecule} {label(rg_param)} seed={seed} rounds={rounds}",
        flush=True,
    )
    positions, energies, acceptance = parallel_tempering(
        potential,
        config["temperatures_kelvin"],
        initial,
        rounds=rounds,
        steps_per_round=steps_per_round,
        timestep_fs=config["timestep_fs"],
        friction_per_ps=config["friction_per_ps"],
        seed=seed,
        platform=config["platform"],
    )
    if not np.isfinite(positions).all() or not np.isfinite(energies).all():
        raise FloatingPointError(f"nonfinite OpenMM output for {molecule} {label(rg_param)}")
    save(destination, expected, positions, energies, acceptance)
    print(
        f"DONE  {destination.relative_to(ROOT)} seconds={time.time() - started:.1f} "
        f"acceptance={np.array2string(acceptance, precision=3)}",
        flush=True,
    )
    del positions, energies, acceptance, potential, physical, initial
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("sanity", "pilot", "candidate", "verify", "extend")
    )
    parser.add_argument("--e", type=float)
    parser.add_argument("--r", type=float)
    parser.add_argument("--base", choices=("pilot", "verification"))
    parser.add_argument("--molecule", choices=("methane", "ethane", "propane", "butane", "pentane", "hexane"))
    args = parser.parse_args()
    config = load_config()
    platform = mm.Platform.getPlatformByName(config["platform"])
    platform.setPropertyDefaultValue("Precision", config["precision"])
    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"OpenMM={mm.version.full_version} platform={config['platform']} "
        f"precision={platform.getPropertyDefaultValue('Precision')}",
        flush=True,
    )

    if args.phase == "sanity":
        run_one(config, "sanity", "hexane", config["seeds"][0], None, sanity=True)
        run_one(config, "sanity", "hexane", config["seeds"][0], (100.0, 0.1), sanity=True)
    elif args.phase == "pilot":
        for seed in config["seeds"]:
            run_one(config, "pilot", "hexane", seed, None)
        for e in config["candidate_e_kj_mol"]:
            for r in config["candidate_r_nm"]:
                for seed in config["seeds"]:
                    run_one(config, "pilot", "hexane", seed, (e, r))
    elif args.phase == "candidate":
        if args.e is None or args.r is None:
            parser.error("candidate requires --e and --r")
        for seed in config["seeds"]:
            run_one(config, "pilot", "hexane", seed, (args.e, args.r))
    elif args.phase == "verify":
        if args.e is None or args.r is None:
            parser.error("verify requires --e and --r")
        for molecule in config["molecules"]:
            for seed in config["seeds"]:
                run_one(config, "verification", molecule, seed, None)
            for seed in config["seeds"]:
                run_one(config, "verification", molecule, seed, (args.e, args.r))
    else:
        if args.base is None or args.e is None or args.r is None:
            parser.error("extend requires --base, --e, and --r")
        molecule = "hexane" if args.base == "pilot" else args.molecule
        if molecule is None:
            parser.error("verification extension requires --molecule")
        phase = f"{args.base}_extended"
        for seed in config["seeds"]:
            run_one(
                config,
                phase,
                molecule,
                seed,
                None,
                rounds_override=config["extension_rounds"],
                start_phase=args.base,
            )
        for seed in config["seeds"]:
            run_one(
                config,
                phase,
                molecule,
                seed,
                (args.e, args.r),
                rounds_override=config["extension_rounds"],
                start_phase=args.base,
            )


if __name__ == "__main__":
    main()

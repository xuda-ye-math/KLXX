#!/usr/bin/env python
"""Run resumable raw/regularized OpenMM sampling for three achiral molecules."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import openmm as mm

from jflows_md.openmm import OpenMM_Potential, langevin, parallel_tempering
from jflows_md.system import Molecular_Bundle


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
BUNDLE_FILES = (
    "coordinates.json",
    "manifest.json",
    "reference.pdb",
    "system.json",
    "system.xml",
    "validation.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def bundle_path(config: dict, molecule: str) -> Path:
    return Path(config["bundle_root"]) / config["molecules"][molecule]["bundle"]


def bundle_hashes(config: dict, molecule: str) -> dict[str, str]:
    path = bundle_path(config, molecule)
    return {name: sha256(path / name) for name in BUNDLE_FILES}


def regularization(config: dict) -> tuple[float, float]:
    return tuple(map(float, config["regularization"]))


def condition_label(rg_param: tuple[float, float] | None) -> str:
    if rg_param is None:
        return "raw"
    e, r = rg_param
    return f"e{e:g}_r{r:g}".replace(".", "p")


def phase_settings(config: dict, phase: str) -> tuple[int, int, int]:
    if phase == "sanity":
        return 20, 10, 0
    rounds = config["extension_rounds"] if phase == "extended" else config["rounds"]
    return int(rounds), int(config["steps_per_round"]), int(config["burnin_rounds"])


def run_metadata(
    config: dict,
    phase: str,
    molecule: str,
    seed: int,
    rg_param: tuple[float, float] | None,
) -> dict:
    rounds, steps_per_round, burnin = phase_settings(config, phase)
    return {
        "schema_version": 1,
        "phase": phase,
        "molecule": molecule,
        "dimension": config["molecules"][molecule]["dimension"],
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
        "jflows_version": version("jflows"),
        "jflows_md_version": version("jflows_md"),
        "openmm_version": mm.version.full_version,
    }


def artifact_path(phase: str, molecule: str, seed: int, rg_param) -> Path:
    name = f"{molecule}__{condition_label(rg_param)}__seed{seed}.npz"
    return ROOT / "data" / phase / name


def artifact_complete(path: Path, expected: dict, n_atoms: int) -> bool:
    if not path.is_file():
        return False
    replicas = len(expected["temperatures_kelvin"])
    try:
        with np.load(path, allow_pickle=False) as data:
            actual = json.loads(str(data["metadata"]))
            positions = data["positions_nm"]
            energies = data["energies_kj_mol"]
            acceptance = data["swap_acceptance"]
            return bool(
                actual == expected
                and positions.shape == (expected["rounds"], replicas, n_atoms, 3)
                and energies.shape == (expected["rounds"], replicas)
                and acceptance.shape == (replicas - 1,)
                and np.isfinite(positions).all()
                and np.isfinite(energies).all()
                and np.isfinite(acceptance).all()
                and np.all((acceptance >= 0.0) & (acceptance <= 1.0))
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def atomic_save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def start_phase(phase: str) -> str:
    return "sanity" if phase == "sanity" else "production"


def starts(config: dict, phase: str, molecule: str, seed: int) -> np.ndarray:
    source_phase = start_phase(phase)
    path = ROOT / "data" / source_phase / "starts" / f"{molecule}__seed{seed}.npz"
    is_sanity = phase == "sanity"
    steps = 800 if is_sanity else int(config["seed_steps"])
    interval = 100 if is_sanity else int(config["seed_sample_interval"])
    expected = {
        "schema_version": 1,
        "molecule": molecule,
        "seed": seed,
        "temperature_kelvin": config["seed_temperature_kelvin"],
        "steps": steps,
        "sample_interval": interval,
        "config_sha256": sha256(CONFIG_PATH),
        "bundle_sha256": bundle_hashes(config, molecule),
    }
    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    replicas = len(config["temperatures_kelvin"])
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                positions = np.asarray(data["positions_nm"])
                energies = np.asarray(data["energies_kj_mol"])
                if (
                    json.loads(str(data["metadata"])) == expected
                    and positions.shape == (replicas, bundle.n_atoms, 3)
                    and energies.shape == (replicas,)
                    and np.isfinite(positions).all()
                    and np.isfinite(energies).all()
                ):
                    return positions
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    trajectory, energies = langevin(
        physical,
        steps=steps,
        sample_interval=interval,
        timestep_fs=config["timestep_fs"],
        friction_per_ps=config["friction_per_ps"],
        temperature_kelvin=config["seed_temperature_kelvin"],
        seed=seed + 100000,
        platform=config["platform"],
    )
    if trajectory.shape != (replicas, bundle.n_atoms, 3):
        raise ValueError(f"seeding produced {trajectory.shape}, expected {(replicas, bundle.n_atoms, 3)}")
    atomic_save(
        path,
        metadata=json.dumps(expected, sort_keys=True),
        positions_nm=trajectory,
        energies_kj_mol=energies,
    )
    return trajectory


def run_one(config: dict, phase: str, molecule: str, seed: int, rg_param) -> None:
    expected = run_metadata(config, phase, molecule, seed, rg_param)
    destination = artifact_path(phase, molecule, seed, rg_param)
    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    if artifact_complete(destination, expected, bundle.n_atoms):
        print(f"SKIP {destination.relative_to(ROOT)}", flush=True)
        return
    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    potential = physical if rg_param is None else physical.regularized(rg_param)
    initial = starts(config, phase, molecule, seed)
    started = time.time()
    print(
        f"START {phase} {molecule} {condition_label(rg_param)} seed={seed} "
        f"rounds={expected['rounds']}",
        flush=True,
    )
    positions, energies, acceptance = parallel_tempering(
        potential,
        config["temperatures_kelvin"],
        initial,
        rounds=expected["rounds"],
        steps_per_round=expected["steps_per_round"],
        timestep_fs=config["timestep_fs"],
        friction_per_ps=config["friction_per_ps"],
        seed=seed,
        platform=config["platform"],
    )
    if not (np.isfinite(positions).all() and np.isfinite(energies).all()):
        raise FloatingPointError(f"nonfinite output for {molecule} {condition_label(rg_param)}")
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


def selected(values, requested):
    return values if requested is None else [requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("sanity", "production", "extended"))
    parser.add_argument("--molecule")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition", choices=("raw", "regularized"))
    args = parser.parse_args()
    config = load_config()
    if args.molecule is not None and args.molecule not in config["molecules"]:
        parser.error(f"unknown molecule: {args.molecule}")
    if args.seed is not None and args.seed not in config["seeds"]:
        parser.error(f"seed must be one of {config['seeds']}")
    platform = mm.Platform.getPlatformByName(config["platform"])
    platform.setPropertyDefaultValue("Precision", config["precision"])
    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"OpenMM={mm.version.full_version} platform={config['platform']} "
        f"precision={platform.getPropertyDefaultValue('Precision')}",
        flush=True,
    )
    molecules = selected(list(config["molecules"]), args.molecule)
    seeds = selected(config["seeds"], args.seed)
    conditions = [None, regularization(config)]
    if args.condition is not None:
        conditions = [None if args.condition == "raw" else regularization(config)]
    if args.phase == "sanity" and args.seed is None:
        seeds = [config["seeds"][0]]
    for molecule in molecules:
        for seed in seeds:
            for rg_param in conditions:
                run_one(config, args.phase, molecule, seed, rg_param)


if __name__ == "__main__":
    main()

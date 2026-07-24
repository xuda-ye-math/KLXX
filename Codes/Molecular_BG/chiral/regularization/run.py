#!/usr/bin/env python
"""Run traced, stereochemistry-audited chiral regularization trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import time

import jflows_md
import numpy as np
import openmm as mm
from openmm import unit

from jflows_md.openmm import OpenMM_Potential, langevin
from jflows_md.system import Molecular_Bundle


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
KB_KJ_MOL_K = 0.00831446261815324
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


def verified_jflows_md_source(config: dict) -> dict:
    source_root = Path(jflows_md.__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != config["jflows_md_commit"]:
        raise ValueError(
            f"live jflows_md commit {commit} does not match frozen "
            f"{config['jflows_md_commit']}"
        )
    if status:
        raise ValueError(
            f"live jflows_md worktree is dirty, so commit provenance is insufficient: {status}"
        )
    return {
        "root": str(source_root),
        "commit": commit,
        "worktree_clean": True,
    }


def bundle_path(config: dict, molecule: str) -> Path:
    return Path(config["bundle_root"]) / config["molecules"][molecule]["bundle"]


def bundle_hashes(config: dict, molecule: str) -> dict[str, str]:
    path = bundle_path(config, molecule)
    return {name: sha256(path / name) for name in BUNDLE_FILES}


def condition_label(rg_param: tuple[float, float] | None) -> str:
    if rg_param is None:
        return "raw"
    e, r = rg_param
    return f"e{e:g}_r{r:g}".replace(".", "p")


def candidate_order(config: dict) -> list[tuple[float, float]]:
    return [tuple(map(float, pair)) for pair in config["candidate_order"]]


def tail_exponent(config: dict, rg_param: tuple[float, float]) -> float:
    hottest = max(map(float, config["temperatures_kelvin"]))
    return 2.0 * rg_param[0] / (KB_KJ_MOL_K * hottest)


def candidate_eligible(config: dict, molecule: str, rg_param: tuple[float, float]) -> bool:
    if rg_param not in candidate_order(config):
        return False
    default_e, default_r = candidate_order(config)[0]
    if rg_param[0] < default_e or rg_param[1] > default_r:
        return False
    dimension = int(config["molecules"][molecule]["dimension"])
    return bool(tail_exponent(config, rg_param) > dimension + config["tail_moment_order"])


def normalized_stereocenters(bundle: Molecular_Bundle) -> list[dict]:
    coordinates = bundle.coordinates
    if "fixed_stereocenters" in coordinates:
        source = coordinates["fixed_stereocenters"]
    elif int(coordinates.get("chirality_sign", 0)) != 0:
        source = [
            {
                "label": "alanine_ca_L",
                "atoms": coordinates["chirality_atoms"],
                "volume_sign": coordinates["chirality_sign"],
            }
        ]
    else:
        source = []
    return [
        {
            "label": str(item["label"]),
            "atoms": [int(value) for value in item["atoms"]],
            "volume_sign": int(item["volume_sign"]),
        }
        for item in source
    ]


def checked_stereocenters(config: dict, molecule: str, bundle: Molecular_Bundle) -> list[dict]:
    actual = normalized_stereocenters(bundle)
    expected = config["molecules"][molecule]["stereocenters"]
    if actual != expected:
        raise ValueError(
            f"bundle stereocenters for {molecule} do not match frozen config: "
            f"actual={actual}, expected={expected}"
        )
    return actual


def signed_volume(positions: np.ndarray, atoms) -> np.ndarray:
    center, first, second, third = map(int, atoms)
    a = positions[..., first, :] - positions[..., center, :]
    b = positions[..., second, :] - positions[..., center, :]
    c = positions[..., third, :] - positions[..., center, :]
    return np.sum(a * np.cross(b, c), axis=-1)


def stereo_margins(positions: np.ndarray, centers: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            float(np.min(item["volume_sign"] * signed_volume(positions, item["atoms"])))
            for item in centers
        ],
        dtype=float,
    )


def require_stereochemistry(
    positions: np.ndarray,
    centers: list[dict],
    tolerance_nm3: float,
    scope: str,
) -> np.ndarray:
    margins = stereo_margins(positions, centers)
    if not np.isfinite(margins).all() or np.any(margins <= tolerance_nm3):
        details = {item["label"]: float(value) for item, value in zip(centers, margins, strict=True)}
        raise ValueError(
            f"stereochemistry check failed for {scope}; minimum signed-volume "
            f"margins (nm^3)={details}, required>{tolerance_nm3}"
        )
    return margins


def _context(potential, temperature, friction, timestep, seed, platform):
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        timestep * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(seed))
    selected = mm.Platform.getPlatformByName(platform)
    context = mm.Context(potential.create_system(), integrator, selected)
    return context, integrator


def _state(context):
    state = context.getState(getPositions=True, getEnergy=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return np.asarray(positions), float(energy)


def parallel_tempering_traced(
    potential,
    temperatures_kelvin,
    positions_nm,
    *,
    rounds: int,
    steps_per_round: int,
    timestep_fs: float,
    friction_per_ps: float,
    seed: int,
    platform: str,
    progress_label: str,
):
    """Match jflows_md replica exchange while recording walker identities."""
    temperatures = np.asarray(temperatures_kelvin, dtype=float)
    replicas = len(temperatures)
    initial = np.asarray(positions_nm, dtype=float)
    if initial.ndim == 2:
        initial = np.repeat(initial[None], replicas, axis=0)
    if initial.shape[0] != replicas:
        raise ValueError(f"expected {replicas} starts, got shape {initial.shape}")

    contexts = []
    integrators = []
    try:
        for index, temperature in enumerate(temperatures):
            context, integrator = _context(
                potential,
                temperature,
                friction_per_ps,
                timestep_fs,
                seed + index,
                platform,
            )
            context.setPositions(initial[index] * unit.nanometer)
            context.setVelocitiesToTemperature(
                temperature * unit.kelvin, seed + replicas + index
            )
            contexts.append(context)
            integrators.append(integrator)

        rng = np.random.default_rng(seed)
        attempts = np.zeros(replicas - 1, dtype=int)
        accepted = np.zeros(replicas - 1, dtype=int)
        walker = np.arange(replicas, dtype=int)
        walker_history = [walker.copy()]
        trajectory = []
        energy_history = []
        beta = 1.0 / (KB_KJ_MOL_K * temperatures)
        progress_interval = max(1, rounds // 20)

        for round_index in range(rounds):
            for integrator in integrators:
                integrator.step(steps_per_round)
            states = [_state(context) for context in contexts]
            positions = [state[0] for state in states]
            energies = np.asarray([state[1] for state in states])
            for left in range(round_index % 2, replicas - 1, 2):
                right = left + 1
                attempts[left] += 1
                log_acceptance = (beta[left] - beta[right]) * (
                    energies[left] - energies[right]
                )
                if np.log(rng.random()) < min(0.0, log_acceptance):
                    positions[left], positions[right] = positions[right], positions[left]
                    energies[left], energies[right] = energies[right], energies[left]
                    walker[left], walker[right] = walker[right], walker[left]
                    contexts[left].setPositions(positions[left] * unit.nanometer)
                    contexts[right].setPositions(positions[right] * unit.nanometer)
                    accepted[left] += 1
            trajectory.append(np.asarray(positions))
            energy_history.append(energies.copy())
            walker_history.append(walker.copy())
            completed = round_index + 1
            if completed == rounds or completed % progress_interval == 0:
                print(
                    f"PROGRESS {progress_label} round={completed}/{rounds}",
                    flush=True,
                )
    finally:
        contexts.clear()
        integrators.clear()

    acceptance = np.divide(
        accepted,
        attempts,
        out=np.zeros_like(accepted, dtype=float),
        where=attempts > 0,
    )
    return (
        np.asarray(trajectory),
        np.asarray(energy_history),
        acceptance,
        np.asarray(walker_history),
    )


def atomic_save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def start_metadata(config: dict, molecule: str, seed: int, sanity: bool) -> dict:
    source = verified_jflows_md_source(config)
    return {
        "schema_version": 1,
        "study": "chiral_regularization_starts",
        "kind": "sanity" if sanity else "full",
        "molecule": molecule,
        "seed": int(seed),
        "temperature_kelvin": float(config["start_temperature_kelvin"]),
        "steps": int(config["sanity_start_steps"] if sanity else config["start_steps"]),
        "sample_interval": int(
            config["sanity_start_sample_interval"]
            if sanity
            else config["start_sample_interval"]
        ),
        "config_sha256": sha256(CONFIG_PATH),
        "bundle_sha256": bundle_hashes(config, molecule),
        "jflows_md_source_root": source["root"],
        "jflows_md_commit_observed": source["commit"],
        "jflows_md_worktree_clean": source["worktree_clean"],
    }


def start_path(molecule: str, seed: int, sanity: bool) -> Path:
    kind = "sanity" if sanity else "full"
    return ROOT / "data" / "starts" / kind / f"{molecule}__seed{seed}.npz"


def starts(config: dict, phase: str, molecule: str, seed: int) -> np.ndarray:
    sanity = phase == "sanity"
    expected = start_metadata(config, molecule, seed, sanity)
    path = start_path(molecule, seed, sanity)
    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    centers = checked_stereocenters(config, molecule, bundle)
    replicas = len(config["temperatures_kelvin"])
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                positions = np.asarray(data["positions_nm"])
                energies = np.asarray(data["energies_kj_mol"])
                metadata = json.loads(str(data["metadata"]))
            if (
                metadata == expected
                and positions.shape == (replicas, bundle.n_atoms, 3)
                and energies.shape == (replicas,)
                and np.isfinite(positions).all()
                and np.isfinite(energies).all()
            ):
                require_stereochemistry(
                    positions,
                    centers,
                    config["stereo_min_margin_nm3"],
                    f"cached starts {molecule} seed {seed}",
                )
                return positions
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing start artifact {path}: {exc}") from exc
        raise ValueError(f"stale or invalid existing start artifact: {path}")

    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    trajectory, energies = langevin(
        physical,
        steps=expected["steps"],
        sample_interval=expected["sample_interval"],
        timestep_fs=config["timestep_fs"],
        friction_per_ps=config["friction_per_ps"],
        temperature_kelvin=expected["temperature_kelvin"],
        seed=seed + 100000,
        platform=config["platform"],
    )
    if trajectory.shape != (replicas, bundle.n_atoms, 3):
        raise ValueError(
            f"seeding produced {trajectory.shape}, expected {(replicas, bundle.n_atoms, 3)}"
        )
    margins = require_stereochemistry(
        trajectory,
        centers,
        config["stereo_min_margin_nm3"],
        f"new starts {molecule} seed {seed}",
    )
    atomic_save(
        path,
        metadata=json.dumps(expected, sort_keys=True),
        positions_nm=trajectory,
        energies_kj_mol=energies,
        stereo_min_margin_nm3=margins,
    )
    return trajectory


def run_metadata(
    config: dict,
    phase: str,
    molecule: str,
    seed: int,
    rg_param: tuple[float, float] | None,
) -> dict:
    settings = config["phases"][phase]
    source = verified_jflows_md_source(config)
    return {
        "schema_version": 1,
        "study": "chiral_regularization",
        "phase": phase,
        "molecule": molecule,
        "dimension": int(config["molecules"][molecule]["dimension"]),
        "seed": int(seed),
        "rg_param": None if rg_param is None else list(map(float, rg_param)),
        "rounds": int(settings["rounds"]),
        "burnin_rounds": int(settings["burnin_rounds"]),
        "temperatures_kelvin": list(map(float, config["temperatures_kelvin"])),
        "steps_per_round": int(settings["steps_per_round"]),
        "timestep_fs": float(config["timestep_fs"]),
        "friction_per_ps": float(config["friction_per_ps"]),
        "platform": config["platform"],
        "precision": config["precision"],
        "config_sha256": sha256(CONFIG_PATH),
        "bundle_sha256": bundle_hashes(config, molecule),
        "jflows_version": version("jflows"),
        "jflows_md_version": version("jflows_md"),
        "jflows_md_source_root": source["root"],
        "jflows_md_commit_observed": source["commit"],
        "jflows_md_worktree_clean": source["worktree_clean"],
        "openmm_version": mm.version.full_version,
    }


def artifact_path(
    phase: str,
    molecule: str,
    seed: int,
    rg_param: tuple[float, float] | None,
) -> Path:
    name = f"{molecule}__{condition_label(rg_param)}__seed{seed}.npz"
    return ROOT / "data" / phase / name


def artifact_complete(path: Path, expected: dict, bundle: Molecular_Bundle, centers: list[dict], tolerance: float) -> bool:
    if not path.is_file():
        return False
    replicas = len(expected["temperatures_kelvin"])
    rounds = expected["rounds"]
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            positions = np.asarray(data["positions_nm"])
            energies = np.asarray(data["energies_kj_mol"])
            acceptance = np.asarray(data["swap_acceptance"])
            walker = np.asarray(data["walker_index"])
            stored_margins = np.asarray(data["stereo_min_margin_nm3"])
        permutations = np.sort(walker, axis=1)
        expected_permutation = np.arange(replicas)[None]
        margins = stereo_margins(positions, centers)
        return bool(
            metadata == expected
            and positions.shape == (rounds, replicas, bundle.n_atoms, 3)
            and energies.shape == (rounds, replicas)
            and acceptance.shape == (replicas - 1,)
            and walker.shape == (rounds + 1, replicas)
            and stored_margins.shape == margins.shape
            and np.isfinite(positions).all()
            and np.isfinite(energies).all()
            and np.isfinite(acceptance).all()
            and np.all((acceptance >= 0.0) & (acceptance <= 1.0))
            and np.array_equal(permutations, np.broadcast_to(expected_permutation, permutations.shape))
            and np.array_equal(stored_margins, margins)
            and np.isfinite(margins).all()
            and np.all(margins > tolerance)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_one(
    config: dict,
    phase: str,
    molecule: str,
    seed: int,
    rg_param: tuple[float, float] | None,
) -> None:
    bundle = Molecular_Bundle.load(bundle_path(config, molecule))
    centers = checked_stereocenters(config, molecule, bundle)
    expected = run_metadata(config, phase, molecule, seed, rg_param)
    destination = artifact_path(phase, molecule, seed, rg_param)
    if artifact_complete(
        destination,
        expected,
        bundle,
        centers,
        config["stereo_min_margin_nm3"],
    ):
        print(f"SKIP complete {destination.relative_to(ROOT)}", flush=True)
        return
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite stale or invalid artifact: {destination}")

    physical = OpenMM_Potential.from_bundle(bundle_path(config, molecule))
    potential = physical if rg_param is None else physical.regularized(rg_param)
    initial = starts(config, phase, molecule, seed)
    label = f"{phase} {molecule} {condition_label(rg_param)} seed={seed}"
    started = time.time()
    print(f"START {label} rounds={expected['rounds']}", flush=True)
    positions, energies, acceptance, walker = parallel_tempering_traced(
        potential,
        config["temperatures_kelvin"],
        initial,
        rounds=expected["rounds"],
        steps_per_round=expected["steps_per_round"],
        timestep_fs=config["timestep_fs"],
        friction_per_ps=config["friction_per_ps"],
        seed=seed,
        platform=config["platform"],
        progress_label=label,
    )
    if not (
        np.isfinite(positions).all()
        and np.isfinite(energies).all()
        and np.isfinite(acceptance).all()
    ):
        raise FloatingPointError(f"nonfinite output for {label}")
    margins = require_stereochemistry(
        positions,
        centers,
        config["stereo_min_margin_nm3"],
        label,
    )
    atomic_save(
        destination,
        metadata=json.dumps(expected, sort_keys=True),
        positions_nm=positions,
        energies_kj_mol=energies,
        swap_acceptance=acceptance,
        walker_index=walker,
        stereo_min_margin_nm3=margins,
    )
    print(
        f"DONE {destination.relative_to(ROOT)} seconds={time.time() - started:.1f} "
        f"acceptance={np.array2string(acceptance, precision=3)} "
        f"stereo_min_nm3={np.array2string(margins, precision=6)}",
        flush=True,
    )
    del positions, energies, acceptance, walker, potential, physical, initial
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("sanity", "pilot", "verification"))
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition", choices=("raw", "candidate"), required=True)
    parser.add_argument("--e", type=float)
    parser.add_argument("--r", type=float)
    args = parser.parse_args()

    config = load_config()
    source = verified_jflows_md_source(config)
    if args.molecule not in config["molecules"]:
        parser.error(f"unknown molecule: {args.molecule}")
    phase_seeds = list(map(int, config["phases"][args.phase]["seeds"]))
    seeds = phase_seeds if args.seed is None else [args.seed]
    if any(seed not in phase_seeds for seed in seeds):
        parser.error(f"seed must be one of {phase_seeds} for {args.phase}")
    if args.condition == "raw":
        if args.e is not None or args.r is not None:
            parser.error("raw condition does not accept --e or --r")
        rg_param = None
    else:
        if args.e is None or args.r is None:
            parser.error("candidate condition requires --e and --r")
        rg_param = (float(args.e), float(args.r))
        if rg_param not in candidate_order(config):
            parser.error(f"{rg_param} is not in the frozen candidate order")
        if not candidate_eligible(config, args.molecule, rg_param):
            exponent = tail_exponent(config, rg_param)
            threshold = config["molecules"][args.molecule]["dimension"] + config["tail_moment_order"]
            parser.error(
                f"{rg_param} is ineligible for {args.molecule}: tail exponent "
                f"{exponent:.6f} must exceed {threshold}"
            )

    platform = mm.Platform.getPlatformByName(config["platform"])
    platform.setPropertyDefaultValue("Precision", config["precision"])
    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"OpenMM={mm.version.full_version} platform={platform.getName()} "
        f"precision={platform.getPropertyDefaultValue('Precision')} "
        f"jflows_md={source['commit']} clean={source['worktree_clean']}",
        flush=True,
    )
    for seed in seeds:
        run_one(config, args.phase, args.molecule, seed, rg_param)


if __name__ == "__main__":
    main()

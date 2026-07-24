#!/usr/bin/env python
"""Probe Ac-Pro-NHMe cis/trans and ring-puckering modes with raw OpenMM PT."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import openmm as mm

from jflows_md.openmm import OpenMM_Potential


HERE = Path(__file__).resolve().parent
TARGET = HERE.parent
BUNDLE = TARGET / "bundle"
SHARED_RUN = HERE.parents[1] / "regularization" / "run.py"
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


def shared_protocol():
    spec = importlib.util.spec_from_file_location(
        "_shared_chiral_regularization_run_for_probe", SHARED_RUN
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared OpenMM protocol: {SHARED_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--max-temperature", type=float, default=1200.0)
    parser.add_argument("--steps-per-round", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--output", type=Path, default=HERE / "data" / "openmm_probe.npz")
    args = parser.parse_args()
    if args.rounds <= 0 or args.replicas < 2 or args.steps_per_round <= 0:
        parser.error("rounds and steps must be positive, with at least two replicas")
    if args.max_temperature <= 300.0:
        parser.error("max temperature must exceed 300 K")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing trajectory: {output}")
    protocol = shared_protocol()
    platform = mm.Platform.getPlatformByName("CUDA")
    platform.setPropertyDefaultValue("Precision", "mixed")
    potential = OpenMM_Potential.from_bundle(BUNDLE)
    temperatures = np.geomspace(300.0, args.max_temperature, args.replicas)
    metadata = {
        "schema_version": 1,
        "target": "N-acetyl-L-proline N-methylamide",
        "force_field": "Amber ff96/OBC1",
        "potential": "raw",
        "rounds": args.rounds,
        "replicas": args.replicas,
        "temperatures_kelvin": temperatures.tolist(),
        "steps_per_round": args.steps_per_round,
        "timestep_fs": 1.0,
        "friction_per_ps": 1.0,
        "seed": args.seed,
        "platform": "CUDA",
        "precision": "mixed",
        "openmm_version": mm.version.full_version,
        "jflows_md_commit": "df0da9c9c194aeb9971f77711480aa91bdb33b9e",
        "bundle_sha256": {name: sha256(BUNDLE / name) for name in BUNDLE_FILES},
    }
    centers = [
        {
            "label": "proline_ca_L",
            "atoms": [16, 6, 18, 13],
            "volume_sign": 1,
        }
    ]
    print(
        f"START raw OpenMM PT rounds={args.rounds} replicas={args.replicas} "
        f"temperatures=300..{args.max_temperature:g} K",
        flush=True,
    )
    positions, energies, acceptance, walker = protocol.parallel_tempering_traced(
        potential,
        temperatures,
        potential.reference_positions_nm,
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        timestep_fs=1.0,
        friction_per_ps=1.0,
        seed=args.seed,
        platform="CUDA",
        progress_label="Ac-Pro-NHMe raw multimodality",
    )
    margins = protocol.require_stereochemistry(
        positions,
        centers,
        1e-6,
        "Ac-Pro-NHMe raw multimodality trajectory",
    )
    protocol.atomic_save(
        output,
        metadata=json.dumps(metadata, sort_keys=True),
        positions_nm=positions,
        energies_kj_mol=energies,
        swap_acceptance=acceptance,
        walker_index=walker,
        stereo_min_margin_nm3=margins,
    )
    print(
        f"DONE {output} acceptance={np.array2string(acceptance, precision=3)} "
        f"stereo_min_nm3={np.array2string(margins, precision=6)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

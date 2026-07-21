#!/usr/bin/env python
"""Direct macroscopic observables from the best final KLXX populations."""

import json
import math
import os
import re
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows_md import Molecular_Potential
from jflows_md.system import Molecular_Bundle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "macroscopic.json"
RG_PARAM = (100.0, 0.15)
MOLECULES = {
    9: "Methane",
    18: "Ethane",
    27: "Propane",
    36: "Butane",
    45: "Pentane",
    54: "Hexane",
}


@eqx.filter_jit
def batch_observables(samples, target, regularized, carbon_indices):
    positions = target.cartesian(samples)
    carbons = positions[:, carbon_indices]
    center = jnp.mean(carbons, axis=1, keepdims=True)
    radius = jnp.sqrt(jnp.mean(
        jnp.sum(jnp.square(carbons - center), axis=2), axis=1
    ))
    end_to_end = jnp.linalg.norm(carbons[:, -1] - carbons[:, 0], axis=1)
    torsions = []
    for first in range(carbons.shape[1] - 3):
        a, b, c, d = (carbons[:, first + index] for index in range(4))
        b0 = -(b - a)
        b1 = c - b
        b2 = d - c
        b1 = b1 / jnp.linalg.norm(b1, axis=1, keepdims=True)
        v = b0 - jnp.sum(b0 * b1, axis=1, keepdims=True) * b1
        w = b2 - jnp.sum(b2 * b1, axis=1, keepdims=True) * b1
        torsions.append(jnp.arctan2(
            jnp.sum(jnp.cross(b1, v) * w, axis=1),
            jnp.sum(v * w, axis=1),
        ))
    phi = (
        jnp.stack(torsions, axis=1)
        if torsions else jnp.zeros((samples.shape[0], 0), dtype=samples.dtype)
    )
    return (
        target.physical_energy(samples),
        regularized.regularized_energy(samples),
        radius,
        end_to_end,
        phi,
    )


def bundle(folder):
    local = folder / "bundle"
    if local.exists():
        return local
    return ROOT / "regularization" / "bundle" / folder.name.replace("_raw", "")


def records(run_dir, run):
    return [
        json.loads((run_dir / item["path"] / "stage.json").read_text())
        for item in run["stages"]
    ]


def factor(stages):
    value = math.prod(1.0 / stage["valid_selected_ess"] for stage in stages)
    value *= math.prod(
        1.0 / stage["sharpen_ess"]
        for stage in stages if stage["rg_start"] != stage["rg_end"]
    )
    return value


def best_runs():
    candidates = {}
    for folder in ROOT.iterdir():
        match = re.fullmatch(r".+_([0-9]+)d(?:_raw)?", folder.name)
        if not match:
            continue
        run_dir = folder / "artifacts" / "klxx"
        run_file = run_dir / "run.json"
        if not run_file.exists():
            continue
        run = json.loads(run_file.read_text())
        if run["status"] != "complete" or run["stages"][-1]["t"] != 1.0:
            continue
        stages = records(run_dir, run)
        dimension = int(match.group(1))
        candidate = (factor(stages), folder, run_dir, run, stages)
        if dimension not in candidates or candidate[0] < candidates[dimension][0]:
            candidates[dimension] = candidate
    return candidates


def compute(dimension, candidate):
    total_factor, folder, run_dir, run, stages = candidate
    final = stages[-1]
    samples = np.load(run_dir / final["validation_samples_path"], mmap_mode="r")
    bundle_path = bundle(folder)
    molecular_bundle = Molecular_Bundle.load(bundle_path)
    target = Molecular_Potential(molecular_bundle)
    regularized = target.regularized(RG_PARAM)
    carbon_indices = jnp.asarray(
        np.flatnonzero(np.asarray(molecular_bundle.system["atomic_numbers"]) == 6)
    )
    values = [[], [], [], [], []]
    for part in np.array_split(samples, int(run["config"]["chunks"]), axis=0):
        result = jax.block_until_ready(batch_observables(
            jnp.asarray(part), target, regularized, carbon_indices
        ))
        for collected, array in zip(values, result):
            collected.append(np.asarray(array))
    physical_energy, energy, radius, end_to_end, phi = (
        np.concatenate(collected, axis=0) for collected in values
    )
    torsion_count = phi.shape[1]
    if torsion_count:
        trans = np.abs(phi) >= 2 * math.pi / 3
        gauche_plus = (phi >= 0) & ~trans
        gauche_minus = (phi < 0) & ~trans
        populations = {
            "trans_fraction": float(np.mean(trans)),
            "gauche_plus_fraction": float(np.mean(gauche_plus)),
            "gauche_minus_fraction": float(np.mean(gauche_minus)),
        }
    else:
        populations = {
            "trans_fraction": None,
            "gauche_plus_fraction": None,
            "gauche_minus_fraction": None,
        }
    sharpening = any(stage["rg_start"] != stage["rg_end"] for stage in stages)
    result = {
        "molecule": MOLECULES[dimension],
        "dimension": dimension,
        "source": folder.name,
        "benchmark": "Sharpening KLXX" if sharpening else "Raw KLXX",
        "total_factor": total_factor,
        "sample_count": int(samples.shape[0]),
        "rg_param": list(RG_PARAM),
        "physical_energy_mean_kj_mol": float(
            np.mean(physical_energy, dtype=np.float64)
        ),
        "physical_energy_sd_kj_mol": float(
            np.std(physical_energy, dtype=np.float64)
        ),
        "regularized_energy_mean_kj_mol": float(np.mean(energy, dtype=np.float64)),
        "regularized_energy_sd_kj_mol": float(np.std(energy, dtype=np.float64)),
        "carbon_radius_mean_nm": float(np.mean(radius, dtype=np.float64)),
        "carbon_radius_sd_nm": float(np.std(radius, dtype=np.float64)),
        "carbon_end_to_end_mean_nm": float(np.mean(end_to_end, dtype=np.float64)),
        "carbon_end_to_end_sd_nm": float(np.std(end_to_end, dtype=np.float64)),
        "backbone_torsion_count": torsion_count,
        **populations,
    }
    print(
        f"{result['molecule']} ({dimension}d) {result['benchmark']} "
        f"factor={total_factor:.6g} N={samples.shape[0]}",
        flush=True,
    )
    del samples, target, regularized, physical_energy, energy, radius, end_to_end, phi
    jax.clear_caches()
    return result


def main():
    selected = best_runs()
    results = {
        str(dimension): compute(dimension, selected[dimension])
        for dimension in sorted(MOLECULES)
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compute one endpoint-to-physical regularization ESS for each alkane."""

import json
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows.utils import compute_ESS_log
from jflows_md import Molecular_Potential


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "regularization_ess.json"
RG_PARAM = (100.0, 0.15)
RUNS = (
    ("Methane", 9, "methane_9d_raw"),
    ("Ethane", 18, "ethane_18d_raw"),
    ("Propane", 27, "propane_27d_raw"),
    ("Butane", 36, "butane_36d"),
    ("Pentane", 45, "pentane_45d"),
    ("Hexane", 54, "hexane_54d"),
)


@eqx.filter_jit
def log_weight(samples, regularized, exact):
    return regularized(samples) - exact(samples)


def bundle(folder):
    molecule = folder.name.removesuffix("_raw")
    return ROOT / f"{molecule}_raw" / "bundle"


def compute(name, dimension, folder_name):
    folder = ROOT / folder_name
    run_dir = folder / "artifacts" / "klxx"
    run = json.loads((run_dir / "run.json").read_text())
    final = run["stages"][-1]
    stage = json.loads((run_dir / final["path"] / "stage.json").read_text())
    samples = np.load(run_dir / stage["validation_samples_path"], mmap_mode="r")
    exact = Molecular_Potential.from_bundle(bundle(folder))
    regularized = exact.regularized(RG_PARAM)
    chunks = int(run["config"]["chunks"])
    weights = []
    for part in np.array_split(samples, chunks, axis=0):
        weights.append(np.asarray(jax.block_until_ready(
            log_weight(jnp.asarray(part), regularized, exact)
        )))
    weights = jnp.asarray(np.concatenate(weights))
    ess = float(jax.block_until_ready(compute_ESS_log(weights)))
    result = {
        "molecule": name,
        "dimension": dimension,
        "rg_param": list(RG_PARAM),
        "source": folder_name,
        "sample_count": int(samples.shape[0]),
        "ess": ess,
    }
    print(
        f"{name} ({dimension}d) RESS={ess:.6g} "
        f"N={samples.shape[0]} source={folder_name}",
        flush=True,
    )
    del weights, samples, regularized, exact
    jax.clear_caches()
    return result


def main():
    results = {
        str(dimension): compute(name, dimension, folder)
        for name, dimension, folder in RUNS
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()

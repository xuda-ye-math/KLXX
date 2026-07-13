#!/usr/bin/env python
"""Distributional checks for saved alkane SMC target pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp

from jflows_md import Molecular_Potential

import parameters as P


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def js_bits(first: np.ndarray, second: np.ndarray) -> float:
    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    mean = 0.5 * (p + q)

    def kl(value):
        keep = value > 0
        return float(np.sum(value[keep] * np.log2(value[keep] / mean[keep])))

    return 0.5 * (kl(p) + kl(q))


def periodic_histogram(value: np.ndarray, bins: int = 180) -> np.ndarray:
    wrapped = (value + np.pi) % (2.0 * np.pi) - np.pi
    return np.histogram(wrapped, bins=bins, range=(-np.pi, np.pi))[0] + 0.5


def pairwise_periodic_histogram(
    first: np.ndarray, second: np.ndarray, bins: int = 36
) -> np.ndarray:
    first = (first + np.pi) % (2.0 * np.pi) - np.pi
    second = (second + np.pi) % (2.0 * np.pi) - np.pi
    return np.histogram2d(
        first, second, bins=bins,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
    )[0].ravel() + 0.5


def chunked_energy(target, values: np.ndarray, chunk: int = 16) -> np.ndarray:
    output = []
    array = jnp.asarray(values)
    for part in jnp.array_split(array, chunk, axis=0):
        output.append(np.asarray(jax.block_until_ready(target(part))))
    return np.concatenate(output)


def compare(first: np.ndarray, second: np.ndarray, target, euclidean: int) -> dict:
    if first.shape != second.shape:
        raise ValueError(f"pool shapes differ: {first.shape} != {second.shape}")
    euclidean_ks = [
        float(ks_2samp(first[:, index], second[:, index]).statistic)
        for index in range(euclidean)
    ]
    periodic_js = [
        js_bits(
            periodic_histogram(first[:, index]),
            periodic_histogram(second[:, index]),
        )
        for index in range(euclidean, first.shape[1])
    ]
    pairwise_js = []
    for index in range(euclidean, first.shape[1] - 1):
        pairwise_js.append(
            js_bits(
                pairwise_periodic_histogram(first[:, index], first[:, index + 1]),
                pairwise_periodic_histogram(second[:, index], second[:, index + 1]),
            )
        )
    first_energy = chunked_energy(target, first)
    second_energy = chunked_energy(target, second)
    return {
        "samples_per_pool": int(first.shape[0]),
        "maximum_euclidean_ks": max(euclidean_ks, default=0.0),
        "euclidean_ks": euclidean_ks,
        "maximum_periodic_marginal_js_bits": max(periodic_js, default=0.0),
        "periodic_marginal_js_bits": periodic_js,
        "maximum_adjacent_periodic_pair_js_bits": max(pairwise_js, default=0.0),
        "adjacent_periodic_pair_js_bits": pairwise_js,
        "target_energy_ks": float(ks_2samp(first_energy, second_energy).statistic),
        "target_energy_mean_difference": float(
            np.mean(first_energy) - np.mean(second_energy)
        ),
    }


def load_run(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    summary = json.loads((path / "summary.json").read_text())
    with h5py.File(path / "samples.h5", "r") as handle:
        train = handle["target_train"][...]
        holdout = handle["target_holdout"][...]
    return summary, train, holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary, train, holdout = load_run(args.run.resolve())
    bundle = HERE / "bundles" / summary["bundle"]["path"]
    physical = Molecular_Potential.from_bundle(bundle)
    target = physical.regularized(
        summary["target"]["energy_cut_kj_mol"],
        energy_scale_kj_mol=summary["target"]["energy_scale_kj_mol"],
        tail_fraction=summary["target"]["tail_fraction"],
    )
    comparisons = {
        "run_train_vs_holdout": compare(
            train, holdout, target, physical.domain.euclidean_dim
        )
    }
    inputs = {
        "run": str(args.run.resolve()),
        "run_summary_sha256": sha256(args.run.resolve() / "summary.json"),
        "run_samples_sha256": sha256(args.run.resolve() / "samples.h5"),
    }
    if args.reference:
        reference_summary, reference_train, reference_holdout = load_run(
            args.reference.resolve()
        )
        if reference_summary["bundle"]["manifest_sha256"] != summary["bundle"]["manifest_sha256"]:
            raise ValueError("run and reference use different molecular bundles")
        comparisons["reference_train_vs_holdout"] = compare(
            reference_train, reference_holdout, target, physical.domain.euclidean_dim
        )
        comparisons["run_vs_reference_train"] = compare(
            train, reference_train, target, physical.domain.euclidean_dim
        )
        comparisons["run_vs_reference_holdout"] = compare(
            holdout, reference_holdout, target, physical.domain.euclidean_dim
        )
        inputs.update(
            reference=str(args.reference.resolve()),
            reference_summary_sha256=sha256(args.reference.resolve() / "summary.json"),
            reference_samples_sha256=sha256(args.reference.resolve() / "samples.h5"),
        )
    gates = {
        "maximum_euclidean_ks": P.EUCLIDEAN_KS_MAX,
        "maximum_periodic_marginal_js_bits": P.MARGINAL_JS_BITS_MAX,
        "maximum_adjacent_periodic_pair_js_bits": P.JOINT_ROTAMER_JS_BITS_MAX,
        "target_energy_ks": P.ENERGY_KS_MAX,
    }
    passed = all(
        comparison[name] <= threshold
        for comparison in comparisons.values()
        for name, threshold in gates.items()
    )
    result = {
        "schema_version": 1,
        "complete": True,
        "passed": passed,
        "inputs": inputs,
        "gates": gates,
        "comparisons": comparisons,
        "runtime": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "driver_sha256": sha256(Path(__file__)),
            "parameters_sha256": sha256(HERE / "parameters.py"),
        },
    }
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'} target-pool audit: {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

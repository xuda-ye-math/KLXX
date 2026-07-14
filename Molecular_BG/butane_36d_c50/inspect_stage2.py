#!/usr/bin/env python
"""Locate the heavy-tail samples responsible for the archived stage-2 failure."""

from __future__ import annotations

import json
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import linear_combination
from jflows_md import Molecular_Potential, load_mixed_flow

import parameters as P


QUANTILES = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)


@eqx.filter_jit
def evaluate_chunk(flow, samples, previous, current):
    latent, ladj = flow.call_and_ladj(samples)
    return (
        previous(samples),
        current(samples),
        previous(latent),
        ladj,
    )


def summary(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "count": int(values.size),
        "finite_count": int(finite.sum()),
        "quantiles": {
            str(value): float(result)
            for value, result in zip(
                QUANTILES, np.quantile(finite_values, QUANTILES)
            )
        },
    }


def inspect_pool(name, samples, flow, previous, current, energy_origin):
    pieces = [[], [], [], []]
    for part in jnp.array_split(samples, P.CHUNKS, axis=0):
        values = jax.block_until_ready(
            evaluate_chunk(flow, part, previous, current)
        )
        for destination, value in zip(pieces, values):
            destination.append(np.asarray(value, dtype=np.float32))
    previous_y, current_y, previous_latent, ladj = [
        np.concatenate(value) for value in pieces
    ]
    warm_z = previous_latent - current_y - ladj
    identity_z = previous_y - current_y
    array = np.asarray(samples, dtype=np.float32)
    row_abs_max = np.max(np.abs(array[:, : P.DIMENSION - 11]), axis=1)
    top = np.argsort(np.abs(warm_z))[-10:][::-1]
    return {
        "coordinates_row_abs_max": summary(row_abs_max),
        "previous_energy_at_samples": summary(previous_y),
        "current_energy_at_samples": summary(current_y),
        "current_relative_energy": summary(current_y - energy_origin),
        "previous_energy_at_warm_latent": summary(previous_latent),
        "warm_ladj": summary(ladj),
        "warm_z": summary(warm_z),
        "identity_z": summary(identity_z),
        "energy_keep_fraction": float(
            np.mean(np.isfinite(current_y) & (current_y - energy_origin <= P.E_CLIP))
        ),
        "top_warm_z_samples": [
            {
                "index": int(index),
                "warm_z": float(warm_z[index]),
                "identity_z": float(identity_z[index]),
                "coordinate_row_abs_max": float(row_abs_max[index]),
                "current_relative_energy": float(current_y[index] - energy_origin),
                "previous_latent_energy": float(previous_latent[index]),
                "ladj": float(ladj[index]),
                "energy_kept": bool(
                    np.isfinite(current_y[index])
                    and current_y[index] - energy_origin <= P.E_CLIP
                ),
            }
            for index in top
        ],
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    reproduction = here / "reproduction"
    metadata = json.loads(
        (reproduction / "stage2_t0175.json").read_text(encoding="utf-8")
    )
    with h5py.File(reproduction / metadata["data"], "r") as handle:
        pools = {
            "stage1_source": jnp.asarray(handle["stage1_particles"][:]),
            "stage2_smc": jnp.asarray(handle["stage2_smc_particles"][:]),
            "stage2_hat": jnp.asarray(handle["stage2_hat_particles"][:]),
        }

    physical = Molecular_Potential.from_bundle(
        here / "bundle", temperature_kelvin=300.0
    )
    molecular_target = physical.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    source = physical.source()
    previous_t = float(metadata["previous_t"])
    failed_t = float(metadata["failed_t"])
    previous = linear_combination(
        [molecular_target, source], [previous_t, 1.0 - previous_t]
    )
    current = linear_combination(
        [molecular_target, source], [failed_t, 1.0 - failed_t]
    )
    flow = load_mixed_flow(reproduction / metadata["accepted_flow"])
    energy_origin = float(metadata["energy_origin"])
    output = {
        "schema_version": 1,
        "data": metadata["data"],
        "previous_t": previous_t,
        "failed_t": failed_t,
        "energy_origin": energy_origin,
        "pools": {
            name: inspect_pool(
                name, samples, flow, previous, current, energy_origin
            )
            for name, samples in pools.items()
        },
    }
    output_path = reproduction / "pool_stats.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

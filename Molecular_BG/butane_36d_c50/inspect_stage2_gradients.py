#!/usr/bin/env python
"""Audit source-potential force finiteness on the archived level-2 pools."""

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


@eqx.filter_jit
def latent_chunk(flow, samples):
    return flow(samples)


@eqx.filter_jit
def energy_and_gradient_chunk(potential, samples):
    energy = potential(samples)
    gradient = jax.grad(lambda value: jnp.sum(potential(value)))(samples)
    return energy, gradient


@eqx.filter_jit
def energy_chunk(potential, samples):
    return potential(samples)


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    levels = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)
    return {
        str(level): float(value)
        for level, value in zip(levels, np.quantile(values, levels))
    }


def inspect(samples, flow, previous, current, energy_origin, start):
    energy_values = []
    gradient_values = []
    keep_values = []
    for part in jnp.array_split(samples, P.CHUNKS, axis=0):
        current_energy = jax.block_until_ready(energy_chunk(current, part))
        latent = (
            part
            if start == "identity"
            else jax.block_until_ready(latent_chunk(flow, part))
        )
        source_energy, source_gradient = jax.block_until_ready(
            energy_and_gradient_chunk(previous, latent)
        )
        energy_values.append(np.asarray(source_energy, dtype=np.float32))
        gradient_values.append(np.asarray(source_gradient, dtype=np.float32))
        keep_values.append(
            np.asarray(
                np.isfinite(current_energy)
                & (current_energy - energy_origin <= P.E_CLIP)
            )
        )
    energy = np.concatenate(energy_values)
    gradient = np.concatenate(gradient_values)
    keep = np.concatenate(keep_values)
    finite_row = np.all(np.isfinite(gradient), axis=1)
    max_abs = np.max(np.abs(gradient), axis=1)
    finite_max = max_abs[np.isfinite(max_abs)]
    return {
        "sample_count": int(samples.shape[0]),
        "energy_kept_count": int(keep.sum()),
        "finite_gradient_row_count": int(finite_row.sum()),
        "nonfinite_gradient_row_count": int((~finite_row).sum()),
        "nonfinite_gradient_among_energy_kept": int((keep & ~finite_row).sum()),
        "source_energy_quantiles": quantiles(energy),
        "gradient_row_max_abs_quantiles": quantiles(finite_max),
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
        "pools": {},
    }
    for name, samples in pools.items():
        output["pools"][name] = {
            start: inspect(
                samples, flow, previous, current, energy_origin, start
            )
            for start in ("identity", "warm")
        }
    output_path = reproduction / "gradient_stats.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

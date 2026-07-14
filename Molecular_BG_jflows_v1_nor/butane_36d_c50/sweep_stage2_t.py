#!/usr/bin/env python
"""Measure identity and warm-start incremental ESS near the accepted t=0.07."""

from __future__ import annotations

import json
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import jax.numpy as jnp

from jflows.potential import linear_combination
from jflows.utils import compute_ESS_log, importance_weights_log
from jflows_md import Molecular_Potential, load_mixed_flow

import parameters as P


T_VALUES = (0.071, 0.075, 0.08, 0.09, 0.10, 0.12, 0.14, 0.175)


def ess(samples, source, target, flow) -> float:
    return float(
        compute_ESS_log(
            importance_weights_log(
                samples,
                source,
                target,
                flow,
                type="G",
                chunks=P.CHUNKS,
            )
        )
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    reproduction = here / "reproduction"
    metadata = json.loads(
        (reproduction / "stage2_t0175.json").read_text(encoding="utf-8")
    )
    with h5py.File(reproduction / metadata["data"], "r") as handle:
        samples = jnp.asarray(handle["stage1_particles"][:])
    physical = Molecular_Potential.from_bundle(
        here / "bundle", temperature_kelvin=300.0
    )
    molecular_target = physical.regularized(
        P.ENERGY_CUT_KJ_MOL,
        energy_scale_kj_mol=P.ENERGY_SCALE_KJ_MOL,
        tail_fraction=P.TAIL_FRACTION,
    )
    original_source = physical.source()
    previous_t = float(metadata["previous_t"])
    previous = linear_combination(
        [molecular_target, original_source],
        [previous_t, 1.0 - previous_t],
    )
    warm = load_mixed_flow(reproduction / metadata["accepted_flow"])
    identity = warm.zeros()
    rows = []
    for value in T_VALUES:
        current = linear_combination(
            [molecular_target, original_source], [value, 1.0 - value]
        )
        record = {
            "t": value,
            "delta_t": value - previous_t,
            "identity_ess": ess(samples, previous, current, identity),
            "warm_ess": ess(samples, previous, current, warm),
        }
        rows.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "schema_version": 1,
        "data": metadata["data"],
        "previous_t": previous_t,
        "sample_count": int(samples.shape[0]),
        "rows": rows,
    }
    (reproduction / "t_sweep.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

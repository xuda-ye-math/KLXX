#!/usr/bin/env python
"""Track the ACE-PRO cis population across every accepted KLXX stage.

Tests whether the final-stage cis excess is inherited from the softer early
stages.  If the cis fraction starts high, falls as the regularization sharpens,
and then stalls above the OpenMM reference value, the excess is residual
intermodal mass that stage rejuvenation never cleared.  If it is flat across
stages, that explanation is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jflows_md import Molecular_Potential
from jflows_md.boltzmann.load import load_validation_samples, validate_run

from plot_conformational_landscape import (
    BUNDLE,
    OMEGA_ATOMS,
    RUN_DIR,
    cartesian_positions,
    dihedral,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "stage_cis_fraction.json"
REFERENCE_MANIFEST = HERE / "reference" / "pt_reference_32000.json"


def cis_fraction(
    samples: np.ndarray, target: Molecular_Potential, chunk_size: int
) -> float:
    """Return the fraction of samples with |omega| < pi/2."""

    cis = 0
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(start + chunk_size, samples.shape[0])
        internal = np.asarray(samples[start:stop])
        positions = np.asarray(
            jax.block_until_ready(cartesian_positions(target, jnp.asarray(internal)))
        )
        omega = dihedral(positions, OMEGA_ATOMS)
        cis += int(np.count_nonzero(np.abs(omega) < np.pi / 2.0))
    return cis / samples.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    record = validate_run(run_dir)
    if record.get("status") != "complete":
        raise ValueError("KLXX run is not complete")
    target = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=300.0)

    reference = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    reference_cis = float(reference["omega"]["cis_fraction"])
    print(f"OpenMM reference cis = {reference_cis:.4f}", flush=True)
    print(f"{'stage':>5} {'t':>7} {'rho':>16} {'samples':>9} {'cis':>7}", flush=True)

    rows = []
    for stage_ref in record["stages"]:
        stage_number = int(stage_ref["stage"])
        stage_root = (run_dir / stage_ref["path"]).resolve()
        metadata = json.loads((stage_root / "stage.json").read_text(encoding="utf-8"))
        samples = load_validation_samples(run_dir, stage_number, mmap_mode="r")
        value = cis_fraction(samples, target, args.chunk_size)
        rho = metadata["rg_end"]
        rows.append(
            {
                "stage": stage_number,
                "t": float(metadata["t"]),
                "rg_end": [float(rho[0]), float(rho[1])],
                "samples": int(samples.shape[0]),
                "cis_fraction": value,
            }
        )
        print(
            f"{stage_number:>5} {float(metadata['t']):>7.3f} "
            f"({float(rho[0]):>6.1f},{float(rho[1]):>5.2f}) "
            f"{samples.shape[0]:>9,} {value:>7.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "reference_cis_fraction": reference_cis,
                "stages": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Read-only CPU chirality probe for persisted molecular BG stages.

Run once over all currently completed stages:

    python probe_stage_chirality.py adp_60d/artifacts/klxx \
        --bundle adp_60d/bundle

Watch a run and probe each stage after it is atomically published:

    python probe_stage_chirality.py adp_60d/artifacts/klxx \
        --bundle adp_60d/bundle --watch
"""

from __future__ import annotations

import os


# These settings are process-local: the probe cannot reserve or execute on a GPU.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
import sys
import time


sys.dont_write_bytecode = True

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from jflows_md import Molecular_Potential
from jflows_md.system import Molecular_Bundle


TERMINAL_RUN_STATES = {"complete", "exhausted"}


@eqx.filter_jit
def cartesian_positions(
    target: Molecular_Potential, internal: jax.Array
) -> jax.Array:
    """Map internal coordinates to Cartesian positions on the probe's CPU."""

    return target.cartesian(internal)


def fixed_stereocenters(bundle: Molecular_Bundle) -> tuple[dict, ...]:
    """Return explicit and legacy fixed-stereocenter metadata uniformly."""

    spec = bundle.coordinates
    if "fixed_stereocenters" in spec:
        raw = spec["fixed_stereocenters"]
    elif int(spec.get("chirality_sign", 0)):
        raw = ({
            "label": "alanine_ca_L",
            "atoms": spec["chirality_atoms"],
            "volume_sign": spec["chirality_sign"],
        },)
    else:
        raw = ()
    return tuple({
        "label": str(item.get("label", f"stereocenter_{index}")),
        "atoms": tuple(map(int, item["atoms"])),
        "volume_sign": int(item["volume_sign"]),
    } for index, item in enumerate(raw))


def signed_volume(positions: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    center, first, second, third = atoms
    a = positions[:, first] - positions[:, center]
    b = positions[:, second] - positions[:, center]
    c = positions[:, third] - positions[:, center]
    return np.einsum("ij,ij->i", a, np.cross(b, c))


def safe_path(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f"absolute artifact path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact path escapes run directory: {relative}")
    return path


def discover_bundle(run_dir: Path) -> Path:
    for parent in (run_dir, *run_dir.parents):
        candidate = parent / "bundle"
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        "no ancestor bundle found; pass --bundle explicitly"
    )


def probe_stage(
    run_dir: Path,
    item: dict,
    target: Molecular_Potential,
    centers: tuple[dict, ...],
    *,
    chunk_size: int,
    minimum_margin_nm3: float,
) -> bool:
    stage = int(item["stage"])
    stage_dir = safe_path(run_dir, item["path"])
    metadata = json.loads(
        (stage_dir / "stage.json").read_text(encoding="utf-8")
    )
    if int(metadata["stage"]) != stage:
        raise ValueError(f"stage number mismatch for {stage_dir}")
    samples_path = safe_path(run_dir, metadata["validation_samples_path"])
    samples = np.load(samples_path, mmap_mode="r", allow_pickle=False)
    if samples.ndim != 2 or samples.shape[1] != target.dimension:
        raise ValueError(
            f"stage {stage} samples have shape {samples.shape}, expected "
            f"[N, {target.dimension}]"
        )

    statistics = {
        center["label"]: {"minimum": np.inf, "violations": 0}
        for center in centers
    }
    nonfinite_internal = 0
    for start in range(0, samples.shape[0], chunk_size):
        internal = np.asarray(samples[start : start + chunk_size])
        finite = np.isfinite(internal).all(axis=1)
        nonfinite_internal += int((~finite).sum())
        if not finite.any():
            continue
        positions = np.asarray(
            jax.device_get(
                cartesian_positions(target, jnp.asarray(internal[finite]))
            )
        )
        for center in centers:
            margin = center["volume_sign"] * signed_volume(
                positions, center["atoms"]
            )
            invalid = (~np.isfinite(margin)) | (margin <= minimum_margin_nm3)
            record = statistics[center["label"]]
            record["violations"] += int(invalid.sum())
            finite_margin = margin[np.isfinite(margin)]
            if finite_margin.size:
                record["minimum"] = min(
                    record["minimum"], float(finite_margin.min())
                )

    passed = nonfinite_internal == 0 and all(
        record["violations"] == 0 for record in statistics.values()
    )
    status = "PASS" if passed else "FAIL"
    details = " ".join(
        f"{label}:min={record['minimum']:.6e}nm^3,"
        f"violations={record['violations']}"
        for label, record in statistics.items()
    )
    print(
        f"[{status}] stage={stage:03d} t={float(item['t']):.6f} "
        f"samples={samples.shape[0]} nonfinite_internal={nonfinite_internal} "
        f"{details}",
        flush=True,
    )
    return passed


def read_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--minimum-margin-nm3", type=float, default=0.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0.0 or args.chunk_size <= 0:
        parser.error("--poll-seconds and --chunk-size must be positive")

    try:
        os.nice(10)
    except OSError:
        pass

    run_dir = args.run_dir.expanduser().resolve()
    bundle_path = (
        args.bundle.expanduser().resolve()
        if args.bundle is not None
        else discover_bundle(run_dir)
    )
    bundle = Molecular_Bundle.load(bundle_path)
    centers = fixed_stereocenters(bundle)
    if not centers:
        raise ValueError(f"bundle {bundle.name!r} has no fixed stereocenters")
    target = Molecular_Potential(bundle)
    devices = jax.devices()
    if not devices or any(device.platform != "cpu" for device in devices):
        raise RuntimeError(f"probe is not CPU-only: {devices}")
    print(
        f"CPU-only chirality probe | bundle={bundle.name} | "
        f"centers={','.join(center['label'] for center in centers)} | "
        f"devices={devices}",
        flush=True,
    )

    seen: set[int] = set()
    while True:
        try:
            manifest = read_manifest(run_dir)
        except FileNotFoundError:
            if not args.watch:
                raise
            time.sleep(args.poll_seconds)
            continue

        for item in manifest["stages"]:
            stage = int(item["stage"])
            if stage in seen:
                continue
            if not probe_stage(
                run_dir,
                item,
                target,
                centers,
                chunk_size=args.chunk_size,
                minimum_margin_nm3=args.minimum_margin_nm3,
            ):
                raise SystemExit(1)
            seen.add(stage)

        if not args.watch:
            if not seen:
                print("No completed stages are listed yet.", flush=True)
            return
        if manifest["status"] in TERMINAL_RUN_STATES:
            print(
                f"Run status={manifest['status']}; checked {len(seen)} stages.",
                flush=True,
            )
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

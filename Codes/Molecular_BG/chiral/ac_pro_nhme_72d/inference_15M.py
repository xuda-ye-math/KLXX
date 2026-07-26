#!/usr/bin/env python
"""Replay the frozen Ac-Pro-NHMe KLXX stage flows on fifteen million particles.

This is inference only: every saved stage flow is loaded without modification.
Each stage performs pushforward, reweight/resample/MALA, sharpening, and a
second reweight/resample/MALA before its population is persisted to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import linear_combination
from jflows.utils import compute_ESS_log, linear_weights_from_log
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann.load import load_stage_flow, validate
from jflows_md.utils import mixed_mala

import parameters as P


HERE = Path(__file__).resolve().parent
TRAINING_RUN = HERE / "artifacts" / "klxx"
BUNDLE = HERE / "bundle"
DEFAULT_OUTPUT = HERE / "artifacts" / "inference_15M"
DEFAULT_SAMPLE_COUNT = 15_000_000
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_LOG_EVERY_CHUNKS = 25
INFERENCE_SEED = 20260724


class ActiveLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def progress(
        self,
        label: str,
        completed: int,
        total: int,
        started: float,
    ) -> None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0.0 else float("inf")
        self(
            f"{label}: {completed:,}/{total:,} "
            f"({100.0 * completed / total:.1f}%) | "
            f"{rate:,.0f} samples/s | ETA {eta / 60.0:.1f} min"
        )


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def operation_key(namespace: int, stage: int, chunk: int = 0):
    key = jax.random.key(INFERENCE_SEED)
    key = jax.random.fold_in(key, namespace)
    key = jax.random.fold_in(key, stage)
    return jax.random.fold_in(key, chunk)


def numpy_rng(namespace: int, stage: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([INFERENCE_SEED, namespace, stage])
    )


def bridge(source, target, t: float):
    return linear_combination([target, source], [t, 1.0 - t])


@eqx.filter_jit
def proposal_and_log_weight(samples, source, target, flow):
    proposal, inverse_ladj = flow.inv_and_ladj(samples)
    return proposal, source(samples) - target(proposal) + inverse_ladj


@eqx.filter_jit
def potential_log_weight(samples, source, target):
    return source(samples) - target(samples)


def chunk_ranges(total: int, chunk_size: int):
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def should_log(index: int, count: int, every: int) -> bool:
    return index == 1 or index == count or index % every == 0


def validate_population(path: Path, shape: tuple[int, int]) -> np.memmap:
    population = np.load(path, mmap_mode="r", allow_pickle=False)
    if population.shape != shape or population.dtype != np.float32:
        raise ValueError(
            f"invalid population {path}: {population.shape} {population.dtype}"
        )
    return population


def generate_source(
    path: Path,
    source,
    shape: tuple[int, int],
    chunk_size: int,
    log_every: int,
    log: ActiveLog,
) -> None:
    total, dimension = shape
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=shape
    )
    ranges = list(chunk_ranges(total, chunk_size))
    started = time.perf_counter()
    for index, (start, stop) in enumerate(ranges, start=1):
        values = source.samples(
            operation_key(101, 0, index), N=stop - start
        )
        values = np.asarray(jax.block_until_ready(values))
        if values.shape != (stop - start, dimension) or not np.isfinite(values).all():
            raise ValueError("invalid source samples")
        output[start:stop] = values
        if should_log(index, len(ranges), log_every):
            output.flush()
            log.progress("source sampling", stop, total, started)
    output.flush()


def flow_proposal_to_disk(
    input_path: Path,
    proposal_path: Path,
    log_weight_path: Path,
    source,
    target,
    flow,
    shape: tuple[int, int],
    chunk_size: int,
    log_every: int,
    stage: int,
    log: ActiveLog,
) -> None:
    if input_path.resolve() == proposal_path.resolve():
        raise ValueError("proposal output aliases its input")
    population = validate_population(input_path, shape)
    proposals = np.lib.format.open_memmap(
        proposal_path, mode="w+", dtype=np.float32, shape=shape
    )
    log_weights = np.lib.format.open_memmap(
        log_weight_path, mode="w+", dtype=np.float32, shape=(shape[0],)
    )
    ranges = list(chunk_ranges(shape[0], chunk_size))
    started = time.perf_counter()
    for index, (start, stop) in enumerate(ranges, start=1):
        proposal, weight = proposal_and_log_weight(
            jnp.asarray(np.asarray(population[start:stop])), source, target, flow
        )
        proposal, weight = jax.device_get(
            jax.block_until_ready((proposal, weight))
        )
        proposal, weight = np.asarray(proposal), np.asarray(weight)
        if not np.isfinite(proposal).all() or not np.isfinite(weight).all():
            raise ValueError(f"nonfinite flow inference at stage {stage}")
        proposals[start:stop] = proposal
        log_weights[start:stop] = weight
        if should_log(index, len(ranges), log_every):
            proposals.flush()
            log_weights.flush()
            log.progress(f"stage {stage:02d} pushforward", stop, shape[0], started)
    proposals.flush()
    log_weights.flush()


def potential_weights_to_disk(
    input_path: Path,
    log_weight_path: Path,
    source,
    target,
    shape: tuple[int, int],
    chunk_size: int,
    log_every: int,
    label: str,
    log: ActiveLog,
) -> None:
    population = validate_population(input_path, shape)
    log_weights = np.lib.format.open_memmap(
        log_weight_path, mode="w+", dtype=np.float32, shape=(shape[0],)
    )
    ranges = list(chunk_ranges(shape[0], chunk_size))
    started = time.perf_counter()
    for index, (start, stop) in enumerate(ranges, start=1):
        weight = potential_log_weight(
            jnp.asarray(np.asarray(population[start:stop])), source, target
        )
        weight = np.asarray(jax.device_get(jax.block_until_ready(weight)))
        if not np.isfinite(weight).all():
            raise ValueError(f"nonfinite weights during {label}")
        log_weights[start:stop] = weight
        if should_log(index, len(ranges), log_every):
            log_weights.flush()
            log.progress(label, stop, shape[0], started)
    log_weights.flush()


def weights_and_ress(path: Path, count: int) -> tuple[np.ndarray, float]:
    log_weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if log_weights.shape != (count,) or not np.isfinite(log_weights).all():
        raise ValueError(f"invalid log weights: {path}")
    device_log_weights = jnp.asarray(np.asarray(log_weights))
    ress = float(jax.block_until_ready(compute_ESS_log(device_log_weights)))
    weights = np.asarray(
        jax.device_get(linear_weights_from_log(device_log_weights))
    )
    if not math.isfinite(ress) or not 0.0 < ress <= 1.0:
        raise ValueError(f"invalid RESS {ress} for {path}")
    if not np.isfinite(weights).all() or not np.any(weights > 0.0):
        raise ValueError(f"invalid linear weights for {path}")
    return weights, ress


def resample_to_disk(
    input_path: Path,
    output_path: Path,
    weights: np.ndarray,
    shape: tuple[int, int],
    chunk_size: int,
    log_every: int,
    namespace: int,
    stage: int,
    label: str,
    log: ActiveLog,
) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("resampling output aliases its input")
    population = validate_population(input_path, shape)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=shape
    )
    cdf = np.cumsum(weights, dtype=np.float64)
    if not math.isfinite(float(cdf[-1])) or cdf[-1] <= 0.0:
        raise ValueError(f"invalid resampling CDF during {label}")
    rng = numpy_rng(namespace, stage)
    ranges = list(chunk_ranges(shape[0], chunk_size))
    started = time.perf_counter()
    for index, (start, stop) in enumerate(ranges, start=1):
        uniforms = rng.random(stop - start) * cdf[-1]
        selected = np.searchsorted(cdf, uniforms, side="right")
        output[start:stop] = population[selected]
        if should_log(index, len(ranges), log_every):
            output.flush()
            log.progress(label, stop, shape[0], started)
    output.flush()


def mala_to_disk(
    input_path: Path,
    output_path: Path,
    potential,
    domain,
    shape: tuple[int, int],
    chunk_size: int,
    log_every: int,
    namespace: int,
    stage: int,
    dt: float,
    steps: int,
    image_radius: int,
    label: str,
    log: ActiveLog,
) -> np.ndarray:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("MALA output aliases its input")
    population = validate_population(input_path, shape)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=shape
    )
    acceptance = np.zeros(steps, dtype=np.float64)
    ranges = list(chunk_ranges(shape[0], chunk_size))
    started = time.perf_counter()
    for index, (start, stop) in enumerate(ranges, start=1):
        moved, history = mixed_mala(
            operation_key(namespace, stage, index),
            jnp.asarray(np.asarray(population[start:stop])),
            potential,
            domain,
            dt=dt,
            steps=steps,
            image_radius=image_radius,
            chunks=1,
        )
        moved, history = jax.device_get(
            jax.block_until_ready((moved, history))
        )
        moved, history = np.asarray(moved), np.asarray(history)
        if not np.isfinite(moved).all() or not np.isfinite(history).all():
            raise ValueError(f"nonfinite MALA result during {label}")
        output[start:stop] = moved
        acceptance += history.astype(np.float64) * (stop - start)
        if should_log(index, len(ranges), log_every):
            output.flush()
            log.progress(label, stop, shape[0], started)
    output.flush()
    return (acceptance / shape[0]).astype(np.float32)


def flow_template(target: Molecular_Potential):
    _, flow_key = jax.random.split(jax.random.key(P.SEED))
    return Mixed_NSF(
        flow_key,
        target.domain,
        bins=P.BINS,
        transforms=P.TRANSFORMS,
        euclidean_bound=P.NSF_LIM,
        hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE,
        mask_strategy="balanced",
    ).zeros()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--mc-dt", type=float, default=P.MC_DT)
    parser.add_argument("--mc-steps", type=int, default=P.MC_STEPS)
    parser.add_argument("--mc-image-radius", type=int, default=P.MC_IMAGE_RADIUS)
    parser.add_argument(
        "--log-every-chunks", type=int, default=DEFAULT_LOG_EVERY_CHUNKS
    )
    parser.add_argument("--max-stages", type=int)
    args = parser.parse_args()
    positive = (
        args.sample_count,
        args.chunk_size,
        args.mc_dt,
        args.mc_steps,
        args.mc_image_radius,
        args.log_every_chunks,
    )
    if any(value <= 0 for value in positive):
        parser.error("sample, chunk, MALA, image, and log controls must be positive")
    if args.max_stages is not None and args.max_stages <= 0:
        parser.error("--max-stages must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite nonempty inference root: {output_dir}")
    output_dir.mkdir(parents=True)
    work_dir = output_dir / "work"
    work_dir.mkdir()
    log = ActiveLog(output_dir / "inference.log")

    training = validate(TRAINING_RUN)
    if training["status"] != "complete" or not training["stages"]:
        raise ValueError("saved KLXX training run is incomplete")
    stage_items = training["stages"]
    if args.max_stages is not None:
        stage_items = stage_items[: args.max_stages]
    target = Molecular_Potential.from_bundle(
        BUNDLE, temperature_kelvin=P.TEMPERATURE_KELVIN
    )
    source = target.source()
    template = flow_template(target)
    shape = (args.sample_count, target.dimension)
    work_a = work_dir / "population_a.npy"
    work_b = work_dir / "population_b.npy"

    manifest = {
        "format": "ac-pro-nhme-frozen-flow-inference-15m-1",
        "status": "running",
        "inference_only": True,
        "training_updates": 0,
        "training_run": "../klxx",
        "training_run_manifest_sha256": sha256(TRAINING_RUN / "run.json"),
        "sample_count": args.sample_count,
        "dimension": target.dimension,
        "config": {
            "seed": INFERENCE_SEED,
            "chunk_size": args.chunk_size,
            "mc_dt": args.mc_dt,
            "mc_steps": args.mc_steps,
            "mc_image_radius": args.mc_image_radius,
            "saved_flow_role": "selected",
            "resampling": "multinomial inverse-CDF",
        },
        "stages": [],
        "raw": None,
    }
    atomic_json(output_dir / "run.json", manifest)
    log(
        f"START frozen-flow inference | backend={jax.default_backend()} | "
        f"samples={args.sample_count:,} | dimension={target.dimension} | "
        f"stages={len(stage_items)} | chunk={args.chunk_size:,} | "
        f"MALA dt={args.mc_dt} steps={args.mc_steps} "
        f"image_radius={args.mc_image_radius} | training updates=0"
    )

    generate_source(
        work_a,
        source,
        shape,
        args.chunk_size,
        args.log_every_chunks,
        log,
    )
    current_path = work_a
    last_target = source

    for item in stage_items:
        stage_started = time.perf_counter()
        stage_number = int(item["stage"])
        saved_dir = TRAINING_RUN / item["path"]
        saved = json.loads((saved_dir / "stage.json").read_text(encoding="utf-8"))
        if saved["selected"] not in ("trained", "identity"):
            raise ValueError(f"invalid saved selection at stage {stage_number}")
        stage_dir = output_dir / f"stage_{stage_number:06d}"
        stage_dir.mkdir()
        log(
            f"STAGE {stage_number:02d} START | "
            f"t={saved['t_start']:.6f}->{saved['t']:.6f} | "
            f"rg={tuple(saved['rg_start'])}->{tuple(saved['rg_end'])} | "
            f"saved selection={saved['selected']}"
        )

        rg_start = tuple(map(float, saved["rg_start"]))
        rg_end = tuple(map(float, saved["rg_end"]))
        source_bridge = bridge(
            source, target.regularized(rg_start), float(saved["t_start"])
        )
        target_soft = bridge(
            source, target.regularized(rg_start), float(saved["t"])
        )
        target_sharp = bridge(
            source, target.regularized(rg_end), float(saved["t"])
        )
        selected_flow = load_stage_flow(
            TRAINING_RUN, stage_number, "selected", template
        )

        flow_weights_path = stage_dir / "flow_log_weights.npy"
        flow_proposal_to_disk(
            current_path,
            work_b,
            flow_weights_path,
            source_bridge,
            target_soft,
            selected_flow,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            stage_number,
            log,
        )
        flow_weights, flow_ress = weights_and_ress(
            flow_weights_path, args.sample_count
        )
        log(f"stage {stage_number:02d} flow ESS={flow_ress:.9f}")
        resample_to_disk(
            work_b,
            work_a,
            flow_weights,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            201,
            stage_number,
            f"stage {stage_number:02d} flow resampling",
            log,
        )
        del flow_weights
        soft_acceptance = mala_to_disk(
            work_a,
            work_b,
            target_soft,
            target.domain,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            301,
            stage_number,
            args.mc_dt,
            args.mc_steps,
            args.mc_image_radius,
            f"stage {stage_number:02d} pre-sharpening MALA",
            log,
        )
        np.save(
            stage_dir / "pre_sharpen_acceptance.npy",
            soft_acceptance,
            allow_pickle=False,
        )
        log(
            f"stage {stage_number:02d} pre-sharpening MALA "
            f"mean acceptance={float(soft_acceptance.mean()):.6f}"
        )

        sharpen_weights_path = stage_dir / "sharpen_log_weights.npy"
        potential_weights_to_disk(
            work_b,
            sharpen_weights_path,
            target_soft,
            target_sharp,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            f"stage {stage_number:02d} sharpening weights",
            log,
        )
        sharpen_weights, sharpen_ress = weights_and_ress(
            sharpen_weights_path, args.sample_count
        )
        log(f"stage {stage_number:02d} sharpening ESS={sharpen_ress:.9f}")
        resample_to_disk(
            work_b,
            work_a,
            sharpen_weights,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            401,
            stage_number,
            f"stage {stage_number:02d} sharpening resampling",
            log,
        )
        del sharpen_weights

        stage_samples_path = stage_dir / "samples.npy"
        sharp_acceptance = mala_to_disk(
            work_a,
            stage_samples_path,
            target_sharp,
            target.domain,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            501,
            stage_number,
            args.mc_dt,
            args.mc_steps,
            args.mc_image_radius,
            f"stage {stage_number:02d} post-sharpening MALA",
            log,
        )
        np.save(
            stage_dir / "post_sharpen_acceptance.npy",
            sharp_acceptance,
            allow_pickle=False,
        )
        elapsed = time.perf_counter() - stage_started
        metadata = {
            "stage": stage_number,
            "t_start": float(saved["t_start"]),
            "t": float(saved["t"]),
            "rg_start": list(rg_start),
            "rg_end": list(rg_end),
            "saved_selection": saved["selected"],
            "saved_flow_path": saved["selected_flow_path"],
            "saved_flow_sha256": sha256(
                TRAINING_RUN / saved["selected_flow_path"]
            ),
            "sample_count": args.sample_count,
            "flow_ress": flow_ress,
            "pre_sharpen_mala_acceptance_mean": float(
                soft_acceptance.mean()
            ),
            "sharpen_ress": sharpen_ress,
            "post_sharpen_mala_acceptance_mean": float(
                sharp_acceptance.mean()
            ),
            "elapsed_seconds": elapsed,
            "samples_path": str(stage_samples_path.relative_to(output_dir)),
        }
        atomic_json(stage_dir / "metadata.json", metadata)
        manifest["stages"].append(metadata)
        atomic_json(output_dir / "run.json", manifest)
        current_path = stage_samples_path
        last_target = target_sharp
        log(
            f"STAGE {stage_number:02d} COMPLETE | flow ESS={flow_ress:.6f} | "
            f"sharpening ESS={sharpen_ress:.6f} | "
            f"post-MALA acceptance={float(sharp_acceptance.mean()):.6f} | "
            f"elapsed={elapsed / 60.0:.2f} min | saved={stage_samples_path}"
        )

    reached_final = bool(stage_items and float(stage_items[-1]["t"]) == 1.0)
    if reached_final:
        raw_started = time.perf_counter()
        raw_dir = output_dir / "raw"
        raw_dir.mkdir()
        raw_weights_path = raw_dir / "log_weights.npy"
        log("RAW START | regularized endpoint -> raw physical potential")
        potential_weights_to_disk(
            current_path,
            raw_weights_path,
            last_target,
            target,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            "raw-potential weights",
            log,
        )
        raw_weights, raw_ress = weights_and_ress(
            raw_weights_path, args.sample_count
        )
        log(f"endpoint-to-physical RESS={raw_ress:.9f}")
        resample_to_disk(
            current_path,
            work_a,
            raw_weights,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            601,
            len(stage_items) + 1,
            "raw-potential resampling",
            log,
        )
        del raw_weights
        raw_samples_path = raw_dir / "samples.npy"
        raw_acceptance = mala_to_disk(
            work_a,
            raw_samples_path,
            target,
            target.domain,
            shape,
            args.chunk_size,
            args.log_every_chunks,
            701,
            len(stage_items) + 1,
            args.mc_dt,
            args.mc_steps,
            args.mc_image_radius,
            "raw-potential MALA",
            log,
        )
        np.save(
            raw_dir / "acceptance.npy", raw_acceptance, allow_pickle=False
        )
        raw_metadata = {
            "target": "raw physical reduced potential at 300 K",
            "sample_count": args.sample_count,
            "reweighting_ress": raw_ress,
            "mala_acceptance_mean": float(raw_acceptance.mean()),
            "elapsed_seconds": time.perf_counter() - raw_started,
            "samples_path": str(raw_samples_path.relative_to(output_dir)),
        }
        atomic_json(raw_dir / "metadata.json", raw_metadata)
        manifest["raw"] = raw_metadata
        manifest["status"] = "complete"
        atomic_json(output_dir / "run.json", manifest)
        log(
            f"RAW COMPLETE | endpoint-to-physical RESS={raw_ress:.9f} | "
            f"MALA acceptance={float(raw_acceptance.mean()):.6f} | "
            f"saved={raw_samples_path}"
        )
    else:
        manifest["status"] = "partial"
        atomic_json(output_dir / "run.json", manifest)

    log(
        f"FINISH status={manifest['status']} | "
        f"completed stages={len(manifest['stages'])} | "
        f"training updates={manifest['training_updates']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise

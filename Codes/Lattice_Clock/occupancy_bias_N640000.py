"""Batch-size scaling of staged-sampler occupancy bias at N=640000.

For each trained schedule in ``B_VALUES``, or in the ``--batch`` list, run the same
staged sampler used by ``occupancy_bias_B2000.py`` for both losses (KL+X_pi
and KLXX): load the saved stage flows without retraining, apply each inverse map,
reweight and multinomially resample, then rejuvenate with MALA at the new
stage potential.  Report

    err = (1/6) * sum_s |p_s - 1/6|

at N=640000, averaged over the four matched sampling seeds 60001--60004.
Those are exactly the seeds used by the N=640000 row of
``occupancy_bias_B2000.py``, so the B=2000 result is directly reproducible.

Production is split by method.  Each method writes a raw partial archive
below ``artifacts/occupancy_bias_N640000``; ``--merge`` combines them into a
plot-ready NPZ and writes CSV/Markdown summaries below ``results/``.

Run from the repository root:
    python Codes_New/Lattice_Clock/occupancy_bias_N640000.py --method klx
    python Codes_New/Lattice_Clock/occupancy_bias_N640000.py --method klxx
    python Codes_New/Lattice_Clock/occupancy_bias_N640000.py --merge

``--batch 1500`` restricts a run to one trained batch size, which is how a
schedule trained on another machine is analyzed where its artifacts live.
"""

import argparse
import gc
import math
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"

from jflows.flow import NCSF
from jflows.potential import Nlog_Uniform, linear_combination
from jflows.utils import langevin, resample
from potential import Clock, magnetization

METHODS = ("klx", "klxx")
SCHEDULE_CONTRACT = "paired_klx_t_hist_v1"

# Physics and MALA configuration shared by every trained artifact; any KLXX
# archive carries them, so the first one present is read.
_REF_RUNS = sorted(ARTIFACTS.glob("klxx_B*/data.npz"))
if not _REF_RUNS:
    raise FileNotFoundError(f"no klxx_B*/data.npz archive under {ARTIFACTS}")
with np.load(_REF_RUNS[0], allow_pickle=False) as _ref:
    if "schedule_contract" not in _ref.files or \
            str(_ref["schedule_contract"]) != SCHEDULE_CONTRACT:
        raise ValueError(f"{_REF_RUNS[0]} has an incompatible schedule contract")
    L, D, P = int(_ref["L"]), int(_ref["D"]), int(_ref["P"])
    J, H = float(_ref["J"]), float(_ref["H"])
    MC_DT, MC_STEPS = float(_ref["mc_dt"]), int(_ref["mc_steps"])

# Flow architecture from train.py.
NSF_LIM = math.pi
BINS, TRANSFORMS, HIDDEN_FEATURES = 16, 6, (256, 256)

N_PARTICLES = 640000
B_VALUES = (2000, 1500, 1000, 500)   # --batch overrides this for a single-batch run
# Match occupancy_bias_B2000.py at k=6: seed = 10000*k + repetition + 1.
SEEDS = (60001, 60002, 60003, 60004)
# 160k requires a 16.65 GiB inverse-map allocation under JAX 0.11 and OOMs
# on the 32 GiB production GPU once the remaining buffers are resident.
CHUNK_SIZE = 80000

ANALYSIS = ARTIFACTS / "occupancy_bias_N640000"
STATUS = ANALYSIS / "status.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as fh:
        fh.write(line + "\n")


u0 = Nlog_Uniform(a=[-NSF_LIM] * D, b=[NSF_LIM] * D)
u1 = Clock(L, P, J, H)


@eqx.filter_jit
def _inv_ladj(flow, x):
    return flow.inv_and_ladj(x)


def read_run_config(method: str, batch_size: int) -> tuple[list[float], Path]:
    """Validate one trained run and return its accepted stage schedule and flow path."""
    run_dir = ARTIFACTS / f"{method}_B{batch_size}"
    data_path = run_dir / "data.npz"
    flow_path = run_dir / "flows.eqx"
    if not data_path.is_file() or not flow_path.is_file():
        raise FileNotFoundError(f"incomplete trained artifact directory: {run_dir}")

    with np.load(data_path, allow_pickle=False) as data:
        if "schedule_contract" not in data.files or \
                str(data["schedule_contract"]) != SCHEDULE_CONTRACT:
            raise ValueError(f"{run_dir}: incompatible schedule contract")
        if not bool(data["complete"]):
            raise ValueError(f"{run_dir}: training run is not complete")
        if str(data["method"]) != method:
            raise ValueError(f"{run_dir}: method metadata mismatch")
        if int(data["batch_size"]) != batch_size:
            raise ValueError(f"{run_dir}: batch-size metadata mismatch")
        config = (
            int(data["L"]), int(data["D"]), int(data["P"]),
            float(data["J"]), float(data["H"]),
            float(data["mc_dt"]), int(data["mc_steps"]),
        )
        expected = (L, D, P, J, H, MC_DT, MC_STEPS)
        if config != expected:
            raise ValueError(f"{run_dir}: physics or MALA configuration mismatch")
        t_hist = np.asarray(data["t_hist"], dtype=np.float64)

    if t_hist.ndim != 1 or not len(t_hist) or t_hist[-1] != 1.0:
        raise ValueError(f"{run_dir}: schedule is empty, malformed, or incomplete")
    if np.any(np.diff(t_hist) <= 0.0):
        raise ValueError(f"{run_dir}: schedule is not strictly increasing")
    return t_hist.tolist(), flow_path


def validate_artifacts() -> None:
    """Check every requested KL+X_pi/KLXX artifact pair before GPU work starts."""
    for batch_size in B_VALUES:
        t_hist_kl, _ = read_run_config("klx", batch_size)
        t_hist_klxx, _ = read_run_config("klxx", batch_size)
        if not np.array_equal(t_hist_kl, t_hist_klxx):
            raise ValueError(
                f"B={batch_size}: kl and klxx artifacts use different t_hist values"
            )


def load_run(method: str, batch_size: int):
    """Load the accepted stage schedule and stage flows for one trained run."""
    t_hist, flow_path = read_run_config(method, batch_size)
    like = [
        NCSF(
            jax.random.key(0),
            a=[-NSF_LIM] * D,
            b=[NSF_LIM] * D,
            bins=BINS,
            transforms=TRANSFORMS,
            hidden_features=HIDDEN_FEATURES,
        ).zeros()
        for _ in t_hist
    ]
    flows = eqx.tree_deserialise_leaves(flow_path, like)
    if len(flows) != len(t_hist):
        raise ValueError(
            f"{method} B={batch_size}: {len(flows)} flows != "
            f"{len(t_hist)} stages"
        )
    return t_hist, flows


def sector_bias(y: np.ndarray) -> tuple[float, list[float]]:
    """Return (1/P) sum_s |p_s - 1/P| and the sector probabilities."""
    m = magnetization(y)
    sector = np.round(np.angle(m) * P / (2.0 * math.pi)).astype(np.int64) % P
    p = np.bincount(sector, minlength=P).astype(np.float64) / y.shape[0]
    return float(np.abs(p - 1.0 / P).sum() / P), p.tolist()


def staged_sample(n: int, seed: int, t_hist, flows) -> np.ndarray:
    """Run one independent staged sampler and return its particles on host."""
    key = jax.random.key(seed)
    y = u0.samples(jax.random.fold_in(key, 0), n)
    t_prev = 0.0
    for stage, (flow, t_k) in enumerate(zip(flows, t_hist), start=1):
        u_prev = linear_combination([u1, u0], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([u1, u0], [t_k, 1.0 - t_k])
        key_k = jax.random.fold_in(key, stage)

        # Use one fixed compiled shape for inverse maps and stage log-weights.
        outs, log_weights = [], []
        for start in range(0, n, CHUNK_SIZE):
            xb = y[start:start + CHUNK_SIZE]
            size = xb.shape[0]
            if size < CHUNK_SIZE:
                xb = jnp.concatenate(
                    [xb, jnp.broadcast_to(xb[-1:], (CHUNK_SIZE - size, D))],
                    axis=0,
                )
            y_push, ladj = _inv_ladj(flow, xb)
            outs.append(y_push[:size])
            log_weights.append(
                u_prev(xb[:size]) - u_k(y_push[:size]) + ladj[:size]
            )
        del y
        y_push = jnp.concatenate(outs)
        logw = jnp.concatenate(log_weights)
        del outs, log_weights

        # One multinomial resampling operation per independent sampling seed.
        y = resample(
            jax.random.fold_in(key_k, 1000),
            y_push,
            jnp.exp(logw - logw.max()),
        )
        del y_push, logw

        # MALA rejuvenation at the current stage potential.
        rejuvenated = []
        for chunk, start in enumerate(range(0, n, CHUNK_SIZE)):
            xb = y[start:start + CHUNK_SIZE]
            size = xb.shape[0]
            if size < CHUNK_SIZE:
                xb = jnp.concatenate(
                    [xb, jnp.broadcast_to(xb[-1:], (CHUNK_SIZE - size, D))],
                    axis=0,
                )
            rejuvenated.append(
                langevin(
                    jax.random.fold_in(key_k, 2000 + chunk),
                    xb,
                    u_k,
                    dt=MC_DT,
                    steps=MC_STEPS,
                    adjust=True,
                )[:size]
            )
        del y
        y = jnp.concatenate(rejuvenated)
        y = jax.block_until_ready(y)
        del rejuvenated
        t_prev = t_k

    y_host = np.asarray(y, dtype=np.float32)
    del y
    return y_host


def batch_size_rows(method: str, raw: dict) -> list[dict]:
    """Run all requested B values for one method and retain every seed result."""
    rows = []
    for batch_size in B_VALUES:
        t_hist, flows = load_run(method, batch_size)
        raw[f"t_hist_{method}_B{batch_size}"] = np.asarray(
            t_hist, dtype=np.float64
        )
        log(
            f"--- {method} B={batch_size}: "
            f"t_hist={[f'{t:.4f}' for t in t_hist]} ({len(t_hist)} stages) ---"
        )

        errors, probabilities = [], []
        for repetition, seed in enumerate(SEEDS, start=1):
            started = time.perf_counter()
            y_host = staged_sample(N_PARTICLES, seed, t_hist, flows)
            error, probability = sector_bias(y_host)
            errors.append(error)
            probabilities.append(probability)
            elapsed = time.perf_counter() - started
            log(
                f"[{method} B={batch_size}] seed {repetition}/{len(SEEDS)} "
                f"({seed}) N={N_PARTICLES} bias={error:.5f} "
                f"sector_probs={['%.3f' % value for value in probability]} "
                f"({elapsed:.0f}s)"
            )
            del y_host

        errors_array = np.asarray(errors, dtype=np.float64)
        raw[f"bias_{method}_B{batch_size}"] = errors_array
        raw[f"probs_{method}_B{batch_size}"] = np.asarray(
            probabilities, dtype=np.float64
        )
        sem = (
            float(errors_array.std(ddof=1) / np.sqrt(len(SEEDS)))
            if len(SEEDS) > 1 else float("nan")
        )
        rows.append(
            dict(
                method=method,
                B=batch_size,
                N=N_PARTICLES,
                reps=len(SEEDS),
                bias=float(errors_array.mean()),
                sem=sem,
            )
        )
        log(
            f"[{method} B={batch_size}] DONE mean bias={rows[-1]['bias']:.5f} "
            f"sem={sem:.5f}"
        )

        # Retain compilations within a run, but release them before loading B's
        # distinct schedule and flow chain.
        del flows
        gc.collect()
        jax.clear_caches()
        log(f"{method} B={batch_size}: buffers + compile cache released")
    return rows


def rows_from_raw(method: str, raw: dict) -> list[dict]:
    """Reconstruct summary rows from a merged per-seed archive."""
    n = int(raw["N"])
    seeds = np.asarray(raw["seeds"], dtype=np.int64)
    rows = []
    for batch_size in np.asarray(raw["B_values"], dtype=np.int64):
        batch_size = int(batch_size)
        errors = np.asarray(raw[f"bias_{method}_B{batch_size}"], dtype=np.float64)
        probabilities = np.asarray(
            raw[f"probs_{method}_B{batch_size}"], dtype=np.float64
        )
        if errors.shape != (len(seeds),):
            raise ValueError(
                f"{method} B={batch_size}: {errors.shape} bias array, "
                f"expected {(len(seeds),)}"
            )
        if probabilities.shape != (len(seeds), P):
            raise ValueError(
                f"{method} B={batch_size}: {probabilities.shape} probability "
                f"array, expected {(len(seeds), P)}"
            )
        rows.append(
            dict(
                method=method,
                B=batch_size,
                N=n,
                reps=len(seeds),
                bias=float(errors.mean()),
                sem=(
                    float(errors.std(ddof=1) / np.sqrt(len(seeds)))
                    if len(seeds) > 1 else float("nan")
                ),
            )
        )
    return rows


def write_summary(raw: dict, rows: dict[str, list[dict]], suffix: str) -> None:
    """Write the merged raw archive and machine/human-readable summaries."""
    data_out = ANALYSIS / f"data{suffix}.npz"
    csv_out = RESULTS / f"occupancy_bias_N640000{suffix}.csv"
    md_out = RESULTS / f"occupancy_bias_N640000{suffix}.md"
    np.savez_compressed(data_out, **raw)
    log(f"raw per-seed data saved: {data_out} ({len(raw)} arrays)")

    import csv
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["method", "B", "N", "reps", "bias", "sem"],
            lineterminator="\n",
        )
        writer.writeheader()
        for method in METHODS:
            writer.writerows(rows[method])

    seeds = [int(seed) for seed in np.asarray(raw["seeds"])]
    with open(md_out, "w") as fh:
        fh.write(
            "# Occupancy bias across training batch sizes "
            f"(L=8 clock, staged sampler, N={int(raw['N'])})\n\n"
            "err = (1/6) sum_s |p_s - 1/6|; mean and SEM over "
            f"{len(seeds)} matched sampling seeds {seeds}.\n"
        )
        for method in METHODS:
            method_rows = rows[method]
            fh.write(f"\n## {method}\n\n")
            fh.write(
                "| | " + " | ".join(f"B={row['B']}" for row in method_rows)
                + " |\n"
            )
            fh.write("|---|" + "---|" * len(method_rows) + "\n")
            fh.write(
                "| mean occupancy bias | "
                + " | ".join(f"{row['bias']:.5f}" for row in method_rows)
                + " |\n"
            )
            fh.write(
                "| sem (over seeds) | "
                + " | ".join(f"{row['sem']:.5f}" for row in method_rows)
                + " |\n"
            )
            fh.write(
                "| seeds | "
                + " | ".join(str(row["reps"]) for row in method_rows)
                + " |\n"
            )

    log(
        "##### OCC-BIAS-N640000 MERGED "
        + " ".join(
            f"{method}: biases="
            f"{['%.5f' % row['bias'] for row in rows[method]]}"
            for method in METHODS
        )
        + " #####"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--method",
        choices=METHODS,
        help="run one production method; run both separately, then --merge",
    )
    group.add_argument(
        "--merge",
        action="store_true",
        help="merge the two completed method archives without GPU work",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="small pipeline check: N=2000 and one seed for every B",
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        help="restrict the run to these batch sizes instead of every entry of B_VALUES",
    )
    args = parser.parse_args()
    if args.batch:
        global B_VALUES
        B_VALUES = tuple(args.batch)

    if not args.smoke and args.method is None and not args.merge:
        parser.error("production runs require --method kl, --method klxx, then --merge")

    global N_PARTICLES, SEEDS, CHUNK_SIZE
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        N_PARTICLES = 2000
        SEEDS = (60001,)
        CHUNK_SIZE = 2000

    validate_artifacts()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fresh_log = args.method == METHODS[0] or (args.smoke and args.method is None)
    with open(STATUS, "w" if fresh_log and not args.merge else "a"):
        pass

    if args.merge:
        raw = {}
        shared_keys = {"N", "B_values", "seeds"}
        for method in METHODS:
            part = ANALYSIS / f"data{suffix}_{method}.npz"
            with np.load(part, allow_pickle=False) as data:
                current = dict(data)
            for key in shared_keys:
                if key not in current:
                    raise ValueError(f"{part}: missing shared metadata {key}")
                if key in raw and not np.array_equal(raw[key], current[key]):
                    raise ValueError(f"{part}: incompatible {key}")
                raw[key] = current[key]
            for key, value in current.items():
                if key not in shared_keys:
                    if key in raw:
                        raise ValueError(f"{part}: duplicate key {key}")
                    raw[key] = value
        rows = {method: rows_from_raw(method, raw) for method in METHODS}
        write_summary(raw, rows, suffix)
        return

    methods = (args.method,) if args.method is not None else METHODS
    part_suffix = f"{suffix}_{args.method}" if args.method is not None else suffix
    data_out = ANALYSIS / f"data{part_suffix}.npz"
    log(
        f"##### OCC-BIAS-N640000 START | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | L={L} D={D} P={P} J={J} H={H} | "
        f"MC={MC_DT}x{MC_STEPS} (MALA) | methods={methods} "
        f"N={N_PARTICLES} B_values={B_VALUES} seeds={SEEDS} "
        f"chunk={CHUNK_SIZE} #####"
    )
    started = time.perf_counter()
    raw = {
        "N": np.asarray(N_PARTICLES, dtype=np.int64),
        "B_values": np.asarray(B_VALUES, dtype=np.int64),
        "seeds": np.asarray(SEEDS, dtype=np.int64),
    }
    rows = {method: batch_size_rows(method, raw) for method in methods}
    np.savez_compressed(data_out, **raw)
    log(f"raw method data saved: {data_out} ({len(raw)} arrays)")

    if args.method is not None:
        log(
            f"##### OCC-BIAS-N640000 {args.method} DONE "
            f"wall={time.perf_counter() - started:.0f}s #####"
        )
        return

    write_summary(raw, rows, suffix)


if __name__ == "__main__":
    main()

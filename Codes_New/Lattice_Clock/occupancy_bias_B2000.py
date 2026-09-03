"""Monte Carlo scaling of the staged-sampler occupancy bias (B=2000 runs).

Runs the exact staged sampler of each trained B=2000 stage schedule—forward KL and KLXX,
per stage: load the stage flow G_k (artifacts/<method>_B2000/flows.eqx, no
retraining), push the chunked compiled inverse, logw = U_{k-1}(x) - U_k(y)
+ ladj, multinomial resample PER TEST BLOCK, MALA rejuvenation on U_k — at
N = 10000 * 2^k for k = 0..6 with 2^(8-k) independent tests (equal total
work 2.56M particles per row), and reports the mean occupancy bias

    err = (1/6) * sum_s |p_s - 1/6|

for both losses against the N^{-1/2} (pure finite-size) Monte Carlo
reference.

Memory control: one compiled chunk shape (160k) for every heavy op, eager
per-chunk loops with prompt frees, sector statistics on host, and a full
buffer + compile-cache release between the two methods.

The production workload is split by method so no single GPU process exceeds
two hours. Each method writes a temporary raw archive below
``artifacts/occupancy_bias_B2000``; a zero-GPU merge produces the combined raw
NPZ and final Markdown/CSV tables below ``results/``. The figure is rendered
separately from the merged raw NPZ.
Reads ``artifacts/{kl,klxx}_B2000/data.npz`` (schedule + config) and the
corresponding ``flows.eqx`` files.

Run from the repo root:
    python Codes/Lattice_Clock/occupancy_bias_B2000.py --method kl
    python Codes/Lattice_Clock/occupancy_bias_B2000.py --method klxx
    python Codes/Lattice_Clock/occupancy_bias_B2000.py --merge
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

METHODS = ("kl", "klxx")
METHOD_LABEL = {
    "kl":   "forward KL",
    "klxx": r"KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$",
}
METHOD_COLOR = {"kl": "tab:blue", "klxx": "tab:purple"}
SCHEDULE_CONTRACT = "paired_kl_t_hist_v1"

# physics + MC config from the paired B=2000 artifacts
with np.load(ARTIFACTS / "klxx_B2000" / "data.npz", allow_pickle=False) as _ref:
    if "schedule_contract" not in _ref.files or \
            str(_ref["schedule_contract"]) != SCHEDULE_CONTRACT:
        raise ValueError("B=2000 klxx artifact has an incompatible schedule contract")
    L, D, P = int(_ref["L"]), int(_ref["D"]), int(_ref["P"])
    J, H = float(_ref["J"]), float(_ref["H"])
    MC_DT, MC_STEPS = float(_ref["mc_dt"]), int(_ref["mc_steps"])
    _t_hist_ref = np.asarray(_ref["t_hist"])
with np.load(ARTIFACTS / "kl_B2000" / "data.npz", allow_pickle=False) as _kl:
    if "schedule_contract" not in _kl.files or \
            str(_kl["schedule_contract"]) != SCHEDULE_CONTRACT:
        raise ValueError("B=2000 kl artifact has an incompatible schedule contract")
    if not np.array_equal(np.asarray(_kl["t_hist"]), _t_hist_ref):
        raise ValueError("B=2000 kl and klxx artifacts use different t_hist values")

# flow architecture from train.py
NSF_LIM = math.pi
BINS, TRANSFORMS, HIDDEN_FEATURES = 16, 6, (256, 256)

BASE_SZIE = 10000
MAX_REPORT_K = 6
KS = list(range(MAX_REPORT_K + 1))       # N = BASE_SZIE * 2^k, k = 0..6
REPS = {k: 2 ** (8 - k) for k in KS}     # equal total work per row
CHUNK_SIZE = 160000                      # one compiled shape for the heavy ops (200k OOMs the compiled inverse)

ANALYSIS = ARTIFACTS / "occupancy_bias_B2000"
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


def load_run(method: str):
    """(accepted t_hist, stage flows) of the trained B=2000 run."""
    run_dir = ARTIFACTS / f"{method}_B2000"
    with np.load(run_dir / "data.npz", allow_pickle=False) as data:
        assert (int(data["L"]), int(data["P"]), float(data["J"]), float(data["H"]),
                float(data["mc_dt"]), int(data["mc_steps"])) == \
               (L, P, J, H, MC_DT, MC_STEPS), f"{method}: config mismatch"
        t_hist = [float(t) for t in data["t_hist"]]
    like = [NCSF(jax.random.key(0), a=[-NSF_LIM] * D, b=[NSF_LIM] * D,
                 bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN_FEATURES).zeros()
            for _ in range(len(t_hist))]
    flows = eqx.tree_deserialise_leaves(run_dir / "flows.eqx", like)
    assert len(flows) == len(t_hist), (
        f"{method}: {len(flows)} flows != {len(t_hist)} stages"
    )
    return t_hist, flows


def sector_bias(y: np.ndarray):
    """err = (1/6) sum_s |p_s - 1/6| from magnetization sectors."""
    m = magnetization(y)
    sector = np.round(np.angle(m) * P / (2.0 * math.pi)).astype(np.int64) % P
    p = np.bincount(sector, minlength=P).astype(np.float64) / y.shape[0]
    return float(np.abs(p - 1.0 / P).sum() / P), p.tolist()


def staged_sample(n: int, G: int, seed: int, t_hist, flows) -> np.ndarray:
    """G INDEPENDENT tests of n particles each, stacked into one [G*n, D]
    GPU pass. Map + Langevin act per particle (block-agnostic); the
    reweight/RESAMPLE is done PER TEST BLOCK, so the G tests are exactly
    independent staged samplers. Returns the final set on HOST memory."""
    total = G * n
    key = jax.random.key(seed)
    y = u0.samples(jax.random.fold_in(key, 0), total)
    t_prev = 0.0
    for k, (flow, t_k) in enumerate(zip(flows, t_hist), start=1):
        u_prev = linear_combination([u1, u0], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([u1, u0], [t_k, 1.0 - t_k])
        key_k = jax.random.fold_in(key, k)
        # chunked compiled inverse + stage log-weights (one shape: CHUNK_SIZE)
        outs, lws = [], []
        for i in range(0, total, CHUNK_SIZE):
            xb = y[i:i + CHUNK_SIZE]
            nb = xb.shape[0]
            if nb < CHUNK_SIZE:                           # pad: ONE compile shape
                xb = jnp.concatenate(
                    [xb, jnp.broadcast_to(xb[-1:], (CHUNK_SIZE - nb, D))], axis=0)
            yt, ladj = _inv_ladj(flow, xb)
            outs.append(yt[:nb])
            lws.append(u_prev(xb[:nb]) - u_k(yt[:nb]) + ladj[:nb])
        del y
        y_push = jnp.concatenate(outs)
        logw = jnp.concatenate(lws)
        del outs, lws
        # multinomial resample per test block (independent staged samplers)
        blocks = []
        for b in range(G):
            sl = slice(b * n, (b + 1) * n)
            lw = logw[sl]
            blocks.append(resample(jax.random.fold_in(key_k, 1000 + b),
                                   y_push[sl], jnp.exp(lw - lw.max())))
        del y_push, logw
        y = jnp.concatenate(blocks)
        del blocks
        # MALA rejuvenation at U_k, chunked at the same compile shape
        rejuv = []
        for j, i in enumerate(range(0, total, CHUNK_SIZE)):
            xb = y[i:i + CHUNK_SIZE]
            nb = xb.shape[0]
            if nb < CHUNK_SIZE:
                xb = jnp.concatenate(
                    [xb, jnp.broadcast_to(xb[-1:], (CHUNK_SIZE - nb, D))], axis=0)
            rejuv.append(
                langevin(
                    jax.random.fold_in(key_k, 2000 + j), xb, u_k,
                    dt=MC_DT, steps=MC_STEPS, adjust=True,
                )[:nb]
            )
        del y
        y = jnp.concatenate(rejuv)
        y = jax.block_until_ready(y)
        del rejuv
        t_prev = t_k
    y_np = np.asarray(y, dtype=np.float32)   # sector stats on host
    del y
    return y_np


def scaling_rows(method: str, raw: dict) -> list[dict]:
    """The full N-scaling test of one method's staged sampler. Every
    per-test bias and sector histogram lands in `raw` (saved to npz)."""
    t_hist, flows = load_run(method)
    log(f"--- {method}: t_hist={[f'{t:.4f}' for t in t_hist]} "
        f"({len(t_hist)} stages) ---")
    raw[f"t_hist_{method}"] = np.asarray(t_hist, dtype=np.float64)
    rows = []
    for k in KS:
        n, reps = BASE_SZIE * 2 ** k, REPS[k]
        G = max(1, min(reps, CHUNK_SIZE // n))   # tests stacked per GPU pass
        G = 2 ** int(math.log2(G))               # power of two: reps % G == 0
        assert reps % G == 0, f"{method} k={k}: reps={reps} not divisible by G={G}"
        errs, probs = [], []
        for grp in range(reps // G):
            t1 = time.perf_counter()
            y_np = staged_sample(n, G, seed=10000 * k + grp + 1,
                                 t_hist=t_hist, flows=flows)
            dt = time.perf_counter() - t1
            for b in range(G):
                err, p = sector_bias(y_np[b * n:(b + 1) * n])
                errs.append(err)
                probs.append(p)
                log(f"[{method} k={k}] test {len(errs):>3}/{reps}  N={n}  "
                    f"bias={err:.5f}  sector_probs={['%.3f' % q for q in p]}  "
                    f"({dt / G:.0f}s)")
            del y_np
        errs = np.asarray(errs)
        raw[f"bias_{method}_k{k}"] = errs.astype(np.float64)
        raw[f"probs_{method}_k{k}"] = np.asarray(probs, dtype=np.float64)
        rows.append(dict(method=method, k=k, N=n, reps=reps, bias=errs.mean(),
                         sem=errs.std(ddof=1) / np.sqrt(reps) if reps > 1
                         else float("nan")))
        log(f"[{method} k={k}] DONE  mean bias={errs.mean():.5f}  "
            f"sem={rows[-1]['sem']:.5f}")
    # release CUDA memory before the next method (the earlier OOM cause)
    del flows
    gc.collect()
    jax.clear_caches()
    log(f"{method}: buffers + compile cache released")
    return rows


def rows_from_raw(method: str, raw: dict) -> list[dict]:
    """Reconstruct summary rows from one method's saved per-test arrays."""
    n_base = int(raw["BASE_SZIE"])
    ks = [int(k) for k in raw["ks"]]
    reps_by_k = {
        k: int(reps) for k, reps in zip(ks, np.asarray(raw["reps"]))
    }
    rows = []
    for k in ks:
        errs = np.asarray(raw[f"bias_{method}_k{k}"], dtype=np.float64)
        reps = reps_by_k[k]
        if errs.shape != (reps,):
            raise ValueError(
                f"{method} k={k}: {errs.shape} bias array, expected {(reps,)}"
            )
        rows.append(
            dict(
                method=method,
                k=k,
                N=n_base * 2 ** k,
                reps=reps,
                bias=float(errs.mean()),
                sem=(
                    float(errs.std(ddof=1) / np.sqrt(reps))
                    if reps > 1 else float("nan")
                ),
            )
        )
    return rows


def write_summary(raw: dict, rows: dict[str, list[dict]], suffix: str) -> None:
    """Write the merged raw archive and final human/machine summaries."""
    data_out = ANALYSIS / f"data{suffix}.npz"
    csv_out = RESULTS / f"occupancy_bias_B2000{suffix}.csv"
    md_out = RESULTS / f"occupancy_bias_B2000{suffix}.md"
    np.savez_compressed(data_out, **raw)
    log(f"raw per-test data saved: {data_out} ({len(raw)} arrays)")
    report_rows = {
        method: [row for row in rows[method] if row["k"] <= MAX_REPORT_K]
        for method in METHODS
    }

    import csv
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["method", "k", "N", "reps", "bias", "sem"],
            lineterminator="\n",
        )
        w.writeheader()
        for method in METHODS:
            w.writerows(report_rows[method])

    slopes = {}
    with open(md_out, "w") as f:
        f.write("# Occupancy-bias Monte Carlo scaling (L=8 clock, staged "
                "sampler, B=2000, forward KL and KLXX)\n\nerr = (1/6) sum_s "
                "|p_s - 1/6|; equal total work per row (10000*256 "
                "particles); 2^(8-k) independent tests at N=10000*2^k; "
                "k=0,...,6.\n")
        for method in METHODS:
            r = report_rows[method]
            ratios = [r[i]["bias"] / r[i + 1]["bias"]
                      for i in range(len(r) - 1)]
            slopes[method] = np.polyfit(
                np.log([x["N"] for x in r]),
                np.log([x["bias"] for x in r]), 1,
            )[0]
            f.write(f"\n## {method}\n\n")
            f.write("| | " + " | ".join(
                f"N=1e4*2^{x['k']}" for x in r) + " |\n")
            f.write("|---|" + "---|" * len(r) + "\n")
            f.write("| mean occupancy bias | "
                    + " | ".join(f"{x['bias']:.5f}" for x in r) + " |\n")
            f.write("| sem (over tests) | "
                    + " | ".join(f"{x['sem']:.5f}" for x in r) + " |\n")
            f.write("| tests | " + " | ".join(str(x["reps"]) for x in r)
                    + " |\n\n")
            f.write("Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): "
                    f"{['%.2f' % x for x in ratios]}\n")
            f.write(f"\nLog-log slope of bias vs N: {slopes[method]:.3f} "
                    f"(Monte Carlo rate = -0.5)\n")

    log("##### OCC-BIAS-B2000 MERGED "
        + " ".join(
            f"{method}: biases={['%.5f' % x['bias'] for x in report_rows[method]]} "
            f"slope={slopes[method]:.3f}" for method in METHODS
        ) + " #####")


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--method", choices=METHODS,
        help="run one production method; run both methods separately, then --merge",
    )
    group.add_argument(
        "--merge", action="store_true",
        help="merge the two completed method archives without GPU work",
    )
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sanity run: BASE_SZIE=2000, k in {0,1}, 2/1 reps")
    args = ap.parse_args()

    if not args.smoke and args.method is None and not args.merge:
        ap.error("production runs require --method kl, --method klxx, then --merge")

    global BASE_SZIE, KS, REPS
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        BASE_SZIE, KS, REPS = 2000, [0, 1], {0: 2, 1: 1}

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    open(STATUS, "w" if args.method == METHODS[0] and not args.smoke else "a").close()

    if args.merge:
        raw = {}
        for method in METHODS:
            part = ANALYSIS / f"data{suffix}_{method}.npz"
            with np.load(part, allow_pickle=False) as data:
                current = dict(data)
            for key in ("BASE_SZIE", "ks", "reps"):
                if key in raw and not np.array_equal(raw[key], current[key]):
                    raise ValueError(f"{part}: incompatible {key}")
                raw[key] = current[key]
            for key, value in current.items():
                if key not in {"BASE_SZIE", "ks", "reps"}:
                    if key in raw:
                        raise ValueError(f"{part}: duplicate key {key}")
                    raw[key] = value
        rows = {method: rows_from_raw(method, raw) for method in METHODS}
        write_summary(raw, rows, suffix)
        return

    methods = (args.method,) if args.method is not None else METHODS
    part_suffix = f"{suffix}_{args.method}" if args.method is not None else suffix
    data_out = ANALYSIS / f"data{part_suffix}.npz"

    log(f"##### OCC-BIAS-B2000 START | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | L={L} D={D} P={P} J={J} H={H} | "
        f"MC={MC_DT}x{MC_STEPS} (MALA) | methods={methods} "
        f"BASE_SZIE={BASE_SZIE} ks={KS} reps={REPS} chunk={CHUNK_SIZE} #####")
    t0 = time.perf_counter()
    raw = {
        "BASE_SZIE": BASE_SZIE,
        "ks": np.asarray(KS),
        "reps": np.asarray([REPS[k] for k in KS]),
    }
    rows = {method: scaling_rows(method, raw) for method in methods}
    np.savez_compressed(data_out, **raw)
    log(f"raw method data saved: {data_out} ({len(raw)} arrays)")

    if args.method is not None:
        log(f"##### OCC-BIAS-B2000 {args.method} DONE "
            f"wall={time.perf_counter() - t0:.0f}s #####")
        return

    write_summary(raw, rows, suffix)


if __name__ == "__main__":
    main()

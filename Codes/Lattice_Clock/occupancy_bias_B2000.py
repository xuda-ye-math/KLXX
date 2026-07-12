"""Monte Carlo scaling of the staged-sampler occupancy bias (B=2000 runs).

Runs the EXACT staged sampler of each trained B=2000 ladder — kl and klxx,
per stage: load the stage flow G_k (flows_<method>_B2000.eqx, no
retraining), push the chunked compiled inverse, logw = U_{k-1}(x) - U_k(y)
+ ladj, multinomial resample PER TEST BLOCK, MALA rejuvenation on U_k — at
N = 10000 * 2^k for k = 0..8 with 2^(8-k) independent tests (equal total
work 2.56M particles per row), and reports the mean occupancy bias

    err = (1/6) * sum_s |p_s - 1/6|

for both losses against the N^{-1/2} (pure finite-size) Monte Carlo
reference.

Memory control: one compiled chunk shape (160k) for every heavy op, eager
per-chunk loops with prompt frees, sector statistics on host, and a full
buffer + compile-cache release between the two methods.

Writes (new files only): occupancy_bias_B2000.md/.csv, the raw per-test
data occupancy_bias_B2000_data.npz, and occupancy_bias_B2000_status.log next to
this file. NO figure here — the figure is rendered separately from the
saved data npz.
Reads data_{kl,klxx}_B2000.npz (ladder + config) and
flows_{kl,klxx}_B2000.eqx.

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows python \
        Codes/Lattice_Clock/occupancy_bias_B2000.py [--smoke]
"""

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                             # Lattice_Clock/potential.py

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

# physics + MC config from the klxx npz; asserted identical across methods
_REF = np.load(HERE / "data_klxx_B2000.npz")
L, D, P = int(_REF["L"]), int(_REF["D"]), int(_REF["P"])
J, H = float(_REF["J"]), float(_REF["H"])
MC_STEP, MC_ITERS = float(_REF["mc_step"]), int(_REF["mc_iters"])

# flow architecture (matches train.py, not stored in the npz)
NSF_LIM = math.pi
BINS, TRANSFORMS, HIDDEN_FEATURES = 16, 6, (256, 256)

N_BASE = 10000
KS = list(range(9))                      # N = N_BASE * 2^k, k = 0..8
REPS = {k: 2 ** (8 - k) for k in KS}     # equal total work per row
CHUNK_SIZE = 160000                      # one compiled shape for the heavy ops (200k OOMs the compiled inverse)

STATUS = HERE / "occupancy_bias_B2000_status.log"


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
    """(ladder, stage flows) of the trained B=2000 run of `method`."""
    data = np.load(HERE / f"data_{method}_B2000.npz")
    assert (int(data["L"]), int(data["P"]), float(data["J"]), float(data["H"]),
            float(data["mc_step"]), int(data["mc_iters"])) == \
           (L, P, J, H, MC_STEP, MC_ITERS), f"{method}: config mismatch"
    ladder = [float(t) for t in data["ladder"]]
    like = [NCSF(jax.random.key(0), a=[-NSF_LIM] * D, b=[NSF_LIM] * D,
                 bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN_FEATURES).zeros()
            for _ in range(len(ladder))]
    flows = eqx.tree_deserialise_leaves(HERE / f"flows_{method}_B2000.eqx", like)
    assert len(flows) == len(ladder), f"{method}: {len(flows)} flows != {len(ladder)} rungs"
    return ladder, flows


def sector_bias(y: np.ndarray):
    """err = (1/6) sum_s |p_s - 1/6| from magnetization sectors."""
    m = magnetization(y)
    sector = np.round(np.angle(m) * P / (2.0 * math.pi)).astype(np.int64) % P
    p = np.bincount(sector, minlength=P).astype(np.float64) / y.shape[0]
    return float(np.abs(p - 1.0 / P).sum() / P), p.tolist()


def staged_sample(n: int, G: int, seed: int, ladder, flows) -> np.ndarray:
    """G INDEPENDENT tests of n particles each, stacked into one [G*n, D]
    GPU pass. Map + Langevin act per particle (block-agnostic); the
    reweight/RESAMPLE is done PER TEST BLOCK, so the G tests are exactly
    independent staged samplers. Returns the final set on HOST memory."""
    total = G * n
    key = jax.random.key(seed)
    y = u0.samples(jax.random.fold_in(key, 0), total)
    t_prev = 0.0
    for k, (flow, t_k) in enumerate(zip(flows, ladder), start=1):
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
            rejuv.append(langevin(jax.random.fold_in(key_k, 2000 + j), xb, u_k,
                                  step=MC_STEP, iters=MC_ITERS, adjust=True)[:nb])
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
    ladder, flows = load_run(method)
    log(f"--- {method}: ladder={[f'{t:.4f}' for t in ladder]} "
        f"({len(ladder)} stages) ---")
    raw[f"ladder_{method}"] = np.asarray(ladder, dtype=np.float64)
    rows = []
    for k in KS:
        n, reps = N_BASE * 2 ** k, REPS[k]
        G = max(1, min(reps, CHUNK_SIZE // n))   # tests stacked per GPU pass
        G = 2 ** int(math.log2(G))               # power of two: reps % G == 0
        assert reps % G == 0, f"{method} k={k}: reps={reps} not divisible by G={G}"
        errs, probs = [], []
        for grp in range(reps // G):
            t1 = time.perf_counter()
            y_np = staged_sample(n, G, seed=10000 * k + grp + 1,
                                 ladder=ladder, flows=flows)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sanity run: N_BASE=2000, k in {0,1}, 2/1 reps")
    args = ap.parse_args()

    global N_BASE, KS, REPS
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        N_BASE, KS, REPS = 2000, [0, 1], {0: 2, 1: 1}

    open(STATUS, "a").close()
    log(f"##### OCC-BIAS-B2000 START | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | L={L} D={D} P={P} J={J} H={H} | "
        f"MC={MC_STEP}x{MC_ITERS} (MALA) | methods={METHODS} "
        f"N_BASE={N_BASE} ks={KS} reps={REPS} chunk={CHUNK_SIZE} #####")

    t0 = time.perf_counter()
    raw = {"N_BASE": N_BASE, "ks": np.asarray(KS),
           "reps": np.asarray([REPS[k] for k in KS])}
    rows = {m: scaling_rows(m, raw) for m in METHODS}

    # ---- raw per-test data (figure + stats re-renderable without rerun) ----
    np.savez_compressed(HERE / f"occupancy_bias_B2000{suffix}_data.npz", **raw)
    log(f"raw per-test data saved: occupancy_bias_B2000{suffix}_data.npz "
        f"({len(raw)} arrays)")

    # ---- table ----
    import csv
    with open(HERE / f"occupancy_bias_B2000{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "k", "N", "reps", "bias", "sem"])
        w.writeheader()
        for m in METHODS:
            w.writerows(rows[m])
    slopes = {}
    with open(HERE / f"occupancy_bias_B2000{suffix}.md", "w") as f:
        f.write("# Occupancy-bias Monte Carlo scaling (L=8 clock, staged "
                "sampler, B=2000, kl and klxx)\n\nerr = (1/6) sum_s "
                "|p_s - 1/6|; equal total work per row (10000*256 "
                "particles); 2^(8-k) independent tests at N=10000*2^k.\n")
        for m in METHODS:
            r = rows[m]
            ratios = [r[i]["bias"] / r[i + 1]["bias"] for i in range(len(r) - 1)]
            slopes[m] = np.polyfit(np.log([x["N"] for x in r]),
                                   np.log([x["bias"] for x in r]), 1)[0]
            f.write(f"\n## {m}\n\n")
            f.write("| | " + " | ".join(f"N=1e4*2^{x['k']}" for x in r) + " |\n")
            f.write("|---|" + "---|" * len(r) + "\n")
            f.write("| mean occupancy bias | "
                    + " | ".join(f"{x['bias']:.5f}" for x in r) + " |\n")
            f.write("| sem (over tests) | "
                    + " | ".join(f"{x['sem']:.5f}" for x in r) + " |\n")
            f.write("| tests | " + " | ".join(str(x["reps"]) for x in r)
                    + " |\n\n")
            f.write(f"Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): "
                    f"{['%.2f' % x for x in ratios]}\n")
            f.write(f"\nLog-log slope of bias vs N: {slopes[m]:.3f} "
                    f"(Monte Carlo rate = -0.5)\n")

    log(f"##### OCC-BIAS-B2000 DONE wall={time.perf_counter() - t0:.0f}s "
        + " ".join(f"{m}: biases={['%.5f' % x['bias'] for x in rows[m]]} "
                   f"slope={slopes[m]:.3f}" for m in METHODS) + " #####")


if __name__ == "__main__":
    main()

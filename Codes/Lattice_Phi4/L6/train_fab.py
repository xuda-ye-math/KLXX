"""Train FAB on this lattice and merge its results with the stored runs of train.py.

The FAB method is ``jflows.train.train_FAB_G`` (samples from pi^2/nu by the
two-phase SMC of the package, no replay buffer). Every constant, the source,
the target, and the flow are imported from this folder's ``train.py``, so the
run differs from the four stored methods only in the objective. The FAB rows
are appended to ``results/results_table.csv`` and the FAB magnetizations and
weights are added to ``artifacts/data.npz``, both written by train.py; the log
is ``artifacts/train_fab.log``.
"""

import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from jflows.train import Monitor, train_FAB_G
from jflows.utils import compute_ESS_log, importance_weights_log, linear_weights_from_log

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_phi4_train", HERE / "train.py")
B = importlib.util.module_from_spec(spec)
sys.modules["_phi4_train"] = B
spec.loader.exec_module(B)

ARTIFACTS, RESULTS = B.ARTIFACTS, B.RESULTS
LOG = ARTIFACTS / "train_fab.log"
NAME = "FAB"


def log(message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a") as stream:
        stream.write(line + "\n")


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()
    log(
        f"START Phi4 L={B.L} D={B.D} FAB | {jax.default_backend()} | "
        f"VALID_SZIE={B.VALID_SZIE} BATCH_SZIE={B.BATCH_SZIE} | STEPS_TOTAL={B.STEPS_TOTAL} "
        f"LR={B.LR} LADDER={B.LADDER} MC={B.MC_DT}x{B.MC_STEPS_1} seeds={B.SEEDS} | "
        f"constants imported from {HERE / 'train.py'}"
    )
    reference = np.load(ARTIFACTS / "phi4_reference.npz")
    log(f"reference p(m>0) = {float((reference['m_trace'] > 0).mean()):.4f}")
    x_valid = B.source.samples(jax.random.key(2), B.VALID_SZIE)
    flow0 = B.NSF(
        jax.random.key(0), a=[-B.NSF_LIM] * B.D, b=[B.NSF_LIM] * B.D,
        bins=B.BINS, transforms=B.TRANSFORMS, hidden_features=B.HIDDEN_FEATURES,
    ).zeros()

    arrays = dict(np.load(ARTIFACTS / "data.npz"))
    with open(RESULTS / "results_table.csv", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] != NAME]
    arrays = {key: value for key, value in arrays.items() if not key.endswith(f"_{NAME}")}

    for seed in B.SEEDS:
        started = time.time()
        flow, _ = train_FAB_G(
            x_valid, B.source, B.target, flow0,
            batch_size=B.BATCH_SZIE, steps_total=B.STEPS_TOTAL, lr=B.LR,
            ladder=B.LADDER, mc_dt=B.MC_DT, mc_steps_1=B.MC_STEPS_1, mc_steps_2=B.MC_STEPS_2,
            mc_adjust=True,
            monitor=Monitor(500, f"[s{seed} {NAME}] ", log), seed=seed,
            u_clip=B.U_CLIP, g_clip=B.G_CLIP,
        )
        flow = jax.block_until_ready(flow)
        jax.effects_barrier()
        samples = flow.inv(x_valid)
        log_weights = importance_weights_log(x_valid, B.source, B.target, flow, type="G")
        ess = float(compute_ESS_log(log_weights))
        weights = np.asarray(linear_weights_from_log(log_weights))
        magnetization = np.asarray(samples).mean(axis=1)
        weights = weights / weights.sum()
        p_plus = float(weights[magnetization > 0].sum())
        rows.append({"seed": seed, "method": NAME, "final_ess": round(ess, 4), "p_plus": round(p_plus, 4)})
        arrays[f"mag_{seed}_{NAME}"] = magnetization.astype(np.float32)
        arrays[f"w_{seed}_{NAME}"] = weights.astype(np.float32)
        log(f"[s{seed} {NAME}] DONE after {time.time() - started:.1f}s | ESS={ess:.4f} p(m>0)={p_plus:.4f}")

    rows.sort(key=lambda row: (int(row["seed"]), row["method"] != NAME))
    with open(RESULTS / "results_table.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "method", "final_ess", "p_plus"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(ARTIFACTS / "data.npz", **arrays)
    log("DONE — FAB rows merged into results_table.csv and data.npz")


if __name__ == "__main__":
    main()

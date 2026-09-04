"""HD Product, LDR-L1: 16 sequential tests at d = 256, 240, ..., 16 with `train_forward_KLL1_G`.

Every constant, the source, the target, the flow, and the artifact writers are
imported from this folder's `train_kl.py`; the target-batch construction is the
same as `KL+X_pi`, so `MC_STEPS_1` is the one from that file. The regularizer is
the centered L1 log-dispersion of Schopmans et al. instead of the pairwise
variation. Each dimension writes `artifacts/d{d}/kll1.npz` beside the forward-KL
files; the log is `artifacts/train_kll1.log`.
"""

import gc
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jflows.train import Monitor, train_forward_KLL1_G
from jflows.utils import compute_ESS_log

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_hd_train_kl", HERE / "train_kl.py")
B = importlib.util.module_from_spec(spec)
sys.modules["_hd_train_kl"] = B
spec.loader.exec_module(B)

ARTIFACTS = B.ARTIFACTS
LOG = ARTIFACTS / "train_kll1.log"
NAME = "KL+L1"
SLUG = "kll1"


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a") as stream:
        stream.write(line + "\n")


def run_dimension(d: int) -> None:
    run_dir = ARTIFACTS / f"d{d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = B.Nlog_Gaussian(mean=[0.0] * d, variance=[B.SIGMA**2] * d)
    target = B.make_target(d)
    x_valid = source.samples(jax.random.key(2), B.VALID_SIZE(d))
    x_eval = source.samples(jax.random.key(3), B.EVAL_SIZE)

    flow = B.make_flow(d)
    log(
        f"[d{d} {NAME}] START | backend={jax.default_backend()} | "
        f"VALID_SIZE={B.VALID_SIZE(d)} | EVAL_SIZE={B.EVAL_SIZE}"
    )
    started = time.time()
    flow, history = train_forward_KLL1_G(
        x_valid,
        source,
        target,
        flow,
        batch_size=B.BATCH_SZIE,
        steps_total=B.STEPS_TOTAL,
        lr=B.LR,
        ladder=B.LADDER,
        mc_dt=B.MC_DT,
        mc_steps_1=B.MC_STEPS_1,
        mc_steps_2=B.MC_STEPS_2,
        coeff_lambda=B.COEFF_LAMBDA,
        monitor=Monitor(20, f"[d{d} {NAME}] ", log),
        checkpoint=B.CHECKPOINT,
        g_clip=B.G_CLIP,
    )
    flow = jax.block_until_ready(flow)
    history = np.asarray(jax.device_get(history), dtype=np.float32)
    flow_path = run_dir / f"flow_{SLUG}.eqx"
    samples_path = run_dir / f"samples_{SLUG}.npy"
    B.save_flow(flow_path, flow)
    log(f"[d{d} {NAME}] flow saved; computing EVAL_SIZE ESS")

    log_weights = B.weights_and_samples(samples_path, x_eval, source, target, flow)
    final_ess = float(compute_ESS_log(jnp.asarray(log_weights)))
    data_path = run_dir / f"{SLUG}.npz"
    temporary = run_dir / f"{SLUG}.tmp.npz"
    np.savez_compressed(
        temporary,
        d=d,
        VALID_SIZE=B.VALID_SIZE(d),
        POOL_SIZE=0,
        EVAL_SIZE=B.EVAL_SIZE,
        BATCH_SZIE=B.BATCH_SZIE,
        STEPS_TOTAL=B.STEPS_TOTAL,
        MC_STEPS_1=B.MC_STEPS_1,
        MC_STEPS_2=B.MC_STEPS_2,
        COEFF_LAMBDA=B.COEFF_LAMBDA,
        CHUNKS=B.CHUNKS,
        CHECKPOINT=B.CHECKPOINT,
        jflows_version=B.JFLOWS_VERSION,
        batch_ess_history=history,
        log_weights=log_weights,
        final_ess=final_ess,
        flow_file=flow_path.name,
        samples_file=samples_path.name,
    )
    os.replace(temporary, data_path)
    log(
        f"[d{d} {NAME}] DONE after {time.time() - started:.1f}s | "
        f"ESS={final_ess:.4f} | saved {data_path}"
    )
    del flow, history, log_weights, x_valid, x_eval, source, target
    gc.collect()
    jax.clear_caches()
    log(f"=== d={d} COMPLETE; JAX caches released ===")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()
    log(
        f"START HD_Product KL+L1 | jflows {B.JFLOWS_VERSION} | jax {jax.__version__} | "
        f"dimensions={B.DIMENSIONS} | BATCH_SZIE={B.BATCH_SZIE} STEPS_TOTAL={B.STEPS_TOTAL} "
        f"LR={B.LR} LADDER={B.LADDER} MC_DT={B.MC_DT} MC_STEPS_1={B.MC_STEPS_1} "
        f"MC_STEPS_2={B.MC_STEPS_2} COEFF_LAMBDA={B.COEFF_LAMBDA} G_CLIP={B.G_CLIP} | "
        f"constants imported from {HERE / 'train_kl.py'}"
    )
    for d in B.DIMENSIONS:
        run_dimension(d)
    log("DONE — KL+L1 at all 16 dimensions d=256,240,...,16 complete")


if __name__ == "__main__":
    main()

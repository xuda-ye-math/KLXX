"""High-dimensional product multi-well — dimension sweep, four objectives.

The target in dimension d = 2**k factorizes,

    U(x) = 1/2 sum_i x_i^2 + 12 sum_{i<k} exp(-x_i^2),

so the first k coordinates are double wells (minima at +-sqrt(ln 24), barrier
~9.9 at the origin) and the rest are standard Gaussian: exactly 2**k modes
with an analytically known structure and a diagonal optimal map. Four
objectives per dimension, all trained by the packed jflows drivers from the
same identity-initialized NSF:

    KL                :  train_forward_KLX_G,  coeff_lambda = 0
    KL+X_mu           :  train_forward_KLX_G,  coeff_lambda = 1
    KL+X_mu+X_hat_mu  :  train_forward_KLXX_G, (alpha, beta) = (1, 0)
    KL+X_mu+X_mix     :  train_forward_KLXX_G, (alpha, beta) = (1/2, 1/2)

All Langevin kernels run MALA (mc_adjust = True); float32 throughout. Each
run is evaluated by its final ESS on a fresh capped source set and by the
sign-pattern mode occupancy of its pushforward (the 2**k bucket counts are
saved, so coverage and imbalance are recomputable at any threshold).

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows python \
        Codes/HD_Product/train.py
Writes temporary ``artifacts/k{k}/data.npz`` files (existing completed runs
are skipped, so the sweep resumes). Final tables and figures are built by the
separate analysis scripts under ``results/``.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, potential_from
from jflows.train import Monitor, train_forward_KLX_G, train_forward_KLXX_G
from jflows.utils import compute_ESS_log, importance_weights_log

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "train.log"

# k values to sweep — highest d first; d = 2**k
K_LIST = (8, 7, 6, 5, 4, 3, 2, 1)

# domain / source
SIGMA = 1.0            # std of the isotropic Gaussian source mu_0 = N(0, I)
NSF_LIM = 4.0          # NSF spline box half-width per coordinate

# NSF flow architecture (fixed across k)
BINS: int = 16
TRANSFORMS: int = 6
HIDDEN_FEATURES = (256, 256)

# training (fixed across k)
BATCH_SIZE: int = 250     # source samples per training step
LR: float = 1e-3       # Adam learning rate
G_CLIP: float = 1e2    # global gradient-norm clip (pre-Adam)

# data pipeline (single-hop AIS + MALA rejuvenation)
LADDER: int = 1        # AIS levels per manufactured target batch
MC_DT: float = 2e-3  # Langevin step size
MC_STEPS: int = 50     # Langevin steps per level / per hat_mu freshening

# quench and temper (the wide-coverage measure hat_mu)
MELT: float = 2.0      # melt scale (std of the Gaussian scatter)
OPT_ALPHA: float = 0.5  # L-BFGS trial alpha (armijo)
OPT_STEPS: int = 200   # L-BFGS iterations

# loss coefficients
COEFF_LAMBDA: float = 1.0
COEFF_ALPHA: float = 0.5
COEFF_BETA: float = 0.5

N_EVAL: int = 20000    # evaluation pool cap (ESS stable to ~1e-2 at 20k)

METHODS = (
    "KL",
    "KL+X_mu",
    "KL+X_mu+X_hat_mu",
    "KL+X_mu+X_mix",
)


def dimension(k):       return 2 ** k
def valid_size(k):      return 10000 * 2 ** k
def pool_size(k):       return 100 * 2 ** k


def train_steps(k):
    return 2000


def make_target(k):
    def energy(x):
        quad = 0.5 * (x ** 2).sum(axis=-1)
        well = 12.0 * jnp.exp(-(x[:, :k] ** 2)).sum(axis=-1)
        return quad + well
    return potential_from(energy)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START HD_Product sweep | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"K_LIST={K_LIST} BATCH_SIZE={BATCH_SIZE} LR={LR} g_clip={G_CLIP} MC={MC_DT}x{MC_STEPS} (MALA) "
        f"ladder={LADDER} QT: melt={MELT} opt={OPT_ALPHA}x{OPT_STEPS} N_EVAL={N_EVAL}")
    for k in K_LIST:
        d = dimension(k)
        out = ARTIFACTS / f"k{k}" / "data.npz"
        if out.exists():
            log(f"k={k} d={d}: {out} exists, skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        log(f"=== k={k} d={d} START (train_steps={train_steps(k)}, "
            f"valid_size={valid_size(k)}, pool_size={pool_size(k)}) ===")
        u0 = Nlog_Gaussian(mean=[0.0] * d, variance=[SIGMA**2] * d)
        u1 = make_target(k)
        x_valid = u0.samples(jax.random.key(2), valid_size(k))
        x_eval = u0.samples(jax.random.key(3), N_EVAL)
        flow0 = NSF(jax.random.key(0), a=[-NSF_LIM] * d, b=[NSF_LIM] * d,
                    bins=BINS, transforms=TRANSFORMS,
                    hidden_features=HIDDEN_FEATURES).zeros()

        store = {
            "k": k,
            "d": d,
            "valid_size": valid_size(k),
            "pool_size": pool_size(k),
            "batch_size": BATCH_SIZE,
            "train_steps": train_steps(k),
            "lr": LR,
            "ladder": LADDER,
            "mc_dt": MC_DT,
            "mc_steps": MC_STEPS,
            "mc_adjust": True,
        }
        for name in METHODS:
            t0 = time.time()
            mon = Monitor(200, f"[k{k} {name}] ", log)
            if name == "KL":
                flow, batch_ess_hist = train_forward_KLX_G(
                    x_valid, u0, u1, flow0,
                    batch_size=BATCH_SIZE, train_steps=train_steps(k), lr=LR,
                    ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=0.0, mc_adjust=True,
                    g_clip=G_CLIP, monitor=mon)
            elif name == "KL+X_mu":
                flow, batch_ess_hist = train_forward_KLX_G(
                    x_valid, u0, u1, flow0,
                    batch_size=BATCH_SIZE, train_steps=train_steps(k), lr=LR,
                    ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=1.0, mc_adjust=True,
                    g_clip=G_CLIP, monitor=mon)
            elif name == "KL+X_mu+X_hat_mu":
                flow, batch_ess_hist = train_forward_KLXX_G(
                    x_valid, u0, u1, flow0,
                    pool_size=pool_size(k), batch_size=BATCH_SIZE,
                    train_steps=train_steps(k), lr=LR, ladder=LADDER, melt=MELT,
                    opt_alpha=OPT_ALPHA, opt_steps=OPT_STEPS,
                    mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=COEFF_LAMBDA, coeff_alpha=1.0, coeff_beta=0.0,
                    mc_adjust=True,
                    g_clip=G_CLIP, monitor=mon)
            else:  # KL+X_mu+X_mix
                flow, batch_ess_hist = train_forward_KLXX_G(
                    x_valid, u0, u1, flow0,
                    pool_size=pool_size(k), batch_size=BATCH_SIZE,
                    train_steps=train_steps(k), lr=LR, ladder=LADDER, melt=MELT,
                    opt_alpha=OPT_ALPHA, opt_steps=OPT_STEPS,
                    mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=COEFF_LAMBDA, coeff_alpha=COEFF_ALPHA,
                    coeff_beta=COEFF_BETA, mc_adjust=True,
                    g_clip=G_CLIP, monitor=mon)
            flow = jax.block_until_ready(flow)
            jax.effects_barrier()

            y_push = flow.inv(x_eval)
            log_weights = importance_weights_log(
                x_eval, u0, u1, flow, type="G", chunks=4
            )
            ess = float(compute_ESS_log(log_weights))
            signs = (np.asarray(y_push[:, :k]) > 0).astype(np.int64)
            bucket = signs @ (2 ** np.arange(k))
            counts = np.bincount(bucket, minlength=2 ** k)
            store[f"batch_ess_hist_{name}"] = np.asarray(
                batch_ess_hist, dtype=np.float32
            )
            store[f"final_ess_{name}"] = ess
            store[f"bucket_counts_{name}"] = counts
            found = int((counts >= 0.5 * N_EVAL / 2 ** k).sum())
            log(f"[k{k} {name}] done in {time.time() - t0:.1f}s   "
                f"final ESS = {ess:.4f}   strict modes = {found}/{2 ** k}")
        np.savez_compressed(out, **store)
        log(f"k={k}: saved {out}")
        del x_valid, x_eval
    log(f"DONE — temporary artifacts in {ARTIFACTS}; "
        "build tables with build_table.py")


if __name__ == "__main__":
    main()

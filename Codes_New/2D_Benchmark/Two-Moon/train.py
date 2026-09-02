"""2D Two-Moon benchmark — FAB and the X-regularized forward KL family.

Five objectives, all trained by the packed jflows drivers from the same
identity-initialized NSF on the same validation set:

    FAB               :  train_FAB_G (pi^2/nu samples by two-phase SMC, no replay buffer)
    KL                :  train_forward_KLX_G,  coeff_lambda = 0
    KL+X_pi           :  train_forward_KLX_G,  coeff_lambda = 1
    KL+X_pi+X_hat_pi  :  train_forward_KLXX_G, (theta, alpha) = (1, 1)
    KL+X_pi+X_mix     :  train_forward_KLXX_G, (theta, alpha) = (1, 1/2)

The target approximates the two-moons density by a Gaussian mixture whose
centers sit evenly along the two interleaved crescent arcs (MOON_K centers
per moon, stroke width MOON_SIGMA) — two disjoint curved components, so the
benchmark stresses the flow's expressivity on distorted ridges and mode
discovery across the gap at once. All Langevin kernels run MALA
(mc_adjust = True). Final ESS is the flow importance sampling ESS on the
complete validation set, and coverage (k = 5) is measured against a
quench and temper reference sample set.

Run from the repo root:
    python Codes_New/2D_Benchmark/Two-Moon/train.py
Writes ``artifacts/data.npz`` (target energy grid, source samples, and the
pushforward samples, training ESS history, final ESS and coverage of each
method) and the log below ``artifacts/``. Render the figures from that file
with ``result.py``.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture
from jflows.train import Monitor, train_FAB_G, train_forward_KLX_G, train_forward_KLXX_G
from jflows.utils import (
    compute_ESS_log,
    coverage,
    importance_weights_log,
    quench_and_temper,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "train.log"
DATA = ARTIFACTS / "data.npz"

# boundary of the domain
SIGMA = 1.0            # standard deviation of the isotropic Gaussian source pi_0
PLT_LIM = 5.0          # half-width of the energy grid stored for the figures
NSF_LIM = 6.0          # half-width of the NSF spline domain

# NSF flow architecture
BINS: int = 32
TRANSFORMS: int = 6
HIDDEN_FEATURES = (128, 128)

# training parameters
VALID_SZIE: int = 50000   # complete validation set used for training and final ESS/coverage
BATCH_SZIE: int = 500     # source samples per training step
POOL_SIZE: int = 0        # 0: quench the complete validation set
STEPS_TOTAL: int = 200      # Adam optimization steps
LR: float = 1e-3       # Adam learning rate

# data pipeline (annealing + MALA rejuvenation)
LADDER: int = 1        # SMC levels per target batch (per phase for FAB)
MC_DT: float = 1e-3  # Langevin step size
MC_STEPS_1: int = 100   # Langevin steps per SMC level / per hat_pi MALA refresh
MC_STEPS_2: int = 100   # Langevin steps of the quench-and-temper pool temper

# quench and temper (the wide-coverage measure hat_pi)
REFER_SZIE: int = 1000    # coverage reference sample count
MELT: float = 2.0      # melt scale (std of the Gaussian scatter)
OPT_DT: float = 0.5  # L-BFGS initial trial step size (Armijo)
OPT_STEPS: int = 200   # L-BFGS iterations
QT_MC_STEPS: int = 1000  # temper length for the coverage reference set

COVERAGE_K: int = 5    # k-NN ball of the coverage metric

# two-moon mixture target: MOON_K Gaussian centers evenly spaced along each
# crescent arc (the two-moons parametrization, vertically stretched and with
# a vertical gap pushing the two components apart)
MOON_K: int = 48            # Gaussian centers per moon
MOON_SIGMA: float = 0.15    # per-mode standard deviation (stroke width)
MOON_SCALE: float = 2.0     # horizontal scale of the unit two-moons pair
MOON_Y_STRETCH: float = 1.4  # vertical stretch of the arcs (distortion)
MOON_Y_SEP: float = 0.35    # vertical half-gap between the components (unit arcs)

_t = np.linspace(0.0, np.pi, MOON_K)
_upper = np.stack([np.cos(_t) - 0.5,
                   MOON_Y_STRETCH * (np.sin(_t) + MOON_Y_SEP)], axis=-1)
_lower = np.stack([0.5 - np.cos(_t),
                   -MOON_Y_STRETCH * (np.sin(_t) + MOON_Y_SEP)], axis=-1)
MODE_MEANS = (MOON_SCALE * np.concatenate([_upper, _lower], axis=0)).tolist()

METHODS = (
    "FAB",
    "KL",
    "KL+X_pi",
    "KL+X_pi+X_hat_pi",
    "KL+X_pi+X_mix",
)
GRID_SIZE: int = 300      # energy grid points per axis
PRIOR_SIZE: int = 5000    # source samples drawn for the figure background


# source: Gaussian pi_0
u0 = Nlog_Gaussian(mean=[0.0, 0.0], variance=[SIGMA**2, SIGMA**2])

# target: two-moon Gaussian mixture
u1 = Nlog_Gaussian_Mixture(
    weights=[1.0] * len(MODE_MEANS),
    mean=MODE_MEANS,
    variance=[[MOON_SIGMA**2, MOON_SIGMA**2]] * len(MODE_MEANS),
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START Two-Moon | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"VALID_SZIE={VALID_SZIE} BATCH_SZIE={BATCH_SZIE} STEPS_TOTAL={STEPS_TOTAL} LR={LR} "
        f"MC={MC_DT}x{MC_STEPS_1}/{MC_STEPS_2} (MALA) ladder={LADDER} "
        f"QT: REFER_SZIE={REFER_SZIE} melt={MELT} opt={OPT_DT}x{OPT_STEPS} | "
        f"moons: K={MOON_K}/moon sigma={MOON_SIGMA} scale={MOON_SCALE}")
    x_valid = u0.samples(jax.random.key(2), VALID_SZIE)
    flow0 = NSF(jax.random.key(0), a=[-NSF_LIM, -NSF_LIM], b=[NSF_LIM, NSF_LIM],
                bins=BINS, transforms=TRANSFORMS,
                hidden_features=HIDDEN_FEATURES).zeros()

    # independent coverage reference set: quench and temper on the target
    log("building the quench and temper coverage reference sample set ...")
    y_hat_ref = quench_and_temper(
        jax.random.key(1), u0.samples(jax.random.key(4), REFER_SZIE), u1,
        melt=MELT, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
        mc_dt=MC_DT, mc_steps=QT_MC_STEPS, mc_adjust=True,
    )
    y_hat_ref = jax.block_until_ready(y_hat_ref)
    log(f"QT reference set ready ({REFER_SZIE} samples)")

    # figure background: target energy on a grid and a source sample cloud
    xs = np.linspace(-PLT_LIM, PLT_LIM, GRID_SIZE)
    X1, X2 = np.meshgrid(xs, xs, indexing="xy")
    grid = jnp.stack([jnp.asarray(X1.ravel()), jnp.asarray(X2.ravel())], axis=-1)
    store = {"X1": X1, "X2": X2,
             "U_grid": np.asarray(u1(grid)).reshape(X1.shape),
             "prior": np.asarray(u0.samples(jax.random.key(42), PRIOR_SIZE))}
    np.savez(DATA, **store)
    log(f"wrote the energy grid and source samples to {DATA}")

    for name in METHODS:
        t0 = time.time()
        mon = Monitor(200, f"[{name}] ", log)
        if name == "FAB":
            flow, batch_ess_hist = train_FAB_G(
                x_valid, u0, u1, flow0,
                batch_size=BATCH_SZIE, steps_total=STEPS_TOTAL, lr=LR,
                ladder=LADDER, mc_dt=MC_DT, mc_steps_1=MC_STEPS_1,
                mc_adjust=True, monitor=mon)
        elif name == "KL":
            flow, batch_ess_hist = train_forward_KLX_G(
                x_valid, u0, u1, flow0,
                batch_size=BATCH_SZIE, steps_total=STEPS_TOTAL, lr=LR,
                ladder=LADDER, mc_dt=MC_DT, mc_steps_1=MC_STEPS_1,
                coeff_lambda=0.0, mc_adjust=True, monitor=mon)
        elif name == "KL+X_pi":
            flow, batch_ess_hist = train_forward_KLX_G(
                x_valid, u0, u1, flow0,
                batch_size=BATCH_SZIE, steps_total=STEPS_TOTAL, lr=LR,
                ladder=LADDER, mc_dt=MC_DT, mc_steps_1=MC_STEPS_1,
                coeff_lambda=1.0, mc_adjust=True, monitor=mon)
        elif name == "KL+X_pi+X_hat_pi":
            flow, batch_ess_hist = train_forward_KLXX_G(
                x_valid, u0, u1, flow0,
                pool_size=POOL_SIZE,
                batch_size=BATCH_SZIE, steps_total=STEPS_TOTAL, lr=LR,
                ladder=LADDER, melt=MELT, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
                mc_dt=MC_DT, mc_steps_1=MC_STEPS_1, mc_steps_2=MC_STEPS_2,
                coeff_lambda=1.0, coeff_theta=1.0, coeff_alpha=1.0,
                mc_adjust=True, monitor=mon)
        else:  # KL+X_pi+X_mix
            flow, batch_ess_hist = train_forward_KLXX_G(
                x_valid, u0, u1, flow0,
                pool_size=POOL_SIZE,
                batch_size=BATCH_SZIE, steps_total=STEPS_TOTAL, lr=LR,
                ladder=LADDER, melt=MELT, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
                mc_dt=MC_DT, mc_steps_1=MC_STEPS_1, mc_steps_2=MC_STEPS_2,
                coeff_lambda=1.0, coeff_theta=1.0, coeff_alpha=0.5,
                mc_adjust=True, monitor=mon)
        flow = jax.block_until_ready(flow)
        jax.effects_barrier()

        y_push = flow.inv(x_valid)
        log_weights = importance_weights_log(x_valid, u0, u1, flow, type="G")
        ess = float(compute_ESS_log(log_weights))
        cov = float(coverage(y_push, y_hat_ref, k=COVERAGE_K, chunks=10))
        store[f"samples_{name}"] = np.asarray(y_push)
        store[f"ess_hist_{name}"] = np.asarray(batch_ess_hist)
        store[f"final_ess_{name}"] = np.asarray(ess)
        store[f"coverage_{name}"] = np.asarray(cov)
        np.savez(DATA, **store)   # rewritten as each method completes
        log(f"[{name}] done in {time.time() - t0:.1f}s   final ESS = {ess:.4f}   "
            f"coverage_k={COVERAGE_K} = {cov:.4f}   (VALID_SZIE = {VALID_SZIE})")

    for name in METHODS:
        log(f"{name:<20s} final ESS = {float(store[f'final_ess_{name}']):.4f}   "
            f"coverage_k={COVERAGE_K} = {float(store[f'coverage_{name}']):.4f}")
    log(f"DONE — data at {DATA}; render the figures with result.py")


if __name__ == "__main__":
    main()

"""Train paired KL+X_pi and KLXX generators for the L=8 clock model.

KL+X_pi (the baseline) selects an adaptive stage schedule. KLXX uses the same
accepted stage schedule as a fixed schedule, with ``POOL_SIZE=0`` so quench and
temper acts on the complete validation set. The population returned by each
generator after its last accepted stage is evaluated directly (sector
occupancy, kNN coverage against the quench-and-temper reference set); no
composed stage map is formed. An optional argument from 1 to 5 selects one
batch-size pair. Outputs are written below ``artifacts/`` for the analysis
scripts.
"""

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

from jflows.train import Monitor
from jflows.boltzmann import (
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G_fixed,
)
from jflows.flow import NCSF
from jflows.potential import Nlog_Uniform
from jflows.utils import quench_and_temper
from potential import Clock, sector_occupancy, torus_coverage

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

# clock model (potential-specific)
L = 8                  # lattice side; D = L*L angles
D = L * L

LOG = ARTIFACTS / "train.log"
P = 6                  # number of clock states -> P symmetry-broken sectors
J = 1.0                # nearest-neighbor coupling
H = 0.5                # Z_p anisotropy strength

# domain / flow: NCSF on the torus [-pi, pi)^D
NSF_LIM = math.pi      # box half-width
BINS: int = 16
TRANSFORMS: int = 6
HIDDEN_FEATURES = (256, 256)

# training
VALID_SZIE: int = 500000
BATCH_SZIE_LIST = (2000, 1000, 500, 250, 125)
POOL_SIZE: int = 0
STEPS_TOTAL_LIST = (3000, 3000, 3000, 3000, 3000)
LR: float = 1e-3

# Langevin / annealing (MALA everywhere, at the target on every SMC level)
LADDER: int = 4        # SMC levels of the training batch
MC_DT: float = 1e-3  # Langevin step size
MC_STEPS_1: int = 20  # MALA steps on the intermediate SMC levels only
MC_STEPS_2: int = 100  # MALA steps of every other rejuvenation: last SMC level, QT batch draw, QT temper, stage advance

# quench and temper (the wide-coverage measure hat_pi)
MELT: float = 2 * math.pi  # wrapped-Gaussian melt: uniform on the torus
OPT_DT: float = 1e-2        # L-BFGS trial step size (armijo)
OPT_STEPS: int = 50       # L-BFGS iterations

# loss coefficients (klxx): the 0.5.4 run used coeff_alpha = coeff_beta = 0.5, i.e. a
# mixture weight (alpha + beta)^2 = 1 and a QT proportion alpha / (alpha + beta) = 0.5
COEFF_LAMBDA: float = 1.0
COEFF_THETA: float = 1.0   # weight of the mixture variation
COEFF_ALPHA: float = 0.5   # QT proportion of the mixture

# training screens (forwarded to the forward Boltzmann drivers)
U_CLIP: float = 1e3    # potential screen: samples with target > U_CLIP drop from the loss
G_CLIP: float = 1e2    # global gradient-norm clip (spike guard before Adam)

# adaptive stage schedule (bg_param of the klx driver; klxx inherits accepted t_hist)
BG_PARAM = {
    "t_safe": 0.25,        # first stage point (the safe start)
    "shrink_factor": 0.7,  # rejected stage: t_k <- t_prev + shrink (t_k - t_prev)
    "enlarge_factor": 1.5, # accepted stage: extrapolation growth factor
    "tau_valid": 0.4,      # validation ESS gate
    "t_tol": 1e-3,         # snap the initial guess to t = 1 when 1 - t_k < t_tol
    "max_stages": 30,      # stage-count safety cap
    "max_retry": 12,       # training attempts per stage before giving up
}

# evaluation (on the population returned after the last accepted stage)
MODE_FRAC: float = 0.01  # a sector counts as found if it holds >= MODE_FRAC/P mass
KNN_K: int = 5         # k of the kNN coverage reference set
CHUNKS: int = 16        # chunk count for the drivers' full-set evaluations
MONITOR_EVERY: int = 50

METHODS = ("klx", "klxx")
SCHEDULE_CONTRACT = "paired_klx_t_hist_v1"


# source: uniform on the torus; target: the clock model
u0 = Nlog_Uniform(a=[-NSF_LIM] * D, b=[NSF_LIM] * D)
u1 = Clock(L, P, J, H)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def attempt_history(stages, name, dtype):
    return np.concatenate(
        [np.asarray(stage[name], dtype=dtype) for stage in stages], axis=0
    )


def main(test: int | None = None) -> None:
    pairs = list(zip(BATCH_SZIE_LIST, STEPS_TOTAL_LIST))
    if test is not None:
        pairs = [pairs[test - 1]]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if test is None:
        open(LOG, "w").close()   # fresh log per full sweep; single-pair runs append
    log(f"START Lattice_Clock sweep | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | L={L} D={D} P={P} J={J} H={H} | "
        f"VALID_SZIE={VALID_SZIE} POOL_SIZE={POOL_SIZE} "
        f"pairs(batch_size,steps_total)={pairs} LR={LR} "
        f"ladder={LADDER} MC={MC_DT}x({MC_STEPS_1},{MC_STEPS_2}) (MALA) "
        f"QT: melt={MELT:.4f} opt={OPT_DT}x{OPT_STEPS} "
        f"u_clip={U_CLIP} g_clip={G_CLIP} bg={BG_PARAM}")

    x_valid = u0.samples(jax.random.key(2), VALID_SZIE)
    flow0 = NCSF(jax.random.key(0), a=[-NSF_LIM] * D, b=[NSF_LIM] * D,
                 bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN_FEATURES).zeros()

    # QT reference set on the full target (uniform melt = fresh uniform draw)
    t0 = time.time()
    hat_ref = quench_and_temper(
        jax.random.key(4), u0.samples(jax.random.key(5), VALID_SZIE), u1,
        melt=0.0, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
        mc_dt=MC_DT, mc_steps=MC_STEPS_2, chunks=CHUNKS,
        mc_adjust=True,
    )
    hat_ref = jax.block_until_ready(hat_ref)
    log(f"QT reference set built: {VALID_SZIE} samples on the full target "
        f"in {time.time() - t0:.1f}s")

    for batch_size, steps_total in pairs:
        t_hist_klx = None
        for method in METHODS:
            tag = f"{method}_B{batch_size}"
            run_dir = ARTIFACTS / tag
            out = run_dir / "data.npz"
            run_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== {tag} START (batch_size={batch_size}, "
                f"steps_total={steps_total}) ===")
            t0 = time.time()
            mon = Monitor(MONITOR_EVERY, f"[{tag}] ", log)
            if method == "klx":
                y, stages = boltzmann_forward_KLX_G(
                    x_valid, u0, u1, flow0,
                    batch_size=batch_size,
                    steps_total=steps_total, lr=LR, ladder=LADDER,
                    mc_dt=MC_DT, mc_steps_1=MC_STEPS_1, mc_steps_2=MC_STEPS_2,
                    coeff_lambda=COEFF_LAMBDA,
                    mc_adjust=True,
                    monitor=mon, bg_param=BG_PARAM, chunks=CHUNKS,
                    checkpoint=False, u_clip=U_CLIP, g_clip=G_CLIP)
                t_hist_klx = [float(s["t"]) for s in stages]
            else:  # KLXX: fixed schedule on the accepted KL+X_pi stage schedule
                if not t_hist_klx or t_hist_klx[-1] != 1.0:
                    raise RuntimeError(f"{tag}: completed KL+X_pi t_hist is required")
                log(f"[{tag}] fixed t_hist from KL+X_pi: "
                    f"{[f'{t:.4f}' for t in t_hist_klx]}")
                y, stages = boltzmann_forward_KLXX_G_fixed(
                    x_valid, u0, u1, flow0,
                    pool_size=POOL_SIZE,
                    batch_size=batch_size,
                    steps_total=steps_total, lr=LR, ladder=LADDER, melt=MELT,
                    opt_dt=OPT_DT, opt_steps=OPT_STEPS,
                    # The public fixed-driver keyword remains `t_list`; the
                    # experiment contract stores the accepted stage schedule
                    # under the legacy identifier `t_hist`.
                    mc_dt=MC_DT, mc_steps_1=MC_STEPS_1, mc_steps_2=MC_STEPS_2,
                    t_list=t_hist_klx,
                    coeff_lambda=COEFF_LAMBDA, coeff_theta=COEFF_THETA,
                    coeff_alpha=COEFF_ALPHA,
                    mc_adjust=True,
                    monitor=mon, chunks=CHUNKS,
                    checkpoint=False, u_clip=U_CLIP, g_clip=G_CLIP)
            y = jax.block_until_ready(y)
            jax.effects_barrier()
            wall = time.time() - t0

            t_hist = np.asarray([s["t"] for s in stages], dtype=np.float64)
            if not len(t_hist) or t_hist[-1] != 1.0:
                reached = float(t_hist[-1]) if len(t_hist) else 0.0
                raise RuntimeError(f"{tag}: incomplete stage schedule at t={reached:.4f}")
            if method == "klxx" and not np.array_equal(
                    t_hist, np.asarray(t_hist_klx, dtype=np.float64)):
                raise RuntimeError(f"{tag}: t_hist differs from paired KL+X_pi")
            valid_selected_ess = np.array(
                [s["valid_selected_ess"] for s in stages], dtype=np.float64
            )
            valid_trained_ess = np.array(
                [s["valid_trained_ess"] for s in stages], dtype=np.float64
            )
            valid_identity_ess = np.array(
                [s["valid_identity_ess"] for s in stages], dtype=np.float64
            )
            selected = np.asarray([s["selected"] for s in stages])
            attempt_count = np.asarray(
                [len(s["t_hist"]) for s in stages], dtype=np.int64
            )
            attempt_t_hist = attempt_history(stages, "t_hist", np.float64)
            batch_ess_hist = attempt_history(
                stages, "batch_ess_hist", np.float32
            )
            valid_trained_ess_hist = attempt_history(
                stages, "valid_trained_ess_hist", np.float64
            )
            valid_identity_ess_hist = attempt_history(
                stages, "valid_identity_ess_hist", np.float64
            )
            attempt_status_hist = attempt_history(
                stages, "attempt_status_hist", str
            )

            flows = [stage["flow"] for stage in stages]
            y_np = np.asarray(y)
            sec_valid = sector_occupancy(y_np, P, MODE_FRAC)
            knn_cov = torus_coverage(y_np[:10000], np.asarray(hat_ref[:2000]), k=KNN_K)

            np.savez_compressed(
                out,
                tag=tag, method=method, L=L, D=D, P=P, J=J, H=H,
                valid_size=VALID_SZIE, pool_size=POOL_SIZE,
                initialize_from_identity=True,
                batch_size=batch_size, train_steps=steps_total, lr=LR,
                ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS_2,
                mc_steps_1=MC_STEPS_1, mc_steps_2=MC_STEPS_2,
                mc_adjust=True, chunks=CHUNKS,
                t_hist=t_hist,
                schedule_contract=SCHEDULE_CONTRACT,
                valid_selected_ess=valid_selected_ess,
                valid_trained_ess=valid_trained_ess,
                valid_identity_ess=valid_identity_ess,
                selected=selected,
                attempt_count=attempt_count,
                attempt_t_hist=attempt_t_hist,
                batch_ess_hist=batch_ess_hist,
                valid_trained_ess_hist=valid_trained_ess_hist,
                valid_identity_ess_hist=valid_identity_ess_hist,
                attempt_status_hist=attempt_status_hist,
                complete=True,
                sectors_valid=int(sec_valid[0] * P), tv_valid=sec_valid[1],
                counts_valid=np.array(sec_valid[2]), abs_m_valid=sec_valid[3],
                knn_coverage=knn_cov, wall_s=wall,
                samples_valid=y_np.astype(np.float32),
            )
            eqx.tree_serialise_leaves(run_dir / "flows.eqx", flows)
            log(f"##### {tag} DONE K={len(stages)} complete=True "
                f"t_hist={[f'{t:.3f}' for t in t_hist]} "
                f"last_stage_ESS={valid_selected_ess[-1]:.4f} "
                f"sectors(valid)={int(sec_valid[0] * P)}/{P} tv={sec_valid[1]:.3f} "
                f"|m|={sec_valid[3]:.3f} knn={knn_cov:.3f} wall={wall:.0f}s #####")
            del y, y_np, stages, flows
            gc.collect()
            jax.clear_caches()
            log(f"{tag}: buffers + compile cache released")
    log(f"DONE — temporary artifacts in {ARTIFACTS}; "
        "build the table with build_table.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

"""Train paired forward-KL and KLXX generators for the L=8 clock model.

Forward KL selects an adaptive bridge. KLXX uses the same accepted bridge as
a fixed schedule, with ``POOL_SIZE=0`` so quench-and-temper acts on the full
validation population. An optional argument from 1 to 5 selects one batch-size
pair. Outputs are written below ``artifacts/`` for the analysis scripts.
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
    boltzmann_forward_KL_G,
    boltzmann_forward_KLXX_G_fixed,
)
from jflows.flow import NCSF
from jflows.potential import Nlog_Uniform
from jflows.utils import compute_ESS_log, quench_and_temper
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
VALID_SZIE: int = 400000
BATCH_SZIE_LIST = (2000, 1000, 500, 250, 125)
POOL_SIZE: int = 0
TRAIN_STEPS_LIST = (3000, 3000, 3000, 3000, 3000)
LR: float = 1e-3

# Langevin / AIS (MALA everywhere)
LADDER: int = 4        # levels of the SMC selection gate and the training AIS
MC_DT: float = 1e-3  # Langevin step size
MC_STEPS: int = 100    # Langevin steps per level / rejuvenation / QT temper

# quench and temper (the wide-coverage measure hat_mu)
MELT: float = 2 * math.pi  # wrapped-Gaussian melt: uniform on the torus
OPT_DT: float = 1e-2        # L-BFGS trial step size (armijo)
OPT_STEPS: int = 100       # L-BFGS iterations

# loss coefficients (klxx)
COEFF_LAMBDA: float = 1.0
COEFF_ALPHA: float = 0.5
COEFF_BETA: float = 0.5

# training screens (forwarded to the forward Boltzmann drivers)
U_CLIP: float = 1e3    # potential screen: samples with target > U_CLIP drop from the loss
G_CLIP: float = 1e2    # global gradient-norm clip (spike guard before Adam)

# adaptive ladder (bg_param of the kl driver; klxx inherits accepted t_hist)
BG_PARAM = {
    "t_safe": 0.25,        # stage-1 bridge coefficient (the safe start)
    "shrink_factor": 0.7,  # rejected stage: t_k <- t_prev + shrink (t_k - t_prev)
    "enlarge_factor": 2.0, # accepted stage: extrapolation growth factor
    "tau_smc": 0.7,        # SMC pre-selection gate on t_k
    "tau_ess": 0.4,        # incremental ESS acceptance threshold
    "t_tol": 1e-3,         # snap the initial guess to t = 1 when 1 - t_k < t_tol
    "max_stages": 30,      # ladder-length safety cap
    "max_retry": 12,       # training attempts per stage before giving up
}

# evaluation
EVAL_SZIE: int = 80000    # composed-pushforward evaluation set size
MODE_FRAC: float = 0.01  # a sector counts as found if it holds >= MODE_FRAC/P mass
KNN_K: int = 5         # k of the kNN coverage referee
CHUNKS: int = 16        # chunk count for the drivers' full-set evaluations
MONITOR_EVERY: int = 50

METHODS = ("kl", "klxx")
SCHEDULE_CONTRACT = "paired_kl_t_hist_v1"


# source: uniform on the torus; target: the clock model
u0 = Nlog_Uniform(a=[-NSF_LIM] * D, b=[NSF_LIM] * D)
u1 = Clock(L, P, J, H)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def compose_pushforward(flows, x):
    y, ladj_sum = x, jnp.zeros(x.shape[0], dtype=x.dtype)
    for flow in flows:
        y, ladj = eqx.filter_jit(flow.inv_and_ladj)(y)
        ladj_sum = ladj_sum + ladj
    return y, u0(x) - u1(y) + ladj_sum


def attempt_history(stages, name, dtype):
    return np.concatenate(
        [np.asarray(stage[name], dtype=dtype) for stage in stages], axis=0
    )


def main(test: int | None = None) -> None:
    pairs = list(zip(BATCH_SZIE_LIST, TRAIN_STEPS_LIST))
    if test is not None:
        pairs = [pairs[test - 1]]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if test is None:
        open(LOG, "w").close()   # fresh log per full sweep; single-pair runs append
    log(f"START Lattice_Clock sweep | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | L={L} D={D} P={P} J={J} H={H} | "
        f"VALID_SZIE={VALID_SZIE} POOL_SIZE={POOL_SIZE} "
        f"pairs(batch_size,train_steps)={pairs} LR={LR} "
        f"ladder={LADDER} MC={MC_DT}x{MC_STEPS} (MALA) "
        f"QT: melt={MELT:.4f} opt={OPT_DT}x{OPT_STEPS} "
        f"u_clip={U_CLIP} g_clip={G_CLIP} bg={BG_PARAM}")

    x_valid = u0.samples(jax.random.key(2), VALID_SZIE)
    x_eval = u0.samples(jax.random.key(3), EVAL_SZIE)
    flow0 = NCSF(jax.random.key(0), a=[-NSF_LIM] * D, b=[NSF_LIM] * D,
                 bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN_FEATURES).zeros()

    # QT referee set on the full target (uniform melt = fresh uniform draw)
    t0 = time.time()
    hat_ref = quench_and_temper(
        jax.random.key(4), u0.samples(jax.random.key(5), VALID_SZIE), u1,
        melt=0.0, opt_dt=OPT_DT, opt_steps=OPT_STEPS,
        mc_dt=MC_DT, mc_steps=MC_STEPS, chunks=CHUNKS,
        mc_adjust=True,
    )
    hat_ref = jax.block_until_ready(hat_ref)
    log(f"QT referee set built: {VALID_SZIE} particles on the full target "
        f"in {time.time() - t0:.1f}s")

    for batch_size, train_steps in pairs:
        t_hist_kl = None
        for method in METHODS:
            tag = f"{method}_B{batch_size}"
            run_dir = ARTIFACTS / tag
            out = run_dir / "data.npz"
            run_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== {tag} START (batch_size={batch_size}, "
                f"train_steps={train_steps}) ===")
            t0 = time.time()
            mon = Monitor(MONITOR_EVERY, f"[{tag}] ", log)
            if method == "kl":
                y, stages = boltzmann_forward_KL_G(
                    x_valid, u0, u1, flow0,
                    batch_size=batch_size,
                    train_steps=train_steps, lr=LR, ladder=LADDER,
                    mc_dt=MC_DT, mc_steps=MC_STEPS,
                    mc_adjust=True,
                    monitor=mon, bg_param=BG_PARAM, chunks=CHUNKS,
                    checkpoint=False, u_clip=U_CLIP, g_clip=G_CLIP)
                t_hist_kl = [float(s["t"]) for s in stages]
            else:  # klxx: fixed schedule on the exact accepted KL history
                if not t_hist_kl or t_hist_kl[-1] != 1.0:
                    raise RuntimeError(f"{tag}: completed kl t_hist is required")
                log(f"[{tag}] fixed t_hist from kl: "
                    f"{[f'{t:.4f}' for t in t_hist_kl]}")
                y, stages = boltzmann_forward_KLXX_G_fixed(
                    x_valid, u0, u1, flow0,
                    pool_size=POOL_SIZE,
                    batch_size=batch_size,
                    train_steps=train_steps, lr=LR, ladder=LADDER, melt=MELT,
                    opt_dt=OPT_DT, opt_steps=OPT_STEPS,
                    # The public fixed-driver keyword remains `t_list`; the
                    # experiment contract and saved accepted history are
                    # deliberately named `t_hist`.
                    mc_dt=MC_DT, mc_steps=MC_STEPS, t_list=t_hist_kl,
                    coeff_lambda=COEFF_LAMBDA, coeff_alpha=COEFF_ALPHA,
                    coeff_beta=COEFF_BETA,
                    mc_adjust=True,
                    monitor=mon, chunks=CHUNKS,
                    checkpoint=False, u_clip=U_CLIP, g_clip=G_CLIP)
            y = jax.block_until_ready(y)
            jax.effects_barrier()
            wall = time.time() - t0

            t_hist = np.asarray([s["t"] for s in stages], dtype=np.float64)
            if not len(t_hist) or t_hist[-1] != 1.0:
                reached = float(t_hist[-1]) if len(t_hist) else 0.0
                raise RuntimeError(f"{tag}: incomplete ladder at t={reached:.4f}")
            if method == "klxx" and not np.array_equal(
                    t_hist, np.asarray(t_hist_kl, dtype=np.float64)):
                raise RuntimeError(f"{tag}: t_hist differs from paired KL")
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
            y_push, logw = jax.block_until_ready(
                compose_pushforward(flows, x_eval)
            )
            final_ess = float(compute_ESS_log(logw))
            sec_push = sector_occupancy(np.asarray(y_push), P, MODE_FRAC)
            sec_valid = sector_occupancy(np.asarray(y[:EVAL_SZIE]), P, MODE_FRAC)
            knn_cov = torus_coverage(np.asarray(y_push[:10000]),
                                     np.asarray(hat_ref[:2000]), k=KNN_K)

            np.savez_compressed(
                out,
                tag=tag, method=method, L=L, D=D, P=P, J=J, H=H,
                valid_size=VALID_SZIE, pool_size=POOL_SIZE,
                initialize_from_identity=True,
                batch_size=batch_size, train_steps=train_steps, lr=LR,
                ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS,
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
                complete=True, final_ess=final_ess,
                sectors_push=int(sec_push[0] * P), tv_push=sec_push[1],
                counts_push=np.array(sec_push[2]), abs_m_push=sec_push[3],
                sectors_valid=int(sec_valid[0] * P), tv_valid=sec_valid[1],
                counts_valid=np.array(sec_valid[2]), abs_m_valid=sec_valid[3],
                knn_coverage=knn_cov, wall_s=wall,
                samples_push=np.asarray(y_push, dtype=np.float32),
                samples_valid=np.asarray(y[:EVAL_SZIE], dtype=np.float32),
                logw_push=np.asarray(logw, dtype=np.float32),
            )
            eqx.tree_serialise_leaves(run_dir / "flows.eqx", flows)
            log(f"##### {tag} DONE K={len(stages)} complete=True "
                f"t_hist={[f'{t:.3f}' for t in t_hist]} "
                f"final_ESS={final_ess:.4f} "
                f"sectors(push)={int(sec_push[0] * P)}/{P} tv={sec_push[1]:.3f} "
                f"|m|={sec_push[3]:.3f} knn={knn_cov:.3f} wall={wall:.0f}s #####")
            del y, stages, flows, y_push, logw
            gc.collect()
            jax.clear_caches()
            log(f"{tag}: buffers + compile cache released")
    log(f"DONE — temporary artifacts in {ARTIFACTS}; "
        "build the table with build_table.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

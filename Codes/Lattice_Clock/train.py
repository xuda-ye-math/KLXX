"""p-state clock Boltzmann generator — batch-size sweep, two objectives.

The target is the cold Boltzmann measure of the P=6 clock model on the
periodic 8 x 8 lattice (D = 64 angles, P symmetry-broken sectors), reached
from the uniform-on-torus source by the public jflows Boltzmann drivers. Two
objectives per batch size start from the same identity-initialized NCSF on
[-pi, pi)^D:

    kl   :  boltzmann_forward_KL_G — bare forward KL stages on the adaptive
            ladder (SMC selection gate + acceptance)
    klxx :  boltzmann_forward_KLXX_G_fixed — KL + X_mu + X_{(hat_mu+bar_nu)/2}
            stages, (alpha, beta) = (1/2, 1/2), per-stage QT pool, trained on
            the SAME accepted t_hist as the kl run (fixed schedule, no SMC gate,
            no acceptance), so the two losses see identical bridge increments

The sweep runs the (BATCH_SIZE, TRAIN_STEPS) pairs of BATCH_SIZE_LIST x TRAIN_STEPS_LIST in
order (kl then klxx at each size); an optional argument in {1, .., 5} runs a
single pair, e.g. `train.py 1` for BATCH_SIZE = 1000. When a kl data file
already exists its accepted `t_hist` is loaded from the npz, so the paired klxx run can
start (or resume) without retraining kl. All Langevin kernels run MALA
(mc_adjust = True); float32 throughout. The circular layers of the NCSF wrap
every input into the box, so unwrapped angles from Langevin / L-BFGS are
handled exactly on the torus.

Each run is evaluated by the composed pushforward of its stage flows on a
fresh source set: direct ESS of the composed generator, sector occupancy of
the pushforward and of the final particle set (coverage, TV from uniform,
mean |m|), and kNN coverage against a quench-and-temper reference set on the
full target.

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows python \
        Codes/Lattice_Clock/train.py
Writes temporary run data below ``artifacts/<tag>``. The public ``flow_dir``
interface saves every trained attempt below ``artifacts/<tag>/attempts``.
After the tables and figures have been rendered into ``results/``, the entire
``artifacts/`` directory may be removed. Build the summary table with
``build_table.py``.
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

from jflows.boltzmann import (
    boltzmann_forward_KL_G,
    boltzmann_forward_KLXX_G_fixed,
)
from jflows.flow import NCSF
from jflows.potential import Nlog_Uniform
from jflows.train import Monitor
from jflows.utils import compute_ESS_log, quench_and_temper

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
from potential import Clock, sector_occupancy, torus_coverage

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

# basic training parameters (standard valid / pool / batch sizes)
N_VALID: int = 400000        # validation/particle set size
POOL_SIZE: int = 100000         # pool size: SMC selection pool and QT pool
BATCH_SIZE_LIST = (1000, 500, 250, 125, 2000)  # batch sizes, in test order
TRAIN_STEPS_LIST = (3000, 3000, 3000, 3000, 3000)  # gradient steps per stage, paired with BATCH_SIZE_LIST
LR: float = 1e-3             # Adam learning rate

# Langevin / AIS (MALA everywhere)
LADDER: int = 4        # levels of the SMC selection gate and the training AIS
MC_DT: float = 1e-3  # Langevin step size
MC_STEPS: int = 100    # Langevin steps per level / rejuvenation / QT temper

# quench and temper (the wide-coverage measure hat_mu)
MELT: float = 2 * math.pi  # wrapped-Gaussian melt: uniform on the torus
OPT_ALPHA: float = 1e-2     # L-BFGS trial alpha (armijo)
OPT_STEPS: int = 100       # L-BFGS iterations

# loss coefficients (klxx)
COEFF_LAMBDA: float = 1.0
COEFF_ALPHA: float = 0.5
COEFF_BETA: float = 0.5

# training screens (forwarded to the forward Boltzmann drivers)
E_CLIP: float = 1e3    # energy screen: samples with target > E_CLIP drop from the loss
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
N_EVAL: int = 20000    # composed-pushforward evaluation set size
MODE_FRAC: float = 0.01  # a sector counts as found if it holds >= MODE_FRAC/P mass
KNN_K: int = 5         # k of the kNN coverage referee
CHUNKS: int = 64        # chunk count for the drivers' full-set evaluations
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


@eqx.filter_jit
def _stage_inverse(flow, y):
    return flow.inv_and_ladj(y)


def compose_pushforward(stages, x):
    """Push x ~ mu_0 through the composed stage inverses G_1^-1, ..., G_K^-1
    and return (y, logw) with logw = U_0(x) - U(y) + sum_k ladj_inv_k — the
    direct log importance weight of the composed generator vs the target."""
    y, ladj_sum = x, jnp.zeros(x.shape[0], dtype=x.dtype)
    for s in stages:
        y, ladj = _stage_inverse(s["flow"], y)
        ladj_sum = ladj_sum + ladj
    return y, u0(x) - u1(y) + ladj_sum


def main(test: int | None = None) -> None:
    pairs = list(zip(BATCH_SIZE_LIST, TRAIN_STEPS_LIST))
    if test is not None:
        pairs = [pairs[test - 1]]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if test is None:
        open(LOG, "w").close()   # fresh log per full sweep; single-pair runs append
    log(f"START Lattice_Clock sweep | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | L={L} D={D} P={P} J={J} H={H} | "
        f"N_VALID={N_VALID} POOL_SIZE={POOL_SIZE} "
        f"pairs(batch_size,train_steps)={pairs} LR={LR} "
        f"ladder={LADDER} MC={MC_DT}x{MC_STEPS} (MALA) "
        f"QT: melt={MELT:.4f} opt={OPT_ALPHA}x{OPT_STEPS} "
        f"e_clip={E_CLIP} g_clip={G_CLIP} bg={BG_PARAM}")

    x_valid = u0.samples(jax.random.key(2), N_VALID)
    x_eval = u0.samples(jax.random.key(3), N_EVAL)
    flow0 = NCSF(jax.random.key(0), a=[-NSF_LIM] * D, b=[NSF_LIM] * D,
                 bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN_FEATURES).zeros()

    # QT referee set on the full target (uniform melt = fresh uniform draw)
    t0 = time.time()
    hat_ref = quench_and_temper(
        jax.random.key(4), u0.samples(jax.random.key(5), POOL_SIZE), u1,
        melt=0.0, opt_alpha=OPT_ALPHA, opt_steps=OPT_STEPS,
        mc_dt=MC_DT, mc_steps=MC_STEPS, chunks=CHUNKS,
        mc_adjust=True,
    )
    hat_ref = jax.block_until_ready(hat_ref)
    log(f"QT referee set built: {POOL_SIZE} particles on the full target "
        f"in {time.time() - t0:.1f}s")

    for batch_size, train_steps in pairs:
        t_hist_kl = None       # accepted KL levels, inherited exactly by KLXX
        for method in METHODS:
            tag = f"{method}_B{batch_size}"
            run_dir = ARTIFACTS / tag
            out = run_dir / "data.npz"
            if out.exists():
                with np.load(out, allow_pickle=False) as data:
                    if "schedule_contract" not in data.files:
                        raise ValueError(
                            f"missing schedule contract (pre-fix artifact): {run_dir}"
                        )
                    complete_saved = bool(data["complete"])
                    t_hist_saved = [float(t) for t in data["t_hist"]]
                    schedule_contract = str(data["schedule_contract"])
                    identity = (
                        str(data["tag"]), str(data["method"]),
                        int(data["batch_size"]), int(data["train_steps"]),
                    )
                expected = (tag, method, batch_size, train_steps)
                if schedule_contract != SCHEDULE_CONTRACT:
                    raise ValueError(f"incompatible schedule contract: {run_dir}")
                if identity != expected or not complete_saved or not t_hist_saved \
                        or t_hist_saved[-1] != 1.0:
                    raise ValueError(f"incompatible or incomplete saved run: {run_dir}")
                if method == "kl":
                    t_hist_kl = t_hist_saved
                    log(f"{tag}: completed artifact exists, skipping "
                        f"(fixed t_hist for klxx: "
                        f"{[f'{t:.4f}' for t in t_hist_kl]})")
                else:
                    if t_hist_saved != t_hist_kl:
                        raise ValueError(f"{tag}: saved t_hist differs from paired kl")
                    log(f"{tag}: completed artifact exists, skipping")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== {tag} START (batch_size={batch_size}, "
                f"train_steps={train_steps}) ===")
            t0 = time.time()
            mon = Monitor(MONITOR_EVERY, f"[{tag}] ", log)
            if method == "kl":
                y, stages = boltzmann_forward_KL_G(
                    x_valid, u0, u1, flow0,
                    pool_size=POOL_SIZE, batch_size=batch_size,
                    train_steps=train_steps, lr=LR, ladder=LADDER,
                    mc_dt=MC_DT, mc_steps=MC_STEPS,
                    mc_adjust=True,
                    monitor=mon, bg_param=BG_PARAM, chunks=CHUNKS,
                    checkpoint=False, e_clip=E_CLIP, g_clip=G_CLIP,
                    flow_dir=run_dir / "attempts")
                t_hist_kl = [float(s["t"]) for s in stages]
            else:  # klxx: fixed schedule on the exact accepted KL history
                if not t_hist_kl or t_hist_kl[-1] != 1.0:
                    raise RuntimeError(f"{tag}: completed kl t_hist is required")
                log(f"[{tag}] fixed t_hist from kl: "
                    f"{[f'{t:.4f}' for t in t_hist_kl]}")
                y, stages = boltzmann_forward_KLXX_G_fixed(
                    x_valid, u0, u1, flow0,
                    pool_size=POOL_SIZE, batch_size=batch_size,
                    train_steps=train_steps, lr=LR, ladder=LADDER, melt=MELT,
                    opt_alpha=OPT_ALPHA, opt_steps=OPT_STEPS,
                    # The public fixed-driver keyword remains `t_list`; the
                    # experiment contract and saved accepted history are
                    # deliberately named `t_hist`.
                    mc_dt=MC_DT, mc_steps=MC_STEPS, t_list=t_hist_kl,
                    coeff_lambda=COEFF_LAMBDA, coeff_alpha=COEFF_ALPHA,
                    coeff_beta=COEFF_BETA,
                    mc_adjust=True,
                    monitor=mon, chunks=CHUNKS,
                    checkpoint=False, e_clip=E_CLIP, g_clip=G_CLIP,
                    flow_dir=run_dir / "attempts")
            y = jax.block_until_ready(y)
            jax.effects_barrier()
            wall = time.time() - t0

            t_hist = np.array([s["t"] for s in stages], dtype=np.float64)
            if method == "klxx" and not np.array_equal(
                    t_hist, np.asarray(t_hist_kl, dtype=np.float64)):
                raise RuntimeError(
                    f"{tag}: returned t_hist differs from paired KL t_hist"
                )
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
            attempt_t_hist = np.concatenate(
                [np.asarray(s["t_hist"], dtype=np.float64) for s in stages]
            ) if stages else np.zeros((0,), dtype=np.float64)
            batch_ess_hist = np.concatenate(
                [np.asarray(s["batch_ess_hist"], dtype=np.float32) for s in stages],
                axis=0,
            ) if stages else np.zeros((0, train_steps), dtype=np.float32)
            valid_trained_ess_hist = np.concatenate(
                [np.asarray(s["valid_trained_ess_hist"], dtype=np.float64)
                 for s in stages]
            ) if stages else np.zeros((0,), dtype=np.float64)
            valid_identity_ess_hist = np.concatenate(
                [np.asarray(s["valid_identity_ess_hist"], dtype=np.float64)
                 for s in stages]
            ) if stages else np.zeros((0,), dtype=np.float64)
            attempt_status_hist = np.concatenate(
                [np.asarray(s["attempt_status_hist"]) for s in stages]
            ) if stages else np.zeros((0,), dtype="U1")
            accepted_attempt_t_hist = attempt_t_hist[
                attempt_status_hist == "accepted"
            ]
            expected_attempt_t_hist = t_hist.astype(np.float32).astype(np.float64)
            if not np.array_equal(accepted_attempt_t_hist, expected_attempt_t_hist):
                raise RuntimeError(
                    f"{tag}: accepted attempt_t_hist does not reconstruct t_hist"
                )
            complete = bool(len(t_hist) > 0 and t_hist[-1] == 1.0)
            if not complete:
                reached = float(t_hist[-1]) if len(t_hist) else 0.0
                log(f"!!!!! {tag} INCOMPLETE LADDER: reached t={reached:.4f} < 1 "
                    f"— the metrics below are vs the FULL target and NOT "
                    f"target-faithful !!!!!")

            y_push, logw = compose_pushforward(stages, x_eval)
            y_push = jax.block_until_ready(y_push)
            final_ess = float(compute_ESS_log(logw))
            sec_push = sector_occupancy(np.asarray(y_push), P, MODE_FRAC)
            sec_valid = sector_occupancy(np.asarray(y[:N_EVAL]), P, MODE_FRAC)
            knn_cov = torus_coverage(np.asarray(y_push[:10000]),
                                     np.asarray(hat_ref[:2000]), k=KNN_K)

            np.savez_compressed(
                out,
                tag=tag, method=method, L=L, D=D, P=P, J=J, H=H,
                valid_size=N_VALID, pool_size=POOL_SIZE,
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
                complete=complete, final_ess=final_ess,
                sectors_push=int(sec_push[0] * P), tv_push=sec_push[1],
                counts_push=np.array(sec_push[2]), abs_m_push=sec_push[3],
                sectors_valid=int(sec_valid[0] * P), tv_valid=sec_valid[1],
                counts_valid=np.array(sec_valid[2]), abs_m_valid=sec_valid[3],
                knn_coverage=knn_cov, wall_s=wall,
                samples_push=np.asarray(y_push, dtype=np.float32),
                samples_valid=np.asarray(y[:N_EVAL], dtype=np.float32),
                logw_push=np.asarray(logw, dtype=np.float32),
            )
            eqx.tree_serialise_leaves(run_dir / "flows.eqx",
                                      [s["flow"] for s in stages])
            log(f"##### {tag} DONE K={len(stages)} complete={complete} "
                f"t_hist={[f'{t:.3f}' for t in t_hist]} "
                f"final_ESS={final_ess:.4f} "
                f"sectors(push)={int(sec_push[0] * P)}/{P} tv={sec_push[1]:.3f} "
                f"|m|={sec_push[3]:.3f} knn={knn_cov:.3f} wall={wall:.0f}s #####")
            if not complete:
                log(f"{tag}: incomplete artifact retained at {run_dir}; "
                    "remove it before retrying; paired klxx will not start")
                del y, stages, y_push, logw
                gc.collect()
                jax.clear_caches()
                t_hist_kl = None
                break
            # release CUDA memory before the next method: drop this run's
            # buffers and the compiled executables (VRAM accumulation
            # across methods is what OOMs long sweeps)
            del y, stages, y_push, logw
            gc.collect()
            jax.clear_caches()
            log(f"{tag}: buffers + compile cache released")
    log(f"DONE — temporary artifacts in {ARTIFACTS}; "
        "build the table with build_table.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

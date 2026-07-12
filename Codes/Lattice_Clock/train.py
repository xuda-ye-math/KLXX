"""p-state clock Boltzmann generator — batch-size sweep, two objectives.

The target is the cold Boltzmann measure of the P=6 clock model on the
periodic 8 x 8 lattice (D = 64 angles, P symmetry-broken sectors), reached
from the uniform-on-torus source by the bridge ladder of the monitored
local twins (boltzmann.py) of the jflows Boltzmann drivers — identical
machinery plus per-stage sector-occupancy monitoring (occupancy, per-sector
mean weights, bias) printed to the status log. Two objectives per batch
size, both from the same identity-initialized NCSF on [-pi, pi)^D:

    kl   :  boltzmann_forward_KL_G — bare forward KL stages on the adaptive
            ladder (SMC selection gate + acceptance)
    klxx :  boltzmann_forward_KLXX_G_fixed — KL + X_mu + X_{(hat_mu+bar_nu)/2}
            stages, (alpha, beta) = (1/2, 1/2), per-stage QT pool, trained on
            the SAME t_list the kl run accepted (fixed schedule, no SMC gate,
            no acceptance), so the two losses see identical bridge increments

The sweep runs the (N_BATCH, STEPS) pairs of N_BATCH_LIST x STEPS_LIST in
order (kl then klxx at each size); an optional argument in {1, .., 5} runs a
single pair, e.g. `train.py 1` for N_BATCH = 1000. When a kl data file
already exists its ladder is loaded from the npz, so the paired klxx run can
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
    conda activate jflows && PYTHONPATH=/mnt/projects/jflows python \
        Codes/Lattice_Clock/train.py
Writes data_<tag>.npz + flows_<tag>.eqx per run (existing data files are
skipped, so the sweep resumes) and train_status.log; build the table
with build_table.py.
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

from jflows.flow import NCSF
from jflows.potential import Nlog_Uniform
from jflows.train import Monitor
from jflows.utils import compute_ESS_log, qt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                             # Lattice_Clock/{potential,boltzmann}.py
from boltzmann import boltzmann_forward_KL_G, boltzmann_forward_KLXX_G_fixed
from potential import Clock, sector_occupancy, torus_coverage

# clock model (potential-specific)
L = 8                  # lattice side; D = L*L angles
D = L * L

LOG = HERE / "train_status.log"
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
N_POOL: int = 100000         # pool size: SMC selection pool and QT pool
N_BATCH_LIST = (1000, 500, 250, 125, 2000)  # batch sizes, in test order
STEPS_LIST = (3000, 3000, 3000, 3000, 3000)  # gradient steps per stage, paired with N_BATCH_LIST
LR: float = 1e-3             # Adam learning rate

# Langevin / AIS (MALA everywhere)
LADDER: int = 4        # rungs of the SMC selection gate and the training AIS
MC_STEP: float = 1e-3  # Langevin step size
MC_ITERS: int = 100    # Langevin steps per rung / rejuvenation / QT temper

# quench and temper (the wide-coverage measure hat_mu)
MELT: float = 2 * math.pi  # wrapped-Gaussian melt: uniform on the torus
OPT_STEP: float = 1e-2     # L-BFGS trial alpha (armijo)
OPT_ITERS: int = 100       # L-BFGS iterations

# loss coefficients (klxx)
COEFF_LAMBDA: float = 1.0
COEFF_ALPHA: float = 0.5
COEFF_BETA: float = 0.5

# training screens (forwarded to the forward Boltzmann drivers)
E_CLIP: float = 1e3    # energy screen: samples with target > E_CLIP drop from the loss
G_CLIP: float = 1e2    # global gradient-norm clip (spike guard before Adam)

# adaptive ladder (bg_param of the kl driver; klxx inherits the accepted t_list)
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
CHUNK: int = 64        # chunk count for the drivers' full-set evaluations
MONITOR_EVERY: int = 50

METHODS = ("kl", "klxx")


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
    pairs = list(zip(N_BATCH_LIST, STEPS_LIST))
    if test is not None:
        pairs = [pairs[test - 1]]
    if test is None:
        open(LOG, "w").close()   # fresh log per full sweep; single-pair runs append
    log(f"START Lattice_Clock sweep | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | L={L} D={D} P={P} J={J} H={H} | "
        f"N_VALID={N_VALID} N_POOL={N_POOL} pairs(B,steps)={pairs} LR={LR} "
        f"ladder={LADDER} MC={MC_STEP}x{MC_ITERS} (MALA) "
        f"QT: melt={MELT:.4f} opt={OPT_STEP}x{OPT_ITERS} "
        f"e_clip={E_CLIP} g_clip={G_CLIP} bg={BG_PARAM}")

    x_valid = u0.samples(jax.random.key(2), N_VALID)
    x_eval = u0.samples(jax.random.key(3), N_EVAL)
    flow0 = NCSF(jax.random.key(0), a=[-NSF_LIM] * D, b=[NSF_LIM] * D,
                 bins=BINS, transforms=TRANSFORMS,
                 hidden_features=HIDDEN_FEATURES).zeros()

    # QT referee set on the full target (uniform melt = fresh uniform draw)
    t0 = time.time()
    hat_ref = qt(jax.random.key(4), u0.samples(jax.random.key(5), N_POOL), u1,
                 melt=0.0, opt_step=OPT_STEP, opt_iters=OPT_ITERS,
                 mc_step=MC_STEP, mc_iters=MC_ITERS, chunk=CHUNK)
    hat_ref = jax.block_until_ready(hat_ref)
    log(f"QT referee set built: {N_POOL} particles on the full target "
        f"in {time.time() - t0:.1f}s")

    for n_batch, steps in pairs:
        t_list_kl = None       # the kl-accepted ladder, inherited by klxx
        for method in METHODS:
            tag = f"{method}_B{n_batch}"
            out = HERE / f"data_{tag}.npz"
            if out.exists():
                if method == "kl":
                    t_list_kl = [float(t) for t in np.load(out)["ladder"]]
                    log(f"{tag}: {out.name} exists, skipping "
                        f"(t_list for klxx: {[f'{t:.4f}' for t in t_list_kl]})")
                else:
                    log(f"{tag}: {out.name} exists, skipping")
                continue
            log(f"=== {tag} START (n_batch={n_batch}, steps={steps}) ===")
            t0 = time.time()
            mon = Monitor(MONITOR_EVERY, f"[{tag}] ", log)
            if method == "kl":
                y, stages = boltzmann_forward_KL_G(
                    x_valid, u0, u1, flow0, n_pool=N_POOL, n_batch=n_batch,
                    steps=steps, lr=LR, ladder=LADDER,
                    mc_step=MC_STEP, mc_iters=MC_ITERS,
                    monitor=mon, bg_param=BG_PARAM, chunk=CHUNK,
                    e_clip=E_CLIP, g_clip=G_CLIP)
                t_list_kl = [float(s["t"]) for s in stages]
            else:  # klxx: fixed schedule on the kl-accepted ladder
                assert t_list_kl, f"{tag}: no kl ladder to inherit"
                log(f"[{tag}] fixed t_list from kl: "
                    f"{[f'{t:.4f}' for t in t_list_kl]}")
                y, stages = boltzmann_forward_KLXX_G_fixed(
                    x_valid, u0, u1, flow0, n_pool=N_POOL, n_batch=n_batch,
                    steps=steps, lr=LR, ladder=LADDER, melt=MELT,
                    opt_step=OPT_STEP, opt_iters=OPT_ITERS,
                    mc_step=MC_STEP, mc_iters=MC_ITERS, t_list=t_list_kl,
                    coeff_lambda=COEFF_LAMBDA, coeff_alpha=COEFF_ALPHA,
                    coeff_beta=COEFF_BETA,
                    monitor=mon, chunk=CHUNK,
                    e_clip=E_CLIP, g_clip=G_CLIP)
            y = jax.block_until_ready(y)
            jax.effects_barrier()
            wall = time.time() - t0

            ladder_ts = np.array([s["t"] for s in stages], dtype=np.float64)
            stage_ess = np.array([s["ess"] for s in stages], dtype=np.float64)
            # full per-step training-surrogate ESS history, one row per stage
            # ([K, steps]); the source for the per-step ESS figure
            stage_ess_history = np.array(
                [np.asarray(s["ess_history"], dtype=np.float32) for s in stages],
                dtype=np.float32) if stages else np.zeros((0, steps), np.float32)
            # per-stage improvement of the accepted map over the identity (SMC)
            # fallback, max(0, trained ESS - identity ESS), one scalar per stage
            imp_history = np.array([s["imp_history"] for s in stages], dtype=np.float32) \
                if stages else np.zeros((0,), np.float32)
            complete = bool(len(ladder_ts) > 0 and ladder_ts[-1] >= 1.0)
            if not complete:
                reached = float(ladder_ts[-1]) if len(ladder_ts) else 0.0
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
                n_valid=N_VALID, n_pool=N_POOL, n_batch=n_batch, steps=steps, lr=LR,
                ladder_rungs=LADDER, mc_step=MC_STEP, mc_iters=MC_ITERS,
                ladder=ladder_ts, stage_ess=stage_ess,
                stage_ess_history=stage_ess_history, imp_history=imp_history,
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
            eqx.tree_serialise_leaves(HERE / f"flows_{tag}.eqx",
                                      [s["flow"] for s in stages])
            log(f"##### {tag} DONE K={len(stages)} complete={complete} "
                f"ladder={[f'{t:.3f}' for t in ladder_ts]} "
                f"final_ESS={final_ess:.4f} "
                f"sectors(push)={int(sec_push[0] * P)}/{P} tv={sec_push[1]:.3f} "
                f"|m|={sec_push[3]:.3f} knn={knn_cov:.3f} wall={wall:.0f}s #####")
            # release CUDA memory before the next method: drop this run's
            # buffers and the compiled executables (VRAM accumulation
            # across methods is what OOMs long sweeps)
            del y, stages, y_push, logw
            gc.collect()
            jax.clear_caches()
            log(f"{tag}: buffers + compile cache released")
    log(f"DONE — data files in {HERE}; build the table with build_table.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

"""2D phi^4 lattice benchmark — X-regularized forward KL in the broken Z2 phase.

Four objectives, three seeds each, all trained by the packed jflows drivers
from the same identity-initialized NSF on the same source pool:

    KL                :  train_forward_KLX_G,  coeff_lambda = 0
    KL+X_mu           :  train_forward_KLX_G,  coeff_lambda = 1
    KL+X_mu+X_hat_mu  :  train_forward_KLXX_G, (alpha, beta) = (1, 0)
    KL+X_mu+X_mix     :  train_forward_KLXX_G, (alpha, beta) = (1/2, 1/2)

The target is the phi^4 action on an L x L periodic lattice flattened to
R^{L*L}, in the broken Z2 phase with a small explicit breaking field h > 0:
two vacua of unequal free energy, the minority vacuum at m > 0. The
observable is the magnetization m(phi) = mean(phi); each run is judged by
its final ESS and its reweighted occupancy p(m > 0) against the reference
ensemble phi4_reference.npz (built by reference.py — run it once before
this script). All Langevin kernels run MALA (mc_adjust = True).

Run from the repo root:
    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows python \
        Codes/Lattice_Phi4/L8/train.py
Writes temporary arrays/logs below ``artifacts/`` and the final table below
``results/``; plotting is separate.
"""

import csv
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
from jflows.utils import (
    compute_ESS_log,
    importance_weights_log,
    linear_weights_from_log,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
LOG = ARTIFACTS / "train.log"

# lattice action S[phi] = sum_x [ -2*KAPPA*phi_x*(phi_{x+e1}+phi_{x+e2}) + phi_x^2
#                                 + LAMBDA*(phi_x^2-1)^2 ] + H*sum_x phi_x
L = 8                   # lattice side; D = L*L sites
D = L * L
KAPPA = 0.40            # hopping
LAMBDA = 0.50           # quartic self-coupling
H = 0.0144              # explicit Z2 breaking (frozen by the pilot scan)

# boundary of the domain
SIGMA = 0.5             # standard deviation of the isotropic Gaussian source mu_0
NSF_LIM = 3.0           # half-width of the NSF spline domain per site

# NSF flow architecture
BINS: int = 16
TRANSFORMS: int = 6
HIDDEN_FEATURES = (256, 256)

# training parameters
N_VALID: int = 100000   # the fixed source set (training pool + final evaluation)
BATCH_SIZE: int = 500      # source samples per training step
TRAIN_STEPS: int = 2000       # Adam optimization steps
LR: float = 1e-3        # Adam learning rate
E_CLIP: float = float("inf")  # energy screen (inf = keep every sample)
G_CLIP: float = 1e3     # global gradient-norm clip (pre-Adam)
SEEDS = (0, 1, 2)       # training seeds (flow batches, AIS, permutations)

# data pipeline (single-hop AIS + MALA rejuvenation)
LADDER: int = 1         # AIS levels per manufactured target batch
MC_DT: float = 2e-3   # Langevin step size
MC_STEPS: int = 50      # Langevin steps per level / per hat_mu freshening

# quench and temper (the wide-coverage measure hat_mu)
POOL_SIZE: int = 2000      # quench-and-temper pool size
MELT: float = 2.0       # melt scale (std of the Gaussian scatter)
OPT_ALPHA: float = 0.1   # L-BFGS trial alpha (armijo)
OPT_STEPS: int = 200    # L-BFGS iterations

METHODS = (
    "KL",
    "KL+X_mu",
    "KL+X_mu+X_hat_mu",
    "KL+X_mu+X_mix",
)


# source: Gaussian mu_0 on R^D
u0 = Nlog_Gaussian(mean=[0.0] * D, variance=[SIGMA**2] * D)


# target: phi^4 lattice action on the L x L periodic lattice
def phi4_energy(x):
    phi = x.reshape(-1, L, L)
    hop = phi * (jnp.roll(phi, 1, axis=1) + jnp.roll(phi, 1, axis=2))
    s = (-2.0 * KAPPA * hop + phi**2
         + LAMBDA * (phi**2 - 1.0) ** 2 + H * phi)
    return s.sum(axis=(1, 2))


u1 = potential_from(phi4_energy)


def magnetization(x):
    return np.asarray(x).mean(axis=1)   # m(phi) = (1/D) sum_x phi_x


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START Phi4 L={L} (D={D}) | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"kappa={KAPPA} lambda={LAMBDA} h={H} | "
        f"N_VALID={N_VALID} BATCH_SIZE={BATCH_SIZE} TRAIN_STEPS={TRAIN_STEPS} LR={LR} e_clip={E_CLIP} g_clip={G_CLIP} seeds={SEEDS} "
        f"MC={MC_DT}x{MC_STEPS} (MALA) ladder={LADDER} "
        f"QT: POOL_SIZE={POOL_SIZE} melt={MELT} opt={OPT_ALPHA}x{OPT_STEPS}")
    ref = np.load(ARTIFACTS / "phi4_reference.npz")
    m_ref = ref["m_trace"].ravel()
    p_plus_ref = float((m_ref > 0).mean())
    log(f"reference: p(m>0) = {p_plus_ref:.4f}")

    x_valid = u0.samples(jax.random.key(2), N_VALID)
    flow0 = NSF(jax.random.key(0), a=[-NSF_LIM] * D, b=[NSF_LIM] * D,
                bins=BINS, transforms=TRANSFORMS,
                hidden_features=HIDDEN_FEATURES).zeros()

    rows = []
    store = {}                          # per-run arrays saved to data.npz
    for seed in SEEDS:
        for name in METHODS:
            t0 = time.time()
            mon = Monitor(500, f"[s{seed} {name}] ", log)
            s = jnp.uint32(seed)
            if name == "KL":
                flow, _ = train_forward_KLX_G(
                    x_valid, u0, u1, flow0,
                    batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, lr=LR,
                    ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=0.0, mc_adjust=True,
                    e_clip=E_CLIP, g_clip=G_CLIP, monitor=mon, seed=s)
            elif name == "KL+X_mu":
                flow, _ = train_forward_KLX_G(
                    x_valid, u0, u1, flow0,
                    batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, lr=LR,
                    ladder=LADDER, mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=1.0, mc_adjust=True,
                    e_clip=E_CLIP, g_clip=G_CLIP, monitor=mon, seed=s)
            elif name == "KL+X_mu+X_hat_mu":
                flow, _ = train_forward_KLXX_G(
                    x_valid, u0, u1, flow0,
                    pool_size=POOL_SIZE, batch_size=BATCH_SIZE,
                    train_steps=TRAIN_STEPS, lr=LR, ladder=LADDER, melt=MELT,
                    opt_alpha=OPT_ALPHA, opt_steps=OPT_STEPS,
                    mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=1.0, coeff_alpha=1.0, coeff_beta=0.0,
                    mc_adjust=True,
                    e_clip=E_CLIP, g_clip=G_CLIP, monitor=mon, seed=s)
            else:  # KL+X_mu+X_mix
                flow, _ = train_forward_KLXX_G(
                    x_valid, u0, u1, flow0,
                    pool_size=POOL_SIZE, batch_size=BATCH_SIZE,
                    train_steps=TRAIN_STEPS, lr=LR, ladder=LADDER, melt=MELT,
                    opt_alpha=OPT_ALPHA, opt_steps=OPT_STEPS,
                    mc_dt=MC_DT, mc_steps=MC_STEPS,
                    coeff_lambda=1.0, coeff_alpha=0.5, coeff_beta=0.5,
                    mc_adjust=True,
                    e_clip=E_CLIP, g_clip=G_CLIP, monitor=mon, seed=s)
            flow = jax.block_until_ready(flow)
            jax.effects_barrier()

            y_push = flow.inv(x_valid)
            log_weights = importance_weights_log(x_valid, u0, u1, flow, type="G")
            ess = float(compute_ESS_log(log_weights))
            w = np.asarray(linear_weights_from_log(log_weights))
            mag = magnetization(y_push)
            wn = w / w.sum()
            p_plus = float(wn[mag > 0].sum())
            rows.append(dict(seed=seed, method=name,
                             final_ess=round(ess, 4), p_plus=round(p_plus, 4)))
            store[f"mag_{seed}_{name}"] = mag.astype(np.float32)
            store[f"w_{seed}_{name}"] = wn.astype(np.float32)
            log(f"[s{seed} {name}] done in {time.time() - t0:.1f}s   "
                f"final ESS = {ess:.4f}   reweighted p(m>0) = {p_plus:.4f}")

    # ── per-seed table ──
    with open(RESULTS / "results_table.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["seed", "method", "final_ess", "p_plus"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    np.savez_compressed(ARTIFACTS / "data.npz", **store)
    log(f"saved {ARTIFACTS / 'data.npz'} ({len(store)} arrays)")
    log(f"DONE — table at {RESULTS / 'results_table.csv'}, "
        f"data at {ARTIFACTS / 'data.npz'}")


if __name__ == "__main__":
    main()

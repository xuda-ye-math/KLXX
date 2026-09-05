"""Train the two flows shown in the talk's mode-collapse figure.

Reverse KL and forward KL on the Himmelblau target, at the parameters of
``Codes/2D_Benchmark/Himmelblau/train.py``, plus four MALA chains started at the
origin. Everything the figure needs is written to

    Slides/figures/artifacts/himmelblau_collapse.npz

so that ``make_himmelblau_collapse.py`` can redraw without retraining. Nothing
under ``Codes/`` or ``Paper/`` is written.

Run detached:
    setsid nohup /home/xuda/.envs/jflows/bin/python \
        /data/projects/KLXX/Slides/figures/train_himmelblau_collapse.py &
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
from jflows.train import Monitor, train_forward_KLX_G, train_reverse_KL_F
from jflows.utils import compute_ESS_log, importance_weights_log

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "train_himmelblau_collapse.log"
DATA = ARTIFACTS / "himmelblau_collapse.npz"

# parameters of Codes/2D_Benchmark/Himmelblau/train.py, unchanged
SIGMA = 1.0
PLT_LIM = 5.5
NSF_LIM = 6.0
BINS: int = 32
TRANSFORMS: int = 6
HIDDEN_FEATURES = (128, 128)
VALID_SZIE: int = 50000
BATCH_SZIE: int = 1000
STEPS_TOTAL: int = 1000
LR: float = 1e-3
LADDER: int = 1
MC_DT: float = 2e-3
MC_STEPS_1: int = 20
MC_STEPS_2: int = 50
GRID_SIZE: int = 300
PRIOR_SIZE: int = 5000

# the four Metropolis-adjusted Langevin chains of the left panel
N_CHAINS: int = 4
CHAIN_STEPS: int = 60000
CHAIN_DT: float = 2.0e-4

u0 = Nlog_Gaussian(mean=[0.0, 0.0], variance=[SIGMA**2, SIGMA**2])


def himmelblau_energy(x):
    x1, x2 = x[..., 0], x[..., 1]
    return (x1**2 + x2 - 11.0) ** 2 + (x1 + x2**2 - 7.0) ** 2


u1 = potential_from(himmelblau_energy)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def grad_U(x):
    x1, x2 = x[..., 0], x[..., 1]
    a = x1**2 + x2 - 11.0
    b = x1 + x2**2 - 7.0
    return np.stack([4.0 * a * x1 + 2.0 * b, 2.0 * a + 4.0 * b * x2], axis=-1)


def mala_chain(x0, n_steps, dt, rng):
    """One Metropolis-adjusted Langevin chain on pi propto exp(-U), in NumPy."""
    x = np.array(x0, dtype=np.float64)
    out = np.empty((n_steps, 2))
    U = lambda z: float(himmelblau_energy(z))
    for k in range(n_steps):
        y = x - dt * grad_U(x) + np.sqrt(2.0 * dt) * rng.standard_normal(2)
        fwd = -np.sum((y - x + dt * grad_U(x)) ** 2) / (4.0 * dt)
        bwd = -np.sum((x - y + dt * grad_U(y)) ** 2) / (4.0 * dt)
        if np.log(rng.random()) < (U(x) - U(y)) + bwd - fwd:
            x = y
        out[k] = x
    return out


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()
    log(f"START himmelblau collapse | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | VALID_SZIE={VALID_SZIE} "
        f"BATCH_SZIE={BATCH_SZIE} STEPS_TOTAL={STEPS_TOTAL} LR={LR}")

    xs = np.linspace(-PLT_LIM, PLT_LIM, GRID_SIZE)
    X1, X2 = np.meshgrid(xs, xs, indexing="xy")
    grid = jnp.stack([jnp.asarray(X1.ravel()), jnp.asarray(X2.ravel())], axis=-1)
    store = {"X1": X1, "X2": X2,
             "U_grid": np.asarray(u1(grid)).reshape(X1.shape),
             "prior": np.asarray(u0.samples(jax.random.key(42), PRIOR_SIZE))}

    log(f"running {N_CHAINS} MALA chains from the origin, "
        f"{CHAIN_STEPS} steps, dt={CHAIN_DT}")
    for k in range(N_CHAINS):
        chain = mala_chain(np.zeros(2), CHAIN_STEPS, CHAIN_DT,
                           np.random.default_rng(k))
        store[f"chain_{k}"] = chain
        log(f"  chain {k + 1}: end {chain[-1].round(2)}")
    np.savez(DATA, **store)

    x_valid = u0.samples(jax.random.key(2), VALID_SZIE)
    flow0 = NSF(jax.random.key(0), a=[-NSF_LIM, -NSF_LIM], b=[NSF_LIM, NSF_LIM],
                bins=BINS, transforms=TRANSFORMS,
                hidden_features=HIDDEN_FEATURES).zeros()

    t0 = time.time()
    log("training reverse KL (F map) ...")
    flow_rev, _ = train_reverse_KL_F(
        x_valid, u0, u1, flow0,
        batch_size=BATCH_SZIE, steps_total=STEPS_TOTAL, lr=LR,
        mc_dt=MC_DT, mc_steps_2=MC_STEPS_2, mc_adjust=True,
        monitor=Monitor(100, "[reverse KL] ", log))
    flow_rev = jax.block_until_ready(flow_rev)
    y_rev = np.asarray(flow_rev(x_valid))
    ess_rev = float(compute_ESS_log(
        importance_weights_log(x_valid, u0, u1, flow_rev, type="F")))
    store["samples_reverse_KL"] = y_rev
    store["final_ess_reverse_KL"] = np.asarray(ess_rev)
    np.savez(DATA, **store)
    log(f"reverse KL done in {time.time() - t0:.1f}s   final ESS = {ess_rev:.4f}")

    t0 = time.time()
    log("training forward KL (G map) ...")
    flow_fwd, _ = train_forward_KLX_G(
        x_valid, u0, u1, flow0,
        batch_size=BATCH_SZIE, steps_total=STEPS_TOTAL, lr=LR,
        ladder=LADDER, mc_dt=MC_DT, mc_steps_1=MC_STEPS_1, mc_steps_2=MC_STEPS_2,
        coeff_lambda=0.0, mc_adjust=True,
        monitor=Monitor(100, "[forward KL] ", log))
    flow_fwd = jax.block_until_ready(flow_fwd)
    y_fwd = np.asarray(flow_fwd.inv(x_valid))
    ess_fwd = float(compute_ESS_log(
        importance_weights_log(x_valid, u0, u1, flow_fwd, type="G")))
    store["samples_forward_KL"] = y_fwd
    store["final_ess_forward_KL"] = np.asarray(ess_fwd)
    np.savez(DATA, **store)
    log(f"forward KL done in {time.time() - t0:.1f}s   final ESS = {ess_fwd:.4f}")
    log(f"DONE, wrote {DATA}")


if __name__ == "__main__":
    main()

"""Train the four Phi4 objectives on the 6 x 6 lattice."""

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

L: int = 6
D: int = L * L
KAPPA: float = 0.40
LAMBDA: float = 0.50
H: float = 0.0257

SIGMA: float = 0.5
NSF_LIM: float = 3.0
BINS: int = 16
TRANSFORMS: int = 6
HIDDEN_FEATURES = (256, 256)

VALID_SZIE: int = 100000
BATCH_SZIE: int = 1000
POOL_SIZE: int = 0
STEPS_TOTAL: int = 2000
LR: float = 1e-3
U_CLIP: float = float("inf")
G_CLIP: float = 1e3
SEEDS = (0, 1, 2)

LADDER: int = 1
MC_DT: float = 2e-3
MC_STEPS_1: int = 20   # Langevin steps per SMC level and per QT batch draw
MC_STEPS_2: int = 50   # Langevin steps of the quench-and-temper pool

MELT: float = 2.0
OPT_DT: float = 0.1
OPT_STEPS: int = 50

# (name, coeff_lambda, (coeff_theta, coeff_alpha)); coeff_alpha is the QT
# proportion of the mixture, 1 - coeff_alpha the detached pushforward share
METHODS = (
    ("KL", 0.0, None),
    ("KL+X_pi", 1.0, None),
    ("KL+X_pi+X_hat_pi", 1.0, (1.0, 1.0)),
    ("KL+X_pi+X_mix", 1.0, (1.0, 0.5)),
)


source = Nlog_Gaussian(mean=[0.0] * D, variance=[SIGMA**2] * D)


def phi4_energy(x):
    phi = x.reshape(-1, L, L)
    neighbors = jnp.roll(phi, 1, axis=1) + jnp.roll(phi, 1, axis=2)
    action = (
        -2.0 * KAPPA * phi * neighbors
        + phi**2
        + LAMBDA * (phi**2 - 1.0) ** 2
        + H * phi
    )
    return action.sum(axis=(1, 2))


target = potential_from(phi4_energy)


def log(message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a") as stream:
        stream.write(line + "\n")


def train(name, coeff_lambda, mixture, seed, x_valid, flow0):
    common = dict(
        batch_size=BATCH_SZIE,
        steps_total=STEPS_TOTAL,
        lr=LR,
        ladder=LADDER,
        mc_dt=MC_DT,
        mc_steps_1=MC_STEPS_1,
        mc_adjust=True,
        monitor=Monitor(500, f"[s{seed} {name}] ", log),
        seed=seed,
        u_clip=U_CLIP,
        g_clip=G_CLIP,
    )
    if mixture is None:
        return train_forward_KLX_G(
            x_valid,
            source,
            target,
            flow0,
            coeff_lambda=coeff_lambda,
            **common,
        )
    return train_forward_KLXX_G(
        x_valid,
        source,
        target,
        flow0,
        pool_size=POOL_SIZE,
        melt=MELT,
        opt_dt=OPT_DT,
        opt_steps=OPT_STEPS,
        mc_steps_2=MC_STEPS_2,
        coeff_lambda=coeff_lambda,
        coeff_theta=mixture[0],
        coeff_alpha=mixture[1],
        **common,
    )


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()

    log(
        f"START Phi4 L={L} D={D} | {jax.default_backend()} | "
        f"VALID_SZIE={VALID_SZIE} BATCH_SZIE={BATCH_SZIE} POOL_SIZE={POOL_SIZE} | "
        f"STEPS_TOTAL={STEPS_TOTAL} LR={LR} MC={MC_DT}x{MC_STEPS_1}/{MC_STEPS_2} "
        f"OPT={OPT_DT}x{OPT_STEPS} seeds={SEEDS}"
    )
    reference = np.load(ARTIFACTS / "phi4_reference.npz")
    log(f"reference p(m>0) = {float((reference['m_trace'] > 0).mean()):.4f}")

    x_valid = source.samples(jax.random.key(2), VALID_SZIE)
    flow0 = NSF(
        jax.random.key(0),
        a=[-NSF_LIM] * D,
        b=[NSF_LIM] * D,
        bins=BINS,
        transforms=TRANSFORMS,
        hidden_features=HIDDEN_FEATURES,
    ).zeros()

    rows = []
    arrays = {}
    for seed in SEEDS:
        for name, coeff_lambda, mixture in METHODS:
            started = time.time()
            flow, _ = train(
                name, coeff_lambda, mixture, seed, x_valid, flow0
            )
            flow = jax.block_until_ready(flow)
            jax.effects_barrier()
            samples = flow.inv(x_valid)
            log_weights = importance_weights_log(
                x_valid, source, target, flow, type="G"
            )
            ess = float(compute_ESS_log(log_weights))
            weights = np.asarray(linear_weights_from_log(log_weights))
            magnetization = np.asarray(samples).mean(axis=1)
            weights = weights / weights.sum()
            p_plus = float(weights[magnetization > 0].sum())

            rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "final_ess": round(ess, 4),
                    "p_plus": round(p_plus, 4),
                }
            )
            arrays[f"mag_{seed}_{name}"] = magnetization.astype(np.float32)
            arrays[f"w_{seed}_{name}"] = weights.astype(np.float32)
            log(
                f"[s{seed} {name}] DONE after {time.time() - started:.1f}s | "
                f"ESS={ess:.4f} p(m>0)={p_plus:.4f}"
            )

    with open(RESULTS / "results_table.csv", "w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("seed", "method", "final_ess", "p_plus"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(ARTIFACTS / "data.npz", **arrays)
    log("DONE")


if __name__ == "__main__":
    main()

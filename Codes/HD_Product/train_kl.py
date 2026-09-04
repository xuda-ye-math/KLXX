"""HD Product, forward KL family: 16 sequential tests at d = 256, 240, ..., 16, four methods each.

FAB runs from `train_fab.py`, which imports everything from this file and uses
its own `MC_STEPS_1`. The two drivers write to the same `artifacts/d{d}` folders
(`plot_ess.py` reads both); this one logs to `artifacts/train_kl.log`.
"""

import gc
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jflows import __version__ as JFLOWS_VERSION
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, potential_from
from jflows.train import Monitor, train_forward_KLX_G, train_forward_KLXX_G
from jflows.utils import compute_ESS_log, importance_weights_log


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
LOG = ARTIFACTS / "train_kl.log"

DIMENSIONS = tuple(range(256, 15, -16))
METHODS = (
    "KL+X_pi+X_mix",
    "KL+X_pi+X_hat_pi",
    "KL+X_pi",
    "KL",
)

SIGMA: float = 1.0
NSF_LIM: float = 4.0
BINS: int = 16
TRANSFORMS: int = 6
HIDDEN_FEATURES = (256, 256)

BATCH_SZIE: int = 300
STEPS_TOTAL: int = 2000
LR: float = 1e-3
G_CLIP: float = 1e2

LADDER: int = 2        # SMC levels; every level is MALA at the target (the forward KL surrogate)
MC_DT: float = 2e-3    # MALA step size everywhere
MC_STEPS_1: int = 50   # MALA steps on the intermediate SMC levels only (at the target: the forward KL surrogate)
MC_STEPS_2: int = 100  # MALA steps of every other rejuvenation: last SMC level, QT batch draw, QT pool temper

MELT: float = 2.0
OPT_DT: float = 0.5
OPT_STEPS: int = 50
CHUNKS: int = 10
CHECKPOINT: bool = False

COEFF_LAMBDA: float = 1.0
COEFF_THETA: float = 1.0   # weight of the mixture variation
COEFF_ALPHA: float = 0.5   # QT proportion of the mixture; 1 - COEFF_ALPHA is the pushforward share

SLUG = {
    "KL+X_pi+X_mix": "klxx_mix",
    "KL+X_pi+X_hat_pi": "klxx_hat_pi",
    "KL+X_pi": "klx",
    "KL": "kl",
}


VALID_SIZE = lambda d: 5000 * d   # noqa: E731
POOL_SIZE = lambda d: 1000 * d    # noqa: E731
EVAL_SIZE: int = 80000


def make_target(d: int):
    k = d.bit_length() - 1

    def energy(x):
        return (
            0.5 * (x**2).sum(axis=-1)
            + 12.0 * jnp.exp(-(x[:, :k] ** 2)).sum(axis=-1)
        )

    return potential_from(energy)


def make_flow(d: int):
    return NSF(
        jax.random.key(0),
        a=[-NSF_LIM] * d,
        b=[NSF_LIM] * d,
        bins=BINS,
        transforms=TRANSFORMS,
        hidden_features=HIDDEN_FEATURES,
    ).zeros()


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a") as stream:
        stream.write(line + "\n")


def train(name, d, x_valid, source, target, flow):
    common = dict(
        batch_size=BATCH_SZIE,
        steps_total=STEPS_TOTAL,
        lr=LR,
        ladder=LADDER,
        mc_dt=MC_DT,
        mc_steps_1=MC_STEPS_1,
        mc_steps_2=MC_STEPS_2,
        monitor=Monitor(20, f"[d{d} {name}] ", log),
        checkpoint=CHECKPOINT,
        g_clip=G_CLIP,
    )
    if name == "KL+X_pi+X_mix":
        return train_forward_KLXX_G(
            x_valid,
            source,
            target,
            flow,
            pool_size=POOL_SIZE(d),
            melt=MELT,
            opt_dt=OPT_DT,
            opt_steps=OPT_STEPS,
            coeff_lambda=COEFF_LAMBDA,
            coeff_theta=COEFF_THETA,
            coeff_alpha=COEFF_ALPHA,
            chunks=CHUNKS,
            **common,
        )
    if name == "KL+X_pi+X_hat_pi":
        return train_forward_KLXX_G(
            x_valid,
            source,
            target,
            flow,
            pool_size=POOL_SIZE(d),
            melt=MELT,
            opt_dt=OPT_DT,
            opt_steps=OPT_STEPS,
            coeff_lambda=COEFF_LAMBDA,
            coeff_theta=COEFF_THETA,
            coeff_alpha=1.0,
            chunks=CHUNKS,
            **common,
        )
    if name == "KL+X_pi":
        return train_forward_KLX_G(
            x_valid,
            source,
            target,
            flow,
            coeff_lambda=COEFF_LAMBDA,
            **common,
        )
    return train_forward_KLX_G(
        x_valid,
        source,
        target,
        flow,
        coeff_lambda=0.0,
        **common,
    )


def save_flow(path: Path, flow) -> None:
    temporary = path.with_suffix(".tmp.eqx")
    eqx.tree_serialise_leaves(temporary, jax.device_get(flow))
    os.replace(temporary, path)


def weights_and_samples(path: Path, x_eval, source, target, flow):
    log_weights = importance_weights_log(
        x_eval,
        source,
        target,
        flow,
        type="G",
        chunks=CHUNKS,
    )
    log_weights = np.asarray(
        jax.device_get(jax.block_until_ready(log_weights)), dtype=np.float32
    )

    temporary = path.with_suffix(".tmp.npy")
    samples = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=x_eval.shape,
    )
    bounds = np.linspace(0, x_eval.shape[0], CHUNKS + 1, dtype=int)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        pushed = flow.inv(x_eval[start:stop])
        samples[start:stop] = np.asarray(
            jax.device_get(jax.block_until_ready(pushed)), dtype=np.float32
        )
    samples.flush()
    del samples
    os.replace(temporary, path)
    return log_weights


def run_dimension(d: int) -> None:
    run_dir = ARTIFACTS / f"d{d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = Nlog_Gaussian(mean=[0.0] * d, variance=[SIGMA**2] * d)
    target = make_target(d)
    x_valid = source.samples(jax.random.key(2), VALID_SIZE(d))
    x_eval = source.samples(jax.random.key(3), EVAL_SIZE)

    for name in METHODS:
        flow = make_flow(d)
        log(
            f"[d{d} {name}] START | backend={jax.default_backend()} | "
            f"VALID_SIZE={VALID_SIZE(d)} | POOL_SIZE={POOL_SIZE(d)} | "
            f"EVAL_SIZE={EVAL_SIZE}"
        )
        started = time.time()
        flow, history = train(name, d, x_valid, source, target, flow)
        flow = jax.block_until_ready(flow)
        history = np.asarray(jax.device_get(history), dtype=np.float32)
        flow_path = run_dir / f"flow_{SLUG[name]}.eqx"
        samples_path = run_dir / f"samples_{SLUG[name]}.npy"
        save_flow(flow_path, flow)
        log(f"[d{d} {name}] flow saved; computing EVAL_SIZE ESS")

        log_weights = weights_and_samples(
            samples_path,
            x_eval,
            source,
            target,
            flow,
        )
        final_ess = float(compute_ESS_log(jnp.asarray(log_weights)))
        data_path = run_dir / f"{SLUG[name]}.npz"
        temporary = run_dir / f"{SLUG[name]}.tmp.npz"
        np.savez_compressed(
            temporary,
            d=d,
            VALID_SIZE=VALID_SIZE(d),
            POOL_SIZE=POOL_SIZE(d),
            EVAL_SIZE=EVAL_SIZE,
            BATCH_SZIE=BATCH_SZIE,
            STEPS_TOTAL=STEPS_TOTAL,
            CHUNKS=CHUNKS,
            CHECKPOINT=CHECKPOINT,
            jflows_version=JFLOWS_VERSION,
            batch_ess_history=history,
            log_weights=log_weights,
            final_ess=final_ess,
            flow_file=flow_path.name,
            samples_file=samples_path.name,
        )
        os.replace(temporary, data_path)
        log(
            f"[d{d} {name}] DONE after {time.time() - started:.1f}s | "
            f"ESS={final_ess:.4f} | saved {data_path}"
        )
        del flow, history, log_weights
        gc.collect()

    del x_valid, x_eval, source, target
    gc.collect()
    jax.clear_caches()
    log(f"=== d={d} COMPLETE; JAX caches released ===")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").close()
    log(
        f"START HD_Product forward KL family | jflows {JFLOWS_VERSION} | jax {jax.__version__} | "
        f"dimensions={DIMENSIONS} | methods={METHODS} | CHUNKS={CHUNKS}"
    )
    for d in DIMENSIONS:
        run_dimension(d)
    log("DONE — forward KL family at all 16 dimensions d=256,240,...,16 complete")


if __name__ == "__main__":
    main()

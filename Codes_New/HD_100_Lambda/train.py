"""HD product target at d = 100: forward KL + lambda X_pi for lambda = 0, 0.5, 1, 1.5, 2.

Every other setting is that of `Codes_New/HD_Product/train_kl.py`; `--lambdas`
selects other values (the log is appended, never truncated). Each lambda
writes `artifacts/lambda_{lambda}/`: the trained flow, the pushforward samples
of `EVAL_SIZE` source points, and `klx.npz` with the batch ESS curve, the
log weights, and the final ESS. The driver logs to `artifacts/train.log`.
"""

import argparse
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
LOG = ARTIFACTS / "train.log"

D: int = 100
LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.0)

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
MC_STEPS_1: int = 50   # MALA steps on the intermediate SMC levels only
MC_STEPS_2: int = 100  # MALA steps of every other rejuvenation (last SMC level)

MELT: float = 2.0
OPT_DT: float = 0.5
OPT_STEPS: int = 50
CHUNKS: int = 10
CHECKPOINT: bool = False

COEFF_LAMBDA: float = 1.0
COEFF_THETA: float = 1.0   # weight of the mixture variation
COEFF_ALPHA: float = 0.5   # QT proportion of the mixture; 1 - COEFF_ALPHA is the pushforward share

VALID_SIZE: int = 5000 * D
POOL_SIZE: int = 1000 * D
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


def save_flow(path: Path, flow) -> None:
    temporary = path.with_suffix(".tmp.eqx")
    eqx.tree_serialise_leaves(temporary, jax.device_get(flow))
    os.replace(temporary, path)


def weights_and_samples(path: Path, x_eval, source, target, flow):
    log_weights = importance_weights_log(
        x_eval, source, target, flow, type="G", chunks=CHUNKS,
    )
    log_weights = np.asarray(
        jax.device_get(jax.block_until_ready(log_weights)), dtype=np.float32
    )
    temporary = path.with_suffix(".tmp.npy")
    samples = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=x_eval.shape,
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


def run_lambda(coeff_lambda, x_valid, x_eval, source, target) -> None:
    """``coeff_lambda`` a number: KL + lambda X_pi; the string "klxx": KL+X_pi+X_mix of HD_Product."""
    klxx = coeff_lambda == "klxx"
    tag = "klxx" if klxx else f"lambda_{coeff_lambda:.1f}"
    slug = "klxx" if klxx else "klx"
    run_dir = ARTIFACTS / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    flow = make_flow(D)
    log(
        f"[{tag}] START {'KL+X_pi+X_mix' if klxx else f'KL + {coeff_lambda} X_pi'} | "
        f"backend={jax.default_backend()} | VALID_SIZE={VALID_SIZE} | EVAL_SIZE={EVAL_SIZE}"
    )
    started = time.time()
    common = dict(
        batch_size=BATCH_SZIE,
        steps_total=STEPS_TOTAL,
        lr=LR,
        ladder=LADDER,
        mc_dt=MC_DT,
        mc_steps_1=MC_STEPS_1,
        mc_steps_2=MC_STEPS_2,
        monitor=Monitor(20, f"[{tag}] ", log),
        checkpoint=CHECKPOINT,
        g_clip=G_CLIP,
    )
    if klxx:
        flow, history = train_forward_KLXX_G(
            x_valid,
            source,
            target,
            flow,
            pool_size=POOL_SIZE,
            melt=MELT,
            opt_dt=OPT_DT,
            opt_steps=OPT_STEPS,
            coeff_lambda=COEFF_LAMBDA,
            coeff_theta=COEFF_THETA,
            coeff_alpha=COEFF_ALPHA,
            chunks=CHUNKS,
            **common,
        )
    else:
        flow, history = train_forward_KLX_G(
            x_valid, source, target, flow, coeff_lambda=coeff_lambda, **common,
        )
    flow = jax.block_until_ready(flow)
    history = np.asarray(jax.device_get(history), dtype=np.float32)
    flow_path = run_dir / f"flow_{slug}.eqx"
    samples_path = run_dir / f"samples_{slug}.npy"
    save_flow(flow_path, flow)
    log(f"[{tag}] flow saved; computing EVAL_SIZE ESS and the pushforward samples")
    log_weights = weights_and_samples(samples_path, x_eval, source, target, flow)
    final_ess = float(compute_ESS_log(jnp.asarray(log_weights)))
    data_path = run_dir / f"{slug}.npz"
    temporary = run_dir / f"{slug}.tmp.npz"
    np.savez_compressed(
        temporary,
        d=D,
        coeff_lambda=COEFF_LAMBDA if klxx else coeff_lambda,
        method="KL+X_pi+X_mix" if klxx else "KL+X_pi",
        VALID_SIZE=VALID_SIZE,
        POOL_SIZE=POOL_SIZE if klxx else 0,
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
        f"[{tag}] DONE after {time.time() - started:.1f}s | "
        f"ESS={final_ess:.4f} | saved {data_path}"
    )
    del flow, history, log_weights
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="*", default=LAMBDAS)
    parser.add_argument("--klxx", action="store_true", help="also run KL+X_pi+X_mix with the HD_Product settings")
    args = parser.parse_args()
    lambdas = tuple(args.lambdas) + (("klxx",) if args.klxx else ())
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log(
        f"START HD_100_Lambda | jflows {JFLOWS_VERSION} | jax {jax.__version__} | "
        f"d={D} | lambdas={lambdas} | CHUNKS={CHUNKS}"
    )
    source = Nlog_Gaussian(mean=[0.0] * D, variance=[SIGMA**2] * D)
    target = make_target(D)
    x_valid = source.samples(jax.random.key(2), VALID_SIZE)
    x_eval = source.samples(jax.random.key(3), EVAL_SIZE)
    for coeff_lambda in lambdas:
        run_lambda(coeff_lambda, x_valid, x_eval, source, target)
    log("DONE - KL + lambda X_pi at d=100 for every lambda")


if __name__ == "__main__":
    main()

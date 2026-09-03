#!/usr/bin/env python
"""Data-driven KL+X_pi for propane 27D: one stage from the source straight to the target.

Every step draws its target batch from a fixed target sample set, the final
validation set of the stored KL+X_pi run (`artifacts/klx`, the population at
t = 1), instead of manufacturing it through the flow with the SMC surrogate.
The loss is the same forward KL plus `coeff_lambda` times the variation of the
log-ratio, with the same energy screen, weight screen, Adam, clips, and
warm-up as `jflows_md.train.train_forward_KLX_G`; the source batch is used
only for the batch ESS of the pushforward. The batch size and the step count
are twice those of `parameters.py`; every other control comes from there, and
the regularization is the fixed endpoint `RG_PARAM_1`.

    python train_data_driven.py                    # writes artifacts/klx_data_driven

Outputs: artifacts/klx_data_driven/ (flow, batch ESS history, validation log
weights), artifacts/klx_data_driven.log, results/klx_data_driven.md. Ctrl+C
stops the training at the next gradient step and evaluates the flow as it is.
"""

import json
import math
import os
import time
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jflows.train import Monitor, _masked_mean, _variation
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann import Manual_Reject, _identity_weights, _push_and_weights
from jflows_md.boltzmann.load import manifest
from jflows_md.train import _adam, _energy_keep, _learning_rate, _run_chunks, _screen_top
from jflows_md.utils.screen import compute_ESS_log

import parameters as P


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
NAME = "klx_data_driven"
SOURCE_RUN = HERE / "artifacts" / "klx"
BATCH_SIZE = 2 * P.BATCH_SIZE     # twice the SMC-based run's batch
TRAIN_STEPS = 2 * P.TRAIN_STEPS   # twice its steps


@eqx.filter_jit
def _chunk(
    x_valid, y_data, source, target, state, static, domain, key, *,
    step_offset, chunk, batch_size, steps_total, lr, coeff_lambda, monitor,
    checkpoint, u_clip, g_clip, lr_warmup, screen_fraction,
):
    """``chunk`` Adam steps of KL + lambda X_pi on target batches drawn from ``y_data``."""
    params, first, second, updates = state
    count, data_count = x_valid.shape[0], y_data.shape[0]

    def body(state, step):
        params, first, second, updates = state
        index_key, data_key = jax.random.split(jax.random.fold_in(key, step))
        x = x_valid[jax.random.choice(index_key, count, (batch_size,), replace=False)]
        y = y_data[jax.random.choice(data_key, data_count, (batch_size,), replace=False)]
        energy = lax.stop_gradient(target(y))
        energy_keep = _energy_keep(energy, u_clip)

        def loss_fn(values):
            latent, ladj = eqx.combine(values, static).call_and_ladj(y)
            z = source(latent) - energy - ladj
            keep = lax.stop_gradient(
                _screen_top(z, energy_keep & jnp.isfinite(z), screen_fraction)
            )
            loss = _masked_mean(z, keep)
            if coeff_lambda != 0.0:
                loss = loss + coeff_lambda * _variation(z, keep)
            return loss, keep

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, keep), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        params, first, second, updates = _adam(
            params, first, second, grads, loss, updates,
            _learning_rate(lr, lr_warmup, step), g_clip, jnp.any(keep),
        )
        # batch ESS of the pushforward of the source batch through the current map
        flow_now = eqx.combine(params, static)
        proposal, ladj = flow_now.inv_and_ladj(x)
        proposal = domain.wrap(proposal)
        log_weight = source(x) - target(proposal) + ladj
        ess = compute_ESS_log(log_weight, screen_fraction)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, 0.0, 1.0)
        return (params, first, second, updates), ess

    steps = step_offset + jnp.arange(1, chunk + 1)
    return lax.scan(body, (params, first, second, updates), steps)


def main():
    artifacts = HERE / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts / NAME
    run_dir.mkdir(exist_ok=True)
    log_path = artifacts / f"{NAME}.log"
    open(log_path, "w").close()

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a") as stream:
            stream.write(line + "\n")

    saved = manifest(SOURCE_RUN)
    if saved["status"] != "complete":
        raise RuntimeError(f"{SOURCE_RUN} is not complete: {saved['status']}")
    last = saved["stages"][-1]
    y_data = jnp.asarray(np.load(SOURCE_RUN / last["path"] / "samples.npy"))
    log(
        f"START propane 27D {NAME} | target data: {SOURCE_RUN.name} stage {last['stage']} "
        f"t={last['t']} ({y_data.shape[0]} rows) | RG_PARAM_1={P.RG_PARAM_1} "
        f"SCREEN_FRACTION={P.SCREEN_FRACTION} | VALID_SIZE={P.VALID_SIZE} "
        f"BATCH_SIZE={BATCH_SIZE} TRAIN_STEPS={TRAIN_STEPS} LR={P.LR} "
        f"LR_WARMUP={P.LR_WARMUP} U_CLIP={P.U_CLIP} G_CLIP={P.G_CLIP} CHUNKS={P.CHUNKS}"
    )

    base = Molecular_Potential.from_bundle(BUNDLE, temperature_kelvin=P.TEMPERATURE_KELVIN)
    target = base.regularized(P.RG_PARAM_1) if P.RG_PARAM_1 is not None else base
    source = base.source()
    domain = base.domain
    source_key, flow_key = jax.random.split(jax.random.key(P.SEED))
    x_valid = source.samples(source_key, N=P.VALID_SIZE)
    flow = Mixed_NSF(
        flow_key, domain, bins=P.BINS, transforms=P.TRANSFORMS,
        euclidean_bound=P.NSF_LIM, hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE, mask_strategy="balanced",
    ).zeros()

    reject = Manual_Reject()
    log(f"stop the training and evaluate: {reject.command}")
    monitor = Monitor(P.MONITOR_EVERY, f"[{P.MOLECULE} {NAME}] ", log)
    started = time.perf_counter()
    with reject:
        trained, history = _run_chunks(
            lambda **kw: _chunk(
                x_valid, y_data, source, target, kw.pop("state"), kw.pop("static"),
                domain, jax.random.fold_in(jax.random.key(31), P.SEED), **kw,
            ),
            flow, TRAIN_STEPS, reject,
            batch_size=BATCH_SIZE, lr=P.LR, coeff_lambda=1.0, monitor=monitor,
            checkpoint=P.CHECKPOINT, u_clip=P.U_CLIP, g_clip=P.G_CLIP,
            lr_warmup=P.LR_WARMUP, screen_fraction=P.SCREEN_FRACTION,
        )
    trained = jax.block_until_ready(trained)
    history = np.asarray(history, dtype=np.float32)
    steps_done = int(np.sum(np.isfinite(history)))
    training_seconds = time.perf_counter() - started
    log(f"training done: {steps_done} of {TRAIN_STEPS} steps in {training_seconds:.1f}s")

    _, trained_log_weight = _push_and_weights(x_valid, source, target, trained, domain, P.CHUNKS)
    identity_log_weight = _identity_weights(x_valid, source, target, P.CHUNKS)
    trained_ess = float(compute_ESS_log(trained_log_weight, P.SCREEN_FRACTION))
    identity_ess = float(compute_ESS_log(identity_log_weight, P.SCREEN_FRACTION))
    log(f"validation ESS={trained_ess:.4f} (identity={identity_ess:.4f}) on {P.VALID_SIZE} source samples")

    eqx.tree_serialise_leaves(run_dir / "flow.eqx", jax.device_get(trained))
    np.save(run_dir / "batch_ess_hist.npy", history)
    np.save(run_dir / "validation_log_weights.npy", np.asarray(trained_log_weight, dtype=np.float32))
    (run_dir / "run.json").write_text(json.dumps({
        "method": "KL+X_pi, data-driven", "target_data": str(SOURCE_RUN.relative_to(HERE)),
        "target_stage": last["stage"], "target_rows": int(y_data.shape[0]),
        "rg_param": P.RG_PARAM_1, "batch_size": BATCH_SIZE, "steps_done": steps_done, "steps_total": TRAIN_STEPS,
        "training_seconds": training_seconds, "valid_trained_ess": trained_ess,
        "valid_identity_ess": identity_ess,
    }, indent=2) + "\n")

    results = HERE / "results"
    results.mkdir(exist_ok=True)
    factor = 1.0 / math.sqrt(trained_ess)
    (results / f"{NAME}.md").write_text("\n".join([
        f"# Propane 27D KL+X_pi, data-driven (target batches from {SOURCE_RUN.name}, stage {last['stage']})",
        "",
        f"- Target data: `{y_data.shape[0]}` samples at t = 1 from `{SOURCE_RUN.relative_to(HERE)}`",
        f"- Regularization: `{P.RG_PARAM_1}`",
        f"- Batch size: `{BATCH_SIZE}`; steps: `{steps_done}` of `{TRAIN_STEPS}`",
        f"- Training time: `{training_seconds / 60:.2f} min`",
        f"- Validation ESS (source population of {P.VALID_SIZE}, one stage 0 -> 1): `{trained_ess:.6f}` (identity `{identity_ess:.6f}`)",
        f"- Factor F_hat = ESS^(-1/2): `{factor:.6g}`",
        "",
    ]) + "\n")
    log(json.dumps({"steps": steps_done, "valid_trained_ess": trained_ess, "factor": factor}))


if __name__ == "__main__":
    main()

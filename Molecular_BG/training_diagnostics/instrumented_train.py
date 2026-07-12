"""Project-local observability wrapper matching the public molecular KLX scan."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jflows.utils import compute_ESS_log
from jflows_md.train import _adam_step, _step_learning_rate, _tree_all_finite


def _stable_norm(tree):
    leaves = [leaf for leaf in jax.tree.leaves(tree) if getattr(leaf, "size", 0)]
    maximum = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    scaled = sum(
        jnp.sum(jnp.where(maximum > 0, leaf / maximum, 0.0) ** 2)
        for leaf in leaves
    )
    return jnp.where(maximum > 0, maximum * jnp.sqrt(scaled), 0.0)


def _tree_delta_max(first, second):
    return jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(left - right))
                for left, right in zip(
                    jax.tree.leaves(first), jax.tree.leaves(second), strict=True
                )
            ]
        )
    )


def instrumented_klx_one_step(
    target_samples,
    source_samples,
    source,
    target,
    flow,
    *,
    n_batch: int,
    lr: float,
    coeff_lambda: float,
    energy_origin=0.0,
    e_clip=float("inf"),
    g_clip=float("inf"),
    seed: int,
):
    """Expose the otherwise hidden arrays for the public trainer's first step."""

    key = jax.random.fold_in(jax.random.key(31), seed)
    target_key, source_key, permutation_key = jax.random.split(
        jax.random.fold_in(key, 1), 3
    )
    target_index = jax.random.choice(
        target_key, target_samples.shape[0], (n_batch,), replace=False
    )
    source_index = jax.random.choice(
        source_key, source_samples.shape[0], (n_batch,), replace=False
    )
    y = target_samples[target_index]
    x_source = source_samples[source_index]
    permutation = jax.random.permutation(permutation_key, n_batch)
    proposal, proposal_ladj = flow.inv_and_ladj(x_source)
    proposal = lax.stop_gradient(proposal)
    proposal_log_weight = lax.stop_gradient(
        source(x_source) - target(proposal) + proposal_ladj
    )
    energy = lax.stop_gradient(target(y))
    energy_keep = jnp.isfinite(energy)
    if e_clip != float("inf"):
        energy_keep = energy_keep & (energy - energy_origin <= e_clip)

    params, static = eqx.partition(flow, eqx.is_inexact_array)

    def loss_fn(trainable):
        candidate = eqx.combine(trainable, static)
        latent, ladj = candidate.call_and_ladj(y)
        ratio = source(latent) - energy - ladj
        keep = lax.stop_gradient(energy_keep & jnp.isfinite(ratio))
        safe = jnp.where(keep, ratio, 0.0)
        pair_keep = keep & keep[permutation]
        difference = jnp.abs(safe - safe[permutation])
        count = jnp.sum(keep)
        pair_count = jnp.sum(pair_keep)
        kl = jnp.sum(safe) / jnp.maximum(count, 1.0)
        x_term = jnp.sum(jnp.where(pair_keep, difference, 0.0)) / jnp.maximum(
            pair_count, 1.0
        )
        return kl + coeff_lambda * x_term, (ratio, keep, kl, x_term)

    (loss, (ratio, keep, kl, x_term)), gradients = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    trained_params, first, second, count, commit = _adam_step(
        params,
        zeros,
        zeros,
        gradients,
        loss,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(lr),
        g_clip,
        jnp.sum(keep) > 0,
    )
    return {
        "flow": eqx.combine(trained_params, static),
        "parameters": trained_params,
        "first_moment": first,
        "second_moment": second,
        "update_count": count,
        "commit": commit,
        "loss": loss,
        "kl": kl,
        "x_term": x_term,
        "ratio": ratio,
        "keep": keep,
        "kept_fraction": jnp.mean(keep.astype(ratio.dtype)),
        "gradients": gradients,
        "raw_gradient_finite": _tree_all_finite(gradients),
        "raw_gradient_norm": _stable_norm(gradients),
        "proposal_log_weight": proposal_log_weight,
        "ess": compute_ESS_log(proposal_log_weight),
        "parameter_delta_max": _tree_delta_max(trained_params, params),
        "target_index": target_index,
        "source_index": source_index,
        "permutation": permutation,
    }


@eqx.filter_jit
def instrumented_klx_scan(
    target_samples,
    source_samples,
    source,
    target,
    flow,
    n_batch: int,
    steps: int,
    lr: float,
    *,
    coeff_lambda: float = 0.0,
    energy_origin=0.0,
    e_clip=float("inf"),
    g_clip=float("inf"),
    seed: int = 0,
    lr_warmup: int = 0,
):
    """Exact public KLX scan plus persistent loss/update diagnostics."""

    key = jax.random.fold_in(jax.random.key(31), seed)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first0 = jax.tree.map(jnp.zeros_like, params)
    second0 = jax.tree.map(jnp.zeros_like, params)
    origin = jnp.asarray(energy_origin)

    def body(carry, step_index):
        current, first, second, update_count = carry
        target_key, source_key, permutation_key = jax.random.split(
            jax.random.fold_in(key, step_index), 3
        )
        y = target_samples[
            jax.random.choice(
                target_key, target_samples.shape[0], (n_batch,), replace=False
            )
        ]
        x_source = source_samples[
            jax.random.choice(
                source_key, source_samples.shape[0], (n_batch,), replace=False
            )
        ]
        permutation = jax.random.permutation(permutation_key, n_batch)
        flow_now = eqx.combine(current, static)
        proposal, proposal_ladj = flow_now.inv_and_ladj(x_source)
        proposal = lax.stop_gradient(proposal)
        proposal_log_weight = lax.stop_gradient(
            source(x_source) - target(proposal) + proposal_ladj
        )
        energy = lax.stop_gradient(target(y))
        energy_keep = jnp.isfinite(energy)
        if e_clip != float("inf"):
            energy_keep = energy_keep & (energy - origin <= e_clip)

        def loss_fn(trainable):
            candidate = eqx.combine(trainable, static)
            latent, ladj = candidate.call_and_ladj(y)
            ratio = source(latent) - energy - ladj
            keep = lax.stop_gradient(energy_keep & jnp.isfinite(ratio))
            safe = jnp.where(keep, ratio, 0.0)
            pair_keep = keep & keep[permutation]
            difference = jnp.abs(safe - safe[permutation])
            kl = jnp.sum(safe) / jnp.maximum(jnp.sum(keep), 1.0)
            x_term = jnp.sum(jnp.where(pair_keep, difference, 0.0)) / jnp.maximum(
                jnp.sum(pair_keep), 1.0
            )
            return kl + coeff_lambda * x_term, (ratio, keep)

        (loss, (ratio, keep)), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(current)
        before = current
        current, first, second, update_count, commit = _adam_step(
            current,
            first,
            second,
            gradients,
            loss,
            update_count,
            _step_learning_rate(lr, lr_warmup, step_index),
            g_clip,
            jnp.sum(keep) > 0,
        )
        return (current, first, second, update_count), (
            loss,
            compute_ESS_log(proposal_log_weight),
            jnp.mean(keep.astype(ratio.dtype)),
            commit,
            _tree_delta_max(current, before),
            _tree_all_finite(gradients),
            _stable_norm(gradients),
        )

    (params, first, second, update_count), history = lax.scan(
        body,
        (params, first0, second0, jnp.asarray(0, dtype=jnp.int32)),
        jnp.arange(1, steps + 1),
    )
    names = (
        "loss",
        "ess",
        "kept_fraction",
        "update_applied",
        "parameter_delta_max",
        "raw_gradient_finite",
        "raw_gradient_norm",
    )
    return (
        eqx.combine(params, static),
        {name: value for name, value in zip(names, history, strict=True)},
        first,
        second,
        update_count,
    )


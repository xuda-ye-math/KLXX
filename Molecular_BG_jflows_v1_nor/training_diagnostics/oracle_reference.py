"""Independent loss, importance-weight, and one-step Adam formulas."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jflows.utils import compute_ESS_log


BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


def tree_all_finite(tree):
    leaves = [leaf for leaf in jax.tree.leaves(tree) if hasattr(leaf, "dtype")]
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def stable_tree_norm(tree):
    leaves = [leaf for leaf in jax.tree.leaves(tree) if getattr(leaf, "size", 0)]
    maximum = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    scaled_square = sum(
        jnp.sum(jnp.where(maximum > 0, leaf / maximum, 0.0) ** 2)
        for leaf in leaves
    )
    return jnp.where(maximum > 0, maximum * jnp.sqrt(scaled_square), 0.0)


def clip_tree(tree, ceiling: float):
    if ceiling == float("inf"):
        return tree
    norm = stable_tree_norm(tree)
    factor = jnp.minimum(1.0, ceiling / (norm + EPS))
    return jax.tree.map(lambda leaf: leaf * factor, tree)


def reference_adam_one_step(params, grads, loss, *, lr: float, g_clip: float, eligible):
    """Independent first-step Adam with atomic finite commit."""

    raw_finite = tree_all_finite(grads)
    finite = jnp.asarray(eligible) & jnp.isfinite(loss) & raw_finite
    clean = jax.tree.map(
        lambda grad: jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad)),
        grads,
    )
    clipped = clip_tree(clean, g_clip)
    first_candidate = jax.tree.map(lambda grad: (1.0 - BETA1) * grad, clipped)
    second_candidate = jax.tree.map(
        lambda grad: (1.0 - BETA2) * grad * grad, clipped
    )
    first_hat = jax.tree.map(lambda value: value / (1.0 - BETA1), first_candidate)
    second_hat = jax.tree.map(
        lambda value: value / (1.0 - BETA2), second_candidate
    )
    parameter_candidate = jax.tree.map(
        lambda value, first, second: value
        - lr * first / (jnp.sqrt(second) + EPS),
        params,
        first_hat,
        second_hat,
    )
    candidate_finite = tree_all_finite(
        (parameter_candidate, first_candidate, second_candidate)
    )
    commit = finite & candidate_finite
    zero = jax.tree.map(jnp.zeros_like, params)
    parameters = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), parameter_candidate, params
    )
    first = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), first_candidate, zero
    )
    second = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), second_candidate, zero
    )
    return {
        "parameters": parameters,
        "parameter_candidate": parameter_candidate,
        "first_moment": first,
        "second_moment": second,
        "first_candidate": first_candidate,
        "second_candidate": second_candidate,
        "clipped_gradients": clipped,
        "raw_gradient_finite": raw_finite,
        "candidate_finite": candidate_finite,
        "commit": commit,
        "update_count": commit.astype(jnp.int32),
        "raw_gradient_norm": stable_tree_norm(grads),
        "clipped_gradient_norm": stable_tree_norm(clipped),
    }


def reference_klx_one_step(
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
    """Known-answer one-step molecular KL/KLX calculation."""

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
    permutation = jax.random.permutation(permutation_key, n_batch)
    y = target_samples[target_index]
    x_source = source_samples[source_index]

    proposal, proposal_ladj = flow.inv_and_ladj(x_source)
    proposal_log_weight = source(x_source) - target(proposal) + proposal_ladj
    energy = lax.stop_gradient(target(y))
    energy_keep = jnp.isfinite(energy)
    if e_clip != float("inf"):
        energy_keep = energy_keep & (energy - energy_origin <= e_clip)

    params, static = eqx.partition(flow, eqx.is_inexact_array)

    def objective(trainable):
        candidate = eqx.combine(trainable, static)
        latent, ladj = candidate.call_and_ladj(y)
        ratio = source(latent) - energy - ladj
        keep = lax.stop_gradient(energy_keep & jnp.isfinite(ratio))
        safe = jnp.where(keep, ratio, 0.0)
        pair_keep = keep & keep[permutation]
        difference = jnp.abs(safe - safe[permutation])
        count = jnp.sum(keep)
        pair_count = jnp.sum(pair_keep)
        kl = jnp.sum(safe) / jnp.maximum(count, 1)
        x_term = jnp.sum(jnp.where(pair_keep, difference, 0.0)) / jnp.maximum(
            pair_count, 1
        )
        return kl + coeff_lambda * x_term, (ratio, keep, kl, x_term)

    (loss, (ratio, keep, kl, x_term)), gradients = jax.value_and_grad(
        objective, has_aux=True
    )(params)
    adam = reference_adam_one_step(
        params,
        gradients,
        loss,
        lr=lr,
        g_clip=g_clip,
        eligible=jnp.sum(keep) > 0,
    )
    trained = eqx.combine(adam["parameters"], static)
    delta = max(
        jnp.max(jnp.abs(new - old))
        for new, old in zip(
            jax.tree.leaves(adam["parameters"]),
            jax.tree.leaves(params),
            strict=True,
        )
    )
    return {
        **adam,
        "flow": trained,
        "loss": loss,
        "kl": kl,
        "x_term": x_term,
        "ratio": ratio,
        "keep": keep,
        "kept_fraction": jnp.mean(keep.astype(ratio.dtype)),
        "gradients": gradients,
        "proposal_log_weight": proposal_log_weight,
        "ess": compute_ESS_log(proposal_log_weight),
        "parameter_delta_max": delta,
        "target_index": target_index,
        "source_index": source_index,
        "permutation": permutation,
        "target_batch": y,
        "source_batch": x_source,
    }


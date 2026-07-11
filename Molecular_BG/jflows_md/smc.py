"""Thin mixed-domain SMC controller built on public jflows concepts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from jflows.potential import linear_combination
from jflows.utils import compute_ESS_log, resample

from .mcmc import mixed_mala


def potential_space_smc(
    key: Array,
    samples: Array,
    source,
    target,
    *,
    t_list,
    step: float,
    iters: int,
    images: int = 3,
) -> tuple[Array, Array, Array]:
    """Reweight/resample/rejuvenate across an explicit potential bridge.

    Returns final samples, per-rung normalized ESS, and MALA acceptance arrays.
    The outer controller is intentionally ordinary Python; each MALA kernel is
    JAX-transformable and no clipped energy enters the weights.
    """

    t_values = [float(value) for value in t_list]
    if not t_values or any(not (0.0 < value <= 1.0) for value in t_values):
        raise ValueError("t_list must be nonempty and lie in (0, 1]")
    if any(right <= left for left, right in zip(t_values, t_values[1:])):
        raise ValueError("t_list must be strictly increasing")
    current = samples
    previous = 0.0
    ess_values, acceptance_values = [], []
    for value in t_values:
        key, resample_key, mala_key = jax.random.split(key, 3)
        delta = value - previous
        log_weight = -delta * (target(current) - source(current))
        log_weight = jnp.where(jnp.isfinite(log_weight), log_weight, -jnp.inf)
        ess_values.append(compute_ESS_log(log_weight))
        shifted = log_weight - jnp.max(log_weight)
        weight = jnp.exp(shifted)
        current = resample(resample_key, current, weight, N=current.shape[0])
        bridge = linear_combination([source, target], [1.0 - value, value])
        current, acceptance = mixed_mala(
            mala_key,
            current,
            bridge,
            target.domain,
            step=step,
            iters=iters,
            images=images,
        )
        acceptance_values.append(acceptance)
        previous = value
    return current, jnp.asarray(ess_values), jnp.stack(acceptance_values)

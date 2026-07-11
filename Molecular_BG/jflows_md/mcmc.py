"""Metropolis-adjusted Langevin dynamics on ``R^p x T^q``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from .core.domain import Mixed_Domain


def _wrapped_log_kernel(delta: Array, variance: float, images: int) -> Array:
    shifts = 2.0 * jnp.pi * jnp.arange(-images, images + 1, dtype=delta.dtype)
    terms = -(delta[..., None] + shifts) ** 2 / (2.0 * variance)
    return jnp.sum(logsumexp(terms, axis=-1), axis=-1)


def _proposal_log_density(
    destination: Array,
    mean: Array,
    domain: Mixed_Domain,
    variance: float,
    images: int,
) -> Array:
    euclidean_delta = destination[:, : domain.euclidean_dim] - mean[:, : domain.euclidean_dim]
    value = -0.5 * jnp.sum(euclidean_delta**2 / variance, axis=-1)
    if domain.periodic_dim:
        periodic_delta = destination[:, domain.euclidean_dim :] - mean[:, domain.euclidean_dim :]
        # Center the truncated image sum on the shortest torus displacement.
        # The drifted proposal mean itself need not lie in the principal box.
        periodic_delta = jnp.mod(periodic_delta + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        value = value + _wrapped_log_kernel(periodic_delta, variance, images)
    return value


def mixed_mala_step(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    step: float,
    images: int = 3,
) -> tuple[Array, Array]:
    """One mixed-domain MALA step; returns samples and per-chain acceptance."""

    if step <= 0 or images < 1:
        raise ValueError("step must be positive and images at least one")
    noise_key, accept_key = jax.random.split(key)
    energy = potential(samples)
    gradient = potential.grad(samples)
    mean_forward = samples - step * gradient
    proposal = domain.wrap(
        mean_forward
        + jnp.sqrt(2.0 * step)
        * jax.random.normal(noise_key, samples.shape, dtype=samples.dtype)
    )
    proposal_energy = potential(proposal)
    proposal_gradient = potential.grad(proposal)
    mean_reverse = proposal - step * proposal_gradient
    log_forward = _proposal_log_density(proposal, mean_forward, domain, 2.0 * step, images)
    log_reverse = _proposal_log_density(samples, mean_reverse, domain, 2.0 * step, images)
    log_alpha = energy - proposal_energy + log_reverse - log_forward
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)
    accepted = jnp.log(jax.random.uniform(accept_key, energy.shape)) < jnp.minimum(log_alpha, 0.0)
    result = jnp.where(accepted[:, None], proposal, samples)
    return result, accepted


def mixed_mala(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    step: float = 1e-4,
    iters: int = 1,
    images: int = 3,
) -> tuple[Array, Array]:
    """Run mixed-domain MALA and return final samples and mean acceptance."""

    if iters < 1:
        raise ValueError("iters must be at least one")
    keys = jax.random.split(key, iters)

    def body(state, subkey):
        updated, accepted = mixed_mala_step(
            subkey, state, potential, domain, step=step, images=images
        )
        return updated, accepted.astype(updated.dtype).mean()

    result, acceptance = jax.lax.scan(body, samples, keys)
    return result, acceptance

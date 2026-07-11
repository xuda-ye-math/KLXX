"""Gaussian-Euclidean x uniform-torus molecular source."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.potential import Potential

from .core.domain import Mixed_Domain


class Molecular_Source(Potential):
    domain: Mixed_Domain
    mean: Array
    variance: Array

    def __init__(
        self,
        domain: Mixed_Domain,
        mean: Array | list[float] | None = None,
        variance: Array | list[float] | None = None,
    ):
        self.domain = domain
        self.mean = jnp.zeros(domain.euclidean_dim) if mean is None else jnp.asarray(mean)
        self.variance = (
            jnp.ones(domain.euclidean_dim) if variance is None else jnp.asarray(variance)
        )
        if self.mean.shape != (domain.euclidean_dim,):
            raise ValueError("source mean has the wrong shape")
        if self.variance.shape != (domain.euclidean_dim,):
            raise ValueError("source variance has the wrong shape")
        if bool(jnp.any(self.variance <= 0)):
            raise ValueError("source variance must be positive")

    @classmethod
    def from_spec(cls, domain: Mixed_Domain, spec: Mapping) -> "Molecular_Source":
        return cls(domain, spec.get("source_mean"), spec.get("source_variance"))

    def __call__(self, q: Array) -> Array:
        euclidean = q[..., : self.domain.euclidean_dim]
        return 0.5 * jnp.sum((euclidean - self.mean) ** 2 / self.variance, axis=-1)

    def samples(self, key: Array, n: int) -> Array:
        gaussian_key, torus_key = jax.random.split(key)
        euclidean = self.mean + jnp.sqrt(self.variance) * jax.random.normal(
            gaussian_key, (n, self.domain.euclidean_dim), dtype=self.mean.dtype
        )
        periodic = jax.random.uniform(
            torus_key,
            (n, self.domain.periodic_dim),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=self.mean.dtype,
        )
        return jnp.concatenate((euclidean, periodic), axis=-1)

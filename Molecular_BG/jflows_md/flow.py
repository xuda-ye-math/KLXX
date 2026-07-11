"""Mixed-domain flow primitives that do not modify :mod:`jflows`.

The first potential milestone only needs an identity increment. It follows the
same public call/inverse/Jacobian protocol as jflows flows and provides a safe
baseline for SMC and later mixed spline couplings.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .core.domain import Mixed_Domain


def conditioner_features(x: Array, domain: Mixed_Domain) -> Array:
    """Raw Euclidean values plus cosine/sine embeddings of torsions."""

    euclidean = x[..., : domain.euclidean_dim]
    periodic = x[..., domain.euclidean_dim :]
    return jnp.concatenate((euclidean, jnp.cos(periodic), jnp.sin(periodic)), axis=-1)


class Mixed_Identity(eqx.Module):
    domain: Mixed_Domain

    def __call__(self, x: Array) -> Array:
        return self.domain.wrap(x)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        return self(x), jnp.zeros(x.shape[:-1], dtype=x.dtype)

    def inv(self, y: Array) -> Array:
        return self.domain.wrap(y)

    def inv_and_ladj(self, y: Array) -> tuple[Array, Array]:
        return self.inv(y), jnp.zeros(y.shape[:-1], dtype=y.dtype)

    def t(self) -> "Mixed_Identity":
        return self

    def zeros(self) -> "Mixed_Identity":
        return self

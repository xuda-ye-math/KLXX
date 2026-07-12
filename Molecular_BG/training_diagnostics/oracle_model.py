"""Known-answer mixed Gaussian/von-Mises target for trainer diagnostics."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from jflows.potential import Potential


EUCLIDEAN_MEAN = (0.4, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1)
EUCLIDEAN_VARIANCE = (0.7, 1.2, 0.8, 1.1, 0.9, 1.3, 0.75)
TORSION_CENTER = (0.6, -0.8)
TORSION_CONCENTRATION = (1.5, 2.0)


class Oracle_Target(Potential):
    """Shifted diagonal Gaussian times two independent von-Mises factors."""

    euclidean_mean: Array
    euclidean_variance: Array
    torsion_center: Array
    torsion_concentration: Array

    def __init__(self, dtype=jnp.float32):
        self.euclidean_mean = jnp.asarray(EUCLIDEAN_MEAN, dtype=dtype)
        self.euclidean_variance = jnp.asarray(EUCLIDEAN_VARIANCE, dtype=dtype)
        self.torsion_center = jnp.asarray(TORSION_CENTER, dtype=dtype)
        self.torsion_concentration = jnp.asarray(
            TORSION_CONCENTRATION, dtype=dtype
        )

    def __call__(self, value: Array) -> Array:
        euclidean = value[:, :7]
        torsion = value[:, 7:]
        gaussian = 0.5 * jnp.sum(
            (euclidean - self.euclidean_mean) ** 2 / self.euclidean_variance,
            axis=-1,
        )
        circular = -jnp.sum(
            self.torsion_concentration
            * jnp.cos(torsion - self.torsion_center),
            axis=-1,
        )
        return gaussian + circular

    def samples(self, seed: int, count: int, *, dtype=np.float32) -> Array:
        rng = np.random.Generator(np.random.PCG64(seed))
        euclidean = rng.normal(
            loc=np.asarray(EUCLIDEAN_MEAN),
            scale=np.sqrt(np.asarray(EUCLIDEAN_VARIANCE)),
            size=(count, 7),
        )
        torsion = np.stack(
            [
                rng.vonmises(center, concentration, size=count)
                for center, concentration in zip(
                    TORSION_CENTER, TORSION_CONCENTRATION, strict=True
                )
            ],
            axis=-1,
        )
        return jnp.asarray(
            np.concatenate((euclidean, torsion), axis=-1).astype(dtype)
        )


class Nonfinite_Row_Target(Potential):
    """Oracle target with +inf energy on a finite-coordinate sentinel row."""

    base: Oracle_Target
    threshold: float = eqx.field(static=True)

    def __init__(self, base: Oracle_Target, threshold: float = 10.0):
        self.base = base
        self.threshold = float(threshold)

    def __call__(self, value: Array) -> Array:
        ordinary = self.base(value)
        return jnp.where(value[:, 0] > self.threshold, jnp.inf, ordinary)


"""Project-local vacuum counterpart of a bundle-backed molecular potential."""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from jflows.potential import Potential
from jflows_md import Molecular_Potential


class Vacuum_Molecular_Potential(Potential):
    """GAFF bonded/nonbonded energy with OBC1/ACE removed exactly."""

    base: Molecular_Potential

    def __init__(self, base: Molecular_Potential):
        self.base = base

    @classmethod
    def from_bundle(cls, path) -> "Vacuum_Molecular_Potential":
        return cls(Molecular_Potential.from_bundle(path))

    @property
    def domain(self):
        return self.base.domain

    @property
    def dimension(self) -> int:
        return self.base.dimension

    @property
    def beta(self) -> Array:
        return self.base.beta

    @property
    def manifest_sha256(self) -> str:
        return self.base.manifest_sha256

    def cartesian(self, q: Array) -> Array:
        return self.base.cartesian(q)

    def source(self):
        return self.base.source()

    def reference_internal(self) -> Array:
        return self.base.reference_internal()

    def support_mask(self, x: Array) -> Array:
        return self.base.support_mask(x)

    def physical_energy(self, q: Array) -> Array:
        self.base._validate_internal(q)
        terms = self.base.forcefield.energy_terms(self.cartesian(q))
        return terms["bond"] + terms["angle"] + terms["torsion"] + terms["nonbonded"]

    def solvent_energy(self, q: Array) -> Array:
        self.base._validate_internal(q)
        return self.base.forcefield.energy_terms(self.cartesian(q))["gb"]

    def __call__(self, q: Array) -> Array:
        self.base._validate_internal(q)
        x, logdet = self.base.coordinates.to_cartesian(q)
        terms = self.base.forcefield.energy_terms(x)
        energy = terms["bond"] + terms["angle"] + terms["torsion"] + terms["nonbonded"]
        return self.beta * energy - logdet

    def regularized(
        self,
        energy_cut_kj_mol: float,
        *,
        energy_scale_kj_mol: float = 50.0,
        tail_fraction: float = 0.0,
    ) -> "Regularized_Vacuum_Molecular_Potential":
        return Regularized_Vacuum_Molecular_Potential(
            self,
            energy_cut_kj_mol=energy_cut_kj_mol,
            energy_scale_kj_mol=energy_scale_kj_mol,
            tail_fraction=tail_fraction,
        )


class Regularized_Vacuum_Molecular_Potential(Potential):
    """C1 lin-log regularization of the explicit vacuum Hamiltonian."""

    base: Vacuum_Molecular_Potential
    reference_energy_kj_mol: Array
    energy_cut_kj_mol: Array
    energy_scale_kj_mol: Array
    tail_fraction: Array

    def __init__(
        self,
        base: Vacuum_Molecular_Potential,
        *,
        energy_cut_kj_mol: float,
        energy_scale_kj_mol: float,
        tail_fraction: float,
    ):
        values = tuple(map(float, (energy_cut_kj_mol, energy_scale_kj_mol, tail_fraction)))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("vacuum regularization values must be finite")
        if energy_cut_kj_mol <= 0 or energy_scale_kj_mol <= 0:
            raise ValueError("vacuum regularization cut and scale must be positive")
        if not 0.0 <= tail_fraction <= 1.0:
            raise ValueError("vacuum regularization tail_fraction must lie in [0, 1]")
        self.base = base
        reference = base.physical_energy(base.reference_internal()[None])[0]
        self.reference_energy_kj_mol = jnp.asarray(reference)
        dtype = self.reference_energy_kj_mol.dtype
        self.energy_cut_kj_mol = jnp.asarray(energy_cut_kj_mol, dtype=dtype)
        self.energy_scale_kj_mol = jnp.asarray(energy_scale_kj_mol, dtype=dtype)
        self.tail_fraction = jnp.asarray(tail_fraction, dtype=dtype)

    @property
    def domain(self):
        return self.base.domain

    @property
    def dimension(self) -> int:
        return self.base.dimension

    @property
    def beta(self) -> Array:
        return self.base.beta

    def source(self):
        return self.base.source()

    def reference_internal(self) -> Array:
        return self.base.reference_internal()

    def cartesian(self, q: Array) -> Array:
        return self.base.cartesian(q)

    def physical_energy(self, q: Array) -> Array:
        return self.base.physical_energy(q)

    def _regularize_energy(self, energy: Array) -> Array:
        excess = energy - self.reference_energy_kj_mol
        over = jnp.maximum(excess - self.energy_cut_kj_mol, 0.0)
        compressed = (
            self.energy_cut_kj_mol
            + (1.0 - self.tail_fraction)
            * self.energy_scale_kj_mol
            * jnp.log1p(over / self.energy_scale_kj_mol)
            + self.tail_fraction * over
        )
        regularized_excess = jnp.where(
            excess > self.energy_cut_kj_mol, compressed, excess
        )
        return self.reference_energy_kj_mol + regularized_excess

    def regularized_physical_energy(self, q: Array) -> Array:
        return self._regularize_energy(self.base.physical_energy(q))

    def __call__(self, q: Array) -> Array:
        self.base.base._validate_internal(q)
        x, logdet = self.base.base.coordinates.to_cartesian(q)
        terms = self.base.base.forcefield.energy_terms(x)
        energy = terms["bond"] + terms["angle"] + terms["torsion"] + terms["nonbonded"]
        return self.beta * self._regularize_energy(energy) - logdet

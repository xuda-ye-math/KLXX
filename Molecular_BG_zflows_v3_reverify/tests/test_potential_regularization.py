"""Focused regression tests for the two coexisting molecular regularizers.

These tests deliberately use the historical glycerol AMBER input.  They verify
the two public potential classes independently:

* ``PDB_Potential`` retains the original absolute ``e_cap`` plus pair-distance
  ``r_floor`` controls.
* ``C_PDB_Potential`` uses the reference-shifted c50 energy map and the common
  c/r convention: ``r_floor=0`` is pure c, while a positive value applies a
  distance floor independently of the energy map.

No Boltzmann-generator training is launched here.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from openmm import unit

from zflows_md.boltzmann import build
from zflows_md.forcefield import c_regularize_energy, build_system, softcap_energy
from zflows_md.potential import C_PDB_Potential, PDB_Potential


DATA = Path(__file__).resolve().parents[1] / "zflows_md" / "data"
PRMTOP = DATA / "glycerol.prmtop"
RST7 = DATA / "glycerol.rst7"


def _glycerol_inputs():
    structure, system = build_system(str(PRMTOP), str(RST7), environment="vacuum")
    reference = np.asarray(
        structure.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    bonds = [(bond.atom1.idx, bond.atom2.idx) for bond in structure.bonds]
    # Two identical frames are sufficient for this construction test.  build()
    # applies its documented positive floor to the whitening standard deviations.
    frames = np.repeat(reference[None, ...], 2, axis=0)
    return system, bonds, reference, frames


def _reference_xi(problem):
    reference = torch.as_tensor(
        problem["reference_positions"],
        dtype=problem["ff"].r_floor.dtype,
    ).unsqueeze(0)
    z, _ = problem["ic"].to_internal(reference)
    stats = problem["whitening"]
    nb = problem["M"] - 1
    na = problem["M"] - 2
    return torch.cat(
        [
            (z[:, :nb] - stats["mu_b"]) / stats["sig_b"],
            (z[:, nb : nb + na] - stats["mu_a"]) / stats["sig_a"],
            z[:, nb + na :],
        ],
        dim=-1,
    )


def _build_problem(regularization, *, r_floor=0.10, distance_floor=None):
    system, bonds, reference, frames = _glycerol_inputs()
    problem = build(
        system=system,
        bonds=bonds,
        md_frames=frames,
        reference_positions=reference,
        dtype=torch.float64,
        environment="vacuum",
        regularization=regularization,
        r_floor=r_floor,
        e_cap=100.0,
        e_cap_scale=50.0,
        c=50.0,
        c_scale=50.0,
        c_tail_fraction=0.0,
        distance_floor=distance_floor,
    )
    problem["reference_positions"] = reference
    return problem


def _two_finite_samples(problem):
    xi = _reference_xi(problem)
    shifted = xi.clone()
    shifted[:, problem["tor_start"] :] += 0.15
    return torch.cat([xi, shifted], dim=0)


def test_historical_e_cap_and_r_floor_potential():
    """The original e/r potential remains constructible and differentiable."""

    problem = _build_problem("er")
    target = problem["u"]
    assert isinstance(target, PDB_Potential)
    assert not isinstance(target, C_PDB_Potential)
    assert target.e_cap.item() == 100.0
    assert target.ff.r_floor.item() == 0.10

    xi = _two_finite_samples(problem).requires_grad_(True)
    z = target.unwhiten(xi)
    cartesian, logdet = target.ic.to_cartesian(z)
    expected = (
        target.beta * softcap_energy(target.ff(cartesian), target.e_cap, 50.0)
        - logdet
        - target.logdet_white
    )
    actual = target(xi)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.sum().backward()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(xi.grad).all()

    target.set_regularization(200.0)
    target.set_r_floor(0.08)
    assert target.e_cap.item() == 200.0
    assert target.ff.r_floor.item() == 0.08


def test_reference_shifted_c50_potential():
    """The new c50 potential is independent of the historical e/r controls."""

    problem = _build_problem("c", r_floor=0.0)
    historical = _build_problem("er")
    target = problem["u"]
    assert isinstance(target, C_PDB_Potential)
    assert not isinstance(target, PDB_Potential)
    assert target.c.item() == 50.0
    assert target.c_scale.item() == 50.0
    assert target.tail_fraction.item() == 0.0
    assert target.ff.r_floor.item() == 0.0
    assert not hasattr(target, "e_cap")

    # Both regularizers wrap the same physical AMBER model.  Every force-field
    # parameter must match exactly; only r_floor itself differs because it is a
    # component of the historical e/r regularization rather than the model.
    assert type(target.ff) is type(historical["ff"])
    for name, value in target.ff.state_dict().items():
        if name != "r_floor":
            torch.testing.assert_close(
                value, historical["ff"].state_dict()[name], rtol=0.0, atol=0.0
            )
    reference = torch.as_tensor(
        problem["reference_positions"], dtype=torch.float64
    ).unsqueeze(0)
    torch.testing.assert_close(
        target.ff(reference), historical["ff"](reference), rtol=1e-12, atol=1e-12
    )

    xi = _two_finite_samples(problem).requires_grad_(True)
    z = target.unwhiten(xi)
    cartesian, logdet = target.ic.to_cartesian(z)
    raw = target.ff(cartesian)
    regularized = c_regularize_energy(
        raw,
        target.reference_energy,
        target.c,
        target.c_scale,
        target.tail_fraction,
    )
    expected = target.beta * regularized - logdet - target.logdet_white
    actual = target(xi)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.sum().backward()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(xi.grad).all()

    # The c50 map is exactly physical through E_ref + 50, compresses above it,
    # and is C1 at the join (both one-sided slopes are one).
    ref = target.reference_energy.detach()
    probe = ref + torch.tensor([0.0, 49.0, 50.0, 100.0], dtype=torch.float64)
    mapped = c_regularize_energy(probe, ref, 50.0, 50.0, 0.0)
    torch.testing.assert_close(mapped[:3], probe[:3], rtol=0.0, atol=1e-12)
    assert mapped[3] < probe[3]

    epsilon = 1e-5
    join_probe = (ref + torch.tensor([50.0 - epsilon, 50.0 + epsilon])).requires_grad_(True)
    join_value = c_regularize_energy(join_probe, ref, 50.0, 50.0, 0.0)
    slopes = torch.autograd.grad(join_value.sum(), join_probe)[0]
    torch.testing.assert_close(slopes, torch.ones_like(slopes), rtol=1e-6, atol=1e-6)

    target.set_c(40.0)
    assert target.c.item() == 40.0


def test_c_r_family_default_zero_member_and_schedule():
    """c/r defaults positive; r_floor=0 is pure c and can be scheduled."""

    default = _build_problem("c")
    pure_c = _build_problem("c", r_floor=0.0)
    hybrid = _build_problem("c", r_floor=0.20)
    target = hybrid["u"]
    assert isinstance(target, C_PDB_Potential)
    assert default["r_floor"] == 0.10
    assert default["ff"].r_floor.item() == 0.10
    assert pure_c["ff"].r_floor.item() == 0.0
    assert hybrid["r_floor"] == 0.20
    assert target.ff.r_floor.item() == 0.20

    target.set_r_floor(0.10)
    assert target.ff.r_floor.item() == 0.10

    # The explicit floor changes only the force-field r_floor buffer.  All
    # physical parameters and all c-map parameters remain exactly matched.
    for name, value in pure_c["ff"].state_dict().items():
        if name != "r_floor":
            torch.testing.assert_close(
                value, target.ff.state_dict()[name], rtol=0.0, atol=0.0
            )
    assert target.c.item() == pure_c["u"].c.item()
    assert target.c_scale.item() == pure_c["u"].c_scale.item()
    assert target.tail_fraction.item() == pure_c["u"].tail_fraction.item()

    with pytest.raises(ValueError, match="finite and nonnegative"):
        target.set_r_floor(-0.01)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        target.set_r_floor(float("nan"))


def test_actual_nonbonded_exact_collision_has_defined_raw_limit():
    """The zero-floor force field returns physical infinities, never NaN."""

    problem = _build_problem("c", r_floor=0.0)
    ff = problem["ff"]
    assert ff.r_floor.item() == 0.0
    positions = torch.zeros((1, ff.M, 3), dtype=torch.float64)

    # Probe one actual force-field pair carrying LJ repulsion.  At an exact
    # collision its raw energy limit is +inf and therefore zero Boltzmann
    # weight; NaN would make resampling undefined.
    candidates = torch.nonzero(ff.npair_eps != 0, as_tuple=False).flatten()
    prefix = "npair"
    if len(candidates) == 0:
        candidates = torch.nonzero(ff.epair_eps != 0, as_tuple=False).flatten()
        prefix = "epair"
    index = int(candidates[0])
    value = ff._pair_nb(
        positions,
        getattr(ff, prefix)[index : index + 1],
        getattr(ff, prefix + "_qq")[index : index + 1],
        getattr(ff, prefix + "_sig")[index : index + 1],
        getattr(ff, prefix + "_eps")[index : index + 1],
    )
    assert torch.isposinf(value).all()
    assert not torch.isnan(value).any()

    # A genuinely excluded pair has exactly zero interaction even when its
    # coordinates coincide; zero coefficients must not form 0 * inf.
    excluded = torch.nonzero(
        (ff.epair_qq == 0) & (ff.epair_eps == 0), as_tuple=False
    ).flatten()
    if len(excluded):
        index = int(excluded[0])
        value = ff._pair_nb(
            positions,
            ff.epair[index : index + 1],
            ff.epair_qq[index : index + 1],
            ff.epair_sig[index : index + 1],
            ff.epair_eps[index : index + 1],
        )
        torch.testing.assert_close(value, torch.zeros_like(value))


if __name__ == "__main__":
    test_historical_e_cap_and_r_floor_potential()
    print("PASS: historical PDB_Potential (e_cap + r_floor)")
    test_reference_shifted_c50_potential()
    print("PASS: C_PDB_Potential (reference-shifted c50)")
    test_c_r_family_default_zero_member_and_schedule()
    print("PASS: C_PDB_Potential c/r family and pure-c zero member")
    test_actual_nonbonded_exact_collision_has_defined_raw_limit()
    print("PASS: raw exact-collision force-field limit")

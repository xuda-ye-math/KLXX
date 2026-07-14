"""Fast checks for the immutable ethane collision-mechanism runner."""

from pathlib import Path

import numpy as np
from openmm import XmlSerializer
import torch

from run_ethane_collision_stress import (
    EXPECTED_PAIR,
    INPUTS,
    _cap_radius,
    _grid,
    _pair_values,
    _reference_energy,
    _selected_pair,
)
from zflows_md.forcefield import Amber_Force_Field


def _forcefield(dtype=torch.float64):
    system = XmlSerializer.deserialize(
        (Path(INPUTS) / "system_vacuum.xml").read_text(encoding="utf-8")
    )
    return Amber_Force_Field(system, dtype=dtype, r_floor=0.0)


def test_frozen_pair_and_grid_are_deterministic():
    ff = _forcefield()
    pair = _selected_pair(ff)
    assert tuple(pair["pair"]) == EXPECTED_PAIR
    reference = _reference_energy(
        pair["qq_e2"], pair["sigma_nm"], pair["epsilon_kj_mol"]
    )
    cap = _cap_radius(
        pair["qq_e2"], pair["sigma_nm"], pair["epsilon_kj_mol"], reference
    )
    grid = _grid(np.float64, pair["sigma_nm"], cap)
    assert grid[0] == 0.0
    assert np.any(grid == 0.10)
    assert np.any(grid == 0.20)
    assert grid[-1] >= 0.50


def test_live_kernel_positive_floor_and_matched_maps():
    ff = _forcefield()
    info = _selected_pair(ff)
    pair = tuple(info["pair"])
    values = dict(
        qq=info["qq_e2"],
        sigma=info["sigma_nm"],
        epsilon=info["epsilon_kj_mol"],
    )
    reference = _reference_energy(**values)
    radii = np.asarray([0.0, 0.05, 0.099, 0.100001, 0.20], dtype=np.float64)
    c_energy, c_force = _pair_values(
        ff,
        radii,
        pair=pair,
        floor=0.10,
        mapping="c",
        reference_energy=reference,
        **values,
    )
    e_energy, e_force = _pair_values(
        ff,
        radii,
        pair=pair,
        floor=0.10,
        mapping="e",
        reference_energy=reference,
        **values,
    )
    assert np.isfinite(c_energy).all()
    assert np.isfinite(c_force).all()
    assert np.all(c_force[radii < 0.10] == 0.0)
    np.testing.assert_allclose(c_energy, e_energy, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c_force, e_force, rtol=1e-13, atol=1e-13)


def test_zero_floor_family_member_is_exact_pure_c():
    ff = _forcefield()
    info = _selected_pair(ff)
    pair = tuple(info["pair"])
    values = dict(
        qq=info["qq_e2"],
        sigma=info["sigma_nm"],
        epsilon=info["epsilon_kj_mol"],
    )
    reference = _reference_energy(**values)
    radii = np.geomspace(1e-5 * values["sigma"], 0.5, 32)
    pure = _pair_values(
        ff,
        radii,
        pair=pair,
        floor=0.0,
        mapping="c",
        reference_energy=reference,
        **values,
    )
    family = _pair_values(
        ff,
        radii,
        pair=pair,
        floor=0.0,
        mapping="c_r0",
        reference_energy=reference,
        **values,
    )
    assert np.array_equal(pure[0], family[0])
    assert np.array_equal(pure[1], family[1])

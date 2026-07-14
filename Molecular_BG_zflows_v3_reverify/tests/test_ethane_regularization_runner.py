"""CPU behavioral tests for the small-system MALA-SMC evidence runner."""

import time

import numpy as np
import pytest
import torch

from run_ethane_regularization_study import (
    SAMPLE_PARAMETERS,
    _adaptive_bridge_sample,
    _adaptive_regularization_sharpen,
    _mala_rejuvenate,
    _normalized_weights_tensor,
    _resample_with_ancestry,
    _wrapped_log_proposal,
)
from zflows_md.potential import Potential, potential_from


def test_weight_guard_and_seeded_ancestry():
    samples = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    logw = torch.linspace(-5.0, 0.0, 20)
    weights, metrics = _normalized_weights_tensor(logw)
    assert metrics["valid"]
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    first = _resample_with_ancestry(samples, logw, 1234)
    second = _resample_with_ancestry(samples, logw, 1234)
    np.testing.assert_array_equal(first[1], second[1])
    with pytest.raises(FloatingPointError):
        _normalized_weights_tensor(torch.tensor([0.0, float("nan")]))
    with pytest.raises(FloatingPointError):
        _normalized_weights_tensor(torch.tensor([0.0, float("inf")]))


def test_wrapped_proposal_is_periodic():
    destination = torch.tensor([[0.2, -2.9]], dtype=torch.float64)
    mean = torch.tensor([[-0.1, 3.0]], dtype=torch.float64)
    shifted = destination.clone()
    shifted[:, 1] += 2.0 * torch.pi
    first = _wrapped_log_proposal(
        destination, mean, tor_start=1, step=0.01, image_radius=2
    )
    second = _wrapped_log_proposal(
        shifted, mean, tor_start=1, step=0.01, image_radius=2
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-12)


def test_mixed_domain_mala_has_finite_acceptance():
    potential = potential_from(
        lambda x: 0.5 * x[:, 0] ** 2 + 1.0 - torch.cos(x[:, 1])
    )
    potential._grad_fn = lambda x: torch.stack((x[:, 0], torch.sin(x[:, 1])), -1)
    potential._eval_fn = potential.forward
    samples = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 64),
            torch.linspace(-torch.pi, torch.pi, 64),
        ),
        dim=-1,
    )
    output, diagnostics = _mala_rejuvenate(
        samples,
        potential,
        tor_start=1,
        step=0.01,
        steps=10,
        chunks=2,
        image_radius=2,
        seed=4321,
        deadline=time.time() + 30.0,
    )
    assert output.shape == samples.shape
    assert torch.isfinite(output).all()
    assert diagnostics["invalid"] == 0
    assert 0.0 < diagnostics["acceptance_rate"] <= 1.0
    assert torch.all(output[:, 1] >= -torch.pi)
    assert torch.all(output[:, 1] < torch.pi)


class _MutableFloorPotential(Potential):
    def __init__(self):
        super().__init__()
        self.register_buffer("r_floor", torch.tensor(0.2))
        self._grad_fn = lambda x: torch.stack(
            (x[:, 0], self.r_floor * torch.sin(x[:, 1])), -1
        )
        self._eval_fn = self.forward

    def forward(self, x):
        return 0.5 * x[:, 0] ** 2 + self.r_floor * (1.0 - torch.cos(x[:, 1]))

    def set_r_floor(self, value):
        self.r_floor.fill_(float(value))


def test_guarded_bridge_and_arithmetic_sharpen_reach_endpoints(tmp_path):
    original = dict(SAMPLE_PARAMETERS)
    SAMPLE_PARAMETERS.update(
        {
            "conditional_ess_min": 0.5,
            "initial_bridge_step": 0.5,
            "initial_sharpen_step": 0.5,
            "minimum_parameter_step": 1e-4,
            "step_growth": 1.0,
            "max_bridge_levels": 4,
            "max_sharpen_levels": 4,
            "mala_step": 0.01,
            "mala_steps": 2,
            "mala_chunks": 2,
            "wrapped_image_radius": 2,
        }
    )
    try:
        source = potential_from(lambda x: 0.5 * (x**2).sum(-1))
        source._grad_fn = lambda x: x
        source._eval_fn = source.forward
        target = _MutableFloorPotential()
        initial = torch.stack(
            (torch.linspace(-1.0, 1.0, 64), torch.linspace(-3.0, 3.0, 64)),
            dim=-1,
        )
        bridge_dir = tmp_path / "bridge"
        bridge_dir.mkdir()
        particles, bridge = _adaptive_bridge_sample(
            initial,
            source,
            target,
            tor_start=1,
            stage_dir=bridge_dir,
            deadline=time.time() + 30.0,
            checkpoint=lambda *_: None,
        )
        assert bridge[-1]["t"] == 1.0
        assert all(record["mala"]["invalid"] == 0 for record in bridge)
        sharpen_dir = tmp_path / "sharpen"
        sharpen_dir.mkdir()
        arm = {
            "initial_floor": 0.2,
            "r_schedule": {"start": 0.2, "end": 0.1},
            "e_schedule": None,
        }
        particles, sharpen = _adaptive_regularization_sharpen(
            particles,
            target,
            arm,
            tor_start=1,
            stage_dir=sharpen_dir,
            deadline=time.time() + 30.0,
            bridge_records=bridge,
            checkpoint=lambda *_: None,
        )
        assert sharpen[-1]["progress"] == 1.0
        assert target.r_floor.item() == pytest.approx(0.1)
        assert list(bridge_dir.glob("bridge-*.npz"))
        assert list(sharpen_dir.glob("sharpen-*.npz"))
    finally:
        SAMPLE_PARAMETERS.clear()
        SAMPLE_PARAMETERS.update(original)

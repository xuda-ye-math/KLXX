"""Regression tests for frozen initial validation populations."""

import pytest
import torch

from zflows_md.boltzmann import (
    _initial_validation_population,
    run_boltzmann,
)
from zflows_md.potential import Source


class Counting_Source(Source):
    def __init__(self):
        super().__init__(n_white=2, n_tor=1)
        self.sample_calls = 0

    def samples(self, count):
        self.sample_calls += 1
        return super().samples(count)


def test_none_path_is_bitwise_and_rng_equivalent_to_historical_draw():
    source = Counting_Source()
    torch.manual_seed(314159)
    expected = source.samples(8)
    expected_rng = torch.random.get_rng_state().clone()

    source.sample_calls = 0
    torch.manual_seed(314159)
    actual = _initial_validation_population(source, 8, "cpu", None)
    actual_rng = torch.random.get_rng_state().clone()

    assert source.sample_calls == 1
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0.0, atol=0.0)


def test_explicit_population_bypasses_sampling_rng_and_clones_caller():
    source = Counting_Source()
    initial = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    original = initial.clone()
    torch.manual_seed(271828)
    rng_before = torch.random.get_rng_state().clone()

    first = _initial_validation_population(source, 8, "cpu", initial)
    second = _initial_validation_population(source, 8, "cpu", initial)
    rng_after = torch.random.get_rng_state().clone()

    assert source.sample_calls == 0
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rng_after, rng_before, rtol=0.0, atol=0.0)
    first[0, 0] = -999.0
    torch.testing.assert_close(initial, original, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "bad, error, message",
    [
        (torch.zeros(8), ValueError, "shape"),
        (torch.zeros(7, 3), ValueError, "sample count"),
        (torch.zeros(8, 4), ValueError, "dimension"),
        (torch.zeros(8, 3, dtype=torch.float64), TypeError, "dtype"),
        (torch.zeros(8, 3, dtype=torch.int64), TypeError, "floating-point"),
        (
            torch.full((8, 3), float("nan"), dtype=torch.float32),
            ValueError,
            "finite",
        ),
        (
            torch.full((8, 3), float("inf"), dtype=torch.float32),
            ValueError,
            "finite",
        ),
        (
            torch.full((8, 3), float("-inf"), dtype=torch.float32),
            ValueError,
            "finite",
        ),
    ],
)
def test_explicit_population_validation(bad, error, message):
    source = Counting_Source()
    with pytest.raises(error, match=message):
        _initial_validation_population(source, 8, "cpu", bad)
    assert source.sample_calls == 0


def test_explicit_population_rejects_non_tensor_without_sampling():
    source = Counting_Source()
    with pytest.raises(TypeError, match="torch.Tensor"):
        _initial_validation_population(source, 8, "cpu", [[0.0] * 3] * 8)
    assert source.sample_calls == 0


def test_invalid_population_fails_before_flow_construction():
    source = Counting_Source()
    called = False

    def flow_factory():
        nonlocal called
        called = True
        raise AssertionError("flow_factory must not run")

    with pytest.raises(ValueError, match="sample count"):
        run_boltzmann(
            source,
            source,
            flow_factory,
            n_valid=8,
            n_pool=4,
            n_batch=2,
            steps=1,
            lr=1e-3,
            lam=1.0,
            mc_step=1e-3,
            mc_iters=1,
            smc_rungs=1,
            smc_rung_iters=1,
            adaptive_tau=0.1,
            validation_tau=0.1,
            shrink_factor=0.7,
            wrap=lambda value: value,
            qt_fn=lambda target: None,
            device="cpu",
            initial_validation=torch.zeros(7, 3),
        )
    assert not called
    assert source.sample_calls == 0

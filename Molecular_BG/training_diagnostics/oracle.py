#!/usr/bin/env python
"""Known-answer loss/gradient/Adam/weight gate before molecular training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import parameters as P


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_error(first, second) -> tuple[float, float]:
    import equinox as eqx
    import jax

    absolute, relative = 0.0, 0.0
    for left, right in zip(
        [leaf for leaf in jax.tree.leaves(first) if eqx.is_inexact_array(leaf)],
        [leaf for leaf in jax.tree.leaves(second) if eqx.is_inexact_array(leaf)],
        strict=True,
    ):
        delta = np.abs(np.asarray(left) - np.asarray(right))
        scale = np.maximum(np.abs(np.asarray(right)), 1e-30)
        absolute = max(absolute, float(np.max(delta)))
        relative = max(relative, float(np.max(delta / scale)))
    return absolute, relative


def _assert_array(name, actual, expected, *, atol: float, rtol: float) -> dict:
    actual_np, expected_np = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual_np, expected_np, atol=atol, rtol=rtol)
    delta = np.abs(actual_np - expected_np)
    scale = np.maximum(np.abs(expected_np), 1e-30)
    return {
        "maximum_absolute_error": float(np.max(delta)) if delta.size else 0.0,
        "maximum_relative_error": float(np.max(delta / scale)) if delta.size else 0.0,
    }


def _assert_tree(name, actual, expected, *, atol: float, rtol: float) -> dict:
    import equinox as eqx
    import jax

    actual_leaves = [
        leaf for leaf in jax.tree.leaves(actual) if eqx.is_inexact_array(leaf)
    ]
    expected_leaves = [
        leaf for leaf in jax.tree.leaves(expected) if eqx.is_inexact_array(leaf)
    ]
    if len(actual_leaves) != len(expected_leaves):
        raise AssertionError(f"{name} leaf count differs")
    errors = []
    for index, (left, right) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        errors.append(
            _assert_array(
                f"{name}[{index}]", left, right, atol=atol, rtol=rtol
            )
        )
    return {
        "leaf_count": len(errors),
        "maximum_absolute_error": max(
            (value["maximum_absolute_error"] for value in errors), default=0.0
        ),
        "maximum_relative_error": max(
            (value["maximum_relative_error"] for value in errors), default=0.0
        ),
    }


def _masked_gradient(flow, source, y, energy, keep, *, sanitize: bool):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    params, static = eqx.partition(flow, eqx.is_inexact_array)
    if sanitize:
        valid_index = jnp.argmax(keep)
        y = jnp.where(keep[:, None], y, y[valid_index])

    def objective(trainable):
        candidate = eqx.combine(trainable, static)
        latent, ladj = candidate.call_and_ladj(y)
        ratio = source(latent) - energy - ladj
        safe = jnp.where(keep, ratio, 0.0)
        return jnp.sum(safe) / jnp.maximum(jnp.sum(keep), 1)

    return jax.value_and_grad(objective)(params)


def _valid_only_gradient(flow, source, y, energy, keep):
    import equinox as eqx
    import jax

    params, static = eqx.partition(flow, eqx.is_inexact_array)
    y_valid, energy_valid = y[keep], energy[keep]

    def objective(trainable):
        candidate = eqx.combine(trainable, static)
        latent, ladj = candidate.call_and_ladj(y_valid)
        return (source(latent) - energy_valid - ladj).mean()

    return jax.value_and_grad(objective)(params)


def run(dtype_name: str, output: Path, full: bool) -> dict:
    import equinox as eqx
    import jax

    jax.config.update("jax_enable_x64", dtype_name == "float64")
    import jax.numpy as jnp

    from jflows.loss import forward_KL_G
    from jflows.train import train_forward_KL_G  # noqa: F401 -- provenance/API check
    from jflows.utils import compute_ESS_log, importance_weights_log
    from jflows_md import Mixed_NSF, Molecular_Source
    from jflows_md.core.domain import Mixed_Domain
    from jflows_md.train import train_molecular_forward_KLX_G

    from instrumented_train import instrumented_klx_one_step, instrumented_klx_scan
    from oracle_model import Nonfinite_Row_Target, Oracle_Target
    from oracle_reference import reference_klx_one_step

    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    numpy_dtype = np.float64 if dtype_name == "float64" else np.float32
    atol = P.ORACLE_ATOL_FLOAT64 if dtype_name == "float64" else P.ORACLE_ATOL_FLOAT32
    rtol = P.ORACLE_RTOL_FLOAT64 if dtype_name == "float64" else P.ORACLE_RTOL_FLOAT32
    domain = Mixed_Domain(7, 2)
    source = Molecular_Source(domain)
    target = Oracle_Target(dtype)
    target_pool = target.samples(P.ORACLE_TARGET_SEED, 128, dtype=numpy_dtype)
    source_pool = source.samples(jax.random.key(P.ORACLE_SOURCE_SEED), 128)

    def new_flow(seed: int):
        return Mixed_NSF(
            jax.random.key(seed),
            domain,
            bins=P.BINS,
            transforms=P.TRANSFORMS,
            euclidean_bound=P.EUCLIDEAN_BOUND,
            hidden_features=P.HIDDEN_FEATURES,
            slope=P.SLOPE,
        ).zeros()

    comparisons = {}
    for label, coefficient, seed in (
        ("bare_kl", 0.0, P.ORACLE_TRAINER_SEED),
        ("klx", 1.0, P.ORACLE_KLX_SEED),
    ):
        flow = new_flow(P.ORACLE_FLOW_SEED)
        reference = reference_klx_one_step(
            target_pool,
            source_pool,
            source,
            target,
            flow,
            n_batch=32,
            lr=1e-3,
            coeff_lambda=coefficient,
            g_clip=100.0,
            seed=seed,
        )
        instrumented = instrumented_klx_one_step(
            target_pool,
            source_pool,
            source,
            target,
            flow,
            n_batch=32,
            lr=1e-3,
            coeff_lambda=coefficient,
            g_clip=100.0,
            seed=seed,
        )
        public_flow, public_ess, public_kept, public_updated = (
            train_molecular_forward_KLX_G(
                target_pool,
                source_pool,
                source,
                target,
                flow,
                n_batch=32,
                steps=1,
                lr=1e-3,
                coeff_lambda=coefficient,
                g_clip=100.0,
                seed=seed,
            )
        )
        jax.block_until_ready((public_flow, public_ess))
        item = {
            "loss": _assert_array(
                f"{label} loss", instrumented["loss"], reference["loss"], atol=atol, rtol=rtol
            ),
            "gradient": _assert_tree(
                f"{label} gradient", instrumented["gradients"], reference["gradients"], atol=atol, rtol=rtol
            ),
            "first_moment": _assert_tree(
                f"{label} first moment", instrumented["first_moment"], reference["first_moment"], atol=atol, rtol=rtol
            ),
            "second_moment": _assert_tree(
                f"{label} second moment", instrumented["second_moment"], reference["second_moment"], atol=atol, rtol=rtol
            ),
            "reference_to_instrumented_flow": _assert_tree(
                f"{label} reference flow", instrumented["flow"], reference["flow"], atol=atol, rtol=rtol
            ),
            "instrumented_to_public_flow": _assert_tree(
                f"{label} public flow", public_flow, instrumented["flow"], atol=atol, rtol=rtol
            ),
            "proposal_log_weight": _assert_array(
                f"{label} log weights", instrumented["proposal_log_weight"], reference["proposal_log_weight"], atol=atol, rtol=rtol
            ),
            "ess": _assert_array(
                f"{label} ESS", public_ess[0], reference["ess"], atol=atol, rtol=rtol
            ),
            "kept": _assert_array(
                f"{label} kept", public_kept[0], reference["kept_fraction"], atol=atol, rtol=rtol
            ),
            "parameter_delta_max": float(reference["parameter_delta_max"]),
            "raw_gradient_norm": float(reference["raw_gradient_norm"]),
        }
        if bool(public_updated[0]) is not bool(reference["commit"]):
            raise AssertionError(f"{label} public/reference update Boolean differs")
        if int(instrumented["update_count"]) != int(reference["update_count"]):
            raise AssertionError(f"{label} update counter differs")
        if item["parameter_delta_max"] <= P.ORACLE_PARAMETER_DELTA_MIN:
            raise AssertionError(f"{label} parameter update is a no-op")
        dot = sum(
            float(jnp.sum((new - old) * grad))
            for new, old, grad in zip(
                jax.tree.leaves(reference["parameters"]),
                jax.tree.leaves(eqx.filter(flow, eqx.is_inexact_array)),
                jax.tree.leaves(reference["gradients"]),
                strict=True,
            )
        )
        if not dot < 0:
            raise AssertionError(f"{label} update is not a descent direction: {dot}")
        item["update_dot_raw_gradient"] = dot
        comparisons[label] = item

    # Eight-step instrumented/public equivalence gate.
    flow = new_flow(P.ORACLE_FLOW_SEED + 1)
    instrumented_flow, history, _, _, update_count = instrumented_klx_scan(
        target_pool,
        source_pool,
        source,
        target,
        flow,
        n_batch=32,
        steps=8,
        lr=1e-3,
        coeff_lambda=0.0,
        g_clip=100.0,
        seed=P.ORACLE_TRAINER_SEED + 1,
    )
    public_flow, public_ess, public_kept, public_updated = train_molecular_forward_KLX_G(
        target_pool,
        source_pool,
        source,
        target,
        flow,
        n_batch=32,
        steps=8,
        lr=1e-3,
        coeff_lambda=0.0,
        g_clip=100.0,
        seed=P.ORACLE_TRAINER_SEED + 1,
    )
    jax.block_until_ready((instrumented_flow, public_flow))
    scan_comparison = {
        "flow": _assert_tree(
            "eight-step flow", instrumented_flow, public_flow, atol=atol, rtol=rtol
        ),
        "ess": _assert_array(
            "eight-step ESS", history["ess"], public_ess, atol=atol, rtol=rtol
        ),
        "kept": _assert_array(
            "eight-step kept", history["kept_fraction"], public_kept, atol=atol, rtol=rtol
        ),
        "update_count": int(update_count),
    }
    if not np.array_equal(np.asarray(history["update_applied"]), np.asarray(public_updated)):
        raise AssertionError("eight-step update histories differ")

    # Masking gates for a finite screened row and a finite-coordinate +inf row.
    masking = {}
    for label, mask_target, sentinel, e_clip in (
        ("finite_eclip", target, 6.0, 5.0),
        ("nonfinite_energy", Nonfinite_Row_Target(target), 12.0, float("inf")),
    ):
        pool = target.samples(P.ORACLE_MASK_SEED, 16, dtype=numpy_dtype)
        pool = pool.at[0, 0].set(sentinel)
        flow = new_flow(P.ORACLE_FLOW_SEED + 2)
        reference = reference_klx_one_step(
            pool,
            source_pool[:16],
            source,
            mask_target,
            flow,
            n_batch=16,
            lr=1e-3,
            coeff_lambda=0.0,
            e_clip=e_clip,
            g_clip=100.0,
            seed=P.ORACLE_MASK_SEED,
        )
        instrumented = instrumented_klx_one_step(
            pool,
            source_pool[:16],
            source,
            mask_target,
            flow,
            n_batch=16,
            lr=1e-3,
            coeff_lambda=0.0,
            e_clip=e_clip,
            g_clip=100.0,
            seed=P.ORACLE_MASK_SEED,
        )
        energy = jax.lax.stop_gradient(mask_target(reference["target_batch"]))
        keep = reference["keep"]
        original_loss, original_grad = _masked_gradient(
            flow, source, reference["target_batch"], energy, keep, sanitize=False
        )
        sanitized_loss, sanitized_grad = _masked_gradient(
            flow, source, reference["target_batch"], energy, keep, sanitize=True
        )
        valid_loss, valid_grad = _valid_only_gradient(
            flow, source, reference["target_batch"], energy, keep
        )
        if int(jnp.sum(~keep)) != 1:
            raise AssertionError(f"{label} did not reject exactly one row")
        masking[label] = {
            "reference_to_instrumented_gradient": _assert_tree(
                f"{label} instrumented gradient", instrumented["gradients"], reference["gradients"], atol=atol, rtol=rtol
            ),
            "original_to_sanitized_gradient": _assert_tree(
                f"{label} sanitized gradient", original_grad, sanitized_grad, atol=atol, rtol=rtol
            ),
            "original_to_valid_only_gradient": _assert_tree(
                f"{label} valid gradient", original_grad, valid_grad, atol=atol, rtol=rtol
            ),
            "original_to_sanitized_loss": _assert_array(
                f"{label} sanitized loss", original_loss, sanitized_loss, atol=atol, rtol=rtol
            ),
            "original_to_valid_only_loss": _assert_array(
                f"{label} valid loss", original_loss, valid_loss, atol=atol, rtol=rtol
            ),
            "kept_fraction": float(jnp.mean(keep)),
        }

    summary = {
        "schema_version": 1,
        "dtype": dtype_name,
        "atol": atol,
        "rtol": rtol,
        "comparisons": comparisons,
        "eight_step_equivalence": scan_comparison,
        "masking": masking,
        "full_gate_run": bool(full),
        "source_hashes": {
            name: _sha256(ROOT / name)
            for name in (
                "oracle.py",
                "oracle_model.py",
                "oracle_reference.py",
                "instrumented_train.py",
                "parameters.py",
            )
        },
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
    }

    if full:
        target_train = target.samples(
            P.ORACLE_TARGET_SEED, P.ORACLE_N_POOL, dtype=numpy_dtype
        )
        source_train = source.samples(
            jax.random.key(P.ORACLE_SOURCE_SEED), P.ORACLE_N_POOL
        )
        audit = source.samples(jax.random.key(P.ORACLE_AUDIT_SEED), P.ORACLE_N_AUDIT)
        initial = new_flow(P.ORACLE_FLOW_SEED)
        trained, full_history, _, _, updates = instrumented_klx_scan(
            target_train,
            source_train,
            source,
            target,
            initial,
            n_batch=P.ORACLE_N_BATCH,
            steps=P.ORACLE_STEPS,
            lr=1e-3,
            coeff_lambda=0.0,
            g_clip=100.0,
            seed=P.ORACLE_TRAINER_SEED,
            lr_warmup=0,
        )
        jax.block_until_ready(trained)
        identity_log_weight = importance_weights_log(
            audit, source, target, initial, "G", chunk=16
        )
        trained_log_weight = importance_weights_log(
            audit, source, target, trained, "G", chunk=16
        )
        identity_ess = compute_ESS_log(identity_log_weight)
        trained_ess = compute_ESS_log(trained_log_weight)
        initial_loss = jnp.mean(forward_KL_G(target_train, source, initial))
        final_loss = jnp.mean(forward_KL_G(target_train, source, trained))
        jax.block_until_ready((identity_ess, trained_ess, initial_loss, final_loss))
        if not bool(jnp.all(full_history["update_applied"])) or int(updates) != P.ORACLE_STEPS:
            raise AssertionError("not every full-oracle Adam step committed")
        if not all(
            bool(jnp.all(jnp.isfinite(full_history[name])))
            for name in ("loss", "ess", "kept_fraction", "parameter_delta_max", "raw_gradient_norm")
        ):
            raise AssertionError("full-oracle history contains nonfinite values")
        ess_delta = float(trained_ess - identity_ess)
        if ess_delta < P.ORACLE_ESS_DELTA_MIN:
            raise AssertionError(f"full-oracle ESS improvement {ess_delta} is below gate")
        if not float(final_loss) < float(initial_loss):
            raise AssertionError("full-oracle frozen forward-KL objective did not fall")
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "oracle_arrays.npz",
            loss=np.asarray(full_history["loss"]),
            ess=np.asarray(full_history["ess"]),
            kept_fraction=np.asarray(full_history["kept_fraction"]),
            update_applied=np.asarray(full_history["update_applied"]),
            parameter_delta_max=np.asarray(full_history["parameter_delta_max"]),
            raw_gradient_finite=np.asarray(full_history["raw_gradient_finite"]),
            raw_gradient_norm=np.asarray(full_history["raw_gradient_norm"]),
            identity_log_weight=np.asarray(identity_log_weight),
            trained_log_weight=np.asarray(trained_log_weight),
        )
        eqx.tree_serialise_leaves(output / "oracle_flow.eqx", trained)
        summary["full_gate"] = {
            "steps": P.ORACLE_STEPS,
            "updates": int(updates),
            "identity_ess": float(identity_ess),
            "trained_ess": float(trained_ess),
            "ess_delta": ess_delta,
            "initial_forward_kl": float(initial_loss),
            "final_forward_kl": float(final_loss),
            "arrays_sha256": _sha256(output / "oracle_arrays.npz"),
            "flow_sha256": _sha256(output / "oracle_flow.eqx"),
        }

    _json_write(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    summary = run(args.dtype, args.output, args.full)
    print(
        f"PASS oracle dtype={args.dtype} full={args.full} "
        f"objectives={list(summary['comparisons'])}"
    )


if __name__ == "__main__":
    main()

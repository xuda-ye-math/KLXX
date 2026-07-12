#!/usr/bin/env python
"""Promote a scientifically validated candidate into the private registry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import parameters as P

from jflows_md import Molecular_Bundle
from jflows_md.system import sha256_file


ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = (ROOT / "candidates").resolve()
BUNDLE_ROOT = ROOT / "bundles"
REGISTRY_PATH = BUNDLE_ROOT / "registry.json"
EXPECTED_REVIEW_SHA256 = "2f56972639bcd5e9fb1df6561ed226a67f8ab2f2f2aabd28c523c5777e0aa53b"
EXPECTED_JFLOWS_HEAD = "0302829fe440b6241172b652ff914db1ebecc273"
EXPECTED_JFLOWS_MD_HEAD = "5896ba275952333267592e7cf7cb3c0313d5b8f8"
EXPECTED_JFLOWS_MD_SOURCE = "07315d370486c3574aac3258a306bc6369e5843c14d72229e029a6360bb2b323"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))

    def require_finite(item, label: str) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"nonfinite JSON value at {label}: {item}")
        if isinstance(item, dict):
            for key, child in item.items():
                require_finite(child, f"{label}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                require_finite(child, f"{label}[{index}]")

    require_finite(value, str(path))
    return value


def _nonnegative(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value}")
    return result


def _require_clean_runtime(runtime: dict, label: str) -> None:
    expected = (
        ("jflows", EXPECTED_JFLOWS_HEAD, None),
        ("jflows_md", EXPECTED_JFLOWS_MD_HEAD, EXPECTED_JFLOWS_MD_SOURCE),
    )
    for name, head, source in expected:
        state = runtime.get(name, {})
        if state.get("dirty") is not False:
            raise ValueError(f"{label} {name} runtime is dirty")
        if state.get("dirty_diff_sha256") != EMPTY_SHA256:
            raise ValueError(f"{label} {name} dirty-diff digest is not empty")
        if state.get("head") != head:
            raise ValueError(f"{label} {name} revision mismatch")
        if source is not None and state.get("package_source_sha256") != source:
            raise ValueError(f"{label} {name} source digest mismatch")
    if runtime.get("jax_backend") != "gpu":
        raise ValueError(f"{label} validation did not use the configured GPU backend")


def _load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        return {"schema_version": 1, "bundles": {}}
    value = _load_json(REGISTRY_PATH)
    if value.get("schema_version") != 1 or not isinstance(value.get("bundles"), dict):
        raise ValueError("invalid private molecular bundle registry")
    return value


def _write_registry(value: dict) -> None:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY_PATH.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REGISTRY_PATH)


def _validate_evidence(
    candidate: Path,
    manifest_sha256: str,
    x64_path: Path,
    float32_path: Path,
    review_path: Path,
) -> tuple[dict, dict]:
    x64, float32 = _load_json(x64_path), _load_json(float32_path)
    for value, mode in ((x64, "x64"), (float32, "float32")):
        if value.get("mode") != mode:
            raise ValueError(f"validation mode mismatch in {mode} evidence")
        if value.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"validation manifest mismatch in {mode} evidence")
        if Path(value.get("bundle", "")).resolve() != candidate:
            raise ValueError(f"validation candidate mismatch in {mode} evidence")
        runtime = value.get("runtime", {})
        if runtime.get("validator_sha256") != sha256_file(ROOT / "validate_bundle.py"):
            raise ValueError(f"{mode} evidence was not produced by the live validator")
        if runtime.get("parameters_sha256") != sha256_file(ROOT / "parameters.py"):
            raise ValueError(f"{mode} evidence used different frozen parameters")
        _require_clean_runtime(runtime, mode)

    runtime_fields = (
        "validator_sha256",
        "parameters_sha256",
        "python",
        "numpy",
        "jax",
        "equinox",
        "openmm",
        "parmed",
        "jax_backend",
        "jflows",
        "jflows_md",
        "pythonpath",
    )
    if any(x64["runtime"].get(field) != float32["runtime"].get(field) for field in runtime_fields):
        raise ValueError("x64 and float32 evidence used different runtime/source states")

    if x64.get("formula") != "CH4" or x64.get("dimension") != 9:
        raise ValueError("x64 evidence is not the frozen methane gate")
    if x64.get("permutation_count") != 24:
        raise ValueError("all 24 methane H permutations were not checked")
    if _nonnegative(x64["energy_error_kj_mol"], "x64 energy error") > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise ValueError("x64 energy parity failed")
    if _nonnegative(x64["force_rmse_kj_mol_nm"], "x64 force RMSE") > P.FORCE_RMSE_KJ_MOL_NM_MAX:
        raise ValueError("x64 force RMSE failed")
    if _nonnegative(x64["force_max_kj_mol_nm"], "x64 force max") > P.FORCE_COMPONENT_KJ_MOL_NM_MAX:
        raise ValueError("x64 maximum force error failed")
    term_errors = [
        _nonnegative(value, f"term error {name}")
        for name, value in x64["term_energy_errors_kj_mol"].items()
    ]
    if max(term_errors) > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise ValueError("x64 per-term energy parity failed")
    covariance = x64.get("off_reference_permutation_covariance_errors")
    if not isinstance(covariance, dict):
        raise ValueError("off-reference permutation covariance evidence is missing")
    if _nonnegative(covariance["physical_energy_kj_mol"], "covariant energy error") > P.ENERGY_ERROR_KJ_MOL_MAX:
        raise ValueError("off-reference physical-energy covariance failed")
    covariance_reduced = max(
        _nonnegative(covariance["jacobian_log_volume"], "Jacobian covariance error"),
        _nonnegative(covariance["reduced_target"], "target covariance error"),
    )
    if covariance_reduced > P.REDUCED_PERMUTATION_ERROR_MAX:
        raise ValueError("off-reference chart change-of-variables gate failed")
    asymmetry = x64.get("source_permutation_asymmetry")
    if not isinstance(asymmetry, dict):
        raise ValueError("declared BAT-source asymmetry was not measured")
    for name in (
        "mean_permutation_range",
        "maximum_permutation_range",
        "fraction_range_above_1e-6",
    ):
        _nonnegative(asymmetry[name], f"source asymmetry {name}")
    parity = float(x64["positive_parity_fraction"])
    if not math.isfinite(parity) or not 0.40 <= parity <= 0.60:
        raise ValueError("both methane parity sectors were not covered")
    if not float32.get("finite"):
        raise ValueError("float32 energy/gradient gate failed")
    if {
        float32.get("sample_dtype"),
        float32.get("target_dtype"),
        float32.get("gradient_dtype"),
    } != {"float32"}:
        raise ValueError("float32 gate used a non-float32 array")

    if sha256_file(review_path) != EXPECTED_REVIEW_SHA256:
        raise ValueError("independent review digest is not the pinned audit")
    review = review_path.read_text(encoding="utf-8")
    if (
        "Reviewer: `/root/alkane_builder_code_audit`" not in review
        or "Verdict: **PASS**" not in review
        or manifest_sha256 not in review
    ):
        raise ValueError("independent review does not PASS this exact candidate manifest")
    return x64, float32


def promote(
    candidate: Path,
    x64_path: Path,
    float32_path: Path,
    review_path: Path,
) -> Path:
    candidate = candidate.expanduser().resolve()
    if not candidate.is_relative_to(CANDIDATE_ROOT):
        raise ValueError("only an artifact under the private candidates root may be promoted")
    bundle = Molecular_Bundle.load(candidate, verify=True)
    manifest_sha = sha256_file(candidate / "manifest.json")
    x64, float32 = _validate_evidence(
        candidate,
        manifest_sha,
        x64_path.resolve(),
        float32_path.resolve(),
        review_path.resolve(),
    )
    destination = BUNDLE_ROOT / candidate.name
    if destination.exists():
        existing = Molecular_Bundle.load(destination, verify=True)
        if sha256_file(existing.path / "manifest.json") != manifest_sha:
            raise FileExistsError(f"promoted path contains different content: {destination}")
    else:
        BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(candidate, destination)
        Molecular_Bundle.load(destination, verify=True)

    registry = _load_registry()
    entry = {
        "path": destination.name,
        "manifest_sha256": manifest_sha,
        "semantic_name": bundle.name,
        "molecule": bundle.manifest["target"],
        "formula": bundle.manifest["formula"],
        "dimension": bundle.dimension,
        "model": bundle.manifest["model"],
        "evidence": {
            "validation_x64_sha256": sha256_file(x64_path),
            "validation_float32_sha256": sha256_file(float32_path),
            "independent_review_sha256": sha256_file(review_path),
            "validator_sha256": x64["runtime"]["validator_sha256"],
            "runtime_jflows_md": x64["runtime"]["jflows_md"],
        },
    }
    name = bundle.manifest["target"]
    previous = registry["bundles"].get(name)
    if previous is not None and previous != entry:
        raise ValueError(f"private registry already pins a different {name} bundle")
    registry["bundles"][name] = entry
    _write_registry(registry)
    return Molecular_Bundle.load(destination, verify=True).path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validation-x64", type=Path, required=True)
    parser.add_argument("--validation-float32", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    path = promote(
        args.candidate,
        args.validation_x64,
        args.validation_float32,
        args.review,
    )
    print(f"bundle promoted {path}")


if __name__ == "__main__":
    main()

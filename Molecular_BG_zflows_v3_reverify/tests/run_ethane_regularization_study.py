#!/usr/bin/env python
"""Matched small-system evidence for c, c/r, and historical e/r.

This runner deliberately uses 18D ethane rather than glycerol or ADP.  It has
real nonbonded 1-4 pairs, but is small enough that deterministic stress tests
and short sampling pilots are inexpensive.  Runtime evidence is written below
``.aris/experiments/regularization_comparison`` and is never overwritten.

Commands
--------
``prepare``
    Generate one fixed-seed vacuum trajectory and frozen source populations.
``gate-a``
    Evaluate matched energy/gradient/collision diagnostics without training.
``gate-b-sample --arm NAME``
    Run guarded mixed-domain MALA-SMC for one fixed/scheduled surrogate arm.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import mdtraj as md
import numpy as np
import torch
from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
from openmm import app

from zflows_md.boltzmann import build
from zflows_md.forcefield import c_regularize_energy, softcap_energy
from zflows_md.potential import linear_combination


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "Molecular_BG_zflows"
ETHANE = ROOT / "Molecular_BG" / "ethane_18d_c50"
EVIDENCE = ROOT / ".aris" / "experiments" / "regularization_comparison"
INPUTS = EVIDENCE / "inputs" / "ethane_vacuum_v1"
RUNS = EVIDENCE / "runs"

OPENMM_SEED = 20260714
TORCH_SEED = 271828
MD_BURNIN_STEPS = 10000
MD_FRAMES = 2000
MD_STRIDE = 10
SOURCE_PER_SPLIT = 20000
GRADIENT_SUBSET = 4096
SAMPLE_SEED = 161803

SAMPLE_ARMS = {
    "c50_raw_r": {
        "energy": "c",
        "initial_floor": 0.0,
        "e_cap": None,
        "e_schedule": None,
        "r_schedule": None,
    },
    "c50_fixed_r010": {
        "energy": "c",
        "initial_floor": 0.10,
        "e_cap": None,
        "e_schedule": None,
        "r_schedule": None,
    },
    "c50_hist_r020_to_r010": {
        "energy": "c",
        "initial_floor": 0.20,
        "e_cap": None,
        "e_schedule": None,
        "r_schedule": {"start": 0.20, "end": 0.10},
    },
    "e_hist_r020_to_r010": {
        "energy": "e",
        "initial_floor": 0.20,
        "e_cap": 100.0,
        "e_schedule": {"start": 100.0, "end": 200.0},
        "r_schedule": {"start": 0.20, "end": 0.10},
    },
}

SAMPLE_PARAMETERS = {
    "particles": SOURCE_PER_SPLIT,
    "conditional_ess_min": 0.70,
    "initial_bridge_step": 0.10,
    "initial_sharpen_step": 0.125,
    "minimum_parameter_step": 1e-4,
    "step_growth": 1.5,
    "max_bridge_levels": 50,
    "max_sharpen_levels": 30,
    "mala_step": 5e-4,
    "mala_steps": 30,
    "mala_chunks": 4,
    "wrapped_image_radius": 2,
    "correction_ess_min": 0.10,
    "timeout_seconds": 1800,
    "schedule": "arithmetic",
}

ARM_DESIGN = {
    "c50_raw_r": {"energy": "c50_rho0", "r_floor_nm": 0.0},
    "c50_fixed_r010": {"energy": "c50_rho0", "r_floor_nm": 0.10},
    "c50_start_r020": {"energy": "c50_rho0", "r_floor_nm": 0.20},
    "e_matched_raw_r": {"energy": "softcap_Eref_plus_50", "r_floor_nm": 0.0},
    "e_matched_fixed_r010": {
        "energy": "softcap_Eref_plus_50",
        "r_floor_nm": 0.10,
    },
    "e_matched_start_r020": {
        "energy": "softcap_Eref_plus_50",
        "r_floor_nm": 0.20,
    },
    "e_hist_start_e100_r020": {"energy": "softcap_100", "r_floor_nm": 0.20},
    "e_hist_final_e200_r010": {"energy": "softcap_200", "r_floor_nm": 0.10},
}


def _scientific_code_hashes() -> dict[str, str]:
    files = (
        Path(__file__),
        PACKAGE_ROOT / "zflows_md" / "boltzmann.py",
        PACKAGE_ROOT / "zflows_md" / "potential.py",
        PACKAGE_ROOT / "zflows_md" / "forcefield.py",
        PACKAGE_ROOT / "zflows_md" / "coords.py",
        PACKAGE_ROOT / "zflows_md" / "utils.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _vacuum_system():
    source = ETHANE / "bundle" / "system.xml"
    system = XmlSerializer.deserialize(source.read_text(encoding="utf-8"))
    for index in reversed(range(system.getNumForces())):
        if type(system.getForce(index)).__name__ == "CustomGBForce":
            system.removeForce(index)
    names = [type(system.getForce(i)).__name__ for i in range(system.getNumForces())]
    expected = {
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
    }
    if set(names) != expected:
        raise RuntimeError(f"unexpected vacuum force set: {names}")
    return system


def _bonds(spec_path: Path | None = None) -> list[tuple[int, int]]:
    if spec_path is None:
        frozen = INPUTS / "system.json"
        spec_path = frozen if frozen.exists() else ETHANE / "bundle" / "system.json"
    spec = json.loads(
        spec_path.read_text(encoding="utf-8")
    )
    return [tuple(map(int, pair)) for pair in spec["bonds"]]


def _git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "diff_sha256": hashlib.sha256(
            run("git", "diff", "--binary").encode("utf-8")
        ).hexdigest(),
        "status": run("git", "status", "--short"),
    }


def prepare() -> None:
    if INPUTS.exists():
        raise FileExistsError(
            f"frozen inputs already exist at {INPUTS}; refusing to overwrite"
        )
    partial = INPUTS.with_name(INPUTS.name + f".partial-{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        system = _vacuum_system()
        system_xml = partial / "system_vacuum.xml"
        system_xml.write_text(XmlSerializer.serialize(system), encoding="utf-8")
        shutil.copy2(ETHANE / "bundle" / "reference.pdb", partial / "reference.pdb")
        shutil.copy2(ETHANE / "bundle" / "system.json", partial / "system.json")

        pdb = app.PDBFile(str(partial / "reference.pdb"))
        integrator = LangevinMiddleIntegrator(
            np.float32(300.0) * unit.kelvin,
            np.float32(1.0) / unit.picosecond,
            np.float32(0.5) * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(OPENMM_SEED)
        try:
            platform = Platform.getPlatformByName("CUDA")
            properties = {"Precision": "single"}
        except Exception:
            platform = Platform.getPlatformByName("CPU")
            properties = {}
        simulation = app.Simulation(
            pdb.topology, system, integrator, platform, properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(maxIterations=1000)
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        reference = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float32,
        )
        reference_energy = float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        simulation.context.setVelocitiesToTemperature(
            np.float32(300.0) * unit.kelvin, OPENMM_SEED + 1
        )
        simulation.step(MD_BURNIN_STEPS)
        dcd = partial / "vacuum_md.dcd"
        simulation.reporters.append(
            app.DCDReporter(str(dcd), MD_STRIDE, enforcePeriodicBox=False)
        )
        simulation.step(MD_FRAMES * MD_STRIDE)
        del simulation
        del integrator

        trajectory = md.load_dcd(str(dcd), top=str(partial / "reference.pdb"))
        frames = np.asarray(trajectory.xyz, dtype=np.float32)
        if frames.shape != (MD_FRAMES, 8, 3):
            raise RuntimeError(f"unexpected MD shape {frames.shape}")

        problem = build(
            system=system,
            bonds=_bonds(partial / "system.json"),
            md_frames=frames,
            reference_positions=reference,
            regularization="c",
            distance_floor=0.0,
            c=50.0,
            c_scale=50.0,
            c_tail_fraction=0.0,
            environment="vacuum",
            dtype=torch.float32,
            device="cpu",
        )
        torch.manual_seed(TORCH_SEED)
        rng_before_source = torch.random.get_rng_state().cpu().numpy()
        source = problem["u0"].samples(4 * SOURCE_PER_SPLIT).cpu().numpy()
        rng_after_source = torch.random.get_rng_state().cpu().numpy()
        if source.dtype != np.float32:
            raise TypeError(f"source dtype is {source.dtype}, expected float32")
        np.savez_compressed(
            partial / "frozen_inputs.npz",
            reference_positions_nm=reference,
            whitening_frames_nm=frames,
            source_gate=source[:SOURCE_PER_SPLIT],
            source_select=source[SOURCE_PER_SPLIT : 2 * SOURCE_PER_SPLIT],
            source_train=source[2 * SOURCE_PER_SPLIT : 3 * SOURCE_PER_SPLIT],
            source_audit=source[3 * SOURCE_PER_SPLIT :],
            torch_rng_state_before_source=rng_before_source,
            torch_rng_state_after_source=rng_after_source,
        )
        manifest = {
            "schema_version": 1,
            "system": "ethane GAFF2/AM1-BCC vacuum; OBC force removed",
            "dimension": 18,
            "temperature_kelvin": 300.0,
            "openmm_seed": OPENMM_SEED,
            "torch_seed": TORCH_SEED,
            "md_burnin_steps": MD_BURNIN_STEPS,
            "md_frames": MD_FRAMES,
            "md_stride": MD_STRIDE,
            "source_per_split": SOURCE_PER_SPLIT,
            "platform": platform.getName(),
            "precision": "single",
            "minimized_reference_energy_kj_mol": reference_energy,
            "source_bundle_system_sha256": _sha256(
                ETHANE / "bundle" / "system.xml"
            ),
            "source_bundle_reference_sha256": _sha256(
                ETHANE / "bundle" / "reference.pdb"
            ),
            "source_bundle_system_spec_sha256": _sha256(
                ETHANE / "bundle" / "system.json"
            ),
            "preparation_script_sha256": _sha256(Path(__file__)),
            "git": _git_state(),
        }
        _json(partial / "manifest.json", manifest)
        manifest["files"] = {
            path.name: _sha256(path)
            for path in sorted(partial.iterdir())
            if path.name != "manifest.json"
        }
        _json(partial / "manifest.json", manifest)
        partial.rename(INPUTS)
        print(f"prepared immutable inputs: {INPUTS}", flush=True)
    except Exception:
        # Preserve a failed preparation for diagnosis instead of silently
        # overwriting it on the next attempt.
        failed = partial.with_name(partial.name.replace(".partial-", ".failed-"))
        if partial.exists():
            partial.rename(failed)
        raise


def _problem(
    system,
    frames: np.ndarray,
    reference: np.ndarray,
    *,
    energy: str,
    floor: float,
    e_cap: float | None = None,
    device: str,
):
    common = dict(
        system=system,
        bonds=_bonds(),
        md_frames=frames,
        reference_positions=reference,
        distance_floor=floor,
        environment="vacuum",
        dtype=torch.float32,
        device=device,
    )
    if energy == "c":
        return build(
            regularization="c",
            c=50.0,
            c_scale=50.0,
            c_tail_fraction=0.0,
            **common,
        )
    if energy == "e":
        if e_cap is None:
            raise ValueError("e arm requires e_cap")
        return build(
            regularization="er",
            e_cap=e_cap,
            e_cap_scale=50.0,
            **common,
        )
    raise ValueError(f"unknown energy treatment {energy}")


def _raw_reduced(target, xi: torch.Tensor) -> torch.Tensor:
    z = target.unwhiten(xi)
    cartesian, logdet = target.ic.to_cartesian(z)
    return target.beta * target.ff(cartesian) - logdet - target.logdet_white


def _evaluate(target, xi_np: np.ndarray, *, raw: bool = False) -> tuple[np.ndarray, np.ndarray]:
    values = []
    gradients = np.full((min(GRADIENT_SUBSET, len(xi_np)), xi_np.shape[1]), np.nan, dtype=np.float32)
    for start in range(0, len(xi_np), 2048):
        xb = torch.as_tensor(xi_np[start : start + 2048], device=target.mu_b.device)
        with torch.no_grad():
            vb = _raw_reduced(target, xb) if raw else target(xb)
        values.append(vb.detach().cpu().numpy().astype(np.float32, copy=False))
    grad_x = torch.as_tensor(
        xi_np[: len(gradients)], device=target.mu_b.device
    )
    for start in range(0, len(grad_x), 512):
        xb = grad_x[start : start + 512].detach().clone().requires_grad_(True)
        vb = _raw_reduced(target, xb) if raw else target(xb)
        finite = torch.isfinite(vb)
        if finite.any():
            good = xb[finite].detach().clone().requires_grad_(True)
            vg = _raw_reduced(target, good) if raw else target(good)
            gg = torch.autograd.grad(vg.sum(), good)[0]
            slot = np.flatnonzero(finite.detach().cpu().numpy()) + start
            gradients[slot] = gg.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.concatenate(values), np.linalg.norm(gradients, axis=1).astype(np.float32)


def _cartesian_diagnostics(target, xi_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-map energy, post-map energy, and minimum active pair distance."""

    raw_energy, mapped_energy, minimum_distance = [], [], []
    ff = target.ff
    active_pairs = []
    for prefix in ("npair", "epair"):
        idx = getattr(ff, prefix)
        qq = getattr(ff, prefix + "_qq")
        eps = getattr(ff, prefix + "_eps")
        active = (qq.abs() + eps.abs()) > 0
        if active.any():
            active_pairs.append(idx[active])
    pair_idx = torch.cat(active_pairs, dim=0)
    for start in range(0, len(xi_np), 2048):
        xb = torch.as_tensor(xi_np[start : start + 2048], device=target.mu_b.device)
        with torch.no_grad():
            z = target.unwhiten(xb)
            cartesian, _ = target.ic.to_cartesian(z)
            raw = ff(cartesian)
            if hasattr(target, "regularized_cartesian_energy"):
                mapped = target.regularized_cartesian_energy(cartesian)
            else:
                mapped = softcap_energy(raw, target.e_cap, target.e_cap_scale)
            delta = cartesian[:, pair_idx[:, 0]] - cartesian[:, pair_idx[:, 1]]
            min_pair = delta.norm(dim=-1).amin(dim=1)
        raw_energy.append(raw.cpu().numpy().astype(np.float32, copy=False))
        mapped_energy.append(mapped.cpu().numpy().astype(np.float32, copy=False))
        minimum_distance.append(min_pair.cpu().numpy().astype(np.float32, copy=False))
    return (
        np.concatenate(raw_energy),
        np.concatenate(mapped_energy),
        np.concatenate(minimum_distance),
    )


def _minimum_active_pair(ff, positions_np: np.ndarray) -> np.ndarray:
    active_pairs = []
    for prefix in ("npair", "epair"):
        idx = getattr(ff, prefix)
        qq = getattr(ff, prefix + "_qq")
        eps = getattr(ff, prefix + "_eps")
        active = (qq.abs() + eps.abs()) > 0
        if active.any():
            active_pairs.append(idx[active])
    pair_idx = torch.cat(active_pairs, dim=0)
    output = []
    for start in range(0, len(positions_np), 2048):
        positions = torch.as_tensor(
            positions_np[start : start + 2048], device=ff.r_floor.device
        )
        with torch.no_grad():
            delta = positions[:, pair_idx[:, 0]] - positions[:, pair_idx[:, 1]]
            output.append(delta.norm(dim=-1).amin(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def _summary(values: np.ndarray, gradients: np.ndarray) -> dict:
    finite = np.isfinite(values)
    grad_finite = np.isfinite(gradients)

    def quantiles(array: np.ndarray) -> dict:
        array = array[np.isfinite(array)].astype(np.float64)
        if len(array) == 0:
            return {}
        return {
            name: float(value)
            for name, value in zip(
                ("min", "p50", "p90", "p99", "max"),
                np.quantile(array, (0.0, 0.5, 0.9, 0.99, 1.0)),
            )
        }

    return {
        "count": int(len(values)),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(values).sum()),
        "posinf": int(np.isposinf(values).sum()),
        "neginf": int(np.isneginf(values).sum()),
        "value_quantiles": quantiles(values),
        "gradient_count": int(len(gradients)),
        "gradient_finite": int(grad_finite.sum()),
        "gradient_norm_quantiles": quantiles(gradients),
    }


def _normalized_weight_metrics(logw: np.ndarray) -> dict:
    finite_or_neginf = np.isfinite(logw) | np.isneginf(logw)
    if not finite_or_neginf.all() or not np.isfinite(logw).any():
        return {"valid": False}
    shifted = logw.astype(np.float64) - np.max(logw[np.isfinite(logw)])
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return {"valid": False}
    weights /= total
    ordered = np.sort(weights)[::-1]
    return {
        "valid": True,
        "normalized_ess": float(1.0 / (len(weights) * np.square(weights).sum())),
        "max_weight": float(ordered[0]),
        "top1_mass": float(ordered[:1].sum()),
        "top10_mass": float(ordered[:10].sum()),
        "top100_mass": float(ordered[:100].sum()),
        "zero_weight_fraction": float(np.mean(weights == 0.0)),
    }


def _eval_reduced_chunks(target, samples: torch.Tensor, *, raw: bool = False) -> torch.Tensor:
    pieces = []
    for chunk in torch.tensor_split(samples, 4, dim=0):
        with torch.no_grad():
            value = _raw_reduced(target, chunk) if raw else target.eval(chunk)
        pieces.append(value.detach().clone())
    return torch.cat(pieces)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _verify_frozen_inputs(manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        path = INPUTS / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen input is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen input hash mismatch for {name}: {actual} != {expected}"
            )


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _normalized_weights_tensor(
    logw: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Guard and normalize log weights in float64 before resampling."""

    host = logw.detach().cpu().numpy().astype(np.float64, copy=False)
    metrics = _normalized_weight_metrics(host)
    if not metrics.get("valid", False):
        raise FloatingPointError("invalid importance-weight mass")
    if torch.isposinf(logw).any() or torch.isnan(logw).any():
        raise FloatingPointError("NaN/+inf log weight cannot be resampled")
    finite = torch.isfinite(logw)
    maximum = logw[finite].max().double()
    weights = torch.where(
        finite,
        torch.exp(logw.double() - maximum),
        torch.zeros_like(logw, dtype=torch.float64),
    )
    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        raise FloatingPointError("nonpositive/nonfinite resampling mass")
    return weights / total, metrics


def _resample_with_ancestry(
    samples: torch.Tensor, logw: torch.Tensor, seed: int
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict]:
    weights, metrics = _normalized_weights_tensor(logw)
    generator = _torch_generator(samples.device, seed)
    ancestors = torch.multinomial(
        weights, samples.shape[0], replacement=True, generator=generator
    )
    return (
        samples[ancestors],
        ancestors.cpu().numpy().astype(np.int64, copy=False),
        weights.cpu().numpy().astype(np.float64, copy=False),
        metrics,
    )


def _wrap_torsions(samples: torch.Tensor, tor_start: int) -> torch.Tensor:
    torsions = torch.remainder(
        samples[:, tor_start:] + math.pi, 2.0 * math.pi
    ) - math.pi
    return torch.cat((samples[:, :tor_start], torsions), dim=-1)


def _wrapped_log_proposal(
    destination: torch.Tensor,
    mean: torch.Tensor,
    *,
    tor_start: int,
    step: float,
    image_radius: int,
) -> torch.Tensor:
    """Unnormalized mixed Euclidean/wrapped-Gaussian proposal log density."""

    euclidean = -(
        (destination[:, :tor_start] - mean[:, :tor_start]) ** 2
    ).sum(-1) / (4.0 * step)
    images = (
        torch.arange(
            -image_radius,
            image_radius + 1,
            device=destination.device,
            dtype=destination.dtype,
        )
        * (2.0 * math.pi)
    )
    base = torch.remainder(
        destination[:, tor_start:] - mean[:, tor_start:] + math.pi,
        2.0 * math.pi,
    ) - math.pi
    delta = base[:, :, None] + images[None, None, :]
    periodic = torch.logsumexp(-(delta**2) / (4.0 * step), dim=-1).sum(-1)
    return euclidean + periodic


def _mala_rejuvenate(
    samples: torch.Tensor,
    potential,
    *,
    tor_start: int,
    step: float,
    steps: int,
    chunks: int,
    image_radius: int,
    seed: int,
    deadline: float,
) -> tuple[torch.Tensor, dict]:
    """Exact mixed-domain MALA with saved acceptance/invalid statistics."""

    outputs = []
    accepted_total = 0
    invalid_total = 0
    proposal_total = 0
    accepted_trace = np.zeros((chunks, steps), dtype=np.int64)
    invalid_trace = np.zeros((chunks, steps), dtype=np.int64)
    for chunk_index, x in enumerate(torch.tensor_split(samples, chunks, dim=0)):
        x = x.detach().clone()
        generator = _torch_generator(x.device, seed + chunk_index)
        for iteration in range(steps):
            if iteration % 5 == 0 and time.time() >= deadline:
                raise TimeoutError("30-minute sampling deadline reached")
            fx = potential.grad(x).detach().clone()
            fx_finite = torch.isfinite(fx).all(dim=-1)
            safe_fx = torch.where(fx_finite[:, None], fx, torch.zeros_like(fx))
            mean_xy = x - step * safe_fx
            noise = torch.randn(
                x.shape,
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
            y = _wrap_torsions(
                mean_xy + math.sqrt(2.0 * step) * noise, tor_start
            )
            fy = potential.grad(y).detach().clone()
            fy_finite = torch.isfinite(fy).all(dim=-1)
            mean_yx = y - step * torch.where(
                fy_finite[:, None], fy, torch.zeros_like(fy)
            )
            ux = potential.eval(x).detach().clone()
            uy = potential.eval(y).detach().clone()
            log_q_yx = _wrapped_log_proposal(
                y,
                mean_xy,
                tor_start=tor_start,
                step=step,
                image_radius=image_radius,
            )
            log_q_xy = _wrapped_log_proposal(
                x,
                mean_yx,
                tor_start=tor_start,
                step=step,
                image_radius=image_radius,
            )
            log_alpha = -uy + ux + log_q_xy - log_q_yx
            valid = (
                fx_finite
                & fy_finite
                & torch.isfinite(ux)
                & torch.isfinite(uy)
                & torch.isfinite(log_alpha)
            )
            uniform = torch.rand(
                log_alpha.shape,
                dtype=log_alpha.dtype,
                device=log_alpha.device,
                generator=generator,
            )
            accept = valid & (uniform.log() < torch.minimum(
                log_alpha, torch.zeros_like(log_alpha)
            ))
            x = torch.where(accept[:, None], y, x)
            accepted_total += int(accept.sum().item())
            invalid_total += int((~valid).sum().item())
            proposal_total += int(len(x))
            accepted_trace[chunk_index, iteration] = int(accept.sum().item())
            invalid_trace[chunk_index, iteration] = int((~valid).sum().item())
        outputs.append(x)
    return torch.cat(outputs), {
        "accepted": accepted_total,
        "invalid": invalid_total,
        "proposals": proposal_total,
        "acceptance_rate": accepted_total / proposal_total,
        "invalid_fraction": invalid_total / proposal_total,
        "accepted_per_chunk_step": accepted_trace.tolist(),
        "invalid_per_chunk_step": invalid_trace.tolist(),
    }


def _save_transition_artifact(
    directory: Path,
    *,
    phase: str,
    level: int,
    logw: torch.Tensor,
    normalized_weights: np.ndarray,
    ancestors: np.ndarray,
) -> Path:
    path = directory / f"{phase}-{level:03d}.npz"
    np.savez_compressed(
        path,
        logw=logw.detach().cpu().numpy().astype(np.float32, copy=False),
        normalized_weights=normalized_weights,
        ancestors=ancestors,
    )
    return path


def _write_sampling_checkpoint(
    partial: Path,
    particles: torch.Tensor,
    bridge_records: list[dict],
    sharpen_records: list[dict],
) -> None:
    temporary = partial / "checkpoint.tmp.npz"
    np.savez_compressed(
        temporary,
        particles=particles.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    temporary.replace(partial / "checkpoint.npz")
    _json(
        partial / "checkpoint.json",
        {
            "bridge": bridge_records,
            "sharpen": sharpen_records,
        },
    )


def _adaptive_bridge_sample(
    initial: torch.Tensor,
    source,
    target,
    *,
    tor_start: int,
    stage_dir: Path,
    deadline: float,
    checkpoint,
) -> tuple[torch.Tensor, list[dict]]:
    """Adaptive SMC from the frozen source to one fixed surrogate target."""

    particles = initial.detach().clone()
    records: list[dict] = []
    t_value = 0.0
    parameter_step = SAMPLE_PARAMETERS["initial_bridge_step"]
    threshold = SAMPLE_PARAMETERS["conditional_ess_min"]
    while t_value < 1.0:
        if time.time() >= deadline:
            raise TimeoutError("30-minute sampling deadline reached")
        if len(records) >= SAMPLE_PARAMETERS["max_bridge_levels"]:
            raise RuntimeError("bridge exceeded max_bridge_levels")
        with torch.no_grad():
            source_value = _eval_reduced_chunks(source, particles)
            target_value = _eval_reduced_chunks(target, particles)
        contrast = source_value - target_value
        trial_step = min(parameter_step, 1.0 - t_value)
        while True:
            candidate = min(1.0, t_value + trial_step)
            logw = (candidate - t_value) * contrast
            _, metrics = _normalized_weights_tensor(logw)
            if metrics["normalized_ess"] >= threshold:
                break
            trial_step *= 0.5
            if trial_step < SAMPLE_PARAMETERS["minimum_parameter_step"]:
                raise RuntimeError(
                    f"bridge cannot meet ESS gate after t={t_value:.8f}"
                )
        level = len(records) + 1
        particles, ancestors, weights, metrics = _resample_with_ancestry(
            particles,
            logw,
            SAMPLE_SEED + 100000 + 100 * level,
        )
        bridge = linear_combination(
            [target, source], [candidate, 1.0 - candidate]
        )
        particles, mala = _mala_rejuvenate(
            particles,
            bridge,
            tor_start=tor_start,
            step=SAMPLE_PARAMETERS["mala_step"],
            steps=SAMPLE_PARAMETERS["mala_steps"],
            chunks=SAMPLE_PARAMETERS["mala_chunks"],
            image_radius=SAMPLE_PARAMETERS["wrapped_image_radius"],
            seed=SAMPLE_SEED + 100000 + 100 * level + 1,
            deadline=deadline,
        )
        artifact = _save_transition_artifact(
            stage_dir,
            phase="bridge",
            level=level,
            logw=logw,
            normalized_weights=weights,
            ancestors=ancestors,
        )
        record = {
            "level": level,
            "t_previous": t_value,
            "t": candidate,
            "delta": candidate - t_value,
            "weight_metrics": metrics,
            "mala": mala,
            "artifact": artifact.name,
        }
        records.append(record)
        checkpoint(particles, records)
        if mala["invalid"] != 0:
            raise FloatingPointError(
                f"bridge MALA produced {mala['invalid']} invalid proposals"
            )
        t_value = candidate
        parameter_step = min(
            trial_step * SAMPLE_PARAMETERS["step_growth"], 1.0 - t_value
        )
    return particles, records


def _set_regularization_state(target, arm: dict, progress: float) -> dict:
    progress = float(progress)
    r_schedule = arm["r_schedule"]
    e_schedule = arm["e_schedule"]
    if r_schedule is None:
        r_floor = float(arm["initial_floor"])
    else:
        r_floor = r_schedule["start"] + progress * (
            r_schedule["end"] - r_schedule["start"]
        )
    target.set_r_floor(r_floor)
    state = {"progress": progress, "r_floor_nm": r_floor}
    if e_schedule is not None:
        e_cap = e_schedule["start"] + progress * (
            e_schedule["end"] - e_schedule["start"]
        )
        target.set_regularization(e_cap)
        state["e_cap_kj_mol"] = e_cap
    return state


def _direct_schedule_contrasts(
    target, arm: dict, particles: torch.Tensor
) -> tuple[dict, dict[str, np.ndarray]]:
    """One-jump diagnostics on particles representing the schedule start."""

    start = _set_regularization_state(target, arm, 0.0)
    old = _eval_reduced_chunks(target, particles)
    end = _set_regularization_state(target, arm, 1.0)
    final = _eval_reduced_chunks(target, particles)
    joint = (old - final).detach().cpu().numpy().astype(np.float64, copy=False)
    report = {
        "start": start,
        "end": end,
        "joint_start_to_end": _normalized_weight_metrics(joint),
    }
    arrays = {"joint_start_to_end_logw": joint}
    if arm["e_schedule"] is not None and arm["r_schedule"] is not None:
        _set_regularization_state(target, arm, 0.0)
        target.set_r_floor(arm["r_schedule"]["end"])
        distance_only = (
            old - _eval_reduced_chunks(target, particles)
        ).detach().cpu().numpy().astype(np.float64, copy=False)
        _set_regularization_state(target, arm, 0.0)
        target.set_regularization(arm["e_schedule"]["end"])
        energy_only = (
            old - _eval_reduced_chunks(target, particles)
        ).detach().cpu().numpy().astype(np.float64, copy=False)
        report["distance_only_start_to_end"] = _normalized_weight_metrics(
            distance_only
        )
        report["energy_only_start_to_end"] = _normalized_weight_metrics(
            energy_only
        )
        arrays["distance_only_start_to_end_logw"] = distance_only
        arrays["energy_only_start_to_end_logw"] = energy_only
    _set_regularization_state(target, arm, 0.0)
    return report, arrays


def _adaptive_regularization_sharpen(
    particles: torch.Tensor,
    target,
    arm: dict,
    *,
    tor_start: int,
    stage_dir: Path,
    deadline: float,
    bridge_records: list[dict],
    checkpoint,
) -> tuple[torch.Tensor, list[dict]]:
    """Arithmetic r=.20->.10 (and historical e=100->200) SMC path."""

    if arm["r_schedule"] is None and arm["e_schedule"] is None:
        return particles, []
    progress = 0.0
    parameter_step = SAMPLE_PARAMETERS["initial_sharpen_step"]
    threshold = SAMPLE_PARAMETERS["conditional_ess_min"]
    records: list[dict] = []
    _set_regularization_state(target, arm, progress)
    while progress < 1.0:
        if time.time() >= deadline:
            raise TimeoutError("30-minute sampling deadline reached")
        if len(records) >= SAMPLE_PARAMETERS["max_sharpen_levels"]:
            raise RuntimeError("regularization path exceeded max_sharpen_levels")
        old_state = _set_regularization_state(target, arm, progress)
        old_value = _eval_reduced_chunks(target, particles)
        trial_step = min(parameter_step, 1.0 - progress)
        while True:
            candidate = min(1.0, progress + trial_step)
            new_state = _set_regularization_state(target, arm, candidate)
            new_value = _eval_reduced_chunks(target, particles)
            logw = old_value - new_value
            _, metrics = _normalized_weights_tensor(logw)
            if metrics["normalized_ess"] >= threshold:
                break
            _set_regularization_state(target, arm, progress)
            trial_step *= 0.5
            if trial_step < SAMPLE_PARAMETERS["minimum_parameter_step"]:
                raise RuntimeError(
                    f"sharpening cannot meet ESS gate after s={progress:.8f}"
                )
        level = len(records) + 1
        particles, ancestors, weights, metrics = _resample_with_ancestry(
            particles,
            logw,
            SAMPLE_SEED + 200000 + 100 * level,
        )
        particles, mala = _mala_rejuvenate(
            particles,
            target,
            tor_start=tor_start,
            step=SAMPLE_PARAMETERS["mala_step"],
            steps=SAMPLE_PARAMETERS["mala_steps"],
            chunks=SAMPLE_PARAMETERS["mala_chunks"],
            image_radius=SAMPLE_PARAMETERS["wrapped_image_radius"],
            seed=SAMPLE_SEED + 200000 + 100 * level + 1,
            deadline=deadline,
        )
        artifact = _save_transition_artifact(
            stage_dir,
            phase="sharpen",
            level=level,
            logw=logw,
            normalized_weights=weights,
            ancestors=ancestors,
        )
        record = {
            "level": level,
            "progress_previous": progress,
            "progress": candidate,
            "delta": candidate - progress,
            "state_previous": old_state,
            "state": new_state,
            "weight_metrics": metrics,
            "mala": mala,
            "artifact": artifact.name,
        }
        records.append(record)
        checkpoint(particles, bridge_records, records)
        if mala["invalid"] != 0:
            raise FloatingPointError(
                f"sharpen MALA produced {mala['invalid']} invalid proposals"
            )
        progress = candidate
        parameter_step = min(
            trial_step * SAMPLE_PARAMETERS["step_growth"], 1.0 - progress
        )
    return particles, records


def _pair_probe(problem) -> dict:
    ff = problem["ff"]
    candidates = []
    for prefix in ("npair", "epair"):
        idx = getattr(ff, prefix)
        qq = getattr(ff, prefix + "_qq")
        sig = getattr(ff, prefix + "_sig")
        eps = getattr(ff, prefix + "_eps")
        for i in range(len(idx)):
            if float(abs(qq[i]) + abs(eps[i])) > 0.0:
                candidates.append((prefix, i, qq[i], sig[i], eps[i]))
    prefix, index, _, _, _ = candidates[0]
    idx = getattr(ff, prefix)[index : index + 1]
    qq = getattr(ff, prefix + "_qq")[index : index + 1]
    sig = getattr(ff, prefix + "_sig")[index : index + 1]
    eps = getattr(ff, prefix + "_eps")[index : index + 1]
    atoms = tuple(map(int, idx[0].detach().cpu()))
    original_floor = float(ff.r_floor.detach().cpu())
    result = {
        "implementation": "Amber_Force_Field._pair_nb",
        "pair_kind": prefix,
        "pair_index": int(index),
        "atoms": atoms,
        "points": [],
    }
    specifications = [(0.0, 0.0), (0.0, 1e-12), (0.0, 1e-8)]
    specifications += [
        (floor, floor + delta)
        for floor in (0.10, 0.20)
        for delta in (-1e-6, 0.0, 1e-6)
    ]
    for floor, distance in specifications:
        ff.set_r_floor(floor)
        positions = torch.zeros(
            (1, ff.M, 3), dtype=ff.r_floor.dtype, device=ff.r_floor.device
        )
        positions[0, atoms[1], 0] = distance
        positions.requires_grad_(True)
        energy = ff._pair_nb(positions, idx, qq, sig, eps)[0]
        if torch.isfinite(energy):
            gradient = torch.autograd.grad(energy, positions)[0][0, atoms[1], 0]
            energy_value = float(energy.detach().cpu())
            gradient_value = float(gradient.detach().cpu())
            energy_class = "finite"
        else:
            energy_value = None
            gradient_value = None
            energy_class = (
                "posinf"
                if torch.isposinf(energy)
                else "neginf"
                if torch.isneginf(energy)
                else "nan"
            )
        result["points"].append(
            {
                "floor_nm": floor,
                "r_nm": distance,
                "energy_class": energy_class,
                "energy_kj_mol": energy_value,
                "dE_dr": gradient_value,
            }
        )
    ff.set_r_floor(original_floor)
    return result


def gate_a() -> None:
    if not INPUTS.exists():
        raise FileNotFoundError(f"run prepare first: missing {INPUTS}")
    input_manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
    script_hash = _sha256(Path(__file__))
    code_hashes = _scientific_code_hashes()
    git_state = _git_state()
    identity_payload = {
        "input_manifest": input_manifest,
        "arm_design": ARM_DESIGN,
        "scientific_code_hashes": code_hashes,
        "git": git_state,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    run = RUNS / f"gate-a-{identity}"
    if run.exists():
        raise FileExistsError(f"immutable run already exists: {run}")
    partial = run.with_name(run.name + f".partial-{os.getpid()}")
    partial.mkdir(parents=True)
    (partial / "arms").mkdir()
    started = time.time()
    try:
        with np.load(INPUTS / "frozen_inputs.npz", allow_pickle=False) as saved:
            frames = saved["whitening_frames_nm"]
            reference = saved["reference_positions_nm"]
            xi = saved["source_gate"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        system = XmlSerializer.deserialize(
            (INPUTS / "system_vacuum.xml").read_text(encoding="utf-8")
        )

        c0 = _problem(
            system, frames, reference, energy="c", floor=0.0, device=device
        )
        e_ref = float(c0["u"].reference_energy.detach().cpu())
        matched_cap = e_ref + 50.0
        md_min_pair = _minimum_active_pair(c0["ff"], frames)
        reference_min_pair = _minimum_active_pair(c0["ff"], reference[None, ...])
        problems = {
            "c50_raw_r": c0,
            "c50_fixed_r010": _problem(
                system, frames, reference, energy="c", floor=0.10, device=device
            ),
            "c50_start_r020": _problem(
                system, frames, reference, energy="c", floor=0.20, device=device
            ),
            "e_matched_raw_r": _problem(
                system,
                frames,
                reference,
                energy="e",
                floor=0.0,
                e_cap=matched_cap,
                device=device,
            ),
            "e_matched_fixed_r010": _problem(
                system,
                frames,
                reference,
                energy="e",
                floor=0.10,
                e_cap=matched_cap,
                device=device,
            ),
            "e_matched_start_r020": _problem(
                system,
                frames,
                reference,
                energy="e",
                floor=0.20,
                e_cap=matched_cap,
                device=device,
            ),
            "e_hist_start_e100_r020": _problem(
                system,
                frames,
                reference,
                energy="e",
                floor=0.20,
                e_cap=100.0,
                device=device,
            ),
            "e_hist_final_e200_r010": _problem(
                system,
                frames,
                reference,
                energy="e",
                floor=0.10,
                e_cap=200.0,
                device=device,
            ),
        }
        base_state = c0["ff"].state_dict()
        forcefield_parity = {}
        for name, problem in problems.items():
            mismatches = []
            for key, value in problem["ff"].state_dict().items():
                if key == "r_floor":
                    continue
                if not torch.equal(value, base_state[key]):
                    mismatches.append(key)
            forcefield_parity[name] = {
                "parameter_mismatches_excluding_floor": mismatches,
                "effective_r_floor_nm": float(problem["ff"].r_floor.detach().cpu()),
            }
            if mismatches:
                raise RuntimeError(f"force-field mismatch in {name}: {mismatches}")
        arrays = {}
        arrays["md_minimum_active_pair_nm"] = md_min_pair
        arrays["reference_minimum_active_pair_nm"] = reference_min_pair
        summaries = {}
        source_value = 0.5 * np.square(
            xi[:, :13], dtype=np.float64
        ).sum(axis=1)
        for name, problem in problems.items():
            values, gradients = _evaluate(problem["u"], xi)
            cart_raw, cart_mapped, min_pair = _cartesian_diagnostics(problem["u"], xi)
            source_logw = source_value - values.astype(np.float64)
            arrays[name + "_reduced"] = values
            arrays[name + "_finite_mask"] = np.isfinite(values)
            arrays[name + "_grad_norm"] = gradients
            arrays[name + "_grad_finite_mask"] = np.isfinite(gradients)
            arrays[name + "_cartesian_pre_map_kj_mol"] = cart_raw
            arrays[name + "_cartesian_mapped_kj_mol"] = cart_mapped
            arrays[name + "_minimum_active_pair_nm"] = min_pair
            arrays[name + "_source_logw"] = source_logw
            summaries[name] = _summary(values, gradients)
            summaries[name]["cartesian_pre_map"] = _summary(
                cart_raw, np.empty(0, dtype=np.float32)
            )["value_quantiles"]
            summaries[name]["cartesian_mapped"] = _summary(
                cart_mapped, np.empty(0, dtype=np.float32)
            )["value_quantiles"]
            summaries[name]["minimum_active_pair_nm"] = _summary(
                min_pair, np.empty(0, dtype=np.float32)
            )["value_quantiles"]
            summaries[name]["floor_active_fraction"] = float(
                np.mean(min_pair < float(problem["ff"].r_floor.detach().cpu()))
            )
            np.savez_compressed(
                partial / "arms" / f"{name}.npz",
                reduced_potential=values,
                finite_mask=np.isfinite(values),
                gradient_norm=gradients,
                gradient_finite_mask=np.isfinite(gradients),
                cartesian_pre_map_kj_mol=cart_raw,
                cartesian_mapped_kj_mol=cart_mapped,
                minimum_active_pair_nm=min_pair,
                source_logw=source_logw,
            )
        raw_values, raw_gradients = _evaluate(c0["u"], xi, raw=True)
        raw_source_logw = source_value - raw_values.astype(np.float64)
        arrays["raw_physical_reduced"] = raw_values
        arrays["raw_physical_finite_mask"] = np.isfinite(raw_values)
        arrays["raw_physical_grad_norm"] = raw_gradients
        arrays["raw_physical_grad_finite_mask"] = np.isfinite(raw_gradients)
        arrays["raw_physical_source_logw"] = raw_source_logw
        summaries["raw_physical"] = _summary(raw_values, raw_gradients)

        comparisons = {}
        for label in ("raw_r", "fixed_r010", "start_r020"):
            c_value = arrays[f"c50_{label}_reduced"].astype(np.float64)
            e_value = arrays[f"e_matched_{label}_reduced"].astype(np.float64)
            delta = c_value - e_value
            finite = np.isfinite(delta)
            comparisons[f"c50_vs_e_matched_{label}"] = {
                "finite_count": int(finite.sum()),
                "different_count": int(np.count_nonzero(delta[finite] != 0.0)),
                "max_abs_reduced_difference": float(np.max(np.abs(delta[finite]))),
            }

        # Direct source-overlap metrics are diagnostics, not claims that the
        # source represents either target.  They use matched xi for every arm.
        for name in problems:
            comparisons[f"source_to_{name}"] = _normalized_weight_metrics(
                source_value - arrays[name + "_reduced"].astype(np.float64)
            )
        comparisons["source_to_raw_physical"] = _normalized_weight_metrics(
            raw_source_logw
        )

        gate_pass = {
            name: bool(
                summary["finite"] == summary["count"]
                and summary["gradient_finite"] == summary["gradient_count"]
                and comparisons[f"source_to_{name}"]["valid"]
            )
            for name, summary in summaries.items()
            if name in problems
        }
        gate_violations = {
            name: [
                reason
                for condition, reason in (
                    (
                        summaries[name]["finite"] == summaries[name]["count"],
                        "nonfinite reduced potential",
                    ),
                    (
                        summaries[name]["gradient_finite"]
                        == summaries[name]["gradient_count"],
                        "nonfinite gradient norm",
                    ),
                    (
                        comparisons[f"source_to_{name}"]["valid"],
                        "invalid source-to-target weight mass",
                    ),
                )
                if not condition
            ]
            for name in problems
        }
        raw_finite_subset = np.isfinite(raw_values[: len(raw_gradients)])
        raw_gate_checks = {
            "no_nan_or_negative_infinity": bool(
                not np.isnan(raw_values).any() and not np.isneginf(raw_values).any()
            ),
            "source_weight_mass_valid": bool(
                comparisons["source_to_raw_physical"]["valid"]
            ),
            "conditional_gradients_finite": bool(
                np.isfinite(raw_gradients[raw_finite_subset]).all()
            ),
        }
        raw_gate_violations = [
            name for name, passed in raw_gate_checks.items() if not passed
        ]

        # Pure energy-map checks independent of molecular geometry.
        probe = torch.linspace(-1000.0, 1e6, 200003, dtype=torch.float64)
        ref64 = torch.tensor(e_ref, dtype=torch.float64)
        c64 = c_regularize_energy(probe, ref64, 50.0, 50.0, 0.0)
        e64 = softcap_energy(probe, e_ref + 50.0, 50.0)
        shift = 12345.0
        c_shift = c_regularize_energy(
            probe + shift, ref64 + shift, 50.0, 50.0, 0.0
        )
        e_shift = softcap_energy(probe + shift, e_ref + 50.0, 50.0)
        algebra = {
            "float64_c_vs_matched_e_max_abs_kj_mol": float(
                (c64 - e64).abs().max()
            ),
            "c_additive_shift_covariance_max_abs_kj_mol": float(
                (c_shift - c64 - shift).abs().max()
            ),
            "fixed_e_additive_shift_covariance_max_abs_kj_mol": float(
                (e_shift - e64 - shift).abs().max()
            ),
        }
        pair_probe = _pair_probe(c0)
        probe_points = pair_probe["points"]
        below_gradients = [
            abs(point["dE_dr"])
            for point in probe_points
            if point["floor_nm"] > 0.0
            and point["r_nm"] < point["floor_nm"]
            and point["dE_dr"] is not None
        ]
        above_gradients = [
            abs(point["dE_dr"])
            for point in probe_points
            if point["floor_nm"] > 0.0
            and point["r_nm"] > point["floor_nm"]
            and point["dE_dr"] is not None
        ]
        control_checks = {
            "forcefield_parameters_match": all(
                not value["parameter_mismatches_excluding_floor"]
                for value in forcefield_parity.values()
            ),
            "c_equals_matched_e_float64": (
                algebra["float64_c_vs_matched_e_max_abs_kj_mol"] <= 1e-10
            ),
            "c_equals_matched_e_float32_within_reduced_tolerance": all(
                value["max_abs_reduced_difference"] <= 1e-2
                for key, value in comparisons.items()
                if key.startswith("c50_vs_e_matched")
            ),
            "c_additive_shift_covariant": (
                algebra["c_additive_shift_covariance_max_abs_kj_mol"] <= 1e-9
            ),
            "fixed_absolute_e_not_shift_covariant": (
                algebra["fixed_e_additive_shift_covariance_max_abs_kj_mol"] > 1.0
            ),
            "reference_floor_inactive_at_0p20": bool(reference_min_pair[0] >= 0.20),
            "md_floor_inactive_at_0p20": bool(np.all(md_min_pair >= 0.20)),
            "hard_floor_zero_gradient_below": bool(
                below_gradients and max(below_gradients) == 0.0
            ),
            "hard_floor_nonzero_gradient_above": bool(
                above_gradients and min(above_gradients) > 0.0
            ),
            "raw_exact_collision_is_positive_infinity": any(
                point["floor_nm"] == 0.0
                and point["r_nm"] == 0.0
                and point["energy_class"] == "posinf"
                for point in probe_points
            ),
        }
        control_violations = [
            name for name, passed in control_checks.items() if not passed
        ]

        np.savez_compressed(partial / "gate_a_arrays.npz", **arrays)
        report = {
            "schema_version": 1,
            "terminal_status": "complete",
            "system": "ethane 18D GAFF2/AM1-BCC vacuum",
            "device": device,
            "dtype": "float32",
            "input_manifest_sha256": _sha256(INPUTS / "manifest.json"),
            "script_sha256": script_hash,
            "scientific_code_hashes": code_hashes,
            "resolved_arm_design": ARM_DESIGN,
            "reference_energy_kj_mol": e_ref,
            "matched_absolute_cap_kj_mol": matched_cap,
            "reference_md_floor_activity": {
                "reference_minimum_active_pair_nm": float(reference_min_pair[0]),
                "md_minimum_active_pair_nm": _summary(
                    md_min_pair, np.empty(0, dtype=np.float32)
                )["value_quantiles"],
                "md_fraction_below_0p10": float(np.mean(md_min_pair < 0.10)),
                "md_fraction_below_0p20": float(np.mean(md_min_pair < 0.20)),
            },
            "arm_summaries": summaries,
            "arm_gate_pass": gate_pass,
            "arm_gate_violations": gate_violations,
            "raw_gate_checks": raw_gate_checks,
            "raw_gate_violations": raw_gate_violations,
            "scientific_gate_status": (
                "pass"
                if all(gate_pass.values())
                and not raw_gate_violations
                and not control_violations
                else "fail"
            ),
            "control_checks": control_checks,
            "control_violations": control_violations,
            "forcefield_parity": forcefield_parity,
            "comparisons": comparisons,
            "algebra": algebra,
            "hard_floor_pair_probe": pair_probe,
            "wall_seconds": time.time() - started,
            "git": git_state,
        }
        _json(partial / "report.json", report)
        _json(
            partial / "manifest.json",
            {
                "schema_version": 1,
                "terminal_status": "complete",
                "files": {
                    str(path.relative_to(partial)): _sha256(path)
                    for path in sorted(partial.rglob("*"))
                    if path.is_file() and path.name != "manifest.json"
                },
            },
        )
        partial.rename(run)
        print(f"completed immutable Gate A: {run}", flush=True)
    except Exception as exc:
        _json(
            partial / "failure.json",
            {"terminal_status": "failed", "error": repr(exc)},
        )
        failed = partial.with_name(partial.name.replace(".partial-", ".failed-"))
        partial.rename(failed)
        raise


def gate_b_sample(arm_name: str) -> None:
    """Run one immutable MALA-SMC arm and audit its raw corrections."""

    if not INPUTS.exists():
        raise FileNotFoundError(f"run prepare first: missing {INPUTS}")
    if arm_name not in SAMPLE_ARMS:
        raise ValueError(f"unknown sampling arm {arm_name}")
    arm = SAMPLE_ARMS[arm_name]
    input_manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
    code_hashes = _scientific_code_hashes()
    git_state = _git_state()
    identity_payload = {
        "input_manifest": input_manifest,
        "arm_name": arm_name,
        "arm": arm,
        "parameters": SAMPLE_PARAMETERS,
        "sample_seed": SAMPLE_SEED,
        "code_hashes": code_hashes,
        "git": git_state,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    run = RUNS / f"gate-b-sample-{arm_name}-{identity}"
    if run.exists():
        raise FileExistsError(f"immutable run already exists: {run}")
    partial = run.with_name(run.name + f".partial-{os.getpid()}")
    partial.mkdir(parents=True)
    log_path = partial / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started = time.time()
    deadline = started + SAMPLE_PARAMETERS["timeout_seconds"]
    try:
        _verify_frozen_inputs(input_manifest)
        with np.load(INPUTS / "frozen_inputs.npz", allow_pickle=False) as saved:
            frames = saved["whitening_frames_nm"]
            reference = saved["reference_positions_nm"]
            initial_np = saved["source_select"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        system = XmlSerializer.deserialize(
            (INPUTS / "system_vacuum.xml").read_text(encoding="utf-8")
        )
        problem = _problem(
            system,
            frames,
            reference,
            energy=arm["energy"],
            floor=float(arm["initial_floor"]),
            e_cap=arm["e_cap"],
            device=device,
        )
        source, target = problem["u0"], problem["u"]
        for potential in (source, target):
            potential.enable_grad(mode="default")
            potential.enable_eval(mode="default")
        initial = torch.as_tensor(initial_np, device=device)
        stage_dir = partial / "transitions"
        stage_dir.mkdir()

        log(
            f"START arm={arm_name} device={device} seed={SAMPLE_SEED} "
            f"parameters={SAMPLE_PARAMETERS}"
        )
        _set_regularization_state(target, arm, 0.0)

        def bridge_checkpoint(particles, records):
            _write_sampling_checkpoint(partial, particles, records, [])
            latest = records[-1]
            log(
                f"bridge level={latest['level']} t={latest['t']:.6f} "
                f"nESS={latest['weight_metrics']['normalized_ess']:.4f} "
                f"MALA={latest['mala']['acceptance_rate']:.3f}"
            )

        start_particles, bridge_records = _adaptive_bridge_sample(
            initial,
            source,
            target,
            tor_start=problem["tor_start"],
            stage_dir=stage_dir,
            deadline=deadline,
            checkpoint=bridge_checkpoint,
        )
        direct_schedule, direct_schedule_arrays = _direct_schedule_contrasts(
            target, arm, start_particles
        )

        def sharpen_checkpoint(particles, bridge, sharpen):
            _write_sampling_checkpoint(partial, particles, bridge, sharpen)
            latest = sharpen[-1]
            log(
                f"sharpen level={latest['level']} "
                f"s={latest['progress']:.6f} "
                f"nESS={latest['weight_metrics']['normalized_ess']:.4f} "
                f"MALA={latest['mala']['acceptance_rate']:.3f}"
            )

        particles, sharpen_records = _adaptive_regularization_sharpen(
            start_particles,
            target,
            arm,
            tor_start=problem["tor_start"],
            stage_dir=stage_dir,
            deadline=deadline,
            bridge_records=bridge_records,
            checkpoint=sharpen_checkpoint,
        )
        particles = particles.detach()
        final_state = _set_regularization_state(target, arm, 1.0)
        final_floor = float(final_state["r_floor_nm"])
        final_surrogate_value = _eval_reduced_chunks(target, particles)
        target.set_r_floor(0.0)
        raw_distance_mapped_value = _eval_reduced_chunks(target, particles)
        raw_physical_on_final = _eval_reduced_chunks(target, particles, raw=True)
        target.set_r_floor(final_floor)

        floor_logw_tensor = final_surrogate_value - raw_distance_mapped_value
        floor_logw = floor_logw_tensor.cpu().numpy().astype(np.float64, copy=False)
        direct_raw_logw_tensor = final_surrogate_value - raw_physical_on_final
        direct_raw_logw = direct_raw_logw_tensor.cpu().numpy().astype(
            np.float64, copy=False
        )
        floor_metrics = _normalized_weight_metrics(floor_logw)
        direct_raw_metrics = _normalized_weight_metrics(direct_raw_logw)
        correction = {
            "final_floor_to_raw_distance": {
                "from_floor_nm": final_floor,
                "metrics": floor_metrics,
            },
            "direct_final_surrogate_to_raw_physical": {
                "metrics": direct_raw_metrics
            },
        }
        arrays = {
            "start_surrogate_particles": start_particles.cpu().numpy().astype(
                np.float32, copy=False
            ),
            "final_surrogate_particles": particles.cpu().numpy().astype(
                np.float32, copy=False
            ),
            "floor_to_raw_distance_logw": floor_logw,
            "direct_final_surrogate_to_raw_physical_logw": direct_raw_logw,
            **direct_schedule_arrays,
        }
        correction_mala = None
        if floor_metrics.get("valid", False) and (
            floor_metrics["normalized_ess"]
            >= SAMPLE_PARAMETERS["correction_ess_min"]
        ):
            if final_floor == 0.0:
                raw_distance_particles = particles.detach().clone()
            else:
                raw_distance_particles, ancestors, weights, _ = (
                    _resample_with_ancestry(
                        particles,
                        floor_logw_tensor,
                        SAMPLE_SEED + 300000,
                    )
                )
                _save_transition_artifact(
                    stage_dir,
                    phase="floor-to-raw",
                    level=1,
                    logw=floor_logw_tensor,
                    normalized_weights=weights,
                    ancestors=ancestors,
                )
                target.set_r_floor(0.0)
                raw_distance_particles, correction_mala = _mala_rejuvenate(
                    raw_distance_particles,
                    target,
                    tor_start=problem["tor_start"],
                    step=SAMPLE_PARAMETERS["mala_step"],
                    steps=SAMPLE_PARAMETERS["mala_steps"],
                    chunks=SAMPLE_PARAMETERS["mala_chunks"],
                    image_radius=SAMPLE_PARAMETERS["wrapped_image_radius"],
                    seed=SAMPLE_SEED + 300001,
                    deadline=deadline,
                )
                if correction_mala["invalid"] != 0:
                    raise FloatingPointError(
                        "floor-to-raw MALA produced invalid proposals"
                    )
            target.set_r_floor(0.0)
            regularized_value = _eval_reduced_chunks(
                target, raw_distance_particles
            )
            raw_physical_value = _eval_reduced_chunks(
                target, raw_distance_particles, raw=True
            )
            energy_logw = (
                regularized_value - raw_physical_value
            ).cpu().numpy().astype(np.float64, copy=False)
            correction["energy_map_to_raw_physical"] = {
                "metrics": _normalized_weight_metrics(energy_logw),
                "mala_after_floor_removal": correction_mala,
            }
            arrays["raw_distance_particles"] = (
                raw_distance_particles.cpu().numpy().astype(np.float32, copy=False)
            )
            arrays["energy_map_to_raw_physical_logw"] = energy_logw

        mala_records = [record["mala"] for record in bridge_records + sharpen_records]
        if correction_mala is not None:
            mala_records.append(correction_mala)
        sampling_checks = {
            "bridge_reached_target": bool(
                bridge_records and bridge_records[-1]["t"] == 1.0
            ),
            "sharpen_reached_final": bool(
                not (arm["r_schedule"] or arm["e_schedule"])
                or (
                    sharpen_records
                    and sharpen_records[-1]["progress"] == 1.0
                )
            ),
            "all_transition_ess_pass": bool(
                all(
                    record["weight_metrics"]["normalized_ess"]
                    >= SAMPLE_PARAMETERS["conditional_ess_min"]
                    for record in bridge_records + sharpen_records
                )
            ),
            "all_mala_proposals_finite": bool(
                all(record["invalid"] == 0 for record in mala_records)
            ),
            "all_mala_acceptance_nonzero": bool(
                all(record["acceptance_rate"] > 0.0 for record in mala_records)
            ),
        }

        def correction_pass(metrics: dict) -> bool:
            return bool(
                metrics.get("valid", False)
                and metrics["normalized_ess"]
                >= SAMPLE_PARAMETERS["correction_ess_min"]
                and metrics["max_weight"] <= 0.01
                and metrics["top100_mass"] <= 0.50
            )

        correction_checks = {
            name: correction_pass(value["metrics"])
            for name, value in correction.items()
            if "metrics" in value
        }

        np.savez_compressed(partial / "sampling_arrays.npz", **arrays)
        summary = {
            "schema_version": 1,
            "process_status": "complete",
            "sampling_gate_status": (
                "pass" if all(sampling_checks.values()) else "fail"
            ),
            "sampling_checks": sampling_checks,
            "raw_correction_gate_status": (
                "pass" if correction_checks and all(correction_checks.values()) else "fail"
            ),
            "correction_checks": correction_checks,
            "arm_name": arm_name,
            "arm": arm,
            "parameters": SAMPLE_PARAMETERS,
            "sample_seed": SAMPLE_SEED,
            "bridge": bridge_records,
            "direct_schedule": direct_schedule,
            "sharpen": sharpen_records,
            "final_state": final_state,
            "correction": correction,
            "wall_seconds": time.time() - started,
            "device": device,
            "dtype": "float32",
            "input_manifest_sha256": _sha256(INPUTS / "manifest.json"),
            "scientific_code_hashes": code_hashes,
            "git": git_state,
        }
        _json(partial / "summary.json", summary)
        _json(
            partial / "manifest.json",
            {
                "schema_version": 1,
                "files": {
                    str(path.relative_to(partial)): _sha256(path)
                    for path in sorted(partial.rglob("*"))
                    if path.is_file() and path.name != "manifest.json"
                },
            },
        )
        partial.rename(run)
        print(f"completed immutable sampling arm: {run}", flush=True)
    except Exception as exc:
        _json(
            partial / "failure.json",
            {
                "process_status": "failed",
                "arm_name": arm_name,
                "error": repr(exc),
            },
        )
        failed = partial.with_name(partial.name.replace(".partial-", ".failed-"))
        partial.rename(failed)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "gate-a", "gate-b-sample"))
    parser.add_argument("--arm", choices=tuple(SAMPLE_ARMS))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "gate-a":
        gate_a()
    else:
        if args.arm is None:
            parser.error("gate-b-sample requires --arm")
        gate_b_sample(args.arm)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Sample and plot glycerol torsions across molecular energy cutoffs.

This is a sampler-only target diagnostic, not Boltzmann-generator training.
It uses potential-space SMC with exact mixed-domain MALA, saves every endpoint
population to HDF5, and plots only from those saved samples.

Run from the X-regularization repository root:

    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
        python Molecular_BG/glycerol_36d/dihedrals.py sample

Replot without any sampling or molecular-energy evaluation:

    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
        python Molecular_BG/glycerol_36d/dihedrals.py replot

Add Cartesian geometry and joint-mode diagnostics to an existing HDF5 file
without changing its samples:

    source ~/.envs/jflows/bin/activate
    PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
        python Molecular_BG/glycerol_36d/dihedrals.py audit
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
EXPLORATORY_DATA_DEFAULT = HERE / "dihedrals_samples.h5"
NEXT_DATA_DEFAULT = HERE / "dihedrals_samples_full.h5"
FIGURE_DEFAULT = HERE / "dihedrals.png"

import equinox as eqx  # noqa: E402
import h5py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from jflows.potential import linear_combination  # noqa: E402
from jflows.utils import linear_weights_from_log, resample  # noqa: E402
from jflows_md import (  # noqa: E402
    Molecular_Bundle,
    Molecular_Potential,
    mixed_mala,
    package_source_sha256,
)

import parameters as P  # noqa: E402


TORSIONS = (
    ((0, 1, 2, 3), "O-C-C-O"),
    ((1, 2, 4, 5), "C-C-C-O"),
    ((1, 2, 3, 10), "C-C-O-H"),
)
DIAGNOSTIC_VOLUME_ATOMS = (2, 1, 3, 4)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cap_label(cut: float | None) -> str:
    if cut is None:
        return "exact"
    return f"cap_{cut:g}".replace(".", "p")


def bridge(source, target, coefficient: float):
    if coefficient <= 0.0:
        return source
    if coefficient >= 1.0:
        return target
    return linear_combination(
        [source, target], [1.0 - coefficient, coefficient]
    )


@eqx.filter_jit
def _energy_difference_chunk(source, target, samples):
    return target(samples) - source(samples)


@eqx.filter_jit
def _physical_diagnostics_chunk(target: Molecular_Potential, samples):
    positions, log_jacobian = target.coordinates.to_cartesian(samples)
    physical_energy = target.forcefield(positions)
    angles = []
    for quad, _ in TORSIONS:
        p0, p1, p2, p3 = (positions[:, atom] for atom in quad)
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        n1 = jnp.cross(b1, b2)
        n2 = jnp.cross(b2, b3)
        b2_unit = b2 / jnp.linalg.norm(b2, axis=-1, keepdims=True)
        sine = jnp.sum(jnp.cross(n1, n2) * b2_unit, axis=-1)
        cosine = jnp.sum(n1 * n2, axis=-1)
        angles.append(jnp.arctan2(sine, cosine))
    return physical_energy, log_jacobian, jnp.stack(angles, axis=-1)


@eqx.filter_jit
def _geometry_diagnostics_chunk(
    target: Molecular_Potential, samples, active_exception_idx
):
    positions = target.coordinates.to_cartesian(samples)[0]
    center, first, second, third = DIAGNOSTIC_VOLUME_ATOMS
    center_position = positions[:, center]
    a = positions[:, first] - center_position
    b = positions[:, second] - center_position
    c = positions[:, third] - center_position
    signed_volume = jnp.sum(a * jnp.cross(b, c), axis=-1)

    pair_idx = target.forcefield.pair_idx
    pair_distance = jnp.linalg.norm(
        positions[:, pair_idx[:, 0]] - positions[:, pair_idx[:, 1]], axis=-1
    )
    exception_distance = jnp.linalg.norm(
        positions[:, active_exception_idx[:, 0]]
        - positions[:, active_exception_idx[:, 1]],
        axis=-1,
    )
    bond_idx = target.forcefield.bond_idx
    bond_distance = jnp.linalg.norm(
        positions[:, bond_idx[:, 0]] - positions[:, bond_idx[:, 1]], axis=-1
    )
    return (
        signed_volume,
        jnp.min(pair_distance, axis=-1),
        jnp.min(exception_distance, axis=-1),
        jnp.min(bond_distance, axis=-1),
        jnp.max(bond_distance, axis=-1),
    )


def chunked_energy_difference(source, target, samples, chunk: int) -> np.ndarray:
    pieces = []
    for part in jnp.array_split(samples, chunk, axis=0):
        value = jax.block_until_ready(
            _energy_difference_chunk(source, target, part)
        )
        pieces.append(np.asarray(value))
    return np.concatenate(pieces)


def physical_diagnostics(
    target: Molecular_Potential, samples, chunk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_parts, logdet_parts, torsion_parts = [], [], []
    for part in jnp.array_split(samples, chunk, axis=0):
        energy, logdet, torsions = jax.block_until_ready(
            _physical_diagnostics_chunk(target, part)
        )
        energy_parts.append(np.asarray(energy))
        logdet_parts.append(np.asarray(logdet))
        torsion_parts.append(np.asarray(torsions))
    return (
        np.concatenate(energy_parts),
        np.concatenate(logdet_parts),
        np.concatenate(torsion_parts),
    )


def geometry_diagnostics(
    target: Molecular_Potential, samples: np.ndarray, chunk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exception_charge = np.asarray(target.forcefield.exception_chargeprod)
    exception_epsilon = np.asarray(target.forcefield.exception_epsilon)
    active = (exception_charge != 0.0) | (exception_epsilon != 0.0)
    active_exception_idx = np.asarray(target.forcefield.exception_idx)[active]
    if active_exception_idx.shape[0] == 0:
        raise RuntimeError("glycerol has no active nonbonded exception pairs")
    active_exception_idx = jnp.asarray(active_exception_idx)
    output = [[], [], [], [], []]
    for part in np.array_split(samples, chunk, axis=0):
        values = jax.block_until_ready(
            _geometry_diagnostics_chunk(
                target, jnp.asarray(part), active_exception_idx
            )
        )
        for pieces, value in zip(output, values, strict=True):
            pieces.append(np.asarray(value))
    return tuple(np.concatenate(pieces) for pieces in output)


def normalized_ess(log_weight: np.ndarray) -> float:
    """Normalized ESS in [0, 1], including limiting infinite weights."""

    value = np.asarray(log_weight)
    if value.ndim != 1 or value.size == 0 or np.isnan(value).any():
        return 0.0
    positive_infinity = np.isposinf(value)
    if positive_infinity.any():
        return float(positive_infinity.sum() / value.size)
    finite = np.isfinite(value)
    if not finite.any():
        return 0.0
    maximum = np.max(value[finite])
    weight = np.zeros_like(value, dtype=np.float64)
    weight[finite] = np.exp(value[finite].astype(np.float64) - maximum)
    denominator = value.size * np.square(weight).sum()
    if not np.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return float(np.clip(weight.sum() ** 2 / denominator, 0.0, 1.0))


def choose_level(
    delta_energy: np.ndarray,
    current: float,
    tau_ess: float,
    min_step: float,
) -> tuple[float, float, float]:
    """Choose the largest next bridge coefficient meeting the ESS gate."""

    direct_ess = normalized_ess(-(1.0 - current) * delta_energy)
    if direct_ess >= tau_ess:
        return 1.0, direct_ess, direct_ess

    low, high = current, 1.0
    for _ in range(32):
        middle = 0.5 * (low + high)
        ess = normalized_ess(-(middle - current) * delta_energy)
        if ess >= tau_ess:
            low = middle
        else:
            high = middle

    candidate = low
    if candidate - current < min_step:
        candidate = min(1.0, current + min_step)
    level_ess = normalized_ess(-(candidate - current) * delta_energy)
    return candidate, level_ess, direct_ess


def adaptive_transition(
    key,
    samples,
    source,
    target,
    domain,
    *,
    name: str,
    tau_ess: float,
    min_step: float,
    max_levels: int,
    mala_step: float,
    mala_iters: int,
    images: int,
    chunk: int,
) -> tuple[jax.Array, dict[str, np.ndarray | float]]:
    """Advance one endpoint pair by adaptive reweight/resample/mixed MALA."""

    current = 0.0
    coefficients, ess_values, direct_values, acceptances = [], [], [], []
    for level_index in range(max_levels):
        delta_energy = chunked_energy_difference(source, target, samples, chunk)
        next_value, expected_ess, direct_ess = choose_level(
            delta_energy, current, tau_ess, min_step
        )
        if expected_ess + 1e-6 < tau_ess:
            raise RuntimeError(
                f"{name}: minimum bridge step violates the ESS gate "
                f"({expected_ess:.4f} < {tau_ess:.4f})"
            )
        increment = next_value - current
        if not increment > 0.0:
            raise RuntimeError(f"{name}: adaptive bridge made no progress at {current}")

        log_weight = -increment * delta_energy
        weight = jax.block_until_ready(
            linear_weights_from_log(jnp.asarray(log_weight))
        )
        if not bool(jnp.sum(weight) > 0):
            raise RuntimeError(f"{name}: all incremental weights vanished")

        level_key = jax.random.fold_in(key, level_index)
        resample_key, mala_key = jax.random.split(level_key)
        samples = resample(
            resample_key, samples, weight, N=samples.shape[0]
        )
        next_potential = bridge(source, target, next_value)
        samples, acceptance = mixed_mala(
            mala_key,
            samples,
            next_potential,
            domain,
            step=mala_step,
            iters=mala_iters,
            images=images,
            chunk=chunk,
        )
        samples, acceptance = jax.block_until_ready((samples, acceptance))
        measured_ess = normalized_ess(log_weight)
        coefficients.append(next_value)
        ess_values.append(measured_ess)
        direct_values.append(direct_ess)
        acceptances.append(np.asarray(acceptance))
        log(
            f"{name}: level {level_index + 1:02d} t={next_value:.6f} "
            f"ESS={measured_ess:.4f} direct={direct_ess:.4f} "
            f"MALA={float(np.mean(np.asarray(acceptance))):.3f}"
        )
        if abs(measured_ess - expected_ess) > 2e-5:
            raise RuntimeError(
                f"{name}: ESS selection mismatch {expected_ess} != {measured_ess}"
            )
        current = next_value
        if current >= 1.0:
            return samples, {
                "coefficients": np.asarray(coefficients),
                "ess": np.asarray(ess_values),
                "direct_ess": np.asarray(direct_values),
                "mala_acceptance": np.stack(acceptances),
                "initial_direct_ess": float(direct_values[0]),
            }

    raise RuntimeError(
        f"{name}: did not reach its endpoint within {max_levels} levels; "
        f"last coefficient={current:.6f}"
    )


def regularized_energy(
    physical: np.ndarray,
    reference: float,
    cut: float | None,
    scale: float,
    tail_fraction: float,
) -> np.ndarray:
    if cut is None:
        return physical.copy()
    excess = physical - reference
    over = np.maximum(excess - cut, 0.0)
    compressed = (
        cut
        + (1.0 - tail_fraction) * scale * np.log1p(over / scale)
        + tail_fraction * over
    )
    return reference + np.where(excess > cut, compressed, excess)


def create_dataset(group, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    first_chunk = min(array.shape[0], 4096) if array.ndim else None
    chunks = None if array.ndim == 0 else (first_chunk, *array.shape[1:])
    group.create_dataset(
        name,
        data=array,
        chunks=chunks,
        compression="gzip" if array.ndim else None,
        compression_opts=4 if array.ndim else None,
        shuffle=bool(array.ndim),
    )


def deformation_specification(
    target: Molecular_Potential,
    reference_energy: float,
    cut: float | None,
    scale: float,
    tail_fraction: float,
) -> tuple[str, str]:
    value = {
        "schema": 1,
        "kind": "exact" if cut is None else "lin_log_excess_energy",
        "base_manifest_sha256": target.manifest_sha256,
        "reference_energy_kj_mol": reference_energy,
        "energy_cut_excess_kj_mol": cut,
        "energy_scale_kj_mol": scale,
        "tail_fraction": tail_fraction,
        "jacobian": "unchanged",
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode()).hexdigest()


def save_endpoint(
    file: h5py.File,
    seed: int,
    label: str,
    samples,
    physical_target: Molecular_Potential,
    endpoint,
    transition: dict[str, np.ndarray | float],
    endpoint_acceptance: np.ndarray,
    *,
    cut: float | None,
    reference_energy: float,
    scale: float,
    tail_fraction: float,
    chunk: int,
) -> None:
    parent = file.require_group(f"seed_{seed}")
    temporary = f"_writing_{label}"
    if temporary in parent:
        del parent[temporary]
    if label in parent:
        raise RuntimeError(f"refusing to overwrite completed HDF5 group seed_{seed}/{label}")
    group = parent.create_group(temporary)

    q = np.asarray(jax.device_get(samples))
    physical, log_jacobian, torsions = physical_diagnostics(
        physical_target, samples, chunk
    )
    deformed = regularized_energy(
        physical, reference_energy, cut, scale, tail_fraction
    )
    reduced = float(np.asarray(physical_target.beta)) * deformed - log_jacobian
    active = np.zeros(physical.shape, dtype=np.uint8)
    if cut is not None:
        active = ((physical - reference_energy) > cut).astype(np.uint8)

    check_count = min(256, q.shape[0])
    expected = np.asarray(
        jax.block_until_ready(endpoint(jnp.asarray(q[:check_count])))
    )
    max_error = float(np.max(np.abs(expected - reduced[:check_count])))
    if not np.isfinite(max_error) or max_error > 5e-4:
        raise RuntimeError(
            f"{label}: saved regularized-energy formula disagrees with endpoint "
            f"potential (max error {max_error:.3e})"
        )

    create_dataset(group, "q", q)
    create_dataset(group, "torsions_rad", torsions)
    create_dataset(group, "physical_energy_kj_mol", physical)
    create_dataset(group, "deformed_energy_kj_mol", deformed)
    create_dataset(group, "target_reduced_energy", reduced)
    create_dataset(group, "log_jacobian", log_jacobian)
    create_dataset(group, "cap_active", active)
    create_dataset(group, "transition_coefficients", transition["coefficients"])
    create_dataset(group, "transition_ess", transition["ess"])
    create_dataset(group, "transition_direct_ess", transition["direct_ess"])
    create_dataset(
        group, "transition_mala_acceptance", transition["mala_acceptance"]
    )
    create_dataset(group, "endpoint_mala_acceptance", endpoint_acceptance)
    group.attrs["cut_kj_mol"] = np.nan if cut is None else cut
    deformation_json, deformation_sha256 = deformation_specification(
        physical_target, reference_energy, cut, scale, tail_fraction
    )
    group.attrs["deformation_spec_json"] = deformation_json
    group.attrs["deformation_spec_sha256"] = deformation_sha256
    group.attrs["cap_active_fraction"] = float(active.mean())
    group.attrs["incoming_direct_ess"] = float(transition["initial_direct_ess"])
    group.attrs["incoming_min_level_ess"] = float(
        np.min(np.asarray(transition["ess"]))
    )
    group.attrs["incoming_levels"] = len(np.asarray(transition["ess"]))
    group.attrs["endpoint_mala_mean"] = float(np.mean(endpoint_acceptance))
    group.attrs["endpoint_formula_max_error"] = max_error
    group.attrs["complete"] = True
    file.flush()
    parent.move(temporary, label)
    file.flush()


def sampling_configuration(
    target: Molecular_Potential,
    *,
    n_samples: int,
    seeds: tuple[int, ...],
    cuts: tuple[float, ...],
    include_exact: bool,
    tau_ess: float,
    min_step: float,
    max_levels: int,
    mala_step: float,
    mala_iters: int,
    endpoint_iters: int,
    chunk: int,
) -> dict[str, object]:
    return {
        "schema": 1,
        "bundle": target.bundle_name,
        "manifest_sha256": target.manifest_sha256,
        "dimension": target.dimension,
        "n_samples": n_samples,
        "seeds": list(seeds),
        "cuts_kj_mol": list(cuts),
        "include_exact": include_exact,
        "scale_kj_mol": P.DIHEDRAL_ENERGY_SCALE,
        "tail_fraction": P.DIHEDRAL_TAIL_FRACTION,
        "tau_ess": tau_ess,
        "min_level_step": min_step,
        "max_levels": max_levels,
        "mala_step": mala_step,
        "mala_iters": mala_iters,
        "endpoint_iters": endpoint_iters,
        "wrapped_images": P.WRAPPED_IMAGES,
        "chunk": chunk,
        "torsions": [
            {"atoms": list(quad), "name": name} for quad, name in TORSIONS
        ],
        "jflows_sha256": package_source_sha256("jflows"),
        "jflows_md_sha256": package_source_sha256("jflows_md"),
    }


def initialize_file(path: Path, config: dict[str, object], overwrite: bool) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    mode = "r+" if path.exists() else "w"
    file = h5py.File(path, mode)
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_json.encode()).hexdigest()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if mode == "w":
        file.attrs["schema"] = 1
        file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        file.attrs["config_json"] = config_json
        file.attrs["config_sha256"] = config_sha256
        file.attrs["script_sha256"] = script_sha256
        file.flush()
    elif file.attrs.get("config_sha256", "") != config_sha256:
        file.close()
        raise RuntimeError(
            "saved data use a different sampling configuration; choose another "
            "--data path or pass --overwrite"
        )
    elif file.attrs.get("script_sha256", "") != script_sha256:
        saved = file.attrs.get("script_sha256", "unknown")
        file.close()
        raise RuntimeError(
            "saved data were generated by a different sampling script "
            f"({saved} != {script_sha256}); resume is forbidden. Replot or "
            "audit the existing artifact, or choose a new --data path"
        )
    return file


def sample(args: argparse.Namespace) -> None:
    target = Molecular_Potential.from_bundle(P.BUNDLE)
    source = target.source()
    if target.dimension != 36 or (
        target.domain.euclidean_dim,
        target.domain.periodic_dim,
    ) != (25, 11):
        raise ValueError("glycerol diagnostic requires the R^25 x T^11 bundle")

    if args.pilot:
        n_samples = args.n_samples or 2048
        seeds = (P.DIHEDRAL_SEEDS[0],)
        cuts = tuple(P.DIHEDRAL_ENERGY_CUTS[:3])
        include_exact = False
        tau_ess = 0.5
        min_step = 1e-3
        max_levels = 20
        mala_iters = 2
        endpoint_iters = 2
        chunk = min(P.DIHEDRAL_CHUNK, n_samples)
    else:
        n_samples = args.n_samples or P.DIHEDRAL_N_SAMPLES
        seeds = tuple(P.DIHEDRAL_SEEDS)
        cuts = tuple(P.DIHEDRAL_ENERGY_CUTS)
        include_exact = not args.no_exact
        tau_ess = P.DIHEDRAL_TAU_ESS
        min_step = P.DIHEDRAL_MIN_LEVEL_STEP
        max_levels = P.DIHEDRAL_MAX_LEVELS
        mala_iters = P.DIHEDRAL_MALA_ITERS
        endpoint_iters = P.DIHEDRAL_ENDPOINT_ITERS
        chunk = P.DIHEDRAL_CHUNK
    if n_samples < chunk or n_samples % chunk:
        raise ValueError("n_samples must be divisible by the physical chunk count")

    config = sampling_configuration(
        target,
        n_samples=n_samples,
        seeds=seeds,
        cuts=cuts,
        include_exact=include_exact,
        tau_ess=tau_ess,
        min_step=min_step,
        max_levels=max_levels,
        mala_step=P.DIHEDRAL_MALA_STEP,
        mala_iters=mala_iters,
        endpoint_iters=endpoint_iters,
        chunk=chunk,
    )
    path = Path(args.data)
    file = initialize_file(path, config, args.overwrite)
    reference_energy = float(
        np.asarray(target.forcefield(target.reference_positions_nm[None])[0])
    )
    file.attrs["reference_energy_kj_mol"] = reference_energy
    file.attrs["cutoff_convention"] = "excess_cartesian_energy_above_E_ref"
    file.flush()
    endpoints = [
        (
            cap_label(cut),
            cut,
            target.regularized(
                cut,
                energy_scale_kj_mol=P.DIHEDRAL_ENERGY_SCALE,
                tail_fraction=P.DIHEDRAL_TAIL_FRACTION,
            ),
        )
        for cut in cuts
    ]
    if include_exact:
        endpoints.append(("exact", None, target))

    log(
        f"START glycerol torsion diagnostic | backend={jax.default_backend()} "
        f"N={n_samples} seeds={seeds} cuts={cuts} exact={include_exact} "
        f"bundle={target.bundle_name} manifest={target.manifest_sha256}"
    )
    try:
        for seed in seeds:
            seed_group = file.require_group(f"seed_{seed}")
            root_key = jax.random.key(seed)
            samples = source.samples(jax.random.fold_in(root_key, 0), n_samples)
            previous = source
            blocked_by_gap = False
            for endpoint_index, (label, cut, endpoint) in enumerate(endpoints, start=1):
                temporary = f"_writing_{label}"
                if temporary in seed_group:
                    del seed_group[temporary]
                if label in seed_group and bool(seed_group[label].attrs.get("complete", False)):
                    if blocked_by_gap:
                        raise RuntimeError(
                            f"seed {seed}: completed {label} follows a missing endpoint"
                        )
                    log(f"seed {seed}: resume from saved {label}")
                    samples = jnp.asarray(seed_group[label]["q"][:])
                    previous = endpoint
                    continue
                blocked_by_gap = True
                transition_name = f"seed {seed} {cap_label(cuts[endpoint_index - 2]) if endpoint_index > 1 and endpoint_index - 2 < len(cuts) else 'source'}->{label}"
                transition_key = jax.random.fold_in(root_key, endpoint_index)
                samples, transition = adaptive_transition(
                    transition_key,
                    samples,
                    previous,
                    endpoint,
                    target.domain,
                    name=transition_name,
                    tau_ess=tau_ess,
                    min_step=min_step,
                    max_levels=max_levels,
                    mala_step=P.DIHEDRAL_MALA_STEP,
                    mala_iters=mala_iters,
                    images=P.WRAPPED_IMAGES,
                    chunk=chunk,
                )
                refresh_key = jax.random.fold_in(root_key, 100000 + endpoint_index)
                samples, endpoint_acceptance = mixed_mala(
                    refresh_key,
                    samples,
                    endpoint,
                    target.domain,
                    step=P.DIHEDRAL_MALA_STEP,
                    iters=endpoint_iters,
                    images=P.WRAPPED_IMAGES,
                    chunk=chunk,
                )
                samples, endpoint_acceptance = jax.block_until_ready(
                    (samples, endpoint_acceptance)
                )
                log(
                    f"seed {seed} {label}: endpoint MALA="
                    f"{float(np.mean(np.asarray(endpoint_acceptance))):.3f}; saving"
                )
                save_endpoint(
                    file,
                    seed,
                    label,
                    samples,
                    target,
                    endpoint,
                    transition,
                    np.asarray(endpoint_acceptance),
                    cut=cut,
                    reference_energy=reference_energy,
                    scale=P.DIHEDRAL_ENERGY_SCALE,
                    tail_fraction=P.DIHEDRAL_TAIL_FRACTION,
                    chunk=chunk,
                )
                previous = endpoint
                blocked_by_gap = False
    except Exception as exc:
        file.attrs["last_failure_utc"] = datetime.now(timezone.utc).isoformat()
        file.attrs["last_failure"] = f"{type(exc).__name__}: {exc}"
        file.flush()
        raise
    finally:
        file.close()

    plot(Path(args.data), Path(args.figure))
    log(f"DONE data={args.data} figure={args.figure}")


def replace_dataset(group, name: str, value: np.ndarray) -> None:
    if name in group:
        del group[name]
    create_dataset(group, name, value)


def audit(data_path: Path, figure_path: Path) -> None:
    """Augment saved particles with geometry and provenance; run no dynamics."""

    if not data_path.exists():
        raise FileNotFoundError(f"saved diagnostic data do not exist: {data_path}")
    with h5py.File(data_path, "r+") as file:
        config = json.loads(file.attrs["config_json"])
        energy_scale = float(config["scale_kj_mol"])
        tail_fraction = float(config["tail_fraction"])
        bundle = Molecular_Bundle.load(config["bundle"])
        target = Molecular_Potential(bundle)
        if target.manifest_sha256 != config["manifest_sha256"]:
            raise RuntimeError("saved diagnostic manifest does not match the bundle")
        diagnostic_atoms = tuple(bundle.coordinates["diagnostic_chirality_atoms"])
        if diagnostic_atoms != DIAGNOSTIC_VOLUME_ATOMS:
            raise RuntimeError(
                f"unexpected glycerol diagnostic atoms: {diagnostic_atoms}"
            )
        reference_energy = float(
            np.asarray(target.forcefield(target.reference_positions_nm[None])[0])
        )
        file.attrs["reference_energy_kj_mol"] = reference_energy
        file.attrs["cutoff_convention"] = "excess_cartesian_energy_above_E_ref"
        file.attrs["audit_script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        file.attrs["audited_utc"] = datetime.now(timezone.utc).isoformat()
        file.attrs["sampling_script_matches_audit_script"] = (
            file.attrs.get("script_sha256", "")
            == file.attrs["audit_script_sha256"]
        )

        labels = [cap_label(value) for value in config["cuts_kj_mol"]]
        if config["include_exact"]:
            labels.append("exact")
        chunk = int(config["chunk"])
        for seed in config["seeds"]:
            for label in labels:
                group = file[f"seed_{seed}/{label}"]
                q = group["q"][:]
                (
                    volume,
                    minimum_pair,
                    minimum_exception,
                    minimum_bond,
                    maximum_bond,
                ) = geometry_diagnostics(target, q, chunk)
                replace_dataset(group, "diagnostic_signed_volume_nm3", volume)
                replace_dataset(
                    group, "minimum_nonbonded_pair_distance_nm", minimum_pair
                )
                replace_dataset(
                    group,
                    "minimum_active_exception_pair_distance_nm",
                    minimum_exception,
                )
                replace_dataset(group, "minimum_bond_distance_nm", minimum_bond)
                replace_dataset(group, "maximum_bond_distance_nm", maximum_bond)
                group.attrs["diagnostic_positive_fraction"] = float(
                    np.mean(volume > 0.0)
                )
                group.attrs["minimum_nonbonded_pair_distance_nm"] = float(
                    np.min(minimum_pair)
                )
                group.attrs["minimum_active_exception_pair_distance_nm"] = float(
                    np.min(minimum_exception)
                )
                cut = float(group.attrs["cut_kj_mol"])
                deformation_json, deformation_sha256 = deformation_specification(
                    target,
                    reference_energy,
                    None if math.isnan(cut) else cut,
                    energy_scale,
                    tail_fraction,
                )
                group.attrs["deformation_spec_json"] = deformation_json
                group.attrs["deformation_spec_sha256"] = deformation_sha256
                log(
                    f"audit seed {seed} {label}: positive-volume="
                    f"{np.mean(volume > 0.0):.4f} min-nonbonded="
                    f"{np.min(minimum_pair):.4f} nm min-active-exception="
                    f"{np.min(minimum_exception):.4f} nm"
                )
        file.flush()
    plot(data_path, figure_path)
    log(f"AUDIT DONE data={data_path} figure={figure_path}; no dynamics ran")


def periodic_density(values: np.ndarray, bins: int, sigma: float):
    count, edges = np.histogram(values, bins=bins, range=(-math.pi, math.pi))
    density = count.astype(np.float64)
    density = gaussian_filter1d(density, sigma=sigma, mode="wrap")
    width = edges[1] - edges[0]
    density /= density.sum() * width
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def js_bits(first: np.ndarray, second: np.ndarray, bins: int = 72) -> float:
    p, _ = np.histogram(first, bins=bins, range=(-math.pi, math.pi))
    q, _ = np.histogram(second, bins=bins, range=(-math.pi, math.pi))
    p = p.astype(np.float64) / p.sum()
    q = q.astype(np.float64) / q.sum()
    middle = 0.5 * (p + q)
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_mask = p > 0.0
    q_mask = q > 0.0
    p_term[p_mask] = p[p_mask] * np.log2(p[p_mask] / middle[p_mask])
    q_term[q_mask] = q[q_mask] * np.log2(q[q_mask] / middle[q_mask])
    return float(0.5 * (p_term.sum() + q_term.sum()))


def joint_js_bits(first: np.ndarray, second: np.ndarray, bins: int) -> float:
    ranges = [(-math.pi, math.pi)] * first.shape[1]
    p = np.histogramdd(first, bins=bins, range=ranges)[0].ravel()
    q = np.histogramdd(second, bins=bins, range=ranges)[0].ravel()
    p = p.astype(np.float64) / p.sum()
    q = q.astype(np.float64) / q.sum()
    middle = 0.5 * (p + q)
    value = 0.0
    p_mask = p > 0.0
    q_mask = q > 0.0
    value += 0.5 * np.sum(p[p_mask] * np.log2(p[p_mask] / middle[p_mask]))
    value += 0.5 * np.sum(q[q_mask] * np.log2(q[q_mask] / middle[q_mask]))
    return float(value)


def mode_count(values: np.ndarray) -> int:
    histogram, _ = np.histogram(values, bins=36, range=(-math.pi, math.pi))
    if histogram.max() == 0:
        return 0
    peaks = [
        index
        for index in range(36)
        if histogram[index] >= histogram[(index - 1) % 36]
        and histogram[index] >= histogram[(index + 1) % 36]
        and histogram[index] > 0.08 * histogram.max()
    ]
    populations = sorted((histogram[index] for index in peaks), reverse=True)
    return sum(value > 0.15 * populations[0] for value in populations)


def plot(data_path: Path, figure_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"saved diagnostic data do not exist: {data_path}")
    with h5py.File(data_path, "r+") as file:
        config = json.loads(file.attrs["config_json"])
        energy_scale = float(config["scale_kj_mol"])
        tail_fraction = float(config["tail_fraction"])
        seeds = tuple(config["seeds"])
        requested = [cap_label(value) for value in config["cuts_kj_mol"]]
        if config["include_exact"]:
            requested.append("exact")
        complete = [
            label
            for label in requested
            if all(
                f"seed_{seed}/{label}" in file
                and bool(file[f"seed_{seed}/{label}"].attrs.get("complete", False))
                for seed in seeds
            )
        ]
        if not complete:
            raise RuntimeError("no endpoint is complete for every configured seed")

        torsions_by_level = {
            label: np.concatenate(
                [file[f"seed_{seed}/{label}/torsions_rad"][:] for seed in seeds],
                axis=0,
            )
            for label in complete
        }
        reference_energy = float(
            file.attrs.get("reference_energy_kj_mol", np.nan)
        )
        reference_label = "exact" if "exact" in complete else complete[-1]
        analysis = file.require_group("analysis")
        for dataset_name in (
            "cross_seed_js_bits",
            "level_vs_reference_js_bits",
            "cross_seed_all_internal_torsion_js_bits",
            "cross_seed_joint_selected_js_bits_12",
            "cross_seed_joint_rotamer_js_bits_3",
            "median_excess_energy_kj_mol",
            "diagnostic_positive_fraction",
            "minimum_nonbonded_pair_distance_nm",
            "minimum_active_exception_pair_distance_nm",
        ):
            if dataset_name in analysis:
                del analysis[dataset_name]
        divergence = np.full((len(complete), len(TORSIONS)), np.nan)
        internal_divergence = np.full((len(complete), 11), np.nan)
        joint_divergence = np.full(len(complete), np.nan)
        rotamer_divergence = np.full(len(complete), np.nan)
        median_excess = np.full((len(complete), len(seeds)), np.nan)
        positive_fraction = np.full((len(complete), len(seeds)), np.nan)
        minimum_nonbonded = np.full((len(complete), len(seeds)), np.nan)
        minimum_exception = np.full((len(complete), len(seeds)), np.nan)
        for level_index, label in enumerate(complete):
            for seed_index, seed in enumerate(seeds):
                group = file[f"seed_{seed}/{label}"]
                if np.isfinite(reference_energy):
                    median_excess[level_index, seed_index] = np.median(
                        group["physical_energy_kj_mol"][:] - reference_energy
                    )
                positive_fraction[level_index, seed_index] = group.attrs.get(
                    "diagnostic_positive_fraction", np.nan
                )
                minimum_nonbonded[level_index, seed_index] = group.attrs.get(
                    "minimum_nonbonded_pair_distance_nm", np.nan
                )
                minimum_exception[level_index, seed_index] = group.attrs.get(
                    "minimum_active_exception_pair_distance_nm", np.nan
                )
        if len(seeds) >= 2:
            for level_index, label in enumerate(complete):
                first = file[f"seed_{seeds[0]}/{label}/torsions_rad"][:]
                second = file[f"seed_{seeds[1]}/{label}/torsions_rad"][:]
                for torsion_index in range(len(TORSIONS)):
                    divergence[level_index, torsion_index] = js_bits(
                        first[:, torsion_index], second[:, torsion_index]
                    )
                first_internal = file[f"seed_{seeds[0]}/{label}/q"][:, 25:]
                second_internal = file[f"seed_{seeds[1]}/{label}/q"][:, 25:]
                for torsion_index in range(11):
                    internal_divergence[level_index, torsion_index] = js_bits(
                        first_internal[:, torsion_index],
                        second_internal[:, torsion_index],
                    )
                joint_divergence[level_index] = joint_js_bits(first, second, 12)
                rotamer_divergence[level_index] = joint_js_bits(first, second, 3)
        level_divergence = np.zeros((len(complete), len(TORSIONS)))
        for level_index, label in enumerate(complete):
            for torsion_index in range(len(TORSIONS)):
                level_divergence[level_index, torsion_index] = js_bits(
                    torsions_by_level[label][:, torsion_index],
                    torsions_by_level[reference_label][:, torsion_index],
                )
        analysis.create_dataset("cross_seed_js_bits", data=divergence)
        analysis.create_dataset(
            "level_vs_reference_js_bits", data=level_divergence
        )
        analysis.create_dataset(
            "cross_seed_all_internal_torsion_js_bits", data=internal_divergence
        )
        analysis.create_dataset(
            "cross_seed_joint_selected_js_bits_12", data=joint_divergence
        )
        analysis.create_dataset(
            "cross_seed_joint_rotamer_js_bits_3", data=rotamer_divergence
        )
        analysis.create_dataset("median_excess_energy_kj_mol", data=median_excess)
        analysis.create_dataset(
            "diagnostic_positive_fraction", data=positive_fraction
        )
        analysis.create_dataset(
            "minimum_nonbonded_pair_distance_nm", data=minimum_nonbonded
        )
        analysis.create_dataset(
            "minimum_active_exception_pair_distance_nm", data=minimum_exception
        )
        analysis.attrs["level_labels_json"] = json.dumps(complete)
        analysis.attrs["reference_label"] = reference_label
        file.flush()

    requested_curves = [cap_label(value) for value in P.DIHEDRAL_PLOT_CUTS]
    curve_labels = [
        label for label in requested_curves if label in complete and label != reference_label
    ]
    if not curve_labels:
        curve_labels = [label for label in complete[:-1]][-2:]

    plt.rcParams.update(
        {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13}
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.7), sharey=False)
    reference_name = (
        "exact-potential population (provisional)"
        if reference_label == "exact"
        else r"$c_{\mathrm{excess}}=$"
        + f"{reference_label.removeprefix('cap_').replace('p', '.')} kJ/mol "
        + "(hardest sampled)"
    )
    styles = {
        "cap_50": ("#9467bd", ":"),
        "cap_100": ("#1f77b4", (0, (7, 3))),
        "cap_200": ("#d62728", "-"),
        "cap_400": ("#ff7f0e", "-."),
    }
    for torsion_index, (axis, (_, title)) in enumerate(zip(axes, TORSIONS)):
        centers, density = periodic_density(
            torsions_by_level[reference_label][:, torsion_index],
            P.DIHEDRAL_HISTOGRAM_BINS,
            P.DIHEDRAL_SMOOTH_SIGMA,
        )
        axis.fill_between(
            centers, density, color="0.78", alpha=0.9, label=reference_name
        )
        axis.plot(centers, density, color="0.35", linewidth=1.2)
        for curve_index, label in enumerate(curve_labels):
            centers, density = periodic_density(
                torsions_by_level[label][:, torsion_index],
                P.DIHEDRAL_HISTOGRAM_BINS,
                P.DIHEDRAL_SMOOTH_SIGMA,
            )
            color, linestyle = styles.get(label, (f"C{curve_index}", "-"))
            cut_name = label.removeprefix("cap_").replace("p", ".")
            axis.plot(
                centers,
                density,
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                label=rf"$c_{{\mathrm{{excess}}}}={cut_name}$ kJ/mol",
            )
        axis.set_title(title, fontsize=20, pad=12)
        axis.set_xlim(-math.pi, math.pi)
        axis.set_xticks((-math.pi, 0.0, math.pi), (r"$-\pi$", "0", r"$\pi$"))
        axis.set_yticks([])
        axis.set_xlabel("dihedral (rad)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("density")
    figure.suptitle(
        "glycerol ($d=36$) — implicit-solvent torsions by excess-energy regularization",
        fontsize=21,
        y=0.99,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    reference_text = (
        rf"; $E_{{\mathrm{{ref}}}}$={reference_energy:.3f} kJ/mol"
        if np.isfinite(reference_energy)
        else ""
    )
    figure.text(
        0.5,
        0.105,
        "GAFF2/AM1-BCC/OBC1, 300 K"
        + reference_text
        + rf"; cutoffs above $E_{{\mathrm{{ref}}}}$; $s={energy_scale:g}$ kJ/mol, "
        + rf"$\rho={tail_fraction:g}$; joint convergence pending",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.20, 1, 0.95))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figure_path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)

    for level_index, label in enumerate(complete):
        modes = [
            int(mode_count(torsions_by_level[label][:, index]))
            for index in range(len(TORSIONS))
        ]
        if np.isfinite(divergence[level_index]).any():
            js_text = ",".join(f"{value:.4f}" for value in divergence[level_index])
        else:
            js_text = "n/a"
        reference_js_text = ",".join(
            f"{value:.4f}" for value in level_divergence[level_index]
        )
        all_internal_text = (
            f"{np.nanmax(internal_divergence[level_index]):.4f}"
            if np.isfinite(internal_divergence[level_index]).any()
            else "n/a"
        )
        median_spread = (
            float(np.nanmax(median_excess[level_index]) - np.nanmin(median_excess[level_index]))
            if np.isfinite(median_excess[level_index]).any()
            else np.nan
        )
        log(
            f"plot {label}: N={torsions_by_level[label].shape[0]} "
            f"modes={modes} cross-seed-JS-bits=[{js_text}] "
            f"vs-{reference_label}-JS-bits=[{reference_js_text}] "
            f"joint12-JS={joint_divergence[level_index]:.4f} "
            f"rotamer3-JS={rotamer_divergence[level_index]:.4f} "
            f"all11-max-JS={all_internal_text} "
            f"median-excess-spread={median_spread:.3f} kJ/mol"
        )
    log(f"wrote {figure_path} from saved data {data_path}; plotting ran no dynamics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", choices=("sample", "audit", "replot"), default="sample"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--figure", type=Path, default=FIGURE_DEFAULT)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.data is None:
        args.data = (
            NEXT_DATA_DEFAULT if args.mode == "sample" else EXPLORATORY_DATA_DEFAULT
        )
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "replot":
        plot(Path(args.data), Path(args.figure))
    elif args.mode == "audit":
        audit(Path(args.data), Path(args.figure))
    else:
        sample(args)


if __name__ == "__main__":
    main()

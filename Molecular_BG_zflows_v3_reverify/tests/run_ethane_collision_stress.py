#!/usr/bin/env python
"""Immutable live-kernel collision evidence for the c/r regularizer family."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from openmm import XmlSerializer
from scipy.optimize import brentq
import torch

from zflows_md.forcefield import (
    Amber_Force_Field,
    ONE_4PI_EPS0,
    c_regularize_energy,
    softcap_energy,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / ".aris" / "experiments" / "regularization_comparison"
INPUTS = EVIDENCE / "inputs" / "ethane_vacuum_v1"
RUNS = EVIDENCE / "runs"

C_VALUE = 50.0
SCALE = 50.0
RHO = 0.0
REFERENCE_RADIUS_NM = 0.30
FLOORS_NM = (0.10, 0.20)
EXPECTED_PAIR = (2, 5)
GRID_POINTS = 401
RELATIVE_PROBES = (1e-2, 1e-4, 1e-6)
SLOPE_TOL = 1e-2
LIMIT_TOL = 1e-3
BOUNDARY_TOL = 1e-4


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


def _selected_pair(ff: Amber_Force_Field) -> dict:
    candidates = []
    for prefix in ("npair", "epair"):
        idx = getattr(ff, prefix)
        qq = getattr(ff, prefix + "_qq")
        sig = getattr(ff, prefix + "_sig")
        eps = getattr(ff, prefix + "_eps")
        for offset in range(len(idx)):
            if eps[offset] > 0 and sig[offset] > 0 and qq[offset] != 0:
                pair = tuple(sorted(map(int, idx[offset].tolist())))
                candidates.append((pair, prefix, offset))
    if not candidates:
        raise RuntimeError("frozen ethane force field has no active LJ+Coulomb pair")
    pair, prefix, offset = min(candidates)
    if pair != EXPECTED_PAIR:
        raise RuntimeError(f"selected pair {pair} != frozen expected {EXPECTED_PAIR}")
    return {
        "pair": pair,
        "prefix": prefix,
        "offset": offset,
        "qq_e2": float(getattr(ff, prefix + "_qq")[offset]),
        "sigma_nm": float(getattr(ff, prefix + "_sig")[offset]),
        "epsilon_kj_mol": float(getattr(ff, prefix + "_eps")[offset]),
    }


def _analytic_pair(r, qq, sigma, epsilon):
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * sr6 * (sr6 - 1.0) + ONE_4PI_EPS0 * qq / r


def _reference_energy(qq, sigma, epsilon):
    return _analytic_pair(REFERENCE_RADIUS_NM, qq, sigma, epsilon)


def _cap_radius(qq, sigma, epsilon, reference_energy):
    threshold = reference_energy + C_VALUE
    lower = sigma * 1e-6
    return brentq(
        lambda radius: _analytic_pair(radius, qq, sigma, epsilon) - threshold,
        lower,
        REFERENCE_RADIUS_NM,
    )


def _grid(dtype: np.dtype, sigma: float, cap_radius: float) -> np.ndarray:
    upper = max(2.0 * sigma, 0.50)
    base = np.geomspace(1e-6 * sigma, upper, GRID_POINTS, dtype=np.float64)
    special = [0.0, cap_radius, REFERENCE_RADIUS_NM]
    for center in (*FLOORS_NM, cap_radius):
        special.append(center)
        for relative in RELATIVE_PROBES:
            special.extend((center * (1.0 - relative), center * (1.0 + relative)))
        if dtype == np.float64:
            value = np.float64(center)
            special.extend(
                (
                    np.nextafter(value, np.float64(0.0)),
                    np.nextafter(value, np.float64(np.inf)),
                )
            )
        else:
            value = np.float32(center)
            special.extend(
                (
                    np.nextafter(value, np.float32(0.0), dtype=np.float32),
                    np.nextafter(value, np.float32(np.inf), dtype=np.float32),
                )
            )
    values = np.asarray([*base, *special], dtype=dtype)
    return np.unique(values)


def _pair_values(
    ff: Amber_Force_Field,
    radii_np: np.ndarray,
    *,
    pair: tuple[int, int],
    qq: float,
    sigma: float,
    epsilon: float,
    floor: float,
    mapping: str,
    reference_energy: float,
) -> tuple[np.ndarray, np.ndarray]:
    ff.set_r_floor(floor)
    dtype = ff.r_floor.dtype
    radii = torch.as_tensor(radii_np, dtype=dtype).detach().clone()
    radii.requires_grad_(True)
    selector = torch.zeros((1, ff.M, 3), dtype=dtype)
    selector[0, pair[1], 0] = 1.0
    positions = radii[:, None, None] * selector
    index = torch.tensor([pair], dtype=torch.long)
    raw = ff._pair_nb(
        positions,
        index,
        torch.tensor([qq], dtype=dtype),
        torch.tensor([sigma], dtype=dtype),
        torch.tensor([epsilon], dtype=dtype),
    )
    if mapping == "raw":
        energy = raw
    elif mapping in {"c", "c_r0"}:
        energy = c_regularize_energy(
            raw,
            torch.tensor(reference_energy, dtype=dtype),
            C_VALUE,
            SCALE,
            RHO,
        )
    elif mapping == "e":
        energy = softcap_energy(raw, reference_energy + C_VALUE, SCALE)
    else:
        raise ValueError(mapping)
    derivative = torch.autograd.grad(energy.sum(), radii)[0]
    return (
        energy.detach().cpu().numpy(),
        (-derivative).detach().cpu().numpy(),
    )


def _slope(radius, force, mask):
    selected = mask & np.isfinite(force) & (np.abs(force) > 0)
    return float(
        np.polyfit(
            np.log(radius[selected].astype(np.float64)),
            np.log(np.abs(force[selected]).astype(np.float64)),
            1,
        )[0]
    )


def _relative_error(actual, expected):
    return float(abs(actual - expected) / max(abs(expected), np.finfo(float).tiny))


def _classifications(array):
    return {
        "finite": int(np.isfinite(array).sum()),
        "nan": int(np.isnan(array).sum()),
        "posinf": int(np.isposinf(array).sum()),
        "neginf": int(np.isneginf(array).sum()),
    }


def _bitwise_constant(array: np.ndarray) -> bool:
    if len(array) == 0:
        return False
    contiguous = np.ascontiguousarray(array)
    integer = contiguous.view(np.uint64 if array.dtype == np.float64 else np.uint32)
    return bool(np.all(integer == integer[0]))


def _runtime() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def run() -> Path:
    started = time.time()
    input_manifest_hash = _sha256(INPUTS / "manifest.json")
    identity_payload = {
        "script": _sha256(Path(__file__)),
        "forcefield": _sha256(ROOT / "Molecular_BG_zflows/zflows_md/forcefield.py"),
        "input_manifest": input_manifest_hash,
        "parameters": {
            "c": C_VALUE,
            "scale": SCALE,
            "rho": RHO,
            "reference_radius_nm": REFERENCE_RADIUS_NM,
            "floors_nm": FLOORS_NM,
            "grid_points": GRID_POINTS,
            "relative_probes": RELATIVE_PROBES,
        },
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output = RUNS / f"collision-stress-ethane-{identity}"
    if output.exists():
        raise FileExistsError(f"immutable run already exists: {output}")
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        system = XmlSerializer.deserialize(
            (INPUTS / "system_vacuum.xml").read_text(encoding="utf-8")
        )
        ff64 = Amber_Force_Field(system, dtype=torch.float64, r_floor=0.0)
        pair_info = _selected_pair(ff64)
        pair = tuple(pair_info["pair"])
        actual = {
            "qq": pair_info["qq_e2"],
            "sigma": pair_info["sigma_nm"],
            "epsilon": pair_info["epsilon_kj_mol"],
        }
        cases = {
            "actual": actual,
            "lj": {**actual, "qq": 0.0},
            "coulomb_repulsive": {**actual, "epsilon": 0.0},
            "coulomb_attractive": {
                **actual,
                "qq": -abs(actual["qq"]),
                "epsilon": 0.0,
            },
        }
        references = {
            name: _reference_energy(
                values["qq"], values["sigma"], values["epsilon"]
            )
            for name, values in cases.items()
        }
        cap_radius = _cap_radius(**actual, reference_energy=references["actual"])
        arrays = {}
        summaries = {}
        checks = {}

        for torch_dtype, np_dtype, dtype_name in (
            (torch.float64, np.float64, "float64"),
            (torch.float32, np.float32, "float32"),
        ):
            ff = Amber_Force_Field(system, dtype=torch_dtype, r_floor=0.0)
            radii = _grid(np_dtype, actual["sigma"], cap_radius)
            arrays[f"radius_nm_{dtype_name}"] = radii
            for case_name, values in cases.items():
                reference = references[case_name]
                for floor in (0.0, *FLOORS_NM):
                    floor_tag = f"r{int(round(100 * floor)):03d}"
                    for mapping in ("raw", "c", "e"):
                        energy, force = _pair_values(
                            ff,
                            radii,
                            pair=pair,
                            qq=values["qq"],
                            sigma=values["sigma"],
                            epsilon=values["epsilon"],
                            floor=floor,
                            mapping=mapping,
                            reference_energy=reference,
                        )
                        stem = f"{dtype_name}_{case_name}_{mapping}_{floor_tag}"
                        arrays[stem + "_energy"] = energy
                        arrays[stem + "_force"] = force
                        summaries[stem] = {
                            "energy": _classifications(energy),
                            "force": _classifications(force),
                        }
                # A separately evaluated zero-floor family member must be
                # bitwise identical to pure c, not merely numerically close.
                energy, force = _pair_values(
                    ff,
                    radii,
                    pair=pair,
                    qq=values["qq"],
                    sigma=values["sigma"],
                    epsilon=values["epsilon"],
                    floor=0.0,
                    mapping="c_r0",
                    reference_energy=reference,
                )
                arrays[f"{dtype_name}_{case_name}_c_r0_energy"] = energy
                arrays[f"{dtype_name}_{case_name}_c_r0_force"] = force

            eps_dtype = np.finfo(np_dtype).eps
            for case_name in cases:
                for floor in (0.0, *FLOORS_NM):
                    floor_tag = f"r{int(round(100 * floor)):03d}"
                    c_energy = arrays[f"{dtype_name}_{case_name}_c_{floor_tag}_energy"]
                    e_energy = arrays[f"{dtype_name}_{case_name}_e_{floor_tag}_energy"]
                    c_force = arrays[f"{dtype_name}_{case_name}_c_{floor_tag}_force"]
                    e_force = arrays[f"{dtype_name}_{case_name}_e_{floor_tag}_force"]
                    finite = np.isfinite(c_energy) & np.isfinite(e_energy)
                    scale_bound = np.maximum.reduce(
                        [
                            np.ones(finite.sum()),
                            np.abs(c_energy[finite]).astype(np.float64),
                            np.abs(e_energy[finite]).astype(np.float64),
                            np.full(finite.sum(), abs(references[case_name])),
                            np.full(
                                finite.sum(),
                                abs(references[case_name] + C_VALUE),
                            ),
                        ]
                    )
                    difference = np.abs(
                        c_energy[finite].astype(np.float64)
                        - e_energy[finite].astype(np.float64)
                    )
                    key = f"{dtype_name}_{case_name}_{floor_tag}"
                    checks[key + "_matched_ce"] = bool(
                        np.array_equal(np.isfinite(c_energy), np.isfinite(e_energy))
                        and np.all(difference <= 8.0 * eps_dtype * scale_bound)
                    )
                    checks[key + "_matched_force_class"] = bool(
                        np.array_equal(np.isfinite(c_force), np.isfinite(e_force))
                    )
                checks[f"{dtype_name}_{case_name}_r0_identity"] = bool(
                    np.array_equal(
                        arrays[f"{dtype_name}_{case_name}_c_r000_energy"],
                        arrays[f"{dtype_name}_{case_name}_c_r0_energy"],
                        equal_nan=True,
                    )
                    and np.array_equal(
                        arrays[f"{dtype_name}_{case_name}_c_r000_force"],
                        arrays[f"{dtype_name}_{case_name}_c_r0_force"],
                        equal_nan=True,
                    )
                )

            for floor in FLOORS_NM:
                floor_tag = f"r{int(round(100 * floor)):03d}"
                below = radii < np_dtype(floor)
                energy = arrays[f"{dtype_name}_actual_c_{floor_tag}_energy"]
                force = arrays[f"{dtype_name}_actual_c_{floor_tag}_force"]
                checks[f"{dtype_name}_{floor_tag}_subfloor_constant"] = (
                    _bitwise_constant(energy[below])
                )
                checks[f"{dtype_name}_{floor_tag}_subfloor_zero_force"] = bool(
                    np.all(force[below] == 0)
                )
                checks[f"{dtype_name}_{floor_tag}_subfloor_finite"] = bool(
                    np.isfinite(energy[below]).all() and np.isfinite(force[below]).all()
                )

        r64 = arrays["radius_nm_float64"]
        window = (r64 >= 1e-5 * actual["sigma"]) & (
            r64 <= 1e-4 * actual["sigma"]
        )
        A = 4.0 * actual["epsilon"] * actual["sigma"] ** 12
        lj_raw_e = arrays["float64_lj_raw_r000_energy"]
        lj_raw_f = arrays["float64_lj_raw_r000_force"]
        lj_c_f = arrays["float64_lj_c_r000_force"]
        lj_e_f = arrays["float64_lj_e_r000_force"]
        finite_window = window & (r64 > 0)
        ratios = {
            "lj_raw_energy": r64[finite_window] ** 12
            * lj_raw_e[finite_window]
            / A,
            "lj_raw_force": r64[finite_window] ** 13
            * lj_raw_f[finite_window]
            / (12.0 * A),
            "lj_c_force": r64[finite_window]
            * lj_c_f[finite_window]
            / (12.0 * SCALE),
            "lj_e_force": r64[finite_window]
            * lj_e_f[finite_window]
            / (12.0 * SCALE),
        }
        for name, value in ratios.items():
            arrays["ratio_" + name] = value
            checks[name + "_limit"] = bool(
                np.max(np.abs(value.astype(np.float64) - 1.0)) <= LIMIT_TOL
            )
        slope_metrics = {
            "lj_raw": _slope(r64, lj_raw_f, window),
            "lj_c": _slope(r64, lj_c_f, window),
            "lj_e": _slope(r64, lj_e_f, window),
        }
        checks["lj_raw_slope"] = abs(slope_metrics["lj_raw"] + 13.0) <= SLOPE_TOL
        checks["lj_c_slope"] = abs(slope_metrics["lj_c"] + 1.0) <= SLOPE_TOL
        checks["lj_e_slope"] = abs(slope_metrics["lj_e"] + 1.0) <= SLOPE_TOL

        C = ONE_4PI_EPS0 * actual["qq"]
        rep_e = arrays["float64_coulomb_repulsive_raw_r000_energy"]
        rep_f = arrays["float64_coulomb_repulsive_raw_r000_force"]
        rep_c_f = arrays["float64_coulomb_repulsive_c_r000_force"]
        att_f = arrays["float64_coulomb_attractive_c_r000_force"]
        coulomb_ratios = {
            "repulsive_raw_energy": r64[finite_window] * rep_e[finite_window] / C,
            "repulsive_raw_force": r64[finite_window] ** 2
            * rep_f[finite_window]
            / C,
            "repulsive_c_force": r64[finite_window]
            * rep_c_f[finite_window]
            / SCALE,
        }
        for name, value in coulomb_ratios.items():
            arrays["ratio_" + name] = value
            checks[name + "_limit"] = bool(
                np.max(np.abs(value.astype(np.float64) - 1.0)) <= LIMIT_TOL
            )
        slope_metrics.update(
            {
                "repulsive_raw": _slope(r64, rep_f, window),
                "repulsive_c": _slope(r64, rep_c_f, window),
                "attractive_c": _slope(r64, att_f, window),
            }
        )
        checks["repulsive_raw_slope"] = (
            abs(slope_metrics["repulsive_raw"] + 2.0) <= SLOPE_TOL
        )
        checks["repulsive_c_slope"] = (
            abs(slope_metrics["repulsive_c"] + 1.0) <= SLOPE_TOL
        )
        checks["attractive_c_slope"] = (
            abs(slope_metrics["attractive_c"] + 2.0) <= SLOPE_TOL
        )

        boundary_metrics = {}
        for floor in FLOORS_NM:
            probe = np.asarray([floor * (1.0 + 1e-6)], dtype=np.float64)
            _, observed = _pair_values(
                ff64,
                probe,
                pair=pair,
                qq=actual["qq"],
                sigma=actual["sigma"],
                epsilon=actual["epsilon"],
                floor=floor,
                mapping="c",
                reference_energy=references["actual"],
            )
            V = _analytic_pair(floor, **actual)
            raw_force = (
                48.0 * actual["epsilon"] * actual["sigma"] ** 12 / floor**13
                - 24.0 * actual["epsilon"] * actual["sigma"] ** 6 / floor**7
                + ONE_4PI_EPS0 * actual["qq"] / floor**2
            )
            derivative = (
                1.0
                if V <= references["actual"] + C_VALUE
                else SCALE / (SCALE + V - references["actual"] - C_VALUE)
            )
            expected = derivative * raw_force
            error = _relative_error(float(observed[0]), expected)
            boundary_metrics[f"r{floor:.2f}_right_force_relative_error"] = error
            checks[f"r{floor:.2f}_right_force"] = error <= BOUNDARY_TOL

        cap_probes = np.asarray(
            [cap_radius * (1.0 - 1e-6), cap_radius * (1.0 + 1e-6)],
            dtype=np.float64,
        )
        _, cap_forces = _pair_values(
            ff64,
            cap_probes,
            pair=pair,
            qq=actual["qq"],
            sigma=actual["sigma"],
            epsilon=actual["epsilon"],
            floor=0.0,
            mapping="c",
            reference_energy=references["actual"],
        )
        cap_force_relative = _relative_error(cap_forces[0], cap_forces[1])
        boundary_metrics["cap_join_force_relative_difference"] = cap_force_relative
        checks["cap_join_force_continuity"] = cap_force_relative <= BOUNDARY_TOL

        join_offsets = np.asarray(
            [-1e-2, -1e-4, -1e-6, 0.0, 1e-6, 1e-4, 1e-2],
            dtype=np.float64,
        )
        threshold = references["actual"] + C_VALUE
        u = torch.tensor(threshold + join_offsets, dtype=torch.float64, requires_grad=True)
        c_join = c_regularize_energy(
            u, torch.tensor(references["actual"], dtype=torch.float64), C_VALUE, SCALE, RHO
        )
        c_join_grad = torch.autograd.grad(c_join.sum(), u, retain_graph=True)[0]
        e_join = softcap_energy(u, threshold, SCALE)
        e_join_grad = torch.autograd.grad(e_join.sum(), u)[0]
        arrays["energy_join_offsets_float64"] = join_offsets
        arrays["energy_join_c_value_float64"] = c_join.detach().numpy()
        arrays["energy_join_e_value_float64"] = e_join.detach().numpy()
        arrays["energy_join_c_derivative_float64"] = c_join_grad.detach().numpy()
        arrays["energy_join_e_derivative_float64"] = e_join_grad.detach().numpy()

        np.savez_compressed(partial / "collision_grid.npz", **arrays)

        # Plot only saved primary-case arrays; every quantitative gate comes
        # from the NPZ/JSON rather than visual judgment.
        positive = r64 > 0
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].loglog(
            r64[positive],
            np.abs(arrays["float64_actual_raw_r000_energy"][positive]),
            label="raw",
        )
        axes[0].loglog(
            r64[positive],
            np.abs(arrays["float64_actual_c_r000_energy"][positive]),
            label="c, r=0",
        )
        for floor, color in zip(FLOORS_NM, ("tab:orange", "tab:green")):
            tag = f"r{int(round(100 * floor)):03d}"
            axes[0].loglog(
                r64[positive],
                np.abs(arrays[f"float64_actual_c_{tag}_energy"][positive]),
                color=color,
                label=f"c/r, r={floor:.2f} nm",
            )
        axes[0].set(xlabel="pair separation (nm)", ylabel="|pair energy| (kJ/mol)")
        axes[0].legend(frameon=False)
        axes[1].loglog(
            r64[positive],
            np.abs(arrays["float64_actual_raw_r000_force"][positive]),
            label="raw",
        )
        axes[1].loglog(
            r64[positive],
            np.abs(arrays["float64_actual_c_r000_force"][positive]),
            label="c, r=0",
        )
        for floor, color in zip(FLOORS_NM, ("tab:orange", "tab:green")):
            tag = f"r{int(round(100 * floor)):03d}"
            axes[1].loglog(
                r64[positive],
                np.abs(arrays[f"float64_actual_c_{tag}_force"][positive]),
                color=color,
                label=f"c/r, r={floor:.2f} nm",
            )
        axes[1].set(xlabel="pair separation (nm)", ylabel="|radial force| (kJ/mol/nm)")
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(partial / "collision_stress.png", dpi=180)
        plt.close(figure)

        summary = {
            "schema_version": 1,
            "process_status": "complete",
            "scientific_gate_status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "slope_metrics": slope_metrics,
            "boundary_metrics": boundary_metrics,
            "pair": pair_info,
            "case_reference_energies_kj_mol": references,
            "cap_radius_nm": cap_radius,
            "parameters": identity_payload["parameters"],
            "classifications": summaries,
            "input_manifest_sha256": input_manifest_hash,
            "script_sha256": identity_payload["script"],
            "forcefield_sha256": identity_payload["forcefield"],
            "runtime": _runtime(),
            "wall_seconds": time.time() - started,
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
        partial.rename(output)
        return output
    except Exception as exc:
        _json(partial / "failure.json", {"process_status": "failed", "error": repr(exc)})
        failed = partial.with_name(partial.name.replace(".partial-", ".failed-"))
        partial.rename(failed)
        raise


if __name__ == "__main__":
    print(run())

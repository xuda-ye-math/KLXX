#!/usr/bin/env python
"""Convergence and mixing diagnostics for independent ADP REMD chains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BASINS = {
    "high_psi_negative_phi": lambda p, q: (p < 0) & (q > np.pi / 2),
    "alpha_r": lambda p, q: (p < 0) & (q > -np.pi / 2) & (q < 0),
    "bottom_negative_phi": lambda p, q: (p < 0) & (q < -np.pi / 2),
    "positive_phi": lambda p, q: p > 0,
}


def histogram_probability(phi: np.ndarray, psi: np.ndarray, bins: int) -> np.ndarray:
    counts, _, _ = np.histogram2d(
        phi, psi, bins=bins, range=((-np.pi, np.pi), (-np.pi, np.pi))
    )
    return counts.ravel() / counts.sum()


def js_bits(a: np.ndarray, b: np.ndarray) -> float:
    middle = 0.5 * (a + b)
    value = 0.0
    for distribution in (a, b):
        mask = distribution > 0
        value += 0.5 * np.sum(distribution[mask] * np.log2(distribution[mask] / middle[mask]))
    return float(value)


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def binary_ess(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    centered = x - x.mean()
    variance = np.dot(centered, centered) / n
    if variance == 0 or n < 4:
        return float(n), 1.0
    size = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(centered, size)
    autocov = np.fft.irfft(transform * transform.conjugate(), size)[:n]
    autocov /= np.arange(n, 0, -1)
    rho = autocov / autocov[0]
    tau = 1.0
    for lag in range(1, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
    tau = max(1.0, float(tau))
    return float(n / tau), tau


def wilson(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(center - half), float(center + half)


def mixing_diagnostics(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    history = data["walker_history"]
    nslots = history.shape[1]
    slot_of_walker = np.empty_like(history)
    for index, row in enumerate(history):
        slot_of_walker[index, row] = np.arange(nslots)
    bottom_max = max(0, int(np.ceil(0.2 * nslots)) - 1)
    top_min = int(np.floor(0.8 * nslots))
    round_trips = []
    coverage = []
    for trace in slot_of_walker.T:
        state = 0
        trips = 0
        for slot in trace:
            if state == 0 and slot <= bottom_max:
                state = 1
            elif state == 1 and slot >= top_min:
                state = 2
            elif state == 2 and slot <= bottom_max:
                trips += 1
                state = 1
        round_trips.append(trips)
        coverage.append(int(len(np.unique(trace))))
    rates = np.divide(
        data["exchange_accepts"], data["exchange_attempts"],
        out=np.zeros_like(data["exchange_accepts"], dtype=float),
        where=data["exchange_attempts"] > 0,
    )
    return {
        "exchange_rate_min": float(rates.min()),
        "exchange_rate_max": float(rates.max()),
        "exchange_rates": rates.tolist(),
        "walker_unique_slots_min": min(coverage),
        "walker_unique_slots": coverage,
        "round_trips_total": int(sum(round_trips)),
        "round_trips_min_per_walker": int(min(round_trips)),
        "round_trips_per_walker": round_trips,
    }


def load_chain(path: Path, burn_ns: float) -> dict[str, object]:
    with np.load(path) as data:
        if str(data["angle_units"]) != "radians":
            raise ValueError(f"{path}: expected radians")
        config = json.loads(str(data["config_json"]))
        times_ns = data["sample_steps"] * float(config["timestep_fs"]) / 1e6
        # A burn-in of B ns excludes the boundary frame at exactly B as well.
        keep = times_ns > burn_ns
        if not np.any(keep):
            raise ValueError(f"{path}: no samples remain after {burn_ns} ns burn-in")
        return {
            "path": str(path.resolve()),
            "phi": data["phi"][keep].copy(),
            "psi": data["psi"][keep].copy(),
            "n_total": int(len(data["phi"])),
            "n_postburn": int(keep.sum()),
            "completed_ns": float(int(data["completed_steps"]) * config["timestep_fs"] / 1e6),
            "mixing": mixing_diagnostics(data),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chains", nargs="+", type=Path)
    parser.add_argument("--burn-ns", type=float, default=2.0)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chains = [load_chain(path, args.burn_ns) for path in args.chains]

    reference = None
    if args.reference:
        with np.load(args.reference) as data:
            if "angle_units" in data.files:
                units = str(data["angle_units"])
            else:
                # Legacy Molecular_BG_2 MD caches store degrees without metadata.
                units = "degrees"
            if units == "degrees":
                reference = (np.deg2rad(data["phi"]), np.deg2rad(data["psi"]))
            elif units == "radians":
                reference = (data["phi"].copy(), data["psi"].copy())
            else:
                raise ValueError(f"{args.reference}: unknown angle units {units}")

    metrics: dict[str, object] = {
        "schema_version": 1,
        "burn_ns": args.burn_ns,
        "chains": [],
        "density_comparisons": {},
        "estimator": "raw periodic histogram; no pseudocount; JS base 2",
    }
    basin_rows = []
    for index, chain in enumerate(chains, 1):
        phi, psi = chain["phi"], chain["psi"]
        basin_result = {}
        for name, selector in BASINS.items():
            indicator = selector(phi, psi)
            count = int(indicator.sum())
            ess, tau = binary_ess(indicator)
            low, high = wilson(count, len(indicator))
            basin_result[name] = {
                "count": count, "probability": float(indicator.mean()),
                "wilson95": [low, high], "indicator_ess": ess, "tau_samples": tau,
            }
            basin_rows.append({
                "chain": index, "basin": name, "count": count, "n": len(indicator),
                "probability": float(indicator.mean()), "wilson_low": low,
                "wilson_high": high, "indicator_ess": ess, "tau_samples": tau,
            })
        metrics["chains"].append({
            "path": chain["path"], "n_total": chain["n_total"],
            "n_postburn": chain["n_postburn"], "completed_ns": chain["completed_ns"],
            "mixing": chain["mixing"], "basins": basin_result,
        })

    for bins in (24, 36, 48):
        distributions = [histogram_probability(c["phi"], c["psi"], bins) for c in chains]
        item: dict[str, object] = {}
        if len(distributions) >= 2:
            item["chain1_chain2_js_bits"] = js_bits(distributions[0], distributions[1])
            item["chain1_chain2_tv"] = total_variation(distributions[0], distributions[1])
        if reference is not None:
            ref_distribution = histogram_probability(reference[0], reference[1], bins)
            item["chain_reference_js_bits"] = [js_bits(d, ref_distribution) for d in distributions]
            item["chain_reference_tv"] = [total_variation(d, ref_distribution) for d in distributions]
        metrics["density_comparisons"][str(bins)] = item

    pooled_phi = np.concatenate([c["phi"] for c in chains])
    pooled_psi = np.concatenate([c["psi"] for c in chains])
    np.savez_compressed(
        args.output_dir / "pooled_postburn.npz",
        angle_units=np.asarray("radians"), phi=pooled_phi, psi=pooled_psi,
        burn_ns=np.asarray(args.burn_ns), source_chains=np.asarray([c["path"] for c in chains]),
    )
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with (args.output_dir / "basins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=basin_rows[0].keys())
        writer.writeheader()
        writer.writerows(basin_rows)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

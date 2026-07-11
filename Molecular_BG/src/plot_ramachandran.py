#!/usr/bin/env python
"""Plot FAB-style alanine-dipeptide Ramachandran density from saved angles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter


TICKS = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
TICK_LABELS = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]


def histogram(phi: np.ndarray, psi: np.ndarray, *, degrees: bool, bins: int) -> np.ndarray:
    """Return a [psi-bin, phi-bin] probability-density histogram."""
    phi = np.asarray(phi, dtype=float).reshape(-1)
    psi = np.asarray(psi, dtype=float).reshape(-1)
    if phi.shape != psi.shape:
        raise ValueError(f"phi and psi shapes differ: {phi.shape} != {psi.shape}")
    if not (np.isfinite(phi).all() and np.isfinite(psi).all()):
        raise ValueError("phi and psi must be finite")
    if degrees:
        phi, psi = np.deg2rad(phi), np.deg2rad(psi)
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    density, _, _ = np.histogram2d(phi, psi, bins=(edges, edges), density=True)
    return density.T


def plot(
    phi: np.ndarray,
    psi: np.ndarray,
    output: Path,
    *,
    degrees: bool,
    bins: int,
    smooth_sigma: float,
    title: str | None,
    vmin: float | None,
    vmax: float | None,
    mask_below_vmin: bool,
) -> None:
    if bins <= 1:
        raise ValueError("bins must be greater than one")
    density = histogram(phi, psi, degrees=degrees, bins=bins)
    if smooth_sigma < 0:
        raise ValueError("smooth_sigma must be nonnegative")
    if smooth_sigma > 0:
        density = gaussian_filter(density, smooth_sigma, mode="wrap")
    positive = density[density > 0]
    if positive.size == 0:
        raise ValueError("the histogram has no populated bins in [-pi, pi]^2")

    data_vmax = float(positive.max())
    vmax = data_vmax if vmax is None else vmax
    vmin = max(float(positive.min()), data_vmax * 1e-4) if vmin is None else vmin
    if not (0 < vmin < vmax):
        raise ValueError("color limits must satisfy 0 < vmin < vmax")

    # KDE produces small nonzero tails. Optionally mask tails below the plotted
    # range rather than clipping them to the darkest color as a thick rim.
    threshold = vmin if mask_below_vmin else 0.0
    shown = np.ma.masked_less(density, threshold) if mask_below_vmin else np.ma.masked_less_equal(density, 0.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        shown,
        origin="lower",
        extent=(-np.pi, np.pi, -np.pi, np.pi),
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xticks(TICKS, TICK_LABELS)
    ax.set_yticks(TICKS, TICK_LABELS)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    if title:
        ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("probability density")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("angles", type=Path, help="NPZ containing 1-D phi and psi arrays")
    parser.add_argument("-o", "--output", type=Path, default=Path("ramachandran.png"))
    units = parser.add_mutually_exclusive_group()
    units.add_argument("--radians", action="store_true", help="force input units to radians")
    units.add_argument("--degrees", action="store_true", help="force input units to degrees")
    parser.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="periodic Gaussian smoothing in histogram bins (default: 0)")
    parser.add_argument("--bins", type=int, default=100,
                        help="bins per axis (FAB evaluator: 64; paper final: 100)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="fixed positive lower LogNorm limit")
    parser.add_argument("--vmax", type=float, default=None,
                        help="fixed upper LogNorm limit")
    parser.add_argument("--title", default=None,
                        help="custom title; use an empty string for no title")
    parser.add_argument("--mask-below-vmin", action="store_true",
                        help="render density below vmin as white instead of clipping it")
    args = parser.parse_args()
    with np.load(args.angles) as data:
        if args.radians:
            degrees = False
        elif args.degrees:
            degrees = True
        elif "angle_units" in data.files:
            units_value = str(data["angle_units"])
            if units_value not in {"radians", "degrees"}:
                raise ValueError(f"unknown angle_units metadata: {units_value}")
            degrees = units_value == "degrees"
        else:
            raise ValueError("angle units are ambiguous; pass --radians or --degrees")
        n = len(data["phi"])
        completed = int(data["completed_steps"]) if "completed_steps" in data.files else None
        config = json.loads(str(data["config_json"])) if "config_json" in data.files else {}
        timestep = config.get("timestep_fs")
        ns = completed * timestep / 1e6 if completed is not None and timestep is not None else None
        title = args.title
        if title is None:
            title = f"OpenMM REMD, 300 K (N={n:,}" + (f", {ns:g} ns)" if ns is not None else ")")
            if args.smooth_sigma > 0:
                title += rf"; periodic $\sigma={args.smooth_sigma:g}$ bins"
        plot(data["phi"], data["psi"], args.output, degrees=degrees, bins=args.bins,
             smooth_sigma=args.smooth_sigma, title=title, vmin=args.vmin, vmax=args.vmax,
             mask_below_vmin=args.mask_below_vmin)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Rejuvenated inference sets of the data-driven flows and their Ramachandran figure.

Usage
    python ramachandran.py infer --method {kl,klx,klxm} [--run-dir DIR] [--samples 40000000] [--resample 10000000] [--mala-steps 100] [--seed 1]
    python ramachandran.py plot

`infer` loads artifacts/<method>_data_driven/flow.eqx, pushes --samples source
samples through it, weights them against the physical potential (the target
of the reference; the driver's regularized target only enters the training),
draws --resample rows by those weights, moves them by --mala-steps MALA steps
under the physical potential (MC_DT, MC_IMAGE_RADIUS of parameters.py), and
writes

    artifacts/<method>_data_driven/inference/samples.npy      the rejuvenated set, internal coordinates (float32)
    artifacts/<method>_data_driven/inference/inference.json   seed, sizes, screened ESS of the pushforward, MALA acceptance

The pushforward set itself is discarded. `plot` reads the stored sets only,
never the flows, and writes results/ramachandran.png: one row of four
Ramachandran free energy surfaces, the MD reference first, then forward KL,
KL+X_pi, and KL+X_pi+X_{(pi+nu_bar)/2}, on the conventions of
Codes/Molecular_BG/chiral/adp_60d (100 bins, wrapped smoothing, 11 kT scale).
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from jflows_md import Mixed_NSF, Molecular_Bundle, Molecular_Potential
from jflows_md.boltzmann import _push_and_weights
from jflows_md.utils.screen import compute_ESS_log
from jflows_md.utils.rejuvenation import mixed_mala

import parameters as P

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"

# Panel titles by method (the run name prefixes of train.py).
TITLES = {"kl": "forward KL", "klx": r"forward KL+$\mathrm{X}_\pi$",
          "klxm": r"forward KL+$\mathrm{X}_{(\pi+\bar{\nu})/2}$",
          "klxxt": r"forward KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\pi+\bar{\nu})/2}$"}

# Zero-based atom indices of bundle/reference.pdb for the backbone angles:
# phi = C(ACE)-N-CA-C and psi = N-CA-C-N(NME).
PHI_ATOMS = (4, 6, 8, 14)
PSI_ATOMS = (6, 8, 14, 16)

FREE_ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "reference_free_energy",
    ("#1b3569", "#286e86", "#369893", "#4aac8e", "#73c077", "#a6d656",
     "#dcdc47", "#ffd443", "#ffb450", "#ef894b", "#cc5338"),
)
FREE_ENERGY_CMAP.set_bad("white")
VMAX = 11.0
BLOCK = 1_000_000   # source samples pushed per block


def dihedral(positions, atoms):
    """Right-handed a-b-c-d dihedrals in radians, wrapped to (-pi, pi]."""
    a, b, c, d = (positions[:, index] for index in atoms)
    b0, b1, b2 = -(b - a), c - b, d - c
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    angle = np.arctan2(np.sum(np.cross(b1, v) * w, axis=1), np.sum(v * w, axis=1))
    return np.remainder(angle.astype(np.float64) + np.pi, 2.0 * np.pi) - np.pi


def weighted_histogram(phi_psi_chunks, bins):
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    counts = np.zeros((bins, bins))
    for phi, psi, weight in phi_psi_chunks:
        chunk, _, _ = np.histogram2d(phi, psi, bins=(edges, edges), weights=weight)
        counts += chunk
    return edges, counts


def free_energy_surface(counts, smoothing):
    density = gaussian_filter(counts, sigma=smoothing, mode="wrap")
    with np.errstate(divide="ignore"):
        free_energy = -np.log(density / density.max())
    free_energy[counts == 0] = np.nan
    return free_energy


def screened_weights(log_weight, fraction):
    """Normalized importance weights with the top ``fraction`` of the log weights screened to zero."""
    log_weight = np.asarray(log_weight, dtype=np.float64)
    finite = np.isfinite(log_weight)
    n_screen = int(np.ceil(fraction * log_weight.shape[0]))
    order = np.argsort(np.where(finite, log_weight, -np.inf))
    keep = np.zeros_like(finite)
    keep[order[: log_weight.shape[0] - n_screen]] = True
    keep &= finite
    weight = np.zeros_like(log_weight)
    weight[keep] = np.exp(log_weight[keep] - log_weight[keep].max())
    return weight / weight.sum(), int(keep.sum())


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def infer(args):
    run_dir = Path(args.run_dir).resolve() if args.run_dir else HERE / "artifacts" / f"{args.method}_data_driven"
    out_dir = run_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = Molecular_Bundle.load(BUNDLE)
    base = Molecular_Potential(bundle, temperature_kelvin=P.TEMPERATURE_KELVIN)
    source = base.source()
    domain = base.domain
    # The construction key selects the coupling masks, which flow.eqx does not
    # store: rebuild with the driver's flow key so the weights load onto the same masks.
    _, flow_key = jax.random.split(jax.random.key(P.SEED))
    flow = Mixed_NSF(
        flow_key, domain, bins=P.BINS, transforms=P.TRANSFORMS,
        euclidean_bound=P.NSF_LIM, hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE, mask_strategy="balanced",
    ).zeros()
    flow = eqx.tree_deserialise_leaves(run_dir / "flow.eqx", flow)
    log(f"flow {run_dir / 'flow.eqx'} loaded; physical potential at T={P.TEMPERATURE_KELVIN} K")

    # Pushforward in blocks of BLOCK source samples into a temporary
    # disk-backed array, with the log weights against the physical potential.
    started = time.perf_counter()
    temporary = out_dir / "pushforward.tmp.npy"
    y = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(args.samples, domain.dimension))
    log_weight = np.empty(args.samples, dtype=np.float32)
    keys = jax.random.split(jax.random.key(args.seed), -(-args.samples // BLOCK))
    for block, key in enumerate(keys):
        start, stop = block * BLOCK, min((block + 1) * BLOCK, args.samples)
        x = source.samples(key, N=stop - start)
        y_block, w_block = _push_and_weights(x, source, base, flow, domain, P.CHUNKS)
        y[start:stop] = np.asarray(jax.device_get(y_block), dtype=np.float32)
        log_weight[start:stop] = np.asarray(jax.device_get(w_block), dtype=np.float32)
        log(f"block {block + 1} of {len(keys)}: rows {start}:{stop} pushed, {time.perf_counter() - started:.1f}s")
    y.flush()
    ess = float(compute_ESS_log(jnp.asarray(log_weight), P.SCREEN_FRACTION))
    log(f"pushforward of {args.samples} source samples (seed {args.seed}) in {time.perf_counter() - started:.1f}s; "
        f"screened ESS={ess:.4f}; finite log weights {int(np.isfinite(log_weight).sum())}")

    # Multinomial resampling by the screened weights, then MALA under the
    # physical potential; the rows are read in index order from the pushforward.
    weight, kept = screened_weights(log_weight, P.SCREEN_FRACTION)
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(args.samples, size=args.resample, replace=True, p=weight))
    log(f"resampled {args.resample} rows ({np.unique(chosen).shape[0]} distinct) from {kept} screened rows")
    out = np.lib.format.open_memmap(out_dir / "samples.npy", mode="w+", dtype=np.float32, shape=(args.resample, domain.dimension))
    keys = jax.random.split(jax.random.key(args.seed + 1), -(-args.resample // args.chunk))
    accepted, started = 0.0, time.perf_counter()
    for block, key in enumerate(keys):
        start, stop = block * args.chunk, min((block + 1) * args.chunk, args.resample)
        rows = jnp.asarray(y[chosen[start:stop]])
        moved, history = mixed_mala(key, rows, base, domain, dt=P.MC_DT, steps=args.mala_steps,
                                    image_radius=P.MC_IMAGE_RADIUS, chunks=1)
        out[start:stop] = np.asarray(jax.block_until_ready(moved), dtype=np.float32)
        accepted += float(np.mean(np.asarray(history))) * (stop - start)
        if (block + 1) % 50 == 0 or block + 1 == len(keys):
            log(f"MALA {args.mala_steps} steps, dt={P.MC_DT}: rows {stop} of {args.resample}, "
                f"mean acceptance {accepted / stop:.3f}, {time.perf_counter() - started:.0f}s")
    out.flush()
    del y
    os.remove(temporary)
    (out_dir / "inference.json").write_text(json.dumps({
        "flow": str(run_dir / "flow.eqx"), "seed": args.seed,
        "pushforward_samples": args.samples, "screen_fraction": P.SCREEN_FRACTION, "screened_ess": ess,
        "resampled": args.resample, "mala_steps": args.mala_steps, "mala_dt": P.MC_DT,
        "mala_acceptance": accepted / args.resample,
    }, indent=2) + "\n")
    log(f"rejuvenated set written: {out_dir / 'samples.npy'} ({args.resample} rows); pushforward discarded")


def plot(args):
    bundle = Molecular_Bundle.load(BUNDLE)
    base = Molecular_Potential(bundle, temperature_kelvin=P.TEMPERATURE_KELVIN)
    cartesian = eqx.filter_jit(lambda q: base.cartesian(q))

    def reference_chunks():
        with h5py.File(HERE / P.REFERENCE, "r") as handle:
            frames = handle["coordinates"]
            for start in range(0, frames.shape[0], P.REFERENCE_CHUNK):
                positions = np.asarray(frames[start:start + P.REFERENCE_CHUNK], dtype=np.float64)
                yield dihedral(positions, PHI_ATOMS), dihedral(positions, PSI_ATOMS), np.ones(positions.shape[0])

    def sample_chunks(samples):
        for start in range(0, samples.shape[0], args.chunk):
            positions = np.asarray(cartesian(jnp.asarray(samples[start:start + args.chunk])))
            yield dihedral(positions, PHI_ATOMS), dihedral(positions, PSI_ATOMS), np.ones(positions.shape[0])

    edges, counts = weighted_histogram(reference_chunks(), args.bins)
    log(f"reference histogram: {int(counts.sum())} frames")
    panels = [("MD reference", free_energy_surface(counts, args.smoothing))]
    for method in ("kl", "klx", "klxxt"):
        path = HERE / "artifacts" / f"{method}_data_driven" / "inference" / "samples.npy"
        samples = np.load(path, mmap_mode="r")
        _, counts = weighted_histogram(sample_chunks(samples), args.bins)
        log(f"{method}: {int(counts.sum())} stored rejuvenated samples histogrammed")
        panels.append((TITLES[method], free_energy_surface(counts, args.smoothing)))

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 13, "axes.titlesize": 13,
        "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    })
    figure, axes = plt.subplots(1, 4, figsize=(15.0, 3.9), sharex=True, sharey=True, layout="constrained")
    for column, (axis, (title, free_energy)) in enumerate(zip(axes.flat, panels)):
        image = axis.pcolormesh(edges, edges, free_energy.T, cmap=FREE_ENERGY_CMAP,
                                vmin=0.0, vmax=VMAX, shading="flat", rasterized=True)
        axis.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), xlabel=r"$\phi$ (rad)", title=title)
        if column == 0:
            axis.set_ylabel(r"$\psi$ (rad)")
        axis.set_aspect("equal")
        axis.set_xticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
        axis.set_yticks((-np.pi, 0.0, np.pi), (r"$-\pi$", "0", r"$\pi$"))
    colorbar = figure.colorbar(image, ax=axes, pad=0.02, fraction=0.03, shrink=0.94)
    colorbar.set_ticks(np.arange(0.0, VMAX, 2.0))
    colorbar.set_label(r"free energy / $k_{\mathrm{B}}T$")
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    output = results / "ramachandran.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    log(f"figure written: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("infer", "plot"))
    parser.add_argument("--method", choices=tuple(TITLES), default="klx", help="infer: run name prefix artifacts/<method>_data_driven")
    parser.add_argument("--run-dir", default=None, help="infer: a run directory holding flow.eqx, in place of artifacts/<method>_data_driven")
    parser.add_argument("--samples", type=int, default=40_000_000, help="infer: pushforward samples")
    parser.add_argument("--resample", type=int, default=10_000_000, help="infer: rows of the rejuvenated set")
    parser.add_argument("--mala-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1, help="infer: source seed (the validation set used SEED)")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=20_000)
    args = parser.parse_args()
    infer(args) if args.command == "infer" else plot(args)


if __name__ == "__main__":
    main()

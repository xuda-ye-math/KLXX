"""Angle-density figure for the B=2000 klxx staged sampler — three curves,
all centered at theta = 0 on [-pi, pi):

    top panel      the single-site marginal p(theta_j) (six clock wells),
    bottom panel   the pair-difference densities p(theta_j - theta_l) at
                   lattice offsets (0,1), (1,1), (0,2), with a
                   ratio-preserving zoom on the [pi/6, pi/2] shoulder.

The densities are reconstructed from N = 2,000,000 fresh source draws
advanced through the trained klxx B=2000 ladder by the FULL staged sampler
(map -> reweight -> resample -> MALA at every rung). Both expensive
artifacts are saved and reused:

    rebuild_klxx_B2000_N2000000.npz          the 2M-sample staged rebuild (GPU, once)
    clock_marginals_B2000_N2000000.npz       the figure's density data

so reruns re-render the figure with no GPU work and no recomputation.

Run from the repo root:
    PYTHONPATH=/mnt/projects/jflows ~/.envs/jax/bin/python \
        Codes/Lattice_Clock/plot_clock_marginals.py
Writes clock_marginals.png next to this file; progress in
plot_clock_marginals_status.log.
"""

import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_klxx_B2000.npz"
FLOWS = HERE / "flows_klxx_B2000.eqx"
N_REBUILD = 2000000
REBUILD_SEED = 202
REBUILD = HERE / f"rebuild_klxx_B2000_N{N_REBUILD}.npz"
DENS = HERE / f"clock_marginals_B2000_N{N_REBUILD}.npz"
STATUS = HERE / "plot_clock_marginals_status.log"
BINS = 241
CHUNK_SIZE = 160000     # one compiled shape for the staged rebuild
BLOCK = 200000          # histogram accumulation block (memory control)
L = 8


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as fh:
        fh.write(line + "\n")


def wrap(a: np.ndarray) -> np.ndarray:
    return np.remainder(a + np.pi, 2.0 * np.pi) - np.pi


def staged_rebuild() -> np.ndarray:
    """N=2M fresh source draws advanced through the trained klxx B2000
    ladder by the full staged sampler (map -> reweight -> resample -> MALA
    at every rung); GPU."""
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import sys
    sys.path.insert(0, str(HERE))

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from jflows.flow import NCSF
    from jflows.potential import Nlog_Uniform, linear_combination
    from jflows.utils import langevin, resample
    from potential import Clock

    data = np.load(DATA)
    D, P = int(data["L"]) ** 2, int(data["P"])
    mc_step, mc_iters = float(data["mc_step"]), int(data["mc_iters"])
    ladder = [float(t) for t in data["ladder"]]
    u0 = Nlog_Uniform(a=[-math.pi] * D, b=[math.pi] * D)
    u1 = Clock(int(data["L"]), P, float(data["J"]), float(data["H"]))
    like = [NCSF(jax.random.key(0), a=[-math.pi] * D, b=[math.pi] * D,
                 bins=16, transforms=6, hidden_features=(256, 256)).zeros()
            for _ in range(len(ladder))]
    flows = eqx.tree_deserialise_leaves(FLOWS, like)
    assert len(flows) == len(ladder)

    key = jax.random.key(REBUILD_SEED)
    y = u0.samples(jax.random.fold_in(key, 0), N_REBUILD)
    t_prev = 0.0
    t0 = time.time()
    for k, (flow, t_k) in enumerate(zip(flows, ladder), start=1):
        u_prev = linear_combination([u1, u0], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([u1, u0], [t_k, 1.0 - t_k])
        key_k = jax.random.fold_in(key, k)
        outs, lws = [], []
        for i in range(0, N_REBUILD, CHUNK_SIZE):
            xb = y[i:i + CHUNK_SIZE]
            nb = xb.shape[0]
            if nb < CHUNK_SIZE:
                xb = jnp.concatenate(
                    [xb, jnp.broadcast_to(xb[-1:], (CHUNK_SIZE - nb, D))], axis=0)
            yt, ladj = flow.inv_and_ladj(xb)
            outs.append(yt[:nb])
            lws.append(u_prev(xb[:nb]) - u_k(yt[:nb]) + ladj[:nb])
        y_push = jnp.concatenate(outs)
        logw = jnp.concatenate(lws)
        del outs, lws
        y = resample(jax.random.fold_in(key_k, 1), y_push,
                     jnp.exp(logw - logw.max()))
        del y_push, logw
        rejuv = []
        for j, i in enumerate(range(0, N_REBUILD, CHUNK_SIZE)):
            xb = y[i:i + CHUNK_SIZE]
            nb = xb.shape[0]
            if nb < CHUNK_SIZE:
                xb = jnp.concatenate(
                    [xb, jnp.broadcast_to(xb[-1:], (CHUNK_SIZE - nb, D))], axis=0)
            rejuv.append(langevin(jax.random.fold_in(key_k, 100 + j), xb, u_k,
                                  step=mc_step, iters=mc_iters,
                                  adjust=True)[:nb])
        y = jnp.concatenate(rejuv)
        y = jax.block_until_ready(y)
        del rejuv
        log(f"rebuild stage {k}/{len(ladder)} (t={t_k:.4f}) done "
            f"[{time.time() - t0:.0f}s]")
        t_prev = t_k
    return np.asarray(y, dtype=np.float32)


def get_rebuild() -> np.ndarray:
    if REBUILD.exists():
        log(f"rebuild found: {REBUILD.name} (no GPU work)")
        return np.load(REBUILD)["y"]
    log(f"rebuild missing — running the staged sampler at N={N_REBUILD} ...")
    t0 = time.time()
    y = staged_rebuild()
    np.savez_compressed(REBUILD, y=y, seed=REBUILD_SEED)
    log(f"rebuild done in {time.time() - t0:.0f}s — saved {REBUILD.name}")
    return y


def densities() -> dict:
    """The three angle densities (computed once, then loaded)."""
    if DENS.exists():
        d = dict(np.load(DENS))
        if "distance2" in d and "sitemean" in d:
            log(f"densities found: {DENS.name} (no recomputation)")
            return d
        log(f"{DENS.name} lacks a curve — recomputing all curves from the "
            f"saved rebuild (no GPU)")
    y = get_rebuild()
    log(f"accumulating histograms over {y.shape[0]} samples ...")
    edges = np.linspace(-np.pi, np.pi, BINS + 1)
    counts = {k: np.zeros(BINS, dtype=np.int64)
              for k in ("marginal", "adjacent", "diagonal", "distance2",
                        "sitemean")}
    for i in range(0, y.shape[0], BLOCK):
        th = y[i:i + BLOCK].astype(np.float64).reshape(-1, L, L)
        blocks = {
            "marginal": th,
            "adjacent": np.concatenate([th - np.roll(th, 1, axis=1),
                                        th - np.roll(th, 1, axis=2)]),
            "diagonal": np.concatenate([th - np.roll(th, (1, 1), axis=(1, 2)),
                                        th - np.roll(th, (1, -1), axis=(1, 2))]),
            "distance2": np.concatenate([th - np.roll(th, 2, axis=1),
                                         th - np.roll(th, 2, axis=2)]),
            # site mean = circular mean over the 64 sites (the
            # magnetization phase; an arithmetic mean is ill-defined on
            # the torus)
            "sitemean": np.angle(np.exp(1j * th).mean(axis=(1, 2))),
        }
        for k, v in blocks.items():
            h, _ = np.histogram(wrap(v).ravel(), bins=edges)
            counts[k] += h
    width = edges[1] - edges[0]
    out = {k: c / (c.sum() * width) for k, c in counts.items()}
    out["centers"] = 0.5 * (edges[:-1] + edges[1:])
    out["n_samples"] = y.shape[0]
    out["L"] = L
    out["source"] = str(REBUILD.name)
    np.savez_compressed(DENS, **out)
    log(f"densities saved: {DENS.name}")
    return out


def main() -> None:
    open(STATUS, "a").close()
    d = densities()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "mathtext.fontset": "cm", "font.family": "serif",
    })

    COL = {"adjacent": "tab:red", "diagonal": "tab:green",
           "distance2": "tab:purple"}
    LBL = {"adjacent": r"$(0,1)$, $d=1.00$", "diagonal": r"$(1,1)$, $d=1.41$",
           "distance2": r"$(0,2)$, $d=2.00$"}
    UNIF = 1.0 / (2.0 * np.pi)
    TICKS = [-np.pi, -2 * np.pi / 3, -np.pi / 3, 0,
             np.pi / 3, 2 * np.pi / 3, np.pi]
    LABELS = [r"$-\pi$", r"$-2\pi/3$", r"$-\pi/3$", r"$0$",
              r"$\pi/3$", r"$2\pi/3$", r"$\pi$"]
    FIGW, FIGH = 6.5, 3.76

    c = d["centers"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(FIGW, FIGH), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.8],
                                            "hspace": 0.08})
    for k in range(-3, 4):
        a.axvline(k * np.pi / 3, color="0.88", lw=0.7, zorder=0)
        b.axvline(k * np.pi / 3, color="0.88", lw=0.7, zorder=0)

    # (top) single-site marginal: the six clock wells (the site-mean curve
    # is computed and saved in the density npz but not drawn — the figure
    # is the reviewed three-curve deliverable)
    a.plot(c, d["marginal"], color="tab:blue", lw=1.2)
    a.axhline(UNIF, ls="--", lw=0.8, color="gray")
    a.set_ylim(0.05, 0.30)
    a.set_ylabel("density")
    a.text(0.02, 0.88, r"single site $\theta_j$",
           transform=a.transAxes, fontsize=8.5)

    # (bottom) pair-difference densities with the pi/3 shoulder marked
    for k in ("adjacent", "diagonal", "distance2"):
        b.plot(c, d[k], color=COL[k], lw=1.2, label=LBL[k])
        for sgn in (-1, 1):
            j = int(np.argmin(np.abs(c - sgn * np.pi / 3)))
            b.plot(c[j], d[k][j], "o", ms=3.5, color=COL[k])
    b.axhline(UNIF, ls="--", lw=0.8, color="gray")
    b.set_ylim(0, 0.57)
    b.set_ylabel("density")
    b.set_xlabel("angle")
    b.text(0.02, 0.97, r"pair difference $\theta_j - \theta_l$",
           transform=b.transAxes, fontsize=8.5, va="top")
    b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.91), fontsize=8)
    b.set_xticks(TICKS)
    b.set_xticklabels(LABELS)
    b.set_xlim(-np.pi, np.pi)

    # ratio-preserving zoom on the shoulder region [pi/6, pi/2]: the inset
    # height follows from the panel's display geometry, so the horizontal :
    # vertical magnification matches the original figure exactly
    ZX0, ZX1, ZY0, ZY1 = np.pi / 6, np.pi / 2, 0.2, 0.3
    bb = b.get_position()
    W_in, H_in = bb.width * FIGW, bb.height * FIGH
    w_reg = (ZX1 - ZX0) / (2 * np.pi) * W_in
    h_reg = (ZY1 - ZY0) / 0.57 * H_in
    WF = 0.26
    mag = WF * W_in / w_reg
    HF = h_reg * mag / H_in
    zi = b.inset_axes([0.97 - WF, 0.94 - HF, WF, HF])
    for k in ("adjacent", "diagonal", "distance2"):
        zi.plot(c, d[k], color=COL[k], lw=1.1)
        j = int(np.argmin(np.abs(c - np.pi / 3)))
        zi.plot(c[j], d[k][j], "o", ms=3, color=COL[k])
    zi.axvline(np.pi / 3, color="0.85", lw=0.7, zorder=0)
    zi.set_xlim(ZX0, ZX1)
    zi.set_ylim(ZY0, ZY1)
    zi.set_xticks([np.pi / 6, np.pi / 3, np.pi / 2])
    zi.set_xticklabels([r"$\pi/6$", r"$\pi/3$", r"$\pi/2$"], fontsize=7)
    zi.set_yticks([0.2, 0.25, 0.3])
    zi.tick_params(labelsize=7)
    b.indicate_inset_zoom(zi, edgecolor="0.5", lw=0.7)

    out = HERE / "clock_marginals.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"DONE — saved {out}")


if __name__ == "__main__":
    main()

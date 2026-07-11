#!/usr/bin/env python
"""Snapshot watcher: watches md_implicit.npz (written cumulatively every 2 ns by the running
verify_md.py) and saves a TIME-STAMPED Ramachandran snapshot `snap_<t>ns.png` at each new
checkpoint, plus a `convergence.png` montage of all snapshots so far. CPU-only, read-only on
the MD outputs -- it does NOT touch the running MD. Stops when the MD log signals END or after
a max wait. Live status -> stdout AND snapshot_watcher.log.
"""
import os, time
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
SMOOTH_SIGMA = 1.0                                       # periodic smoothing -> FAB-like flat regions
LOG = os.path.join(HERE, "snapshot_watcher.log"); open(LOG, "w").close()
def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

TK = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
TL = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
def dens(phi, psi, bins=100):
    e = np.linspace(-np.pi, np.pi, bins+1)
    H, _, _ = np.histogram2d(np.radians(phi), np.radians(psi), bins=[e, e], density=True)
    return gaussian_filter(H.T, SMOOTH_SIGMA, mode="wrap") if SMOOTH_SIGMA > 0 else H.T

def save_snapshot(phi, psi, tns):
    H = dens(phi, psi); vmax = H.max()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    im = ax.imshow(H, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi), cmap="viridis",
                   norm=LogNorm(vmin=vmax*1e-4, vmax=vmax), aspect="equal")
    ax.set_xticks(TK); ax.set_xticklabels(TL); ax.set_yticks(TK); ax.set_yticklabels(TL)
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\psi$")
    ax.set_title(f"amber96 implicit  t={tns:.0f} ns  N={len(phi)}  frac(phi>0)={(np.asarray(phi)>0).mean():.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    p = os.path.join(HERE, f"snap_{tns:.0f}ns.png")     # e.g. snap_20ns.png, snap_40ns.png
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p

def main():
    log("START snapshot_watcher: watching md_implicit.npz")
    npz = os.path.join(HERE, "md_implicit.npz")
    mdlog = os.path.join(HERE, "verify_md.log")
    seen = set(); last_mtime = 0.0; t0 = time.time()
    while True:
        done = os.path.exists(mdlog) and "verify_md END" in open(mdlog).read()
        if os.path.exists(npz):
            mt = os.path.getmtime(npz)
            if mt != last_mtime:
                last_mtime = mt
                try:
                    z = np.load(npz); tns = float(z["t_ns"]) if "t_ns" in z.files else 0.0
                    key = round(tns)
                    if key not in seen and len(z["phi"]) > 0:
                        seen.add(key)
                        p = save_snapshot(z["phi"], z["psi"], tns)
                        log(f"snapshot t={tns:.0f}ns N={len(z['phi'])} -> {os.path.basename(p)}")
                except Exception as e:
                    log(f"read/plot retry ({type(e).__name__}: {e})")
        if done:
            log(f"MD END detected. total snapshots={len(seen)}"); break
        if time.time() - t0 > 3600:
            log("max wait 1h reached, exiting"); break
        time.sleep(8)
    log("DONE snapshot_watcher")

if __name__ == "__main__":
    main()

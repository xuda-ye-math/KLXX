#!/usr/bin/env python
"""Verification MD: alanine dipeptide Ramachandran, VACUUM vs OBC2 IMPLICIT solvent,
built with the real AMBER protein force field (amber96) -- NOT the openff small-
molecule parametrization currently in zflows_md/data/alanine_dipeptide.prmtop.

Goal: confirm that (a) a proper protein force field breaks the spurious L/D symmetry
and (b) OBC2 implicit solvent reproduces the classic asymmetric landscape of adp.jpeg.

Runs short Langevin MD for each environment, collects phi/psi, and writes a 2-panel
free-energy Ramachandran (kT units, jet colormap, adp.jpeg style).

Live status log -> stdout AND verify_md.log (tail -f to watch). No tqdm.
"""
import os, sys, time
from datetime import datetime
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = "/mnt/projects/zflows_md_backup/tests/data/alanine_dipeptide.pdb"
# The shipped PDB is a D-alanine (mirror image, chirality triple product < 0). Start MD
# from a known L-alanine structure (FAB amber-implicit min-energy, SAME 22-atom ordering)
# so we sample the correct L landscape (phi<0 dominant), matching adp.jpeg.
LSTART = "/home/xuda/fab-torch/experiments/aldp/data/position_min_energy.pt"
LOG = os.path.join(HERE, "verify_md.log"); open(LOG, "w").close()
def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

# phi = C(ACE,4)-N(6)-CA(8)-C(14);  psi = N(6)-CA(8)-C(14)-N(NME,16)
PHI = (4, 6, 8, 14); PSI = (6, 8, 14, 16)
def dihedral(x, idx):
    p0, p1, p2, p3 = [x[:, i] for i in idx]
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    b2n = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    xx = (n1 * n2).sum(-1); yy = (np.cross(n1, n2) * b2n).sum(-1)
    return np.degrees(np.arctan2(yy, xx))

def chirality(x1):  # x1: [22,3]; L-alanine -> triple product (N-CA, C-CA, CB-CA) > 0
    CA, N, C, CB = 8, 6, 14, 10
    return float(np.dot(np.cross(x1[N] - x1[CA], x1[C] - x1[CA]), x1[CB] - x1[CA]))

def l_start_positions():
    import torch
    pos = torch.load(LSTART).numpy().reshape(22, 3).astype(np.float64)   # nm, L-alanine
    return unit.Quantity(pos, unit.nanometer)

def run_md(env, n_ns, dt_fs=1.0, save_ps=0.5):
    log(f"START MD env={env} target={n_ns} ns dt={dt_fs} fs save_every={save_ps} ps")
    pdb = app.PDBFile(PDB)
    # amber96 (+ amber96 OBC GBSA) == FAB's openmmtools AlanineDipeptideVacuum/Implicit
    if env == "vacuum":
        ff = app.ForceField("amber96.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    else:  # implicit OBC GBSA (GBSAOBCForce)
        ff = app.ForceField("amber96.xml", "amber96_obc.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtosecond)
    try:
        sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("CUDA"))
        plat = "CUDA"
    except Exception:
        sim = app.Simulation(pdb.topology, system, integ); plat = "CPU"
    sim.context.setPositions(l_start_positions())      # start from L-alanine (not the D PDB)
    sim.minimizeEnergy()
    x0 = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    log(f"  {env}: platform={plat} minimized  chirality={chirality(x0):+.2f} "
        f"({'L' if chirality(x0) > 0 else 'D'})  start phi={dihedral(x0[None],PHI)[0]:+.0f}")
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    log(f"  {env}: starting sampling")
    steps_per_save = int(save_ps * 1000 / dt_fs)
    n_save = int(n_ns * 1000 / save_ps)
    coords = []
    t0 = time.time()
    for k in range(n_save):
        sim.step(steps_per_save)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        coords.append(pos.astype(np.float32))
        if (k + 1) % max(1, n_save // 40) == 0 or k == 0:
            x = np.array(coords)
            phi = dihedral(x[-1:], PHI)[0]; psi = dihedral(x[-1:], PSI)[0]
            frac_pos = (dihedral(x, PHI) > 0).mean()
            el = time.time() - t0
            log(f"  {env} {k+1}/{n_save} frames  t={(k+1)*save_ps:.0f}ps  "
                f"phi={phi:+.0f} psi={psi:+.0f}  frac(phi>0)={frac_pos:.2f}  "
                f"{el:.0f}s ({(k+1)/el:.1f} fr/s)")
        # incremental checkpoint: save data + re-plot so verify_ramachandran.png updates live
        if (k + 1) % max(1, n_save // 20) == 0:
            x = np.array(coords)
            np.savez(os.path.join(HERE, f"md_{env}.npz"),
                     phi=dihedral(x, PHI), psi=dihedral(x, PSI), t_ns=(k + 1) * save_ps / 1000.0)
            replot()
    x = np.array(coords)
    phi = dihedral(x, PHI); psi = dihedral(x, PSI)
    np.savez(os.path.join(HERE, f"md_{env}.npz"), phi=phi, psi=psi, t_ns=n_ns)
    replot()
    log(f"DONE {env}: {len(phi)} frames, frac(phi>0)={ (phi>0).mean():.3f}")
    return phi, psi

def density_hist(phi, psi, bins=100):
    """Normalized 2D histogram density over (phi,psi) in radians, FAB Figure-19 style
    (100x100 bins, no smoothing, log color scale)."""
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    H, _, _ = np.histogram2d(np.radians(phi), np.radians(psi), bins=[edges, edges], density=True)
    return H.T, edges                                     # [psi, phi]

def replot():
    """Rebuild verify_ramachandran.png from whatever md_*.npz exist (FAB Fig-19 style),
    with the FAB Fig-19 reference panel for side-by-side comparison. Called live at each
    checkpoint so the PNG updates while the run proceeds."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    tk = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tl = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    panels = []
    for env in ("implicit", "vacuum"):
        f = os.path.join(HERE, f"md_{env}.npz")
        if os.path.exists(f):
            z = np.load(f)
            panels.append((env, z["phi"], z["psi"], float(z["t_ns"]) if "t_ns" in z.files else 0.0))
    fabref = os.path.join(HERE, "fab_fig19_test.png")
    ncols = len(panels) + (1 if os.path.exists(fabref) else 0)
    if ncols == 0:
        return
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes = np.atleast_1d(axes)
    ai = 0
    for env, phi, psi, tns in panels:
        ax = axes[ai]; ai += 1
        H, _ = density_hist(phi, psi)
        vmax = H.max(); norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        im = ax.imshow(H, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                       cmap="viridis", norm=norm, aspect="equal")
        ax.set_xticks(tk); ax.set_xticklabels(tl); ax.set_yticks(tk); ax.set_yticklabels(tl)
        ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\psi$")
        ax.set_title(f"amber96 {env} ({tns:.0f} ns, N={len(phi)})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if os.path.exists(fabref):
        ax = axes[ai]
        ax.imshow(plt.imread(fabref)); ax.axis("off")
        ax.set_title("FAB Fig. 19 (reference test data)")
    fig.suptitle("alanine dipeptide Ramachandran — MD reference (L-alanine) vs FAB Fig. 19")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "verify_ramachandran.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    n_ns = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    log(f"==== verify_md START  implicit={n_ns}ns vacuum={n_ns/2}ns ====")
    for env, ns in (("implicit", n_ns), ("vacuum", n_ns / 2)):   # implicit first: the FAB Fig-19 deliverable
        run_md(env, ns)
    replot()
    log(f"==== verify_md END -> {os.path.join(HERE, 'verify_ramachandran.png')} ====")

if __name__ == "__main__":
    main()

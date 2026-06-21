#!/usr/bin/env python
"""Alanine dipeptide (ADP) L- and D-form structures, for the paper.

The classical Amber force field is achiral (mirror-symmetric), so the L- and D-enantiomers of alanine
dipeptide are isoenergetic and the internal-coordinate Boltzmann generator emits both. This renders the
energy-minimized L-form and the D-form (a genuine 3D reflection of it) as depth-sorted ball-and-stick
structures (parmed topology bonds, ASE CPK colours / covalent radii), each from its own viewpoint. The
alpha carbon (the chiral centre) is marked with a halo and stereo bonds -- a SOLID wedge to the substituent
toward the viewer, a HASHED wedge to the one behind, plain lines for the two in-plane -- so the opposite
handedness (R vs S) is explicit in each panel even though the two molecules are shown at different angles.
Saves conformer_L.png, conformer_D.png and a combined conformers.png at 400 dpi, matching the
glycerol / diethanolamine conformer figures.

Usage: python conformer_figure.py [--elev_l 18 --azim_l 210 --elev_d 18 --azim_d 330]
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import parmed as pmd
import openmm as mm
from openmm import app, unit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm", "font.family": "serif"})   # match dihedrals.png

HERE = os.path.dirname(os.path.abspath(__file__))                 # this script's own dir == the molecule folder
REPO = os.path.dirname(HERE)                                      # parent dir holds tests/data with the prmtop/rst7
GOLD = "#E8990C"


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def minimized_positions(prmtop, crd):
    """Energy-minimized Cartesian (Angstrom) + atomic numbers + topology bonds (the L-form)."""
    s = pmd.load_file(prmtop, xyz=crd)
    system = s.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond)
    sim = app.Simulation(s.topology, system, integ, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(s.positions)
    sim.minimizeEnergy()
    pos = (sim.context.getState(getPositions=True).getPositions(asNumpy=True)
           .value_in_unit(unit.angstrom))
    Z = np.array([a.atomic_number for a in s.atoms])
    bonds = [(b.atom1.idx, b.atom2.idx) for b in s.bonds]
    return np.asarray(pos, float), Z, bonds, s


def find_calpha(Z, bonds):
    """The alanine alpha carbon: a C bonded to exactly {N, carbonyl-C, methyl-C, H}.
    Returns (ca, [n, carbonyl_c, methyl_c, h]) ordered, or None."""
    n = len(Z)
    adj = {i: [] for i in range(n)}
    for i, j in bonds:
        adj[i].append(j); adj[j].append(i)
    for c in range(n):
        if Z[c] != 6 or len(adj[c]) != 4:
            continue
        nbr = adj[c]
        if sorted(Z[k] for k in nbr) != [1, 6, 6, 7]:
            continue
        n_idx = next(k for k in nbr if Z[k] == 7)
        h_idx = next(k for k in nbr if Z[k] == 1)
        carbons = [k for k in nbr if Z[k] == 6]
        co = next((k for k in carbons if any(Z[m] == 8 for m in adj[k])), None)        # carbonyl C (has O)
        me = next((k for k in carbons if sum(Z[m] == 1 for m in adj[k]) >= 3), None)    # methyl C (>=3 H)
        if co is not None and me is not None:
            return c, [n_idx, co, me, h_idx]
    return None


def view_matrix(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry


LIGHT = np.array([-0.42, 0.50, 0.76]); LIGHT = LIGHT / np.linalg.norm(LIGHT)   # upper-left, toward viewer


def _sphere(color, depth, n=80):
    """RGBA image of a shaded sphere (diffuse + specular highlight + depth dimming, soft edge)."""
    g = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(g, g)
    rr = np.hypot(xx, yy)
    nz = np.sqrt(np.clip(1 - xx ** 2 - yy ** 2, 0, 1))
    diff = np.clip(xx * LIGHT[0] + yy * LIGHT[1] + nz * LIGHT[2], 0, 1)
    shade = (0.40 + 0.60 * diff) * depth
    rgb = np.clip(np.asarray(color)[None, None, :] * shade[..., None] + 0.75 * (diff ** 16)[..., None], 0, 1)
    alpha = np.clip((1.0 - rr) / 0.045, 0, 1)           # antialiased circular edge
    return np.dstack([rgb, alpha])


def _wedge(ax, x0, y0, x1, y1, filled, halo):
    """Stereo bond from the chiral centre (x0,y0) to a substituent (x1,y1): SOLID filled wedge if the
    substituent is toward the viewer, HASHED wedge if behind."""
    d = np.array([x1 - x0, y1 - y0]); ln = float(np.hypot(*d))
    if ln < 1e-6:
        return
    u = d / ln; perp = np.array([-u[1], u[0]]); w = 0.15        # half-width at the substituent end
    if filled:
        tri = np.array([[x0, y0], [x1, y1] + perp * w, [x1, y1] - perp * w])
        ax.add_patch(plt.Polygon(tri, closed=True, facecolor=GOLD, edgecolor="white",
                                 lw=1.0, zorder=60, joinstyle="round"))
    else:
        for t in np.linspace(0.30, 1.0, 5):                    # hashed wedge: rungs of growing width
            p = np.array([x0, y0]) + d * t; a = p + perp * w * t; b = p - perp * w * t
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GOLD, lw=2.2, solid_capstyle="round",
                    zorder=60, path_effects=halo)


def draw(ax, Z, P, bonds, V, cen, L, chiral=None):
    """Depth-sorted ball-and-stick (shaded CPK spheres + half-coloured sticks), common centre + half-range.
    If `chiral=(ca, [n,co,me,h])`, the four bonds at the alpha carbon are drawn as stereo bonds (wedge/hash)
    and the centre haloed, so the handedness is explicit."""
    q = P @ V.T
    x, y, zc = q[:, 0], q[:, 1], q[:, 2]
    rad = 0.18 + 0.30 * covalent_radii[Z]              # ball radius (Angstrom)
    lo, span = zc.min(), (zc.max() - zc.min()) or 1.0
    dep = lambda z: 0.50 + 0.50 * (z - lo) / span      # depth dim: rear darker
    ca_bonds = set()
    if chiral is not None:
        ca, nbrs = chiral
        ca_bonds = {frozenset((ca, nb)) for nb in nbrs}
    for i, j in bonds:                                 # half-coloured sticks (skip the alpha-C bonds: stereo below)
        if frozenset((i, j)) in ca_bonds:
            continue
        xm, ym, zo = 0.5 * (x[i] + x[j]), 0.5 * (y[i] + y[j]), 0.5 * (zc[i] + zc[j])
        for k, (xe, ye) in ((i, (xm, ym)), (j, (xm, ym))):
            col = np.clip(np.asarray(jmol_colors[Z[k]]) * dep(zc[k]), 0, 1)
            ax.plot([x[k], xe], [y[k], ye], color=col, lw=6.0, solid_capstyle="round",
                    zorder=2 + 2 * (zo - lo) / span)
    for idx in np.argsort(zc):                         # spheres back-to-front
        r = rad[idx]
        ax.imshow(_sphere(jmol_colors[Z[idx]], dep(zc[idx])), extent=(x[idx] - r, x[idx] + r,
                  y[idx] - r, y[idx] + r), origin="lower", interpolation="bilinear",
                  zorder=6 + 4 * (zc[idx] - lo) / span)
    if chiral is not None:                             # stereo marker at the chiral centre
        ca, nbrs = chiral
        halo = [pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()]
        order = sorted(nbrs, key=lambda nb: zc[nb] - zc[ca])   # back -> ... -> front (by depth vs Ca)
        back, front, plane = order[0], order[-1], order[1:3]
        for nb in plane:                              # two in-plane bonds: plain dark lines
            ax.plot([x[ca], x[nb]], [y[ca], y[nb]], color="0.2", lw=3.2, solid_capstyle="round",
                    zorder=59, path_effects=halo)
        _wedge(ax, x[ca], y[ca], x[front], y[front], filled=True, halo=halo)   # toward viewer
        _wedge(ax, x[ca], y[ca], x[back], y[back], filled=False, halo=halo)    # behind
        ring = plt.Circle((x[ca], y[ca]), rad[ca] * 1.45, fill=False, edgecolor=GOLD, lw=2.6,
                          zorder=61, path_effects=halo)
        ax.add_patch(ring)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(cen[0] - L, cen[0] + L); ax.set_ylim(cen[1] - L, cen[1] + L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="adp")
    ap.add_argument("--d", type=int, default=60)
    ap.add_argument("--prmtop", default="alanine_dipeptide")
    ap.add_argument("--elev_l", type=float, default=18.0)
    ap.add_argument("--azim_l", type=float, default=210.0)
    ap.add_argument("--elev_d", type=float, default=18.0)
    ap.add_argument("--azim_d", type=float, default=330.0)
    a = ap.parse_args()

    folder = HERE                                      # run in-place: the script's own dir is the molecule folder
    import shutil
    try:                                               # snapshot the generator into the molecule folder (like run.py)
        shutil.copy(os.path.abspath(__file__), os.path.join(folder, os.path.basename(__file__)))
    except shutil.SameFileError:
        pass
    prm = os.path.join(REPO, "tests", "data", f"{a.prmtop}.prmtop")
    crd = os.path.join(REPO, "tests", "data", f"{a.prmtop}.rst7")
    log(f"START adp L/D conformer figure  L(elev={a.elev_l},azim={a.azim_l}) D(elev={a.elev_d},azim={a.azim_d})")
    P0, Z, bonds, struct = minimized_positions(prm, crd)
    ca = find_calpha(Z, bonds)
    log(f"  minimized: {len(Z)} atoms, {len(bonds)} bonds; alpha-C = atom {ca[0] if ca else None}")

    P0 = P0 - P0.mean(0)                               # centre on centroid
    REFLECT = np.array([-1.0, 1.0, 1.0])               # genuine 3D reflection -> the D enantiomer (a real structure)
    PANELS = [
        ("L", "L-form", P0,           view_matrix(a.elev_l, a.azim_l)),
        ("D", "D-form", P0 * REFLECT, view_matrix(a.elev_d, a.azim_d)),
    ]
    rmax = 0.18 + 0.30 * float(covalent_radii[Z].max())
    projs = [(P @ Vv.T)[:, :2] for _, _, P, Vv in PANELS]
    cens = [pr.mean(0) for pr in projs]
    L = max(np.abs(pr - c).max() for pr, c in zip(projs, cens)) + rmax + 0.05   # common scale -> equal molecule size

    for (tag, label, P, Vv), c in zip(PANELS, cens):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        draw(ax, Z, P, bonds, Vv, c, L, chiral=ca)
        fig.tight_layout(pad=0.05)
        out = os.path.join(folder, f"conformer_{tag}.png")
        fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)
        log(f"  wrote {out}")

    # combined 1x2 panel; per-panel width stays 2.8 (= glycerol/dea 8.4/3) so bond/atom proportions match.
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.5))
    for ax, (tag, label, P, Vv), c in zip(axes, PANELS, cens):
        draw(ax, Z, P, bonds, Vv, c, L, chiral=ca)
        ax.set_title(label, fontsize=13, pad=6)                            # enantiomer name above
    ELEM = {6: "C", 7: "N", 8: "O", 1: "H"}            # legend = elements present in the molecule
    present = [z for z in (6, 7, 8, 1) if z in set(int(x) for x in Z)]
    leg = [Line2D([0], [0], marker="o", linestyle="none", markersize=12, markeredgecolor="0.3",
                  markerfacecolor=jmol_colors[z], label=ELEM[z]) for z in present]
    leg.append(Line2D([0], [0], marker="o", linestyle="none", markersize=12, markerfacecolor="none",
                      markeredgecolor=GOLD, markeredgewidth=2.0, label=r"chiral C$_\alpha$"))
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), frameon=False, fontsize=12,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.5, 0.05))
    fig.suptitle(r"alanine dipeptide — L and D enantiomers", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    out = os.path.join(folder, "conformers.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    log(f"  wrote {out}  (combined)")
    log("DONE")


if __name__ == "__main__":
    main()

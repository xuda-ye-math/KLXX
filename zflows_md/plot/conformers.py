#!/usr/bin/env python
"""Typical glycerol configurations at different torsion states, for the paper.

Takes the energy-minimized molecule and sets ONE backbone torsion (default O-C-C-O) to its three
rotamer states -- gauche- (-60 deg), trans (180 deg), gauche+ (+60 deg) -- by a rigid rotation of the
c-side group about the central b-c bond, so only that torsion changes (the rest of the molecule is held
in its minimized geometry). Each state is drawn as a depth-sorted ball-and-stick (parmed topology bonds,
ASE CPK colours/covalent radii) from a common viewpoint aligned on the b-c bond, so the rotamer is the
visible change. Saves conformer_<state>.png and a combined conformers.png at 400 dpi.

Usage: python conformers.py [--name glycerol --d 36 --torsion O-C-C-O --elev 14 --azim -64]
"""
import argparse
import os
import sys
from collections import deque

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

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Molecular_BG")
from zflows_md.dihedral import proper_torsions          # named proper torsions from the prmtop

STATES = [(-60.0, r"gauche$^-$"), (180.0, "trans"), (60.0, r"gauche$^+$")]


def minimized_positions(prmtop, crd):
    """Energy-minimized Cartesian (Angstrom) + atomic numbers + topology bonds."""
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


def signed_dihedral(p, a, b, c, d):
    """Right-handed dihedral a-b-c-d (deg): the signed angle from the a-direction to the d-direction about
    the b->c axis. Consistent with a +axis (right-hand) rotation of the d-side AND with the project's
    torsion convention (dihedral.dihedral), so a set angle matches the marginal plots."""
    A, B, C, D = p[a], p[b], p[c], p[d]
    axis = C - B; axis = axis / np.linalg.norm(axis)
    u1 = (A - B) - ((A - B) @ axis) * axis; u1 = u1 / np.linalg.norm(u1)
    u2 = (D - C) - ((D - C) @ axis) * axis; u2 = u2 / np.linalg.norm(u2)
    return np.degrees(np.arctan2(np.cross(u1, u2) @ axis, u1 @ u2))


def cside_atoms(bonds, n, b, c):
    """Atoms on c's side of the b-c bond (the rigid group that rotates), via BFS with b-c removed."""
    adj = {i: set() for i in range(n)}
    for i, j in bonds:
        adj[i].add(j); adj[j].add(i)
    adj[b].discard(c); adj[c].discard(b)               # cut the rotatable bond
    seen, q = {c}, deque([c])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); q.append(v)
    return sorted(seen)


def set_torsion(p, a, b, c, d, target_deg, rot):
    """Rotate the rot atoms about the b-c axis so dihedral(a,b,c,d) == target_deg."""
    p = p.copy()
    axis = p[c] - p[b]; axis = axis / np.linalg.norm(axis)
    dtheta = np.radians(target_deg - signed_dihedral(p, a, b, c, d))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(dtheta) * K + (1 - np.cos(dtheta)) * (K @ K)   # Rodrigues
    p[rot] = (p[rot] - p[b]) @ R.T + p[b]
    return p


def align_to_bond(p, a, b, c):
    """Express p in a frame where b is at the origin, c on +x, and a in the +xy half-plane,
    so every state is viewed identically and the rotamer (d swinging about +x) is the change."""
    p = p - p[b]
    x = p[c] / np.linalg.norm(p[c])
    va = p[a] - (p[a] @ x) * x
    y = va / np.linalg.norm(va)
    z = np.cross(x, y)
    return p @ np.stack([x, y, z]).T


GOLD = "#E8990C"


def view_matrix(elev, azim):
    """azim tilts the central bond (x, the dihedral axis) toward the viewer so the torsion arc opens up;
    elev adds a vertical tilt for 3D depth."""
    e, a = np.radians(elev), np.radians(azim)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry


def dihedral_marker(ax, P, quad, V):
    """Mark the a-b-c-d dihedral: a gold arc (with dashed rays + degree label) swept about the central
    b-c bond, in the plane perpendicular to it -- so the torsion is unmistakable."""
    a, b, c, d = quad
    A, B, C, D = P[a], P[b], P[c], P[d]
    axis = C - B; axis = axis / np.linalg.norm(axis)
    M = 0.5 * (B + C)
    u1 = (A - B) - ((A - B) @ axis) * axis; u1 = u1 / np.linalg.norm(u1)
    u2 = (D - C) - ((D - C) @ axis) * axis; u2 = u2 / np.linalg.norm(u2)
    w = np.cross(axis, u1)                              # right-hand perp (CCW from u1 about the axis)
    ang = np.radians(signed_dihedral(P, a, b, c, d))    # matches set_torsion + the title
    halo = [pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()]   # white outline -> pops over atoms
    r = 0.92
    t = np.linspace(0, ang, 56)
    arc = (M + r * (np.cos(t)[:, None] * u1 + np.sin(t)[:, None] * w)) @ V.T
    ax.plot(arc[:, 0], arc[:, 1], color=GOLD, lw=3.0, zorder=60, solid_capstyle="round", path_effects=halo)
    for u in (u1, u2):                                  # dashed rays from the bond out to the arc ends
        p = np.vstack([M, M + r * u]) @ V.T
        ax.plot(p[:, 0], p[:, 1], color=GOLD, lw=1.9, ls=(0, (3, 2)), zorder=60, path_effects=halo)
    lp = (M + 1.5 * r * (np.cos(ang / 2) * u1 + np.sin(ang / 2) * w)) @ V.T   # label pushed outside the arc
    val = int(round(np.degrees(ang))); val = 180 if val == -180 else val      # -180 == +180 (trans)
    ax.text(lp[0], lp[1], rf"$\phi \,\approx\, {val}^\circ$", color=GOLD,
            fontsize=13, fontweight="bold", ha="center", va="center", zorder=61,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])


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


def draw(ax, Z, P, bonds, V, cen, L, quad):
    """Depth-sorted ball-and-stick: shaded spheres (CPK) + half-coloured sticks + the dihedral marker,
    common centre+half-range so the panels are comparable."""
    q = P @ V.T
    x, y, zc = q[:, 0], q[:, 1], q[:, 2]
    rad = 0.18 + 0.30 * covalent_radii[Z]              # ball radius (Angstrom) -- slightly smaller atoms
    lo, span = zc.min(), (zc.max() - zc.min()) or 1.0
    dep = lambda z: 0.50 + 0.50 * (z - lo) / span      # depth dim: rear darker
    for i, j in bonds:                                 # thinner sticks behind, half-coloured, depth-dimmed
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
    dihedral_marker(ax, P, quad, V)                    # gold arc + degree label on the torsion
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(cen[0] - L, cen[0] + L); ax.set_ylim(cen[1] - L, cen[1] + L)   # common centre + zoom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="glycerol")
    ap.add_argument("--d", type=int, default=36)
    ap.add_argument("--torsion", default="O-C-C-O")
    ap.add_argument("--elev", type=float, default=14.0)
    ap.add_argument("--azim", type=float, default=72.0)   # near-Newman: opens the dihedral arcs
    a = ap.parse_args()

    folder = os.path.join(REPO, f"{a.name}_{a.d}d")
    import shutil; shutil.copy(os.path.abspath(__file__), os.path.join(folder, os.path.basename(__file__)))  # snapshot into the molecule folder (like run.py)
    prm = os.path.join(REPO, "tests", "data", f"{a.name}.prmtop")
    crd = os.path.join(REPO, "tests", "data", f"{a.name}.rst7")
    P0, Z, bonds, struct = minimized_positions(prm, crd)

    proper = proper_torsions(prm)
    names = ["-".join(t[2]) for t in proper]
    quad = [t[0] for t in proper][names.index(a.torsion)]   # first match
    ai, bi, ci, di = quad
    rot = cside_atoms(bonds, len(Z), bi, ci)
    V = view_matrix(a.elev, a.azim)
    print(f"[{a.name}] torsion {a.torsion} = atoms {quad}; rotating {len(rot)} atoms; "
          f"view elev={a.elev} azim={a.azim}", flush=True)

    tagmap = {-60.0: "gauche_minus", 180.0: "trans", 60.0: "gauche_plus"}
    # build all three states first, then a COMMON zoom (all share the b atom at the origin)
    confs = [(align_to_bond(set_torsion(P0, ai, bi, ci, di, t, rot), ai, bi, ci), t, lab)
             for t, lab in STATES]
    rmax = 0.18 + 0.30 * float(covalent_radii[Z].max())
    centers = [(P @ V.T)[:, :2].mean(0) for P, _, _ in confs]   # per-conformer centre -> each molecule is centred
    L = max(np.abs((P @ V.T)[:, :2] - c).max() for (P, _, _), c in zip(confs, centers)) + rmax + 0.05  # common scale

    for (P, target, label), c in zip(confs, centers):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        draw(ax, Z, P, bonds, V, c, L, (ai, bi, ci, di))
        fig.tight_layout(pad=0.05)
        out = os.path.join(folder, f"conformer_{tagmap[target]}.png")
        fig.savefig(out, dpi=600, bbox_inches="tight"); plt.close(fig)
        struct.coordinates = P                                 # aligned coords -> PDB for the PyMOL renderer
        struct.save(os.path.join(folder, f"conformer_{tagmap[target]}.pdb"), overwrite=True)
        print(f"  wrote {out}", flush=True)

    # combined 1x3 panel for the paper, with a C/O/H element legend
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.0))
    for ax, (P, target, label), c in zip(axes, confs, centers):
        draw(ax, Z, P, bonds, V, c, L, (ai, bi, ci, di))
        ax.set_title(label, fontsize=13, pad=6)                            # state name above
        ax.text(0.5, -0.03, rf"{a.torsion} $\,\approx\, {int(target)}^\circ$", transform=ax.transAxes,
                ha="center", va="top", fontsize=12)                        # named dihedral + value below
    ELEM = {6: "C", 7: "N", 8: "O", 1: "H"}                           # legend = elements present in the molecule
    present = [z for z in (6, 7, 8, 1) if z in set(int(x) for x in Z)]
    leg = [Line2D([0], [0], marker="o", linestyle="none", markersize=12, markeredgecolor="0.3",
                  markerfacecolor=jmol_colors[z], label=ELEM[z]) for z in present]
    fig.legend(handles=leg, loc="lower center", ncol=len(leg), frameon=False, fontsize=12,
               handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(rf"{a.name} — backbone {a.torsion} rotamers", y=1.0, fontsize=13)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    out = os.path.join(folder, "conformers.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}  (combined)", flush=True)


if __name__ == "__main__":
    main()

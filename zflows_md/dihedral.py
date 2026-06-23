#!/usr/bin/env python
"""Multimodal-torsion scan: short MD of candidate molecules, rank every proper torsion by
how clearly multimodal it is (>=2 comparable, well-separated wells). Picks the example for
the dihedral-marginal figure -- the trans-dominated alkane backbone was too trivial (unimodal).

Metric per torsion: histogram (36 bins over 360 deg) -> peaks -> 'balance' = 2nd-mode / 1st-mode
population (1.0 = perfectly balanced bimodal; ~0.25 = trans-dominated alkane), plus #modes.

Platform default CPU (GPU busy with the training campaign). Pass --platform CUDA after it ends.
"""
import argparse
import os
import numpy as np
import parmed
import openmm as mm
from openmm import app, unit

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Molecular_BG")
CANDIDATES = {
    "phenol":     ("tests/data/phenol.prmtop",     "tests/data/phenol.rst7"),
    "glycerol":   ("tests/data/glycerol.prmtop",   "tests/data/glycerol.rst7"),
    "morpholine": ("tests/data/morpholine.prmtop", "tests/data/morpholine.rst7"),
}


def run_md(prm_path, rst_path, n_frames, stride, platform):
    prm = app.AmberPrmtopFile(prm_path)
    crd = app.AmberInpcrdFile(rst_path)
    system = prm.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                        1.0 * unit.femtosecond)
    sim = app.Simulation(prm.topology, system, integ, mm.Platform.getPlatformByName(platform))
    sim.context.setPositions(crd.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.step(20000)
    fr = []
    for _ in range(n_frames):
        sim.step(stride)
        fr.append(sim.context.getState(getPositions=True).getPositions(asNumpy=True)
                  .value_in_unit(unit.nanometer))
    return np.array(fr)


def proper_torsions(prm_path):
    p = parmed.load_file(prm_path)
    out, seen = [], set()
    for d in p.dihedrals:
        if d.improper:
            continue
        cb = tuple(sorted((d.atom2.idx, d.atom3.idx)))     # one torsion per central bond
        if cb in seen:
            continue
        seen.add(cb)
        atoms = (d.atom1, d.atom2, d.atom3, d.atom4)
        out.append((tuple(x.idx for x in atoms),
                    tuple(x.name for x in atoms),
                    tuple(x.element_name for x in atoms)))
    return out


def dihedral(fr, quad):
    p0, p1, p2, p3 = (fr[:, a] for a in quad)
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    b2n = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.degrees(np.arctan2((np.cross(n1, n2) * b2n).sum(-1), (n1 * n2).sum(-1)))


def multimodality(phi):
    h, _ = np.histogram(phi, bins=36, range=(-180, 180))
    h = h.astype(float)
    if h.sum() == 0:
        return 0.0, 0
    peaks = [i for i in range(36)
             if h[i] >= h[(i - 1) % 36] and h[i] >= h[(i + 1) % 36] and h[i] > 0.08 * h.max()]
    pops = sorted((h[i] for i in peaks), reverse=True)
    nmode = sum(1 for p in pops if p > 0.15 * pops[0]) if pops else 0
    balance = pops[1] / pops[0] if len(pops) >= 2 else 0.0
    return balance, nmode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--n_frames", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=1000)       # 2000*1000*1fs = 2 ns/molecule
    a = ap.parse_args()
    print(f"multimodal scan  platform={a.platform}  "
          f"{a.n_frames}x{a.stride}fs = {a.n_frames * a.stride * 1e-6:.1f} ns/molecule", flush=True)
    best = []
    for name, (prm, rst) in CANDIDATES.items():
        print(f"\n=== {name} ===", flush=True)
        fr = run_md(os.path.join(ROOT, prm), os.path.join(ROOT, rst), a.n_frames, a.stride, a.platform)
        print(f"  MD done: {fr.shape[0]} frames", flush=True)
        ranked = []
        for quad, names, elems in proper_torsions(os.path.join(ROOT, prm)):
            phi = dihedral(fr, quad)
            bal, nmode = multimodality(phi)
            ranked.append((bal, nmode, names, elems, phi))
        ranked.sort(key=lambda x: -x[0])
        for bal, nmode, names, elems, _ in ranked[:4]:
            print(f"  {'-'.join(elems):<12} ({'-'.join(names)}):  modes={nmode}  balance={bal:.2f}", flush=True)
        if ranked:
            best.append((ranked[0][0], name, ranked[0]))
    best.sort(key=lambda x: -x[0])
    print("\n=== RANKING (clearest multimodal first) ===", flush=True)
    for bal, name, r in best:
        print(f"  {name:<12} {'-'.join(r[3])}  balance={bal:.2f}  modes={r[1]}", flush=True)
    if best:
        bal, name, r = best[0]
        np.save(os.path.join(ROOT, f"multimodal_winner_{name}.npy"), r[4])
        print(f"\n  WINNER: {name}  torsion {'-'.join(r[3])}  (balance={bal:.2f})  -> saved phi", flush=True)


if __name__ == "__main__":
    main()

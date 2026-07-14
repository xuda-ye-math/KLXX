#!/usr/bin/env python
"""HARD GATE: the torch Amber_Force_Field (with the new OBC2 GB term) must reproduce
OpenMM's amber96 + amber96_obc (GBSAOBCForce) IMPLICIT-solvent energy to < 1e-2 kJ/mol,
per force term AND in total, on physical alanine-dipeptide frames.

Do NOT train / run ASMC until this passes. Builds the implicit system exactly as
verify_md.py (the validated MD reference), draws physical frames by short implicit MD
from the L-alanine min structure, then compares torch energy_terms() against OpenMM
per-force-group energies (Reference platform, double precision).

Live status log -> stdout AND validate_obc.log (tail -f to watch). No tqdm.
"""
import os, sys, time
from datetime import datetime
import numpy as np
import torch
import openmm as mm
import openmm.app as app
from openmm import unit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zflows_md.forcefield import Amber_Force_Field

# External fixed assets (same as verify_md.py): amber96 ADP topology + L-alanine min config.
# Resolved against candidates so a move of fab-torch under /mnt/projects doesn't break it.
def _first_existing(cands, what):
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{what} not found in any of: {cands}")

PDB = _first_existing([
    "/mnt/projects/zflows_md_backup/tests/data/alanine_dipeptide.pdb",
    "/home/xuda/zflows_md_backup/tests/data/alanine_dipeptide.pdb",
], "ADP PDB")
LSTART = _first_existing([
    "/mnt/projects/fab-torch/experiments/aldp/data/position_min_energy.pt",
    "/home/xuda/fab-torch/experiments/aldp/data/position_min_energy.pt",
], "L-alanine position_min_energy.pt")
N_FRAMES = 64          # physical frames to test
MD_STRIDE = 200        # MD steps between saved frames
TOL = 1e-2             # kJ/mol hard gate on |dE_total|

LOG = os.path.join(HERE, "validate_obc.log"); open(LOG, "w").close()
def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

# torch term -> OpenMM force class it must match
TERM_OF = {"bond": "HarmonicBondForce", "angle": "HarmonicAngleForce",
           "torsion": "PeriodicTorsionForce", "nonbonded": "NonbondedForce",
           "gb": "GBSAOBCForce"}


def build_implicit():
    """amber96 + amber96_obc on the ADP PDB topology -> (system, positions_L[nm])."""
    pdb = app.PDBFile(PDB)
    ff = app.ForceField("amber96.xml", "amber96_obc.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    posL = torch.load(LSTART, weights_only=False).numpy().reshape(-1, 3).astype(np.float64)
    return pdb, system, posL


def main():
    t0 = time.time()
    log(f"START validate_obc  PDB={os.path.basename(PDB)}  tol={TOL} kJ/mol  device=Reference(double)")
    pdb, system, posL = build_implicit()
    M = system.getNumParticles()
    forces = {f.__class__.__name__: f for f in system.getForces()}
    log(f"  M={M} atoms  forces={list(forces.keys())}")

    # torch force field in float64 (isolate formula correctness from float32 roundoff)
    ff = Amber_Force_Field(system, dtype=torch.float64)
    log(f"  torch Amber_Force_Field built  has_gb={ff.has_gb}  gb_pre={getattr(ff,'gb_pre',None):.6f}"
        f"  gb_sa_coeff={getattr(ff,'gb_sa_coeff',None):.6f}")

    # one force-group per force -> read per-term OpenMM energies
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
    group_of = {f.__class__.__name__: i for i, f in enumerate(system.getForces())}

    # --- generate physical frames: minimize from L, then short implicit MD (CUDA if available) ---
    integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 1*unit.femtosecond)
    try:
        sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("CUDA"))
        plat = "CUDA"
    except Exception:
        sim = app.Simulation(pdb.topology, system, integ); plat = "CPU"
    sim.context.setPositions(posL * unit.nanometer)
    sim.minimizeEnergy()
    minpos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    sim.context.setVelocitiesToTemperature(300*unit.kelvin)
    log(f"  MD platform={plat}: collecting {N_FRAMES} frames (stride {MD_STRIDE}) ...")
    frames = [minpos.astype(np.float64)]
    for k in range(N_FRAMES - 1):
        sim.step(MD_STRIDE)
        frames.append(sim.context.getState(getPositions=True)
                      .getPositions(asNumpy=True).value_in_unit(unit.nanometer).astype(np.float64))
    frames = np.array(frames)                                   # [N_FRAMES, M, 3] nm
    log(f"  collected {len(frames)} frames")

    # --- OpenMM reference energies (Reference platform, double) ---
    ctx = mm.Context(system, mm.VerletIntegrator(1*unit.femtosecond), mm.Platform.getPlatformByName("Reference"))
    def omm_terms(pos):
        ctx.setPositions(pos * unit.nanometer)
        out = {}
        for name, gi in group_of.items():
            out[name] = ctx.getState(getEnergy=True, groups={gi}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        out["total"] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return out

    # --- torch energies (batched, float64) ---
    xb = torch.tensor(frames, dtype=torch.float64)
    with torch.no_grad():
        tt = ff.energy_terms(xb)
    torch_terms = {k: v.cpu().numpy() for k, v in tt.items()}

    # --- compare per term + total ---
    log("  --- per-term max|dE| over frames (torch vs OpenMM), kJ/mol ---")
    max_term = {}
    for term, omm_name in TERM_OF.items():
        d = np.array([omm_terms(frames[i])[omm_name] for i in range(len(frames))]) - torch_terms[term]
        max_term[term] = float(np.abs(d).max())
        log(f"    {term:10s} vs {omm_name:18s} max|dE|={max_term[term]:.3e}  mean|dE|={float(np.abs(d).mean()):.3e}")
    d_tot = np.array([omm_terms(frames[i])["total"] for i in range(len(frames))]) - torch_terms["total"]
    max_tot = float(np.abs(d_tot).max())
    log(f"    {'TOTAL':10s} {'':21s} max|dE|={max_tot:.3e}  mean|dE|={float(np.abs(d_tot).mean()):.3e}")

    # --- GB energy at min config (reference sanity) ---
    gb_min_omm = omm_terms(frames[0])["GBSAOBCForce"]
    gb_min_torch = float(torch_terms["gb"][0])
    log(f"  E_GB(min) OpenMM={gb_min_omm:.4f}  torch={gb_min_torch:.4f}  dE={abs(gb_min_omm-gb_min_torch):.3e} kJ/mol")

    # --- float32 spot check (production dtype used by ASMC) ---
    ff32 = Amber_Force_Field(system, dtype=torch.float32)
    with torch.no_grad():
        tot32 = ff32.energy_terms(torch.tensor(frames, dtype=torch.float32))["total"].cpu().numpy().astype(np.float64)
    d32 = np.array([omm_terms(frames[i])["total"] for i in range(len(frames))]) - tot32
    log(f"  float32 total max|dE|={float(np.abs(d32).max()):.3e}  (production ASMC dtype)")

    ok = max_tot < TOL
    log(f"##### {'PASS' if ok else 'FAIL'}: total max|dE|={max_tot:.3e} kJ/mol  (gate {TOL})  wall={time.time()-t0:.0f}s #####")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

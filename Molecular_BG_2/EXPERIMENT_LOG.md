# ADP Ramachandran — recreate FAB Fig. 19 (persistent log)

Goal: locally reproduce the FAB paper's alanine-dipeptide Ramachandran (arXiv 2208.01893,
Fig. 19 = ground-truth MD test data). Reference image copied here: `fab_fig19_test.png`.

## Root-cause diagnosis (why the project's `adp_60d` plot looked wrong)

Three independent bugs, each verified:

1. **Wrong force field in the prmtop.** `zflows_md/data/alanine_dipeptide.prmtop` has atom
   types `C1,C2,N1,O1,H1,H2,H3` — these are **openff-2.1.0 (small-molecule) types**, not AMBER
   protein types (`CT,C,N,O,H,HC`). openff has no protein-backbone torsion terms, so the
   Ramachandran comes out ~symmetric and unlike FAB. (`gen_molecules.py` only builds the
   hetero-atom molecules via GAFF; it never generated ADP.)
2. **Shipped PDB is the D-enantiomer (mirror image).** `alanine_dipeptide.pdb` chirality
   triple product = **−2.62**; a known-L structure (FAB `position_min_energy.pt`) = **+2.56**.
   So MD from the PDB samples φ>0 (mirror of FAB). Fix: start from the L structure.
   (Calibration also confirmed our φ-sign convention is correct: known-L gives φ≈−146°.)
3. **Vacuum, not implicit solvent.** FAB Fig. 19 is implicit solvent (amber96 + OBC) at 300 K
   (`fab_buff.yaml`: `env: implicit`, `temperature: 300`). Vacuum keeps a large C7ax (φ>0)
   population; solvent suppresses it, giving the asymmetric φ<0-dominant landscape.

## Correct setup (this folder)

- `verify_md.py`: OpenMM Langevin MD, **amber96 + amber96_obc** (== FAB
  `openmmtools.AlanineDipeptideImplicit`), 300 K, no constraints, 1 fs.
- Start coordinates: FAB `position_min_energy.pt` (L-alanine, same 22-atom ordering).
- φ = atoms(4,6,8,14), ψ = atoms(6,8,14,16). Plots log-density viridis, φ,ψ∈[−π,π],
  100×100 bins — FAB Fig. 19 style. Incremental replot every 2 ns → `verify_ramachandran.png`.
- Outputs: `md_implicit.npz`, `md_vacuum.npz`, `verify_ramachandran.png`, live log `verify_md.log`.

## Reference-code facts gathered (for the package-integration route)

- **GBSAOBCForce params** (amber96_obc): ε_solvent=78.3, ε_solute=1.0,
  surfaceAreaEnergy=2.25936 kJ/mol/nm²; per-particle (charge, radius, scale) extracted;
  min-energy E_GB ≈ −50.36 kJ/mol (validation target for a torch OBC term).
- **Regularization (boltzgen vs zflows_md).** boltzgen `regularize_energy` works on
  energy **already divided by kBT** (reduced units): log-compress above `energy_cut`
  (`log(E−cut+1)+cut`), hard-cap at `energy_max`, non-finite→NaN. Defaults 1e8/1e20 kT are
  essentially "never active except pathological clashes." zflows `softcap_energy` works on
  **raw kJ/mol before β** (e_cap=100 kJ/mol ≈ 40 kT — far more aggressive) and adds a
  distance-level `r_floor` soft-core (no boltzgen analog). Same *shape* (linear→log, C¹), not
  numerically interchangeable; convert via `e_cap[kT]=e_cap[kJ/mol]/(R·T)`.
- **NaN/singular handling in FAB.** No prevention: boltzgen maps non-finite energy→NaN;
  `fab/train.py` skips the **entire optimizer step** if the batch loss is NaN/Inf
  ("nan loss encountered") or grad-norm non-finite, plus grad clipping. No per-sample
  filtering (`torch.mean`/`logsumexp` over the batch). zflows prevents the singularity at the
  source (r_floor) so no step is ever skipped — better when clash samples are frequent.

## Run history (what was launched / stopped, and why — honest record)

- smoke 0.1–0.15 ns: proved GPU works (~42k MD steps/s on the 5090) and exposed bug #2/#3.
- amber14SB implicit 40 ns (killed ~20 ns): switched FF to amber96 to match FAB exactly.
- amber96 implicit 40 ns (killed ~18 ns): added incremental replot for live observability.
- amber96 implicit 40 ns + vacuum 20 ns (**current, running, NOT to be killed**): task b4n91axn3.
  At 4 ns the φ<0 basin structure already matches FAB Fig. 19.

Mistake to not repeat: killing/relaunching for edits that could have waited; no persistent
log until now. This file is that log — update it, don't restart from scratch.

## Status / next

- [in progress] amber96 implicit 40 ns MD reference → `verify_ramachandran.png`.
- Known hard mode: the **dim αL basin at φ≈+60°** (FAB notes it as "very dim, φ≈1"). A single
  300 K walker may under-sample it; if so, the exact-Boltzmann all-modes route is the project's
  **annealed SMC** on an implicit-solvent torch potential (needs an OBC term added to
  `Amber_Force_Field`, validated against E_GB above).

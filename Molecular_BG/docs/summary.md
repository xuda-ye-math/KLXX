# Alanine-dipeptide reference and local OpenMM comparison

## Outcome

The root `adp_fab.png` is the current reference: a fresh unsmoothed 100 x 100
histogram of exactly 1,000,000 frames from the FAB authors' ground-truth REMD
training split, with a fixed logarithmic color range of `1e-4`--`1`. It is a
replot of the public data, not the byte-identical paper raster.

Two independent local OpenMM REMD chains reproduce the main morphology and
basin weights of that reference. Each chain ran for 20 ns per temperature
slot; the first 2 ns were excluded, leaving 36,000 pooled 300 K frames. These
local chains used OBC2 and are retained as a comparison, not as the exact FAB
Hamiltonian. Their former root-level plots were removed to keep the working
folder focused on the frozen target and the next `jflows_md` phase.

## Protocol

- OpenMM 8.5.2 CUDA 13 build, mixed precision, RTX 5090.
- ACE-ALA-NME, 22 atoms, L-alanine minimum.
- Amber96 plus OBC2 GBSA (`amber96_obc.xml`), NoCutoff, no constraints.
- LangevinMiddle integrator, 1 fs, friction 1/ps.
- 21 temperature slots: 300, 350, ..., 1300 K.
- Neighbor swaps every 200 steps; 300 K frame saved every 1000 steps.
- Independent seeds: 20260711 and 20260712.
- Each chain used two restartable 10 ns segments. Segment runtimes were
  6488.5/6435.1 s and 6465.5/6434.0 s; every segment stayed below two hours.

## Evidence

The exact comparison reference is the authors' one-million-frame `train.h5`
from Zenodo record 6993124. Its MD5 is
`8d34fdda8694ee8d6745fc45b9eb3380`, matching the published checksum. MDTraj
1.11.1 produced `data/reference/fab_train_phi_psi.npz` from those coordinates.

| Diagnostic | Chain 1 | Chain 2 | Gate |
|---|---:|---:|---:|
| Post-burn-in frames | 18,000 | 18,000 | at least 8,000 |
| Exchange acceptance range | 0.543--0.882 | 0.544--0.879 | no bottleneck |
| Walker slots visited, minimum | 21/21 | 21/21 | both ladder ends |
| Round trips, total | 10,380 | 10,483 | report/global travel |
| Round trips, minimum per walker | 461 | 471 | global travel |
| JS to FAB, 36 x 36 (bits) | 0.01104 | 0.01172 | secondary agreement |

Cross-chain JS divergence is 0.01229 bits on the frozen 36 x 36 estimator,
well below the predeclared 0.10-bit gate. Sensitivity values are 0.00522 bits
at 24 x 24 and 0.01944 bits at 48 x 48.

Major-basin probabilities agree closely: alpha-R is 0.1160 vs 0.1253,
high-psi negative-phi is 0.7575 vs 0.7511, and the periodic bottom band is
0.0899 vs 0.0872. Positive phi is rare (0.00250 vs 0.00339; 45 and 61 saved
frames); the exact FAB training-data value is 0.00333.

## Exact model for the next phase

Source and system audits identify the FAB Hamiltonian as Amber ff96 plus OBC1
GBSA (`igb=2`, `mbondi2` radii), the ACE nonpolar term, dielectric 1.0/78.5,
zero salt, `NoCutoff`, no constraints, and 300 K. The approved
`../JFLOWS_MD_PLAN.md` freezes that exact model for `jflows_md`; it explicitly
excludes `amber96_obc.xml` because that OpenMM XML implements OBC2.

## Qualification

The downloaded FAB dataset establishes strong qualitative and distributional
agreement for the completed OBC2 comparison. It does not make those trajectories
samples of the exact OBC1 target, nor is the 1M-sample replot claimed to be a
pixel-identical reconstruction of the authors' separate paper/test raster.
Rare positive-phi fine structure is necessarily noisier in the 36,000-frame
local comparison than in the million-frame reference.

Raw numerical evidence is in `data/analysis/metrics.json` and
`data/analysis/basins.csv`. The replot-ready pooled angles are in
`data/analysis/pooled_postburn.npz`.
Restartable chain checkpoints and run logs are under `data/runs/`.

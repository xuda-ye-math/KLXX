# FAB alanine-dipeptide reference provenance

## Original paper raster

- Paper source archive: `/mnt/projects/arXiv-2208.01893v3.tar.gz`
- Path inside archive: `figures/aldp/ram/test.png`
- Archive path: `figures/aldp/ram/test.png`
- SHA-256: `b53d6bc8607672f48e4347f93593345b8e81a05027dd514be5ba63e5581cf7d9`
- Image dimensions: 2485 x 2079 pixels, RGBA PNG

This archived asset is the ground-truth/test-data Ramachandran panel and the
first panel of `figures/aldp/ram/summary.png`. It is not currently copied into
the clean working root. The file previously called
`Molecular_BG_2/fab_fig19_test.png` has the same SHA-256 checksum, so it is an
exact copy rather than a recreation.

The current `../adp_fab.png` is different by design: it is a fresh 100 x 100
histogram made from exactly 1,000,000 frames in the authors' public training
split, with no kernel smoothing and fixed `LogNorm(vmin=1e-4, vmax=1)`. It is
the reproducible local reference, not a claim of byte identity with the paper
raster or of using the paper's separate test split.

The archive's `main.tex` includes the complete summary as
`figures/aldp/ram/summary.png`. Its caption orders the panels as ground truth
from MD, maximum-likelihood flow trained on MD, FAB with replay buffer, and FAB
with replay buffer after importance reweighting.

## How the reference samples were generated

The authors' public dataset is Zenodo record 6993124
(`https://zenodo.org/records/6993124`, CC BY 4.0). The local
`../data/reference/fab_train.h5` is the published one-million-frame training
split and has MD5
`8d34fdda8694ee8d6745fc45b9eb3380`, equal to Zenodo's checksum. MDTraj 1.11.1
was used to calculate phi/psi from its coordinates and save the compact,
replot-ready `../data/reference/fab_train_phi_psi.npz`; no trajectory frames
were altered.

The following is stated in the FAB paper source (`main.tex`, alanine-dipeptide
experiment and appendix):

- System: 22-atom alanine dipeptide in implicit solvent at 300 K.
- Only the L enantiomer is used for the reported target distribution.
- Ground-truth data are generated with replica-exchange molecular dynamics
  (parallel tempering), rather than one ordinary MD trajectory.
- There are 21 replicas: 300 K, 350 K, ..., 1300 K.
- Replica exchanges are attempted every 200 iterations.
- A state at every multiple of 1000 time steps is retained as a sample.
- Each run is equilibrated for 200,000 iterations and then run for 2,000,000
  iterations.
- Many runs with different random seeds are executed in parallel because every
  run starts from the same minimum-energy configuration.
- The training/test split is 90%/10%.

The arXiv bundle does not contain the trajectories, the phi/psi arrays, or the
Python program that rendered the PNG. Consequently, the exact number of test
points in this panel, histogram normalization call, and explicit color limits
cannot be proven from the tarball alone.

## Exact benchmark Hamiltonian

The FAB implementation constructs OpenMMTools'
`AlanineDipeptideImplicit(constraints=None)`. Auditing the contemporaneous
OpenMMTools test system and its Amber input identifies the target as ff96 with
OBC1 GBSA (`igb=2`, `mbondi2` radii), the ACE nonpolar term, solute/solvent
dielectrics 1.0/78.5, zero salt, `NoCutoff`, no constraints, and 300 K.

An independently built OpenMM system using `amber96.xml` plus
`implicit/obc1.xml` agrees with the canonical system on 100 FAB frames to a
maximum energy difference of `1.90e-4 kJ/mol` and force RMSE
`1.21e-3 kJ/mol/nm`. In contrast, `amber96_obc.xml` implements OBC2; the
completed local OBC2 runs are comparison data and are not the target frozen in
`../JFLOWS_MD_PLAN.md`.

## How the Ramachandran plot was made

The paper explicitly says that each Ramachandran plot is a two-dimensional
histogram of the backbone dihedrals phi and psi with 100 x 100 bins. Inspection
of the published asset establishes the remaining visible conventions:

- phi is horizontal and psi is vertical;
- both axes cover [-pi, pi];
- ticks are -pi, -pi/2, 0, pi/2, pi;
- the colormap is viridis on a logarithmic scale;
- empty histogram bins are white;
- the displayed density spans approximately four logarithmic decades.

For the 22-atom ordering used by the related local ADP setup, the torsions are:

- phi: atoms `(4, 6, 8, 14)`, C(ACE)-N-CA-C;
- psi: atoms `(6, 8, 14, 16)`, N-CA-C-N(NME).

These zero-based indices come from the local molecular setup, not from the
arXiv bundle. Always verify them against the topology before applying them to a
new coordinate file.

`../src/plot_ramachandran.py` implements these plot conventions from an NPZ file
containing one-dimensional `phi` and `psi` arrays. It transposes NumPy's
`histogram2d` result before display so rows map to psi and columns map to phi;
omitting this transpose is a common source of an incorrectly oriented figure.

## Reweighting clarification

The ground-truth panel copied here is not importance reweighted. Reweighting is
used for generated flow/FAB samples in other panels. For those model results,
the paper reports drawing 10^7 samples and clipping the largest 1000 importance
weights to the smallest value among that set before computing reweighted
metrics and histograms.

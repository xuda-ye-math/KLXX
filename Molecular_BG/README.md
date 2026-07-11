# Alanine-dipeptide molecular benchmark

The single figure at the workspace root is the frozen reference target:

- `adp_fab.png`: a fresh 100 x 100, unsmoothed histogram of exactly 1,000,000
  configurations from the FAB authors' ground-truth REMD training split. It
  uses a logarithmic color range from `1e-4` to `1`.

Those configurations were generated for FAB's 22-atom ACE--ALA--NME target:
Amber ff96 with OBC1 implicit solvent (`igb=2`, `mbondi2` radii), ACE nonpolar
solvation, `NoCutoff`, no constraints, and 300 K. This is the model frozen for
the planned `jflows_md` benchmark.

The completed local OpenMM chains used `amber96_obc.xml`, which is OBC2. They
are retained as useful comparison data, but are not relabeled as samples from
the exact FAB Hamiltonian and no root-level OBC2 plot is retained.

## Layout

```text
Molecular_BG/
├── adp_fab.png
├── JFLOWS_MD_PLAN.md   # approved FAB-aligned BG design
├── src/                 # OpenMM REMD, analysis, and plotting programs
├── assets/system/       # ADP topology and L-minimum coordinates
├── data/
│   ├── reference/       # downloaded FAB data and derived phi/psi
│   ├── runs/            # restartable local OBC2 comparison chains
│   ├── analysis/        # OBC2 pooled angles and diagnostics
│   └── legacy/          # initial short calibration artifact
└── docs/                # provenance and benchmark summary
```

## Reproduce the reference figure without rerunning dynamics

```bash
conda run -n jflows python Molecular_BG/src/plot_ramachandran.py \
  Molecular_BG/data/reference/fab_train_phi_psi.npz \
  --bins 100 --smooth-sigma 0 --vmin 1e-4 --vmax 1 \
  --mask-below-vmin --title "Ground-truth REMD (N=1,000,000)" \
  --output Molecular_BG/adp_fab.png
```

See `docs/summary.md` for the benchmark result and `docs/PROVENANCE.md` for
the distinction between the authors' source raster, the 1M-sample replot, and
the local OBC2 comparison.

The approved package and experiment design for the next phase is
`JFLOWS_MD_PLAN.md`. It defines a self-contained `Molecular_Potential` bundle
API and three targets: exact FAB L-ADP (60D, ff96/OBC1), glycerol (36D,
GAFF2/AM1-BCC/OBC1), and explicitly neutral diethanolamine (48D,
GAFF2/AM1-BCC/OBC1). The plan freezes coordinate support together with each
Hamiltonian and specifies mixed spline flow, mixed-domain MALA,
no-sharpening training, and per-target validation gates.

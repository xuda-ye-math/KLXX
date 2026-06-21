# Forward KL Can Be Better: Normalizing Flow Boltzmann Sampling with X-Regularization

Source and numerical experiments for the paper of the same name. Each experiment folder is
self-contained and carries its own `RUN.md` (exact commands + a full file tree); environment
setup is in `PYTHON.md`.

```
Log-Likelihood-Ratio-Discrepancy/
│
├── Paper/                  ◄──── the manuscript
│   ├── main.tex            ◄──── THE paper (start here)
│   ├── main.pdf            #     compiled PDF
│   └── references.bib
│
├── PYTHON.md               #  conda environment setup (zflows / zflows_md, OpenMM, PyTorch)
│
├── 2D_Benchmark/           #  §5.1  six 2D mode-discovery targets
├── Sensor_Array/           #  §5.2  permutation-symmetric source localization
├── HD_Product/             #  §5.3  high-dimensional product multi-well (tables)
├── HD_Product_Ladder/      #  §5.3  AIS-ladder ESS sweep
├── Phi4_Lattice_6/         #  §5.4  tilted φ⁴ lattice field theory, L = 6
├── Phi4_Lattice_8/         #  §5.4  tilted φ⁴ lattice field theory, L = 8
├── Clock_Lattice/          #  §4 + §5.5  adaptive-temperature BG, p-state clock model
├── Poisson_Inverse/        #  §5.6  Bayesian screened-Poisson source inversion
├── Molecular_BG/           #  §6    molecular BGs (glycerol / diethanolamine / alanine dipeptide)
│
└── tests/data/             #  shared MD inputs (prmtop/rst7) for Molecular_BG
```

Each folder maps to the paper section shown; open its `RUN.md` for the commands and details.

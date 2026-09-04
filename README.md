# KLXX

Research code and numerical evidence for *Mode Coverage in Normalizing Flow
Boltzmann Generators via Log-Ratio Variation*.

- Paper: [`Paper/main.pdf`](Paper/main.pdf)
- LaTeX source: [`Paper/main.tex`](Paper/main.tex)
- Python environment: [`PYTHON.md`](PYTHON.md)

The illustration below shows the simplest target the numerical suite covers.

<p align="center">
  <img src="Paper/figures/2d_himmelblau_samples.png" width="1050" alt="Mode discovery on the analytic Himmelblau benchmark under four training objectives">
</p>

Mode discovery on the analytic Himmelblau target, the simplest example. The
columns are four training objectives — forward KL, forward KL+X_μ, forward
KL+X_μ+X_μ̂, and forward KL+X_μ+X_(μ̂+ν̄)/2 — and each panel overlays the
pushforward samples of the trained flow on the target energy, with the Gaussian
source in gray. Forward KL and the target-weighted variation alone settle on
part of the wells and leave the rest unpopulated. Weighting the variation by
quench and temper candidates instead recovers every well, and mixing
pushforward samples into that weighting additionally suppresses the mass left
stranded between modes.

## Active implementation

The experiments use the published JAX packages
[`jflows`](https://github.com/xuda-ye-math/jflows) and
[`jflows_md`](https://github.com/xuda-ye-math/jflows_md), whose sources are
also included under [`External/`](External/).
Install them into a virtual environment and activate it as described in
[`PYTHON.md`](PYTHON.md).

Run a `jflows` experiment from the repository root. The 2D Himmelblau
benchmark of the figure above is the cheapest:

```bash
python Codes/2D_Benchmark/Himmelblau/train.py
```

The molecular drivers train a staged Boltzmann generator and should only be
launched with sufficient resources. They take the loss as an argument:

```bash
python Codes/Achiral/NMA_30D/train.py --method klxx
```

## Repository layout

```text
KLXX/
├── Codes/
│   ├── 1D_QT/              # 1D quench and temper illustration
│   ├── 2D_Benchmark/       # four analytic mode-discovery benchmarks
│   ├── HD_Product/         # high-dimensional product multi-well sweep
│   ├── HD_100_Lambda/      # target-variation coefficient sweep at d=100
│   ├── Lattice_Phi4/       # L=6 and L=8 tilted phi-four experiments
│   ├── Lattice_Clock/      # periodic clock-model experiments
│   └── Achiral/            # NMA, glycerol and diethanolamine generators
├── External/               # jflows and jflows_md sources as git submodules
├── Paper/                  # manuscript and paper figures
└── PYTHON.md               # authoritative pip-only environment guide
```

Each experiment folder keeps its raw arrays under `artifacts/` and its figures
under `results/`. Every file in `Paper/figures/` is a hard link to the figure
in the `results/` folder that produced it, so the manuscript and the code tree
cannot drift apart on the same machine. A fresh clone receives two independent
copies, since Git records content rather than links.

The alkane, chiral, and alanine dipeptide experiments of earlier revisions are
not part of the current project tree.

## Numerical suites

| Folder | Purpose |
|---|---|
| `Codes/1D_QT/` | 1D Rastrigin target used to illustrate quench and temper |
| `Codes/2D_Benchmark/` | Forward KL and X-regularized comparisons on multimodal 2D targets |
| `Codes/HD_Product/` | Dimension scaling for product multi-well targets |
| `Codes/Lattice_Phi4/` | Broken-phase lattice phi-four training and reference diagnostics |
| `Codes/Lattice_Clock/` | Mixed periodic flow, adaptive stage schedule, and occupancy diagnostics |
| `Codes/HD_100_Lambda/` | Sweep of the target-variation coefficient at d=100 |
| `Codes/Achiral/` | Adaptive-staging Boltzmann generators on three achiral molecules |

Every active driver documents its exact invocation at the top of the
file. Long runs save raw numerical arrays and checkpoints separately from
plotting so figures can be regenerated without retraining.

## Molecular boundary

The reported molecular targets are NMA at `d=30`, glycerol at `d=36`, and
neutral diethanolamine at `d=48`, each in its own folder under
`Codes/Achiral/`. A folder holds the runtime bundle in `bundle/`, the driver
`train.py`, its constants in `parameters.py`, and `pt_reference.py`, which
generates the parallel-tempering reference that the dihedral marginals are
compared against.

Training uses the pure-JAX `Molecular_Potential` and does not reconstruct a
Hamiltonian from a PDB. OpenMM/ParmEd are installed for validation and optional
bundle maintenance; AmberTools is optional and is not part of the main runtime
environment.

## Reproducibility conventions

- The drivers depend on the installed Python packages `jflows` and `jflows_md`,
  which use a JAX backend; the specification and the environment for the tests
  are given in [`PYTHON.md`](PYTHON.md).
- The reference molecular data are generated with OpenMM and AmberTools. The
  peptide targets use Amber ff96 and the remaining targets use GAFF2 with
  AM1-BCC charges; every target is solvated by the OBC1 (`igb=2`) implicit
  solvent with the ACE nonpolar term.
- Float32 is the default dtype for both training and evaluation.

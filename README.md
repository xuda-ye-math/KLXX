# KLXX

Research code and numerical evidence for log-ratio variation in forward KL
normalizing flow Boltzmann generators.

- Paper: [`Paper/main.pdf`](Paper/main.pdf)
- LaTeX source: [`Paper/main.tex`](Paper/main.tex)
- Current operational state: [`status.md`](status.md)
- Python environment: [`PYTHON.md`](PYTHON.md)

## Active implementation

The experiments use the published JAX packages `jflows` and `jflows_md`.
Install them into a virtual environment and activate it as described in
[`PYTHON.md`](PYTHON.md); a source checkout, editable installation, or manually
configured `PYTHONPATH` is not required.

Run a `jflows` experiment from the repository root with:

```bash
python Codes/Lattice_Clock/train.py
```

The molecular drivers are full-size and must not be launched without explicit
authorization:

```bash
python Codes/Molecular_BG/alkane_family/methane_9d_raw/train.py
```

Do not launch production molecular training without explicit authorization.

## Repository layout

```text
KLXX/
├── Codes/
│   ├── 2D_Benchmark/       # four analytic mode-discovery benchmarks
│   ├── HD_Product/         # high-dimensional product multi-well sweep
│   ├── Lattice_Clock/      # periodic clock-model experiments
│   ├── Lattice_Phi4/       # L=6 and L=8 tilted phi-four experiments
│   └── Molecular_BG/       # alkane, achiral, and chiral molecular experiments
├── Paper/                  # manuscript and tracked paper figures
├── PYTHON.md               # authoritative pip-only environment guide
└── status.md               # sole operational diary and handoff record
```

The former archived molecular baselines and controls are not part of the
current project tree.

## Numerical suites

| Folder | Purpose |
|---|---|
| `Codes/2D_Benchmark/` | Forward KL and X-regularized comparisons on multimodal 2D targets |
| `Codes/HD_Product/` | Dimension scaling for product multi-well targets |
| `Codes/Lattice_Phi4/` | Broken-phase lattice phi-four training and reference diagnostics |
| `Codes/Lattice_Clock/` | Mixed periodic flow, adaptive stage schedule, and occupancy diagnostics |
| `Codes/Molecular_BG/` | Current regularized molecular experiments |

Every active driver documents its exact invocation at the top of the
file. Long runs save raw numerical arrays and checkpoints separately from
plotting so figures can be regenerated without retraining.

## Molecular boundary

Tracked runtime bundles live within the corresponding experiment directories
under `Codes/Molecular_BG/alkane_family/`, `Codes/Molecular_BG/achiral/`, and
`Codes/Molecular_BG/chiral/`. The reported targets comprise the methane through
hexane alkane family; NMA, glycerol, and neutral diethanolamine; and the three
chiral targets `(2R,3R)`-2,3-butanediol, alanine dipeptide, and Ac-Pro-NHMe.

Training uses the pure-JAX `Molecular_Potential` and does not reconstruct a
Hamiltonian from a PDB. OpenMM/ParmEd are installed for validation and optional
bundle maintenance; AmberTools is optional and is not part of the main runtime
environment.

## Reproducibility conventions

- Commands activate the environment described in [`PYTHON.md`](PYTHON.md) and
  then use ordinary `python` and `pip` names.
- Drivers import the installed `jflows` and `jflows_md` packages directly; no
  `PYTHONPATH` is set.
- Float32 is the normal training dtype.
- MALA is the default Langevin kernel.
- Molecular target energies and reported ESS values remain unclipped; `e_clip`
  is only an optimizer screen.
- The completed molecular runs use fixed-regularization settings or audited
  sharpening paths implemented by `jflows_md`.
- Package smokes run from temporary copies so public repositories stay clean.

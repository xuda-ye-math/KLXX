# X-regularized forward KL

Research code and numerical evidence for X-functional regularization of
forward-KL normalizing-flow Boltzmann generators.

- Paper: [`Paper/main.pdf`](Paper/main.pdf)
- LaTeX source: [`Paper/main.tex`](Paper/main.tex)
- Current operational state: [`status.md`](status.md)
- Python environment: [`PYTHON.md`](PYTHON.md)

## Active implementation

Current experiments use the public JAX packages in two neighboring source
repositories:

- `/data/projects/jflows`
- `/data/projects/jflows_md`

The local pip-only virtual environment is `~/.envs/jflows`. Neither package is
installed into it; private runs select the live checkout explicitly through
`PYTHONPATH`. The former Conda environment and `~/.envs/jax` are retired.
Activate it once in each terminal:

```bash
source "$HOME/.envs/jflows/bin/activate"
```

Run a `jflows` experiment from the repository root with:

```bash
PYTHONPATH=/data/projects/jflows \
  python Codes/Lattice_Clock/train.py
```

The molecular drivers are full-size and must not be launched without explicit
authorization:

```bash
PYTHONPATH=/data/projects/jflows:/data/projects/jflows_md \
  python Molecular_BG/methane_9d/train.py
```

Do not launch production molecular training without explicit authorization.

## Repository layout

```text
X-regularization/
├── Codes/
│   ├── 2D_Benchmark/       # four analytic mode-discovery benchmarks
│   ├── HD_Product/         # high-dimensional product multi-well sweep
│   ├── Lattice_Clock/      # periodic clock-model experiments
│   └── Lattice_Phi4/       # L=6 and L=8 tilted phi-four experiments
├── Molecular_BG/           # completed 9D--45D fixed-e/r alkane experiments
│   ├── methane_9d/
│   ├── ethane_18d/
│   ├── propane_27d/
│   ├── butane_36d/
│   └── pentane_45d/
├── .archive/               # ignored old molecular baselines and controls
├── Paper/                  # manuscript and tracked paper figures
├── PYTHON.md               # authoritative pip-only environment guide
└── status.md               # sole operational diary and handoff record
```

The old JAX baseline and three `Molecular_BG_zflows_*` controls live below
ignored `.archive/`. Their old zflows/PyTorch environment instructions are
historical evidence, not active guidance; the ext4 mirror preserves them.

## Numerical suites

| Folder | Purpose |
|---|---|
| `Codes/2D_Benchmark/` | Forward KL and X-regularized comparisons on multimodal 2D targets |
| `Codes/HD_Product/` | Dimension scaling for product multi-well targets |
| `Codes/Lattice_Phi4/` | Broken-phase lattice phi-four training and reference diagnostics |
| `Codes/Lattice_Clock/` | Mixed periodic flow, adaptive ladder, and occupancy diagnostics |
| `Molecular_BG/` | Current energy/distance-regularized molecular experiments |
| `.archive/` | Ignored historical molecular baselines and PyTorch controls |

Every active driver documents its exact local invocation at the top of the
file. Long runs save raw numerical arrays and checkpoints separately from
plotting so figures can be regenerated without retraining.

## Molecular boundary

Each current experiment carries a complete six-file runtime bundle under its
own `Molecular_BG/<molecule>_<dimension>d/bundle/` directory. The current
targets are the GAFF2/AM1-BCC/OBC1 n-alkane series:

- 9D methane;
- 18D ethane;
- 27D propane;
- 36D n-butane;
- 45D n-pentane.

Training uses the pure-JAX `Molecular_Potential` and does not reconstruct a
Hamiltonian from a PDB. OpenMM/ParmEd are installed for validation and optional
bundle maintenance; AmberTools is optional and is not part of the main runtime
environment.

## Reproducibility conventions

- Current local commands activate `~/.envs/jflows` and then use ordinary
  `python` and `pip` names.
- `PYTHONPATH` always names the live public checkout(s).
- Float32 is the normal training dtype.
- MALA is the default Langevin kernel.
- Molecular target energies and reported ESS values remain unclipped; `e_clip`
  is only an optimizer screen.
- The completed runs use fixed e/r surrogates. Later experiments will adapt
  the separately audited sharpening technique from `../zflows_md`.
- Package smokes run from temporary copies so public repositories stay clean.

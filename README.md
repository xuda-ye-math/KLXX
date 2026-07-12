# X-regularized forward KL

Research code and numerical evidence for X-functional regularization of
forward-KL normalizing-flow Boltzmann generators.

- Paper: [`Paper_Arxiv/main.pdf`](Paper_Arxiv/main.pdf)
- LaTeX source: [`Paper_Arxiv/main.tex`](Paper_Arxiv/main.tex)
- Current operational state: [`status.md`](status.md)
- Python environment: [`PYTHON.md`](PYTHON.md)

## Active implementation

Current experiments use the public JAX packages in two neighboring source
repositories:

- `/mnt/projects/jflows`
- `/mnt/projects/jflows_md`

The local pip-only virtual environment is `~/.envs/jflows`. Neither package is
installed into it; private runs select the live checkout explicitly through
`PYTHONPATH`. The former Conda environment and `~/.envs/jax` are retired.
Activate it once in each terminal:

```bash
source "$HOME/.envs/jflows/bin/activate"
```

Run a `jflows` experiment from the repository root with:

```bash
PYTHONPATH=/mnt/projects/jflows \
  python Codes/Lattice_Clock/train.py
```

Run the bounded molecular pipeline with:

```bash
PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
  python Molecular_BG/glycerol_36d/train.py --smoke
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
├── Molecular_BG/
│   ├── bundles/            # three immutable schema-2 runtime targets
│   ├── glycerol_36d/       # current molecular training driver
│   ├── reference/          # FAB ground-truth reference artifacts
│   └── JFLOWS_MD_PLAN.md
├── Molecular_BG_1/         # archived paper-era PyTorch result tree
├── Molecular_BG_2/         # archived abandoned PyTorch repair tree
├── Paper_Arxiv/            # manuscript and tracked paper figures
├── PYTHON.md               # authoritative pip-only environment guide
└── status.md               # sole operational diary and handoff record
```

`Molecular_BG_1/` and `Molecular_BG_2/` are historical evidence. Their old
zflows/PyTorch environment instructions are intentionally not active guidance.

## Numerical suites

| Folder | Purpose |
|---|---|
| `Codes/2D_Benchmark/` | Forward KL and X-regularized comparisons on multimodal 2D targets |
| `Codes/HD_Product/` | Dimension scaling for product multi-well targets |
| `Codes/Lattice_Phi4/` | Broken-phase lattice phi-four training and reference diagnostics |
| `Codes/Lattice_Clock/` | Mixed periodic flow, adaptive ladder, and occupancy diagnostics |
| `Molecular_BG/` | Bundle-driven molecular potentials and glycerol Boltzmann-generator work |

Every active driver documents its exact local invocation at the top of the
file. Long runs save raw numerical arrays and checkpoints separately from
plotting so figures can be regenerated without retraining.

## Molecular boundary

The molecular runtime consumes complete, hash-verified directories under
`Molecular_BG/bundles/`, mirrored from `/mnt/projects/jflows_md/bundles/`.
The current targets are:

- 60D L-alanine dipeptide, Amber ff96/OBC1;
- 36D neutral glycerol, GAFF2/AM1-BCC/OBC1;
- 48D neutral diethanolamine, GAFF2/AM1-BCC/OBC1.

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
- No sharpening is part of the current molecular target or training design.
- Package smokes run from temporary copies so public repositories stay clean.

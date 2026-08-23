# KLXX

Research code and numerical evidence for log-ratio variation in forward KL
normalizing flow Boltzmann generators.

- Paper: [`Paper/main.pdf`](Paper/main.pdf)
- LaTeX source: [`Paper/main.tex`](Paper/main.tex)
- Python environment: [`PYTHON.md`](PYTHON.md)

The two illustrations below bracket the numerical suite, showing the simplest
and the hardest target it covers.

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

<p align="center">
  <img src="Paper/figures/molecular_adp_ramachandran.png" width="800" alt="Alanine dipeptide Ramachandran free energy surfaces across accepted stages">
</p>

Alanine dipeptide backbone free energy over the φ and ψ torsions at four
accepted stages of the KLXX Boltzmann generator, the hardest example. Every
surface here comes from KLXX and from no other sampler. Each panel bins
inference samples, obtained from the KLXX-trained flow saved at that stage, into
a two-dimensional torsion histogram and reports the negative logarithm of the
normalized occupancy in units of kBT relative to the observed minimum; unvisited
bins are left white. Read across the panels, the surfaces show how the
interpolation carries the generator from a smooth source landscape to the
resolved basins and barriers of the target.

The panels show the path; the table below shows where it ends. The final-stage
KLXX sample set is compared against published ground truth on the same 100-bin
φ/ψ grid with the same wrapped smoothing, with 10⁷ samples on each side.

<div align="center">

<table>
<thead>
<tr><th>quantity</th><th>KLXX</th><th>ground truth</th></tr>
</thead>
<tbody>
<tr><td>α_L basin population (φ &gt; 0)</td><td>0.0031</td><td>0.0033</td></tr>
<tr><td>β/PPII basin population (ψ &gt; 0)</td><td>0.7958</td><td>0.7940</td></tr>
<tr><td>α_R basin population (ψ &lt; 0)</td><td>0.2012</td><td>0.2027</td></tr>
<tr><td>maximum free energy</td><td>10.40</td><td>10.40</td></tr>
</tbody>
</table>

</div>

Free energies are in units of kBT. The three basin populations differ by at
most 0.0018, including the α_L basin that carries only 0.3% of the total, and
both surfaces reach the same maximum. Each surface is normalized to its own
most populated bin, so only per-method values are quoted; a difference between
two such surfaces would carry an arbitrary additive offset.
Together the figure and the table are the accuracy claim for KLXX on this
target: the panels show that the staged construction reaches the right
landscape, and the table shows that the landscape it reaches matches an
independent reference.

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

The molecular drivers are full-size and should only be launched with sufficient resources:

```bash
python Codes/Molecular_BG/alkane_family/methane_9d_raw/train.py
```

## Repository layout

```text
KLXX/
├── Codes/
│   ├── 1D_QT/              # 1D quench and temper illustration
│   ├── 2D_Benchmark/       # four analytic mode-discovery benchmarks
│   ├── HD_Product/         # high-dimensional product multi-well sweep
│   ├── Lattice_Clock/      # periodic clock-model experiments
│   ├── Lattice_Phi4/       # L=6 and L=8 tilted phi-four experiments
│   └── Molecular_BG/       # alkane, achiral, and chiral molecular experiments
├── External/               # jflows and jflows_md sources as git submodules
├── Paper/                  # manuscript and tracked paper figures
└── PYTHON.md               # authoritative pip-only environment guide
```

The former archived molecular baselines and controls are not part of the
current project tree.

## Numerical suites

| Folder | Purpose |
|---|---|
| `Codes/1D_QT/` | 1D Rastrigin target used to illustrate quench and temper |
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

- The drivers depend on the installed Python packages `jflows` and `jflows_md`,
  which use a JAX backend; the specification and the environment for the tests
  are given in [`PYTHON.md`](PYTHON.md).
- The reference molecular data are generated with OpenMM and AmberTools. The
  peptide targets use Amber ff96 and the remaining targets use GAFF2 with
  AM1-BCC charges; every target is solvated by the OBC1 (`igb=2`) implicit
  solvent with the ACE nonpolar term.
- Float32 is the default dtype for both training and evaluation.

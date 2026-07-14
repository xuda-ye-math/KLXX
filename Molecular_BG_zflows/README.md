# zflows_md

Molecular **Boltzmann generators on whitened internal coordinates**, in pure PyTorch — an independent, self-contained **variant** of [`zflows`](https://github.com/xuda-ye-math/zflows) (a separate package, not a branch or fork) that turns an Amber force field into a normalizing-flow sampling target. The zflows flow / loss / potential / utils machinery is **vendored directly into the package** (`zflows_md/{core,flow.py,loss.py,potential.py,utils.py}`), so `zflows_md` depends on nothing outside itself.

> **Status: experimental.** Tested on **Linux + NVIDIA GPU**. The Amber force field is read from a `.prmtop` / `.rst7` (or an OpenMM `System`) with **parmed / openmm only**, then compiled to static tensor buffers — so forward, gradient, and the training loss are pure tensor ops and `torch.compile`-clean. Environment setup is in **[PYTHON.md](PYTHON.md)**.
>
> This project was developed with [Claude Code](https://claude.com/claude-code).
>
> **Always `conda activate zflows` before anything that compiles** — the env's Blackwell-capable `ptxas`
> (CUDA 13.x) is only on `PATH` after activation; without it TorchInductor fails with `ptxas-blackwell`
> on sm_120 GPUs. Do **not** use `TORCHDYNAMO_DISABLE=1`.

## Layout

The package has exactly four subfolders (`core`, `plot`, `data`, `template`); everything else is a plain
top-level module. The project also holds the smoke tests and the three molecule experiments:

```
zflows_md/
├── zflows_md/     boltzmann·potential·forcefield·coords·flow·loss·utils·dihedral·gate  +  core/ plot/ data/ template/
├── tests/         structure + all-method smoke tests (run_all.py, smoke_all_methods.py)
└── Molecular_BG/  per-molecule config.json + train.py + figures (glycerol 36d, diethanolamine 48d, adp 60d) + RUN.md
```

## Features

**The molecular target is a `Potential`.** `zflows_md.PDB_Potential` is the molecule's Boltzmann target **μ ∝ exp(−U)** expressed on **whitened internal bond–angle–torsion (BAT) coordinates** `xi`. It subclasses the vendored `Potential` base (`forward(xi) -> [N]`), so it composes directly with the vendored flows (`zflows_md.flow`), losses (`zflows_md.loss`), and samplers (`zflows_md.utils`). It takes **only internal coordinates** — the Cartesian Amber energy (`zflows_md.Amber_Force_Field`) is an internal helper called after the BAT internal→Cartesian map, with the whitening + BAT volume Jacobians folded into `U(xi)` so `exp(−U)` is the Boltzmann density pulled back to `xi`. Following the zflows naming rule, classes Capitalize_Each_Word (`PDB_Potential`, `Amber_Force_Field`, `Internal_Coordinates`, `Source`) and instances are lowercase (`u`, `u0`).

**Assemble the problem with `build()`.** `build()` takes a prmtop + a handful of MD frames (used only to estimate the bond/angle whitening) and returns the potentials plus the flow box:

```python
from zflows_md import build

B  = build(prmtop="alanine_dipeptide.prmtop", crd="alanine_dipeptide.rst7",
           md_frames=frames, T=300.0, device="cuda")   # frames: [F, natoms, 3] (nm)
u  = B["u"]      # PDB_Potential — the target on whitened internal coords,  u(xi) -> [N]
u0 = B["u0"]     # Source — Gaussian bond/angle + uniform torsion prior
a, b, wrap = B["a"], B["b"], B["wrap"]   # NCSF spline box + domain wrap
```

Because `u` / `u0` are plain `Potential`s they plug straight into `zflows_md.utils.langevin / lbfgs` and the vendored flow-training losses (`zflows_md.loss`).

**Train a Boltzmann generator with `run_boltzmann()`.** The bundled engine trains an **X-regularized adaptive-temperature ladder** end to end — an annealed-SMC Boltzmann generator that grows its own temperature schedule by sizing each step from the per-stage effective sample size:

```python
from zflows_md import run_boltzmann

stages, Y, complete, flow, F_inv = run_boltzmann(
    u0, u, flow_factory, n_valid=..., n_pool=..., n_batch=..., steps=...,
    method="klxx",                          # "klxx" = X-regularized KL + X_mu + X_(mu_hat+nu_bar)/2; "kl" = bare forward KL
    adaptive_tau=0.75, validation_tau=0.4,  # SMC step-select target / stage-accept ESS floor
    shrink_factor=0.7, enlarge_factor=2.0,  # next Δt on FAIL (shrink) / on SUCCESS (enlarge)
    e_min=100, e_max=200,                   # soft cap annealed in sharpening (omit for a fixed cap)
    wrap=wrap, qt_fn=qt_fn, device="cuda")
```

The headline metric is `F = ∏_k (1/ESS_val_k)(1/ESS_sharp_k)` — the Monte-Carlo error-propagation factor through both per-stage importance reweights (smaller is better; `F = 1` is ideal). `run_asmc(...)` runs the same ladder with the flow set to the identity at every stage, the no-flow reference.

**Soft-core regularity for singular targets.** The Lennard-Jones `r → 0` clash makes `U` singular and destabilizes training at high dimension; `build(..., r_floor=, e_cap=, e_cap_scale=)` exposes a soft-core distance floor and a smooth energy cap. In the *sharpening* schedule these are annealed from soft to sharp along the ladder, with a per-stage Monte-Carlo reweight removing the bias; the defaults reproduce the bare Amber energy, so an un-annealed run is unchanged.

**Per-molecule runs.** A full Boltzmann-generator run is driven by a small, self-contained **`train.py`** that reads a per-molecule, read-only `config.json` directly — no central driver, no template indirection. The molecule's `prmtop` / `rst7` load from the shared `zflows_md/data` folder; start from the copy-ready example in `template/`:

```bash
conda activate zflows                                    # required for torch.compile
cp zflows_md/template/{config.json,train.py} Molecular_BG/<mol>/
cd Molecular_BG/<mol> && python train.py --method klxx   # {klxx|kl|asmc} [--raw] [--no-delta] [--smoke]
```

Each run writes `data_<TAG>.pth` + a live `status_<TAG>.log` next to itself; `--raw` / `--no-delta` reproduce the full ablation (see `Molecular_BG/RUN.md`). The headline metric is `F = ∏_k (1/ESS_val_k)(1/ESS_sharp_k)` (smaller is better). `data/gen_molecules.py` generates fresh `prmtop` + `rst7` for new hetero-atom molecules via GAFF.

**Public surface.**

```python
from zflows_md import PDB_Potential, Torsion_Target, Source   # zflows_md.potential (MD targets)
from zflows_md import Potential, linear_combination, NCSF     # vendored zflows base / flow
from zflows_md import Amber_Force_Field                       # zflows_md.forcefield (internal FF)
from zflows_md import Internal_Coordinates                    # zflows_md.coords  (BAT)
from zflows_md import build, run_boltzmann                    # zflows_md.boltzmann
from zflows_md import flow, loss, utils                       # vendored zflows submodules
from zflows_md import dihedral, gate                          # torsion analysis + mode-coverage gate
from zflows_md.boltzmann import run_asmc                      # no-flow annealed-SMC reference
```

- `zflows_md.potential` — the vendored `Potential` base + `linear_combination`, plus the MD targets `PDB_Potential` (the public target), `Torsion_Target` (torsion-only soft block), `Source` (prior). All on whitened internal coords.
- `zflows_md.flow` / `zflows_md.loss` — the **vendored** zflows normalizing flows (`NCSF`, …) and flow-training losses (`loss_compile`, `forward_KL`, …).
- `zflows_md.forcefield` — `Amber_Force_Field` (parmed/openmm → static buffers → batched torch Amber energy on Cartesian, kJ/mol), `build_system`, `softcap_energy`. Internal helper; not a public Cartesian potential.
- `zflows_md.coords` — `Internal_Coordinates` (automatic z-matrix + NeRF, exact internal↔Cartesian with log|det|).
- `zflows_md.boltzmann` — `build` (assemble the BG problem) + `run_boltzmann` (the adaptive-temperature ladder with the staged-ESS gate) + `run_asmc` (no-flow reference) + the stage losses / samplers.
- `zflows_md.utils` — the vendored samplers (`langevin`, `lbfgs`, `resample`, `compute_ESS_log`, …) + `pdb_to_prmtop`, alignment helpers.
- `zflows_md.dihedral` — torsion analysis: `proper_torsions`, `dihedral`, `multimodality`.
- `zflows_md.gate` — `build_mode_checker`, the optional per-stage mode-coverage acceptance gate.
- `zflows_md.plot` — figure generation (kept out of the core import): `marginals`, `dihedrals`, `conformers`, `ablation`, `summary`, `table`, `pymol`.

## Functional form (Amber, NoCutoff / isolated molecule)

bonds `k(r−r0)²` · angles `k(θ−θ0)²` · dihedrals + impropers `Σ Vₙ[1 + cos(nφ − γ)]` · scaled 1-4 (electrostatic /scee, van-der-Waals /scnb) · nonbonded `A/r¹² − B/r⁶ + COUL·qᵢqⱼ/r` (COUL via OpenMM, charges in e). Exclusions (1-2 / 1-3 / 1-4) come from the system; 1-4 interactions are added back scaled. Reproduces OpenMM energies / gradients to numerical tolerance.

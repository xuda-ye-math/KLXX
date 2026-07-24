---
name: jflows
description: Build, explain, debug, benchmark, and test code against the live `jflows` JAX/Equinox package and its `jflows_md` mixed-domain molecular companion. Use for `/jflows`; imports from either public package; NSF/NCSF/CNF/OTFlow/RealNVP or Mixed_NSF construction; KL/KLX/KLXX training and adaptive-staging, fixed-schedule, or identity-only Boltzmann generators; importance sampling, SMC/AIS, MCMC, optimization, artifacts and stage persistence; molecular bundles, regularization and sharpening; Molecular_Potential; or native OpenMM molecular workflows.
---

# JFlows

Use the live roots `/data/projects/jflows` and `/data/projects/jflows_md` and
treat their current public source, examples, and smoke tests as authoritative.
Use `/home/xuda/.envs/jflows` as the **default and only** local Python virtual
environment. It is pip-based; the former Conda environment and `~/.envs/jax`
are retired. Both packages are installed there in editable mode from the live
roots, so ordinary imports resolve directly to the source checkouts. Do not
install a second copy or use another environment.

## Source and interface map

The generic source directory is `/data/projects/jflows/jflows/`; the molecular
source directory is `/data/projects/jflows_md/jflows_md/`. Use this complete
repository map:

```text
/data/projects/jflows/
├── jflows/                     # package source
│   ├── flow.py                 # Flow, NSF, NCSF, CNF, OTFlow, RealNVP
│   ├── potential.py            # Potential and built-in energies
│   ├── loss.py                 # reverse/forward KL and X losses
│   ├── train.py                # direct train_* stage functions
│   ├── artifacts.py            # medium-level save/load helpers
│   ├── boltzmann/              # generators plus write/load persistence
│   ├── utils/                  # metrics, MCMC, SMC/AIS, optimization, QT
│   └── core/                   # private implementation
├── example/                    # executable generic examples
├── smoke/                      # generic smoke tests
└── doc/                        # low/medium/high-level documentation

/data/projects/jflows_md/
├── jflows_md/                  # molecular package source
│   ├── flow.py                 # Mixed_Identity, Mixed_NSF
│   ├── potential.py            # Molecular_Potential and regularization
│   ├── source.py               # Molecular_Source
│   ├── system.py               # Molecular_Bundle and bundle discovery
│   ├── train.py                # molecular KLX/KLXX stage functions
│   ├── artifacts.py            # re-exported generic artifact helpers
│   ├── boltzmann/              # sharpening generators and persistence
│   ├── utils/                  # mixed MALA, QT, SMC/AIS
│   ├── openmm/                 # OpenMM_Potential, Langevin, tempering
│   ├── bundle_build/           # bundle construction CLI
│   └── core/                   # private molecular implementation
├── bundles/                    # separately distributed bundle data
├── smoke/                      # molecular smoke tests
└── doc/                        # molecular API/workflow documentation
```

Import generic public objects from `jflows.flow`, `.potential`, `.loss`,
`.train`, `.boltzmann`, `.artifacts`, and `.utils`. The `jflows_md` root lazily
exports its common objects; its module paths `.flow`, `.potential`, `.source`,
`.system`, `.train`, `.boltzmann`, `.artifacts`, `.utils`, and `.openmm` are
also public. Treat both `core/` trees as private.

## Work from evidence

1. Inspect the relevant public module before writing code.
2. For generic work, read the closest shipped example under
   `/data/projects/jflows/example/`. For molecular work, use
   `/data/projects/jflows_md/doc/05-workflows.md` and the matching smoke test;
   `jflows_md` has no separate example directory.
3. Read the matching smoke test under `/data/projects/jflows/smoke/` or
   `/data/projects/jflows_md/smoke/` for exact calls, shapes, edge cases, and
   expected behavior.
4. Use only public imports from `jflows.flow`, `potential`, `loss`, `train`,
   `boltzmann`, `artifacts`, and `utils`. Direct stage trainers live in
   `jflows.train`; `boltzmann_identity` plus the four adaptive-staging and four
   fixed-schedule trained generators live in `jflows.boltzmann`. Never import
   the retired `jflows.training.*` or private `jflows.core.*` namespaces.
5. For molecular work, inspect the public modules under
   `/data/projects/jflows_md/jflows_md/`; keep its low-level `core` API private.
6. Keep package edits separate from drivers and experiments. Do not modify a
   package merely to make a benchmark or example work unless the user
   explicitly requests a package change.

Read [references/api.md](references/api.md) when exact signatures, direction
conventions, return values, stage records, artifacts, or molecular bundle APIs
matter. Read [references/workflows.md](references/workflows.md) when choosing a
model, trainer, sampler, example, or smoke test.

## Run correctly

Activate the default virtual environment, then use ordinary Python. The
editable installations already point at both live roots:

```bash
source /home/xuda/.envs/jflows/bin/activate
python script.py
python -m smoke.test_flow
python molecular_script.py
```

The environment provides accelerator-backed JAX/Equinox, OpenMM, and the
scientific stack. Verify `jflows.__file__` and `jflows_md.__file__` when source
provenance matters. Set `PYTHONPATH` only when intentionally running copied
temporary source trees; put `/data/projects/jflows` before
`/data/projects/jflows_md` when explicit live-root resolution is useful.
Use `/data/projects/X-regularization/PYTHON.md` only when rebuilding or auditing
the local environment.

For public installations, `jflows_md[bundles]` is the optional construction
extra. It adds OpenMM, ParmEd, and the unofficial AmberTools command-line wheel;
ordinary bundle runtime does not require that extra. Write newly prepared
targets to a new bundle directory rather than overwriting an active input.

Treat live package roots as read-only during verification. Source inspection
and `git status` checks are safe. Copy the needed package, test/example, and
bundle files to temporary repository roots, run from those copied roots, and
remove them afterward. This prevents logs, figures, bytecode, caches, and
compiled artifacts from appearing in public repositories. For molecular
tests, preserve the outer `jflows_md/bundles/` layout and use
`PYTHONPATH=<temporary-jflows-root>:<temporary-jflows-md-root>`.

Prefer the configured accelerator. Do not force `JAX_PLATFORMS=cpu`; the project targets accelerator-backed JAX and normal work uses float32. Smoke tests may opt into x64 themselves. In standalone scripts, set this before importing JAX:

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

Run examples as modules from the **temporary copied repository context**, for
example `python -m example.2D_single`. Expect first calls to include JIT
compilation. Synchronize with `jax.block_until_ready(...)` around timing
boundaries.

## Preserve the package conventions

- Treat a `Potential` as an energy `U` for an unnormalized density proportional to `exp(-U(x))`. Accept and return batches: `[N, d] -> [N]`.
- Pass explicit PRNG keys to every random constructor or operation. Split or fold keys instead of reusing them accidentally.
- Remember equinox modules are immutable. Rebind `flow = flow.zeros()` and trainer outputs.
- Initialize energy-training flows with `.zeros()` unless a nonidentity start
  is intentional. `OTFlow` is the one exception: use `.near_identity()` so
  its quadratic factor has a nonzero gradient. Reserve `OTFlow.zeros()` for
  an exact identity map such as a Boltzmann fallback.
- Distinguish directions rigorously: `F` maps source to target; `G = F^{-1}` maps target to source. A suffix `_F` or `_G` fixes training direction and takes no `type` argument. Metrics and AIS accept `type="F"` or `"G"`.
- For a trained F flow, generate with `y = flow(x)`. For a trained G flow, generate from source samples with `y = flow.inv(x)`.
- Reduce low-level per-sample losses with `.mean()` when a scalar objective is required.
- Call dynamically selected outer Boltzmann construction adaptive-staging.
  Treat `t` as a dimensionless stage-interpolation parameter, call each
  `t_{k-1} -> t_k` transition a stage, and call the accepted endpoints a stage
  schedule. Reserve `ladder` for the inner SMC/AIS levels. In `jflows_md`, keep
  physical `temperature_kelvin` and `beta` thermodynamic, and call
  replica-exchange temperatures a temperature grid.
- Temper generic potentials algebraically, such as `beta * U`. For molecular
  targets, use the explicit `Molecular_Potential.from_bundle(...,
  temperature_kelvin=...)` interface so target beta and source variance stay
  matched.
- Default Langevin is MALA (`adjust=True` / `mc_adjust=True`). Use `adjust=False` for ULA. Never combine `adjust=True` with positive `taming`.
- Use log weights and `compute_ESS_log` for numerically difficult targets. ESS is normalized to `(0, 1]`.
- Increase `chunks` to reduce peak device memory; it means a number of row
  partitions, not a chunk size. Eager full-set weight loops are strictly
  partitioned, but chunk loops nested inside an outer JIT are not a strict
  VRAM bound because XLA may co-schedule buffers. Generic `jflows` accepts
  only the canonical `chunks` spelling.
- Use canonical generic controls: primitive kernels use `dt` and `steps`;
  composite drivers use `mc_dt`/`mc_steps`, `opt_dt`/`opt_steps`,
  `train_steps`, `batch_size`, `pool_size`, and `chunks`. KLXX uses
  `pool_size=0` for QT on the complete validation population and a positive
  value for a separately resampled pool. Use only the single memory-control
  spelling `chunks`; the molecular companion retains its documented
  `opt_alpha` spelling and uses `u_clip`/`g_clip` for optimizer screening.
- Preserve the committed execution schemes. Reverse KL, forward KL, and KLX
  are outer `eqx.filter_jit` trainers containing the complete Adam loop in one
  `lax.scan`. KLXX runs chunked quench-and-temper eagerly before its compiled
  Adam scan. Generic Boltzmann SMC selection and stage advancement are eager
  controllers around compiled chunk kernels. The `jflows_md` molecular
  SMC/AIS/Boltzmann orchestration
  intentionally uses eager level/chunk controllers around fixed-shape kernels,
  while its stage trainer retains its own compiled `lax.scan`. Cross-package
  alignment concerns mathematical interfaces and conventions, not identical
  compilation layout.
- Classical potential-space SMC rejuvenates at its matching intermediate
  potential. Flow-proposal AIS deliberately applies fractional geometric
  weights but rejuvenates at the final target at every level; treat it as a
  biased score-free target surrogate, not exact AIS/SMC.
- Keep constructor and tuning values explicit. Do not silently derive independent experimental knobs.
- Every trained Boltzmann driver uses full-validation ESS to select the better
  of the trained and identity maps. Adaptive-staging trained drivers use that
  selected ESS as the post-training acceptance/retry gate; fixed-schedule
  trained drivers always advance on their prescribed schedule with the better
  map.
  `boltzmann_identity` performs no flow construction or training and gates the
  exact identity reweighting ESS directly. Optimizer batch ESS and molecular
  kept/update diagnostics are monitors, never acceptance gates.
- Keep persistence separate from computation. `jflows.artifacts` provides
  direct flow/sample/history save-load operations, and `jflows_md.artifacts`
  re-exports that format. Complete-stage persistence lives in each package's
  `boltzmann.write` and `boltzmann.load` modules. Computation generators take
  no `run_dir`, `resume`, or `problem_id`; the explicit
  `jflows[_md].boltzmann.load.run(...)` orchestrator owns those controls.
  Never enable resume without explicit user authorization, and never
  overwrite a nonempty artifact directory. Identity runs pass `flow=None`,
  store no `.eqx` artifacts, and load with a `None` continuation flow.
- Molecular Boltzmann generators require `rg_param_0` and `rg_param_1`.
  Trained generators fit the flow at the pre-sharpen endpoint, then gate and
  rejuvenate the linear regularization update with sharpening ESS.
  Molecular `boltzmann_identity` removes only flow training: SMC selection,
  complete-population identity ESS, pre-sharpen mixed MALA, sharpening ESS,
  and post-sharpen mixed MALA remain active. Equal endpoint pairs mean fixed
  regularization; different pairs enable sharpening.
- Keep mixed molecular coordinates in `R^p x T^q`: Euclidean coordinates come
  first and torsions last. Obtain the domain from `target.domain`; do not build
  user code on `jflows_md.core.*`.

## Validate proportionally

For API usage, run the narrowest matching smoke module from a temporary copy. For new workflows, first run a reduced configuration in temporary storage that still exercises compilation, training, weighting, and sampling, then scale up. Check finite arrays, output shapes, forward/inverse reconstruction, opposite log-Jacobian signs, ESS history shape/range, and the final sampling direction. View generated figures before reporting them.

If behavior disagrees with this skill, trust the current source and smoke tests, then update the calling code rather than relying on remembered API behavior.

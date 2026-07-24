# Workflow selection and executable references

## Contents

- [Choose the model and objective](#choose-the-model-and-objective)
- [Minimal generic pattern](#minimal-generic-pattern)
- [Minimal molecular pattern](#minimal-molecular-pattern)
- [Example and smoke-test routing](#example-and-smoke-test-routing)
- [Isolated verification](#isolated-verification)
- [Common failures](#common-failures)

## Choose the model and objective

- Use NSF for bounded, nonperiodic low/moderate-dimensional targets.
- Use NCSF only when every modeled coordinate is periodic on one box/torus.
- Use CNF for flexible continuous dynamics; exact trace costs more memory and
  can benefit from `checkpoint=True`.
- Use OTFlow when closed-form trace and transport-inspired dynamics fit the
  benchmark.
- Use RealNVP for a fast bidirectional affine-coupling baseline; explicit
  mixing such as `"lu"` can improve coordinate communication.
- Use `jflows_md.Mixed_NSF` for molecular `R^p x T^q` coordinates. Obtain the
  domain from a verified `Molecular_Potential`, not from private core modules.

For NSF/NCSF, choose F/G training direction with the expensive autoregressive
inverse in mind. A suffix `_F` or `_G` fixes the trained direction.

- Use reverse KL F for a simple mode-seeking baseline.
- Use forward KL G when mass coverage matters; generic `jflows` manufactures
  target batches with its flow-proposal AIS surrogate.
- Add KLX to penalize spread in the log density ratio.
- Use KLXX when a quench-and-temper coverage pool is worth the extra cost.
  Set `pool_size=0` to quench the complete validation population or use a
  positive value for a separately resampled pool; use `chunks` for QT memory
  partitioning.
- Use an adaptive-staging Boltzmann generator when direct source-to-target
  training is too difficult. Use a fixed schedule when bridge coefficients
  must be controlled exactly.
- Use `boltzmann_identity` when the adaptive-staging reweight/resample/MCMC
  baseline is needed without any flow or optimizer. In `jflows_md`, this
  removes only flow training: regularization sharpening and both mixed-MALA
  transitions remain active.

Pair ESS with coverage or domain-specific mode diagnostics. Training loss
alone does not establish sampling quality. Smoke-size runs establish only that
the pipeline compiles and executes; never use them for scientific ESS claims.

## Minimal generic pattern

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, potential_from
from jflows.train import Monitor, train_forward_KLX_G
from jflows.utils import compute_ESS_log, importance_weights_log

source = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
target = potential_from(lambda x: ((x**2).sum(-1) - 4.0) ** 2)
x_valid = source.samples(jax.random.key(2), 20000)
flow = NSF(
    jax.random.key(0), [-4.0, -4.0], [4.0, 4.0],
    bins=16, transforms=4, hidden_features=(64, 64),
).zeros()

flow, batch_ess_hist = train_forward_KLX_G(
    x_valid, source, target, flow,
    batch_size=500, train_steps=1000, lr=1e-3,
    ladder=1, mc_dt=2e-3, mc_steps=50, coeff_lambda=1.0,
    monitor=Monitor(100, "[KLX] "),
)
log_w = importance_weights_log(
    x_valid, source, target, flow, type="G", chunks=1,
)
valid_ess = compute_ESS_log(log_w)
y = flow.inv(x_valid)
```

## Minimal molecular pattern

Frozen bundles are runtime inputs. The pure-JAX potential does not call OpenMM
during training.

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann import boltzmann_forward_KLXX_G

target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
source = target.source()
x_valid = source.samples(jax.random.key(1), 200000)
flow = Mixed_NSF(
    jax.random.key(0), target.domain,
    bins=32, transforms=6, euclidean_bound=8.0,
    hidden_features=(256, 256), mask_strategy="balanced",
).zeros()

y_valid, stages = boltzmann_forward_KLXX_G(
    x_valid, source, target, flow,
    pool_size=0, batch_size=10000, train_steps=500, lr=1e-3,
    ladder=8, mc_dt=1e-2, mc_steps=50,
    melt=1.0, opt_alpha=1e-2, opt_steps=200,
    rg_param_0=(50.0, 0.15), rg_param_1=(100.0, 0.15),
    coeff_lambda=1.0, coeff_alpha=0.5, coeff_beta=0.5,
    u_clip=1e3, g_clip=1e2, chunks=32,
)
complete = bool(stages and stages[-1]["t"] == 1.0)
valid_ess = stages[-1]["valid_selected_ess"] if stages else 0.0
```

Treat those values as an interface example, not universal molecular tuning.
Keep source and target temperatures matched through
`Molecular_Potential.from_bundle(..., temperature_kelvin=...)` followed by
`target.source(...)`.

This example linearly sharpens `(e,r)` from `(50,0.15)` to `(100,0.15)`.
Use equal endpoint pairs for fixed regularization. For complete-stage
persistence, drive `jflows_md.boltzmann.iterate_boltzmann` through
`jflows_md.boltzmann.load.run`; include `valid_size` in the saved config and
leave `resume=False` unless the user explicitly authorizes resume.

For a flow-free molecular baseline, call `boltzmann_identity` with the same
`x_valid`, source, target, SMC/MALA controls, regularization endpoints, and
`bg_param`, but omit flow, pool, batch, optimizer, checkpoint, and
initialization controls. Sharpening is still required when the endpoint pairs
differ. For persistence use `iterate_identity` and pass `flow=None`.

## Example and smoke-test routing

Generic examples under `/data/projects/jflows/example/`:

- `2D_single.py`: NSF, Gaussian mixture, reverse-vs-forward stage training,
  ESS, and plotting.
- `3D_periodic.py`: NCSF, periodic target, log weights, resampling, and MALA.
- `4D_boltzmann.py`: adaptive-staging reverse/forward Boltzmann generators.
- `CNF_vs_OTFlow.py`: checkpointing, held-out ESS, timing, and CSV output.
- `flow_scaling_law.py`: JIT warmup, synchronization, and flow latency.

Generic smoke tests under `/data/projects/jflows/smoke/`:

- `test_flow`, `test_circular`: constructors, round trips, Jacobians, and seams.
- `test_potential`, `test_linear_combination`: potential algebra and bridges.
- `test_loss`, `test_loss_training`: loss formulas and trainability.
- `test_train`, `test_clip`: compiled trainer kernels, histories, screening,
  clipping, and determinism.
- `test_utils_api`: canonical signatures, stable semantic aliases, and utility
  execution.
- `test_metrics`: weights, ESS, coverage, and resampling.
- `test_rejuvenation`, `test_annealing`, `test_optimization`: MCMC, SMC/AIS,
  optimizers, and quench-and-temper prerequisites.
- `test_boltzmann`: adaptive-staging/fixed-schedule stage and ESS-gate
  contracts.
- `test_boltzmann_identity`: identity-only adaptive-staging computation and
  absence of flow-training calls.
- `test_boltzmann_identity_artifacts`: flow-free stage persistence and resume.
- `test_boltzmann_artifacts`: complete-stage persistence, interruption,
  continuation, reload, and stage readers.
- `test_boltzmann_chunks`: KLXX QT execution and the single `chunks` forwarding
  path through trainer and generator layers.
- `test_public_api`: the split `jflows.train` / `jflows.boltzmann` surface and
  retirement of `jflows.training`.
- `test_chunk`: chunk-count equivalence and memory partitioning.

Molecular smoke tests under `/data/projects/jflows_md/smoke/`:

- `test_bundles`, `test_molecular_potential`: frozen bundle integrity and
  OpenMM-reference energy/force parity.
- `test_mixed_nsf`: mixed flow geometry, seams, and Jacobians.
- `test_mixed_training`: trainer/BG ESS histories, operation-key separation,
  AIS behavior, and linear sharpening.
- `test_float32_training`, `test_float32_glycerol_compile`: default-float32
  checkpoint/rematerialized training and the bounded real glycerol compile
  path.
- `test_support_and_utils`: chirality support and mixed utility contracts.
- `test_jflows_compatibility`, `test_api_consistency`: package boundary and
  canonical interface checks.
- `test_jflows_md_chunking`: fixed-shape eager chunk-controller behavior.
- `test_boltzmann_checkpoints`, `test_edge_cases`: ESS-only retries, identity
  fallback, terminal failures, and molecular safety guards.
- `test_boltzmann_identity`: identity ESS followed by active sharpening and
  pre-/post-sharpen mixed MALA.
- `test_boltzmann_identity_artifacts`: molecular identity save/load/resume
  without `.eqx` artifacts.
- `test_boltzmann_integration`: computed combined-stage persistence and resume
  equivalence.
- `test_regularization`, `test_sharpening_gate`: `(e,r)` potential behavior and
  the combined flow/sharpening ESS gate.
- `test_openmm`: JAX/OpenMM energy-force parity plus native Langevin and
  parallel tempering.
- `test_artifacts`: template-based flow/sample/history persistence through the
  shared artifact format.
- `run_all.py`: complete bounded package suite; no production training.
- `benchmark_compile.py`: opt-in compilation benchmark, excluded from
  `run_all.py`.

## Isolated verification

Activate only the pip environment and copy source/tests to temporary roots:

```bash
source ~/.envs/jflows/bin/activate
tmp=$(mktemp -d /tmp/jflows-smoke.XXXXXX)
mkdir -p "$tmp/jflows" "$tmp/jflows_md"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  /data/projects/jflows/jflows /data/projects/jflows/smoke \
  /data/projects/jflows/pyproject.toml "$tmp/jflows/"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  /data/projects/jflows_md/jflows_md /data/projects/jflows_md/smoke \
  /data/projects/jflows_md/bundles /data/projects/jflows_md/pyproject.toml \
  "$tmp/jflows_md/"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$tmp/jflows:$tmp/jflows_md" \
  python "$tmp/jflows_md/smoke/run_all.py"
rm -rf "$tmp"
```

For one generic test, copy only the generic package and matching smoke file,
then run it as `python -m smoke.test_metrics` from the temporary jflows root.
Run examples the same way from a copied repository context. Never let
verification write into either live public repository.

## Common failures

- Wrong environment: activate `~/.envs/jflows`; do not use Conda or
  `~/.envs/jax`.
- Stale package: activate `/home/xuda/.envs/jflows`, confirm both editable
  locations with `python -m pip show jflows jflows-md`, and verify module
  `__file__` paths when provenance matters.
- Wrong samples: pass source samples to importance-weight functions; `type`
  says whether the trained flow represents F or G.
- Wrong generated map: F uses `flow(x)`; G uses `flow.inv(x)` for
  source-to-target generation.
- Wrong Gaussian scale: built-ins take variance, not standard deviation.
- Scalar custom energy: return `[N]`, not one scalar for a whole batch.
- Mutability assumption: rebind returned immutable flows and states.
- Key reuse: split or fold keys for independent operations.
- MALA/taming conflict: positive taming requires `adjust=False`; molecular
  mixed MALA is always adjusted and exposes no `adjust` switch.
- Weight underflow: retain log weights and use `compute_ESS_log`.
- Falling molecular KL with falling ESS: audit an independent target holdout
  and saved proposal weights. KL does not monotonically control importance
  ESS. Only full-validation trained-versus-identity ESS gates a BG stage.
- Wrong diagnostic: `batch_ess_hist` monitors optimizer batches;
  `valid_selected_ess` is the stage acceptance quantity.
- Memory pressure: increase `chunks`, reduce batch/pool size, or use supported
  checkpointing. `chunks` is a count, so larger means fewer rows per chunk.
- Timing compilation: warm up, synchronize, then time; synchronize after the
  timed region.
- Compile explosion: preserve the established strategies. Reverse KL, forward
  KL, and KLX compile one complete `lax.scan`; KLXX keeps QT eager before its
  compiled scan. Generic and molecular Boltzmann controllers keep eager
  stage/level/chunk loops around fixed-shape kernels.
- Incomplete adaptive-staging run: require final `t == 1.0` before claiming
  target completion.

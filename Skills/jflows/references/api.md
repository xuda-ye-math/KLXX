# Public API and invariants

Use this as a current interface map for the live repositories, then inspect the
implementation and nearest smoke test before changing code.

## Contents

- [Naming conventions](#naming-conventions)
- [jflows flows, potentials, and losses](#jflows-flows-potentials-and-losses)
- [jflows stage trainers](#jflows-stage-trainers)
- [jflows Boltzmann generators](#jflows-boltzmann-generators)
- [jflows utilities](#jflows-utilities)
- [jflows_md molecular objects](#jflows_md-molecular-objects)
- [jflows_md training and Boltzmann generators](#jflows_md-training-and-boltzmann-generators)
- [jflows_md samplers, OpenMM, and artifacts](#jflows_md-samplers-openmm-and-artifacts)

## Naming conventions

Use these names throughout generic `jflows` code:

- primitive integrators: `dt`, `steps`;
- HMC: `dt`, `leapfrog_steps`, `trajectories`;
- composite MCMC: `mc_dt`, `mc_steps`, and molecular
  `mc_image_radius`;
- optimization: `alpha`/`steps` or composite `opt_dt`/`opt_steps`;
- training: `batch_size`, `train_steps`, and KLXX `pool_size`;
- memory partitioning: `chunks`, the number of row partitions;
- adaptive-staging outer schedule: `t`, the dimensionless interpolation
  parameter, with each `t_{k-1} -> t_k` transition called a stage and accepted
  endpoints called a stage schedule;
- SMC/AIS: `ladder`, the number of inner levels;
- molecular thermodynamics: `temperature_kelvin` and `beta`, with
  replica-exchange temperatures called a temperature grid.

Each package accepts only its documented canonical keywords. Both packages use
`pool_size` for KLXX pool selection. The molecular companion retains
`opt_alpha`, while generic `jflows` uses `opt_dt`; both use `u_clip`, `g_clip`,
and separate complete-stage persistence modules. Molecular generators require
`rg_param_0` and `rg_param_1`. Molecular image controls are `image_radius` or
`mc_image_radius`, and `Molecular_Source.samples` accepts only `N` as its
sample count. Experiment drivers may name their population constant
`VALID_SIZE`; the package APIs consume the resulting array as `x_valid`.

## jflows flows, potentials, and losses

Import flows and stable transform primitives from `jflows.flow`:

```python
NSF(key, a, b, bins=8, slope=1e-3, transforms=4, randmask=True,
    hidden_features=(64, 64), activation=jax.nn.silu)
NCSF(key, a, b, bins=8, slope=1e-3, transforms=4, randmask=True,
     hidden_features=(64, 64), activation=jax.nn.silu)
CNF(key, dimension, frequency=3, nt=16, exact=True,
    hidden_features=(64, 64), activation=jax.nn.silu)
OTFlow(key, dimension, hidden=64, layer=3, rank=10, nt=8,
       time_bound=(0.0, 1.0))
RealNVP(key, dimension, transforms=4, randmask=True, mixing=None,
        hidden_features=(64, 64), activation=jax.nn.silu)

Flow
Transform
ComposedTransform(*transforms)
MonotonicRQSTransform(widths, heights, derivatives, bound=1.0,
                      slope=1e-3, circular=False)
CircularRQSTransform(*phi, bound=math.pi, slope=1e-3)
```

Flows expose `flow(x)`, `call_and_ladj(x)`, `inv(y)`, `inv_and_ladj(y)`,
`t()`, `zeros()`, `with_trace_key(key)`, and `needs_trace_key`. Deterministic
flows return themselves and `False` for the trace hooks; approximate CNF uses
them for stochastic probes. NSF/NCSF are MAF-style: forward evaluation is
parallel and inversion autoregressive. NCSF is periodic on `[a,b]`. For
energy training initialize with `.zeros()`, except OTFlow: use
`.near_identity()` because its exact zero quadratic head has zero gradient.

Import potentials from `jflows.potential`:

```python
Potential
potential_from(fn)
Nlog_Uniform(a, b)
Nlog_Gaussian(mean, variance)
Nlog_Gaussian_Mixture(weights, mean, variance)
linear_combination(potentials, coeffs=None)
```

Every potential maps `[N,d] -> [N]`; samplable built-ins provide
`samples(key, N)`. Gaussian arguments are variances. Potential arithmetic
supports scalar multiplication/division, addition, subtraction, negation, and
`sum(...)`.

Import losses from `jflows.loss`; each returns one value per sample:

```python
reverse_KL_F(x, target, flow, trace_key=None)
forward_KL_G(y, source, flow, trace_key=None)
forward_KLX_G(y, source, target, flow, key,
              coeff_lambda=1.0, trace_key=None)
forward_X_G(y, source, target, flow, key, trace_key=None)
```

For `x = G(y)` and `ladj = log|det J_G(y)|`, the KLX density-ratio coordinate
is `z = source(x) - target(y) - ladj`. Reduce low-level losses with `.mean()`
when a scalar objective is required.

## jflows stage trainers

Import from `jflows.train`:

```python
Monitor(every, prefix="", printer=print)

train_reverse_KL_F(
    x_valid, source, target, flow, batch_size, train_steps, lr,
    mc_dt, mc_steps, mc_adjust=True, monitor=None, seed=0,
    checkpoint=False, *, initialize_from_identity=False,
    t_start=0.0, t_end=1.0,
)
train_forward_KL_G(
    x_valid, source, target, flow, batch_size, train_steps, lr,
    ladder, mc_dt, mc_steps, mc_adjust=True, monitor=None, seed=0,
    checkpoint=False, u_clip=inf, g_clip=inf, *,
    initialize_from_identity=False, t_start=0.0, t_end=1.0,
)
train_forward_KLX_G(
    x_valid, source, target, flow, batch_size, train_steps, lr,
    ladder, mc_dt, mc_steps, coeff_lambda=1.0, mc_adjust=True,
    monitor=None, seed=0, checkpoint=False, u_clip=inf, g_clip=inf, *,
    initialize_from_identity=False, t_start=0.0, t_end=1.0,
)
train_forward_KLXX_G(
    x_valid, source, target, flow, pool_size, batch_size, train_steps,
    lr, ladder, melt, opt_dt, opt_steps, mc_dt, mc_steps,
    coeff_lambda=1.0, coeff_alpha=0.5, coeff_beta=0.5,
    mc_adjust=True, monitor=None, seed=0, checkpoint=False,
    u_clip=inf, g_clip=inf, *, chunks=1,
    initialize_from_identity=False,
    t_start=0.0, t_end=1.0,
)
```

Each returns `(trained_flow, batch_ess_hist)`, where
`batch_ess_hist.shape == (train_steps,)`. `Monitor` reports loss and the
pre-update proposal-to-target batch ESS. Reverse KL, forward KL, and KLX are
direct outer-`eqx.filter_jit` implementations containing one Adam `lax.scan`.
KLXX deliberately runs quench-and-temper eagerly before its compiled scan, so
its `chunks` value bounds QT outside an enclosing JIT. `pool_size=0` uses the
complete `x_valid` population; a positive value resamples a separate pool.
`u_clip` screens optimizer loss samples only, and `g_clip` clips the global
gradient norm. Every monitor line includes numeric `t_start -> t_end`.

## jflows Boltzmann generators

Import the identity-only adaptive-staging function, four adaptive-staging
trained functions, and four fixed-schedule trained functions from
`jflows.boltzmann`:

```python
boltzmann_identity(
    x_valid, source, target, ladder, mc_dt, mc_steps, *,
    mc_adjust=True, monitor=None, bg_param=None, chunks=1, seed=0,
)
boltzmann_reverse_KL_F(
    x_valid, source, target, flow, batch_size, train_steps,
    lr, ladder, mc_dt, mc_steps, *, initialize_from_identity=True,
    mc_adjust=True, monitor=None, bg_param=None, chunks=1,
    checkpoint=False, seed=0,
)
boltzmann_forward_KL_G(
    x_valid, source, target, flow, batch_size, train_steps,
    lr, ladder, mc_dt, mc_steps, *, initialize_from_identity=True,
    mc_adjust=True, monitor=None, bg_param=None, chunks=1,
    checkpoint=False, u_clip=inf, g_clip=inf, seed=0,
)
boltzmann_forward_KLX_G(
    x_valid, source, target, flow, batch_size, train_steps,
    lr, ladder, mc_dt, mc_steps, *, initialize_from_identity=True,
    coeff_lambda=1.0, mc_adjust=True, monitor=None, bg_param=None,
    chunks=1, checkpoint=False, u_clip=inf, g_clip=inf, seed=0,
)
boltzmann_forward_KLXX_G(
    x_valid, source, target, flow, pool_size, batch_size, train_steps,
    lr, ladder, melt, opt_dt, opt_steps, mc_dt, mc_steps, *,
    initialize_from_identity=True, coeff_lambda=1.0,
    coeff_alpha=0.5, coeff_beta=0.5,
    mc_adjust=True, monitor=None, bg_param=None, chunks=1,
    checkpoint=False, u_clip=inf, g_clip=inf, seed=0,
)

boltzmann_reverse_KL_F_fixed(
    x_valid, source, target, flow, batch_size, train_steps, lr,
    mc_dt, mc_steps, t_list, *, initialize_from_identity=True,
    mc_adjust=True, monitor=None, chunks=1, checkpoint=False, seed=0,
)
boltzmann_forward_KL_G_fixed(
    x_valid, source, target, flow, batch_size, train_steps, lr,
    ladder, mc_dt, mc_steps, t_list, *, initialize_from_identity=True,
    mc_adjust=True, monitor=None, chunks=1, checkpoint=False,
    u_clip=inf, g_clip=inf, seed=0,
)
boltzmann_forward_KLX_G_fixed(
    x_valid, source, target, flow, batch_size, train_steps, lr,
    ladder, mc_dt, mc_steps, t_list, *, initialize_from_identity=True,
    coeff_lambda=1.0, mc_adjust=True, monitor=None, chunks=1,
    checkpoint=False, u_clip=inf, g_clip=inf, seed=0,
)
boltzmann_forward_KLXX_G_fixed(
    x_valid, source, target, flow, pool_size, batch_size, train_steps,
    lr, ladder, melt, opt_dt, opt_steps, mc_dt, mc_steps, t_list, *,
    initialize_from_identity=True, coeff_lambda=1.0,
    coeff_alpha=0.5, coeff_beta=0.5,
    mc_adjust=True, monitor=None, chunks=1, checkpoint=False,
    u_clip=inf, g_clip=inf, seed=0,
)
```

`boltzmann_identity` has no `flow`, batch, optimizer, checkpoint, or
initialization arguments. Each accepted stage optionally passes the SMC gate,
requires full-validation exact-identity ESS to clear `tau_ess`, resamples, and
applies Langevin at the accepted bridge. It returns `(y_valid, stages)` and
uses the same stage-selection policy keys in `bg_param` as the trained
generators.

The `_fixed` variants replace dynamic stage selection with `t_list`; the
controller consumes the next listed endpoint above the current coefficient.
Supply an increasing schedule in `(0, 1]`. A schedule ending below one
deliberately returns an incomplete run.

Stage-selection policy keys in `bg_param` are `t_safe`, `shrink_factor`,
`enlarge_factor`, `tau_smc`, `tau_ess`, `t_tol`, `max_stages`, and `max_retry`.
Every driver returns `(y_valid, stages)`. Completion requires a nonempty stage
list and `stages[-1]["t"] == 1.0`.

Canonical trained accepted-stage fields are:

```python
{
    "t": float,
    "t_start": float,
    "valid_selected_ess": float,
    "valid_trained_ess": float,
    "valid_identity_ess": float,
    "valid_sample_count": int,
    "selected": "trained" | "identity",
    "flow": Flow,
    "continuation_flow": Flow,
    "t_hist": Array,                         # [attempts]
    "batch_ess_hist": Array,                 # [attempts, train_steps]
    "valid_trained_ess_hist": Array,         # [attempts]
    "valid_identity_ess_hist": Array,        # [attempts]
    "attempt_status_hist": tuple[str, ...],
    "selection_history": tuple[dict, ...],
    "elapsed_seconds": float,
    "selected_flow_path": str | None,
    "continuation_flow_path": str | None,
    "validation_samples_path": str | None,
}
```

Identity-only records omit `valid_trained_ess`, `flow`, `continuation_flow`,
`batch_ess_hist`, and `valid_trained_ess_hist`. They retain `t`, `t_start`,
identity/selected ESS, sample count, `selected="identity"`, attempt and
selection histories, elapsed time, and the optional saved validation path.

Every trained driver uses full-validation ESS to select the better trained or
identity map. Adaptive-staging trained drivers additionally require selected
ESS to clear `tau_ess`; otherwise they shrink and retry. Fixed-schedule trained
drivers never reject or retry and advance once with the better map.
`boltzmann_identity` has no comparison or training step and gates its identity
ESS directly. Before either flow or identity evaluation at a candidate stage,
`tau_smc` independently gates the endpoint using SMC ESS.

Each saved `flow` is the incremental map for that bridge stage. The complete
generator is the ordered chain of all selected stage maps, and a stage's
`valid_selected_ess` is incremental rather than a global source-to-final ESS.
Compute a fresh full-chain importance ESS when that scientific diagnostic is
needed.

The nine generators are pure computation APIs and accept no `run_dir`,
`resume`, or `problem_id` arguments. Generic medium-level artifacts are:

```python
from jflows.artifacts import (
    save_flow, load_flow, save_samples, load_samples,
    save_history, load_history,
)
```

Complete-stage Boltzmann persistence is separate:

```python
from jflows.boltzmann.write import create, stage, finish
from jflows.boltzmann.load import (
    manifest, validate, load, fork,
    load_stage_flow, load_validation_samples, load_training_history, run,
)

run(
    run_dir, problem_id, config, samples, flow, iterate, *, resume=False,
)
```

`write.stage` publishes an accepted stage only after its validation population
and histories are written. Trained stages additionally store selected and
continuation flows; identity stages store no flow artifacts.
`load.run` resumes from the last manifest-listed complete stage and recomputes
an interrupted partial stage. Its `iterate` callback is an advanced
package-level stage iterator; ordinary callers should use the nine public
generators directly unless they explicitly need persistence orchestration.

## jflows utilities

Import from the flat `jflows.utils` namespace:

```python
importance_weights_log(samples, source, target, flow, type,
                       chunks=1, trace_key=None)
importance_weights(samples, source, target, flow, type,
                   chunks=1, trace_key=None)
linear_weights_from_log(log_weights)
compute_ESS_log(log_weights)
compute_ESS(weights)
coverage(y, x, k=5, chunks=1)
resample(key, samples, weights, N=None)

langevin_step(key, x, potential, dt=1e-3, adjust=True, taming=0)
langevin(key, samples, potential, dt=1e-3, steps=100,
         adjust=True, taming=0, chunks=1)
stochastic_heun_step(key, x, potential, dt=1e-3)
stochastic_heun(key, samples, potential, dt=1e-3, steps=100, chunks=1)
leapfrog(x, p, potential, dt, steps)
hmc_step(key, x, potential, dt=1e-2, leapfrog_steps=10)
hamiltonian_monte_carlo(key, samples, potential, dt=1e-2,
                        leapfrog_steps=10, trajectories=10, chunks=1)

sequential_monte_carlo(key, samples, source, target, ladder=1,
                       mc_dt=1e-3, mc_steps=100, adjust=True,
                       taming=0, chunks=1)
annealed_importance_sampling(key, samples, source, target, flow, type,
                             ladder=1, mc_dt=1e-3, mc_steps=100,
                             adjust=True, taming=0, chunks=1,
                             trace_key=None,
                             return_initial_log_weights=False)

lbfgs_init(x, potential, memory=6)
lbfgs_step(state, potential, alpha=1.0, armijo=False)
lbfgs(samples, potential, alpha=1.0, steps=100, memory=6,
      armijo=False, chunks=1)
adamw_init(x)
adamw_step(state, potential, lr=1e-2, beta1=.9, beta2=.999,
           eps=1e-8, weight_decay=0.0)
adamw(samples, potential, lr=1e-2, steps=100, beta1=.9, beta2=.999,
      eps=1e-8, weight_decay=0.0, chunks=1)
quench_and_temper(key, samples, target, melt, opt_dt=1.0,
                   opt_steps=100, mc_dt=1e-3, mc_steps=100,
                   mc_adjust=True, chunks=1)
```

Aliases are `rejuvenation`, `hmc`, `smc`, `ais`, `optimization`, and `qt`.
SMC returns `(samples, per_level_ess)` and rejuvenates at the matching bridge
potential. Flow-proposal AIS optionally returns
`(samples, initial_log_weights)`; it uses fractional proposal weights but
rejuvenates at the final target at every level, so it is a biased score-free
surrogate rather than exact AIS/SMC. MALA is the default; positive taming
requires `adjust=False`.

## jflows_md molecular objects

The root `jflows_md` namespace lazily exports the common public API. Module
imports from `jflows_md.flow`, `.potential`, `.source`, `.system`, `.train`,
`.boltzmann`, `.utils`, `.artifacts`, and `.openmm` are also public. Never import
`jflows_md.core.*` in user code.

Load a verified target and obtain its mixed domain and matched source:

```python
Molecular_Bundle.load(path_or_name, *, root=None, verify=True)
available_bundles(root=None)
Molecular_Potential(bundle, *, temperature_kelvin=None)
Molecular_Potential.from_bundle(path_or_name, *, root=None, verify=True,
                                temperature_kelvin=None)

target.domain
target.dimension
target.cartesian(q)
target.physical_energy(q)
target.energy_terms(q)
target.reference_internal()
target.support_mask(target.cartesian(q))
target.source()
target.regularized(rg_param)
```

The temperature override changes target beta and scales the source Euclidean
variance by the same temperature ratio. The main molecular regularization
constants are `e = energy_threshold_kj_mol` and
`r = pair_distance_floor_nm`; optimizer `u_clip` is separate.
`regularized(...)` applies a
reference-shifted linear/logarithmic energy map and optionally floors Amber
regular-pair and exception distances. Bonded terms, OBC1/ACE, the coordinate
Jacobian, and `physical_energy(q)` remain physical. It returns a
training/diagnostic bridge, whose mixed-chart normalizability must be audited,
and never mutates the physical endpoint. Its `regularized_energy(q)` method
reports the surrogate Cartesian energy.

Built-in source-checkout bundle names are `adp_ff96_obc1`,
`glycerol_gaff2_am1bcc_obc1`, and
`diethanolamine_gaff2_am1bcc_obc1`. Short-name lookup needs the outer
`/data/projects/jflows_md/bundles/` tree. A code-only installed package requires
an external bundle path.

The source and mixed flow interfaces are:

```python
Molecular_Source(domain, mean=None, variance=None)
source.samples(key, N)

Mixed_Identity(domain)
Mixed_NSF(key, domain, *, bins=8, transforms=6, euclidean_bound=5.0,
          hidden_features=(64, 64), slope=1e-3,
          activation=jax.nn.silu, mask_strategy="random")
```

Use `domain = target.domain`. The mixed chart is contiguous `R^p x T^q`, with
Euclidean coordinates first and periodic torsions last. `mask_strategy` is
`"random"` or `"balanced"`.

## jflows_md training and Boltzmann generators

Canonical trainers from `jflows_md.train` consume already prepared,
fixed-shape pools:

```python
train_forward_KLX_G(
    target_samples, source_samples, source, target, flow,
    batch_size, train_steps, lr, coeff_lambda=1.0, monitor=None,
    seed=0, checkpoint=False, *, initialize_from_identity=False,
    u_clip=inf, g_clip=inf, lr_warmup=0, t_start=0.0, t_end=1.0,
)
train_forward_KLXX_G(
    target_samples, source_samples, hat_samples, source, target, flow,
    domain, batch_size, train_steps, lr, coeff_lambda=1.0,
    coeff_alpha=0.5, coeff_beta=0.5, mc_dt=1e-3, mc_steps=1,
    mc_image_radius=3, monitor=None, seed=0, checkpoint=False, *,
    initialize_from_identity=False, u_clip=inf, g_clip=inf,
    lr_warmup=0, t_start=0.0, t_end=1.0,
)
```

Both return `(flow, batch_ess_hist)`. Batch ESS is the pre-update
proposal-to-target monitor and never an acceptance rule.

Canonical adaptive-staging drivers from `jflows_md.boltzmann` are
keyword-oriented:

```python
boltzmann_identity(
    x_valid, source, target, ladder, mc_dt, mc_steps, *,
    rg_param_0, rg_param_1, monitor=None, bg_param=None, chunks=1,
    mc_image_radius=3, seed=0,
)
boltzmann_forward_KLX_G(
    x_valid, source, target, flow, pool_size, batch_size,
    train_steps, lr, ladder, mc_dt, mc_steps, *,
    rg_param_0, rg_param_1, initialize_from_identity=True, coeff_lambda=1.0,
    monitor=None, bg_param=None, chunks=1, mc_image_radius=3,
    seed=0, checkpoint=False, u_clip=inf, g_clip=inf, lr_warmup=0,
)
boltzmann_forward_KLXX_G(
    x_valid, source, target, flow, pool_size, batch_size,
    train_steps, lr, ladder, melt, opt_alpha, opt_steps, mc_dt, mc_steps, *,
    rg_param_0, rg_param_1, initialize_from_identity=True,
    coeff_lambda=1.0, coeff_alpha=0.5, coeff_beta=0.5,
    monitor=None, bg_param=None, chunks=1, mc_image_radius=3,
    seed=0, checkpoint=False, u_clip=inf, g_clip=inf, lr_warmup=0,
)
```

Molecular `boltzmann_identity` has no flow, pool, batch, optimizer,
checkpoint, or initialization controls. It retains the molecular stage
sequence: SMC endpoint selection, complete-population identity ESS,
resampling and mixed MALA at the pre-sharpen endpoint, exact regularization
reweighting with its own `tau_ess` gate, then resampling and mixed MALA at the
post-sharpen endpoint. Sharpening therefore remains active whenever
`rg_param_0 != rg_param_1`.

`rg_param_0 == rg_param_1` gives fixed regularization. Different endpoint
pairs enable linear sharpening: the flow is trained at `rg_start`, then
reweighting and mixed MALA move the population to `rg_end`. Sharpening ESS uses
the same `tau_ess` rejection/shrink gate as the selected flow ESS.

Trained molecular stage records add `rg_start`, `rg_end`, `flow_rg`, `population_rg`,
`flow_endpoint="pre_sharpen"`, `sharpen_ess`, `sharpen_ess_hist`,
`sharpen_mala_acceptance`, `valid_sample_count`, `smc_ess`, `smc_acceptance`,
`mala_acceptance`, `hat_mala_acceptance`, `objective`, and
`initialized_from_identity` to the generic fields.
Post-training, full-validation trained-versus-identity ESS alone selects the
map and controls training-attempt acceptance, shrink, and retry. Before
training, `tau_smc` independently gates candidate bridge levels using SMC ESS.
As with generic BG, each molecular stage map and ESS are incremental; reload
and compose all selected maps in stage order. `hat_mala_acceptance` is `None`
for KLX and an acceptance history for KLXX.
Direct trainers preserve the supplied flow unless
`initialize_from_identity=True`. Molecular Boltzmann generators default it to
`True` for every stage/retry; `False` warm-starts each stage from the preceding
selected flow while retries reuse the immutable stage-entry template. The
independent identity validation fallback is present in both modes.
Molecular identity records instead use `objective="identity"` and
`selected="identity"`; they retain regularization, identity ESS, SMC, both
MALA, and sharpening histories while omitting every flow, trained ESS, batch
ESS, and initialization field.

## jflows_md samplers, OpenMM, and artifacts

Import molecular samplers from `jflows_md.utils`:

```python
wrapped_normal_relative_error_bound(dt=1e-4, image_radius=3)
mixed_mala_step(key, samples, potential, domain, *,
                dt=1e-4, image_radius=3)
mixed_mala(key, samples, potential, domain, *,
           dt=1e-4, steps=1, image_radius=3, chunks=1)
mixed_quench_and_temper(key, samples, potential, domain, *, melt=0.0,
                        opt_alpha=1e-2, opt_steps=100,
                        mc_dt=1e-3, mc_steps=100,
                        mc_image_radius=3, chunks=1)
sequential_monte_carlo(key, samples, source, target, ladder=1,
                       mc_dt=1e-3, mc_steps=100,
                       mc_image_radius=3, domain=None, chunks=1)
potential_space_smc(key, samples, source, target, t_list,
                    mc_dt=1e-3, mc_steps=100,
                    mc_image_radius=3, domain=None, chunks=1)
annealed_importance_sampling(key, samples, source, target, flow,
                             ladder=1, mc_dt=1e-3, mc_steps=100,
                             mc_image_radius=3, domain=None, chunks=1,
                             return_initial_log_weights=False)
```

Mixed MALA is always Metropolis-adjusted and has no `adjust` argument.
`mixed_mala_step` returns `(samples, accepted)` with one Boolean per chain;
`mixed_mala` and `mixed_quench_and_temper` return a chunk-weighted mean
acceptance history of shape `(steps,)` or `(mc_steps,)`. Molecular SMC returns
`(particles, level_ess, level_step_acceptance)`, with acceptance shaped
`(levels, mc_steps)`. G-native molecular AIS takes no direction string and
optionally returns `(particles, initial_log_weights)`; like generic
flow-proposal AIS, it rejuvenates at the final target. `smc` and `ais` are
stable short names shared with `jflows`; the long names remain clearest in
public examples.

Import native OpenMM support from `jflows_md.openmm`:

```python
OpenMM_Potential.from_bundle(path_or_name, *, root=None, verify=True,
                             temperature_kelvin=None)
potential.regularized(rg_param)
langevin(potential, positions_nm=None, *, steps=1000,
         sample_interval=100, timestep_fs=1.0, friction_per_ps=1.0,
         temperature_kelvin=None, seed=0, platform=None)
parallel_tempering(potential, temperatures_kelvin, positions_nm=None, *,
                   rounds=100, steps_per_round=100, timestep_fs=1.0,
                   friction_per_ps=1.0, seed=0, platform=None)
```

`OpenMM_Potential` is Cartesian and uses native OpenMM systems independently of
the JAX internal-coordinate `Molecular_Potential`. Its regularized form uses
the same `(e, r)` convention.

`jflows_md.artifacts` re-exports the generic medium-level helpers:

```python
save_flow(path, flow)
load_flow(path, template)
save_samples(path, samples)
load_samples(path)
save_history(path, **history)
load_history(path)
```

Complete-stage molecular persistence is separate:

```python
from jflows_md.boltzmann.write import create, stage, finish
from jflows_md.boltzmann.load import (
    manifest, validate, load, fork,
    load_stage_flow, load_validation_samples, load_training_history, run,
)

run(run_dir, problem_id, config, samples, flow, iterate, *, resume=False)
```

For identity persistence, pass `flow=None`; no `.eqx` files are stored,
`load(run_dir)` needs no template, and the returned continuation is `None`.
`run.json` stores the exact experiment config; drivers should include
`valid_size` when it is part of the experiment definition. `.eqx` stores flow
leaves, `.npy` stores arrays, and `.npz` stores named stage histories, so those
binary formats do not encode driver constant names such as `VALID_SIZE`.

A molecular bundle contains exactly `manifest.json`, `system.json`,
`coordinates.json`, `validation.json`, `system.xml`, and `reference.pdb`.
The loader accepts only the current quotient-measure coordinate schema and
does not migrate other bundle or run formats.

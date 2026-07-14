# Glycerol KL loss/ESS diagnostic

This interrupted stage-1 run is retained as a reproducible diagnostic, not as
a completed Boltzmann-generator result. It used the production 36D flow,
`N_POOL=200000`, `N_BATCH=40000`, `STEPS=1000`, and `N_VALID=1000000` at bridge
coefficient `t=0.02`. No GPU rerun is needed to inspect the stored trajectory.

`attempt_flows/stage_01_attempt_01_t_0.02000000/` contains:

- `flow_step_*.eqx`: the initial flow and post-update flows at steps 10, 25,
  50, 100, 250, 500, 750, and 1000;
- `diagnostics.npz`: all one-million-sample proposal log weights at those
  checkpoints, the 200000-particle fixed training pool, validation particles,
  SMC histories, and optimizer histories;
- the parent `train_status.log`: the original run transcript.

The honest full-validation proposal ESS trajectory was

```text
step       0       10       25       50      100      250      500      750     1000
ESS   0.00911  0.01268  0.07077  0.14181  0.18959  0.09721  0.00478  0.00218  0.00090
```

The transcript's per-step `ESS` is the standard batch training diagnostic
`compute_ESS_log(z)` evaluated on current-target training samples. It is
distinct from the full-validation proposal ESS reported at sparse checkpoints;
both are retained under explicit ESS names.

## Diagnosis

The loss/ESS behavior is not caused by a sign error, G/F direction error,
Jacobian error, ESS implementation error, or `e_clip` singularity screen.
Independent float64 recomputation agrees with the stored proposal ESS, all
200000 training energies were finite, the kept fraction was exactly one, and
`e_clip=1000` was inactive. Runs with finite and infinite clipping were
parameter-identical; aggressive clipping made ESS worse.

The principal effect is fixed-pool overfitting combined with an objective
mismatch. The approximately 3.1-million-parameter flow sees about 200 effective
passes over the fixed training pool. Its training-pool mean log ratio keeps
falling, while an independently generated 20000-particle target holdout reaches
its best mean near step 100 and then worsens. The generalization gap grows from
about 0.005 at initialization to about 9.6 at step 1000. Forward KL also does
not control the Renyi-2/chi-square moment that determines importance ESS, so a
decreasing KL estimate need not imply monotonically increasing ESS. A smaller
controlled KL+X experiment continued improving proposal ESS through step 500,
where bare KL had already declined.

Rare molecular collision tails improved during training rather than causing
the collapse. The late failure is instead high-weight undercoverage: the top
100 validation weights carried about 0.76% of mass at step 100 and 21.94% at
step 1000.

A separate gradient audit found occasional rejected float32 updates to be a
path-specific numerical event at an unsaved spline state. Replaying the exact
pool and keys produced no rejection; stored flows, values, and gradients were
finite and gradient norms stayed below `g_clip`. These sparse guarded no-ops do
not explain the ESS trajectory and do not justify changing the optimizer or
spline implementation.

The live `jflows_md` driver therefore supports an optional sparse
`selection_steps` schedule. It evaluates exact identity, the pre-update warm
start, requested post-update flows, and the final flow using honest
full-validation proposal ESS, then applies the unchanged `tau_ess` gate. The
empty schedule preserves the original final-versus-identity behavior.

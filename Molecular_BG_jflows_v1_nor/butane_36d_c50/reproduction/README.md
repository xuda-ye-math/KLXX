# n-butane stage-2 singularity reproduction

This capsule preserves the float32 inputs that reproduce the failed adaptive
KLXX bridge from `t = 0.07` to `t = 0.175`. It is independent of later
`artifacts/` cleanup: the accepted level-1 flow and all three particle pools
needed by the trainer live here.

## Contents

- `stage2_t0175.h5`: initial source particles, the accepted level-1 particle
  population, the exact level-2 SMC pool, the exact level-2 quench-and-temper
  pool, and MALA/SMC diagnostics.
- `stage1_selected_flow.eqx` and its JSON sidecar: accepted level-1
  `Mixed_NSF` used by the failed warm start.
- `stage2_t0175.json`: controller parameters, deterministic trainer seed, and
  summary checks.
- `one_step_results.json`: one-step batch-size, learning-rate, and identity
  controls.
- `pool_stats.json`: coordinate, energy, log-Jacobian, and density-ratio tail
  statistics.
- `gradient_stats.json`: source-potential force finiteness and tail statistics.
- `t_sweep.json`: full-population incremental ESS versus the proposed next
  bridge coefficient.

The scripts one directory above regenerate or inspect these files:
`reproduce_stage2.py`, `diagnose_stage2.py`, `inspect_stage2.py`,
`inspect_stage2_gradients.py`, and `sweep_stage2_t.py`.

## Run

```bash
source ~/.envs/jflows/bin/activate
export PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md:/mnt/projects/X-regularization/Molecular_BG
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python diagnose_stage2.py warm_b40000_lr1e-4
python diagnose_stage2.py identity_b40000_lr1e-4
python inspect_stage2.py
python inspect_stage2_gradients.py
python sweep_stage2_t.py
```

`reproduce_stage2.py` reconstructs the HDF5 file from the accepted flow and
the deterministic controller keys. Run it only while a matching accepted
level-1 flow still exists under `artifacts/flows/`.

## Supported diagnosis

The failure is not caused by the size of the proposed `0.07 -> 0.175` bridge.
At `t = 0.071`, only `0.001` beyond the accepted level, the identity map has
ESS `0.999813`, while the reused level-1 flow already has ESS
`0.00000250012`. The warm-start ESS stays at that value for every tested
coefficient through `t = 0.175`.

The molecular controller reuses the accepted incremental map as the next
stage's initial trainable flow. That map is not a safe near-identity transport
for the next bridge. Ordinary level-2 SMC samples have standardized Euclidean
coordinates no larger than `5.60` and benign current energies, but the warm
flow maps some of them to sterically singular molecular configurations. The
previous-bridge energy at the mapped latent reaches `1.86e17`, and its largest
force component reaches `4.20e19`; the flow log-Jacobian remains small. Thus
the large loss comes from the differentiated stage-source term
`U_previous(G(y))`, not from the RQS log determinant or the target samples.

The optimizer's `e_clip` mask checks only the current target energy at `y`.
It does not screen the previous-bridge energy or force at the flow latent.
The quench-and-temper pool also contains 169 rows with nonfinite identity
source forces, including 5 rows that pass the current target-energy mask.
Consequently a 40,000-sample step can have a finite reported loss but a
nonfinite gradient; the atomic optimizer correctly rejects that entire update.

The exact one-step controls show:

| Start | Batch | LR | Pre-step loss | Pre-step full ESS | Update applied |
|:---|---:|---:|---:|---:|:---:|
| warm | 40000 | 1e-4 | 1.4005e13 | 2.5001e-6 | no |
| identity | 40000 | 1e-4 | -19.003 | 0.081062 | no |
| warm | 20000 | 1e-4 | 2.8011e13 | 2.5001e-6 | yes |
| warm | 10000 | 1e-4 | 5.6021e13 | 2.5001e-6 | yes |
| identity | 10000 | 1e-4 | -19.048 | 0.081062 | no |
| warm | 10000 | 1e-5 | 5.6021e13 | 2.5001e-6 | yes |
| warm | 10000 | 1e-6 | 5.6021e13 | 2.5001e-6 | yes |

The nearly exact inverse scaling of warm loss with batch size identifies one
dominant extreme density-ratio sample. Lowering the learning rate scales the
parameter displacement but cannot improve a proposal that is already
singular before the first update.

## Fixes to evaluate next

No package fix is applied by this capsule. The evidence supports testing both
of the following in `jflows_md`:

1. initialize each molecular incremental stage from an exact identity flow,
   rather than reusing the preceding stage's incremental map;
2. add a molecular source-side finite-energy/finite-force guard for
   `U_previous(G(y))`, with an explicitly audited bias and an AD-safe mask.

The first addresses the catastrophic warm proposal. The second is still
needed because the identity control can encounter nonfinite source forces from
the QT pool. SMC/MALA selection, learning rate, and a smaller bridge alone do
not address these two causes.

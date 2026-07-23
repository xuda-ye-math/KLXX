# Achiral-molecule OpenMM regularization study

This standalone folder tests the fixed molecular regularization pair
`rg_param = (100.0, 0.15)` on NMA, glycerol, and neutral diethanolamine. The
alkane family was tested separately and cyclohexane is not part of this study.

The frozen protocol and decision gates are in `.aris/EXPERIMENT_PLAN.md`.
Native OpenMM trajectories are resumable: an existing artifact is reused only
when its metadata, shape, hashes, and finite-value checks match the current
configuration exactly.

Run with the shared `jflows` environment and disabled JAX preallocation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/xuda/.envs/jflows/bin/python run.py sanity
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/xuda/.envs/jflows/bin/python run.py production
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/xuda/.envs/jflows/bin/python analyze.py production
```

If the frozen mixing checks require the preregistered longer trajectories:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/xuda/.envs/jflows/bin/python run.py extended
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/xuda/.envs/jflows/bin/python analyze.py extended
```

The analysis writes `REPORT.md`, machine-readable metrics under `results/`,
and plots of RESS, torsional fidelity, and the force stress panel. Raw `.npz`
trajectories under `data/` are ignored by Git but remain part of the local and
backup artifact record.

After the common-pair screen, NMA was fixed at `(100, 0.15)` and a separate,
hash-preserving candidate addendum was opened for glycerol and neutral
diethanolamine:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/xuda/.envs/jflows/bin/python run_candidates.py pilot
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/xuda/.envs/jflows/bin/python analyze_candidates.py pilot
```

The selected molecule-specific pairs are verified with `run_candidates.py
verification --molecule ... --e ... --r ...`; the exact candidate grid and
gates are frozen in `candidate_config.json`.

The final user-selected mapping and candidate-to-raw RESS report are generated
without changing either frozen configuration:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
  /home/xuda/.envs/jflows/bin/python finalize_selected.py
```

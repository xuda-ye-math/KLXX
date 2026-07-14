# Controlled molecular-training diagnostics

This private experiment isolates molecular target complexity from the
`jflows_md` loss, importance-weight, and Adam implementations.  It proceeds in
the fixed order

1. independent `R^7 x T^2` loss/gradient/Adam/weight oracle;
2. explicit-H methane bundle and all-24-H-permutation validation;
3. direct bare-KL methane training against the soft `c50_rho001` target;
4. ethane, propane, and n-butane only after every earlier gate passes.

The public package repositories are live, read-only dependencies.  Local runs
activate the pip-only environment and expose both source roots explicitly:

```bash
source ~/.envs/jflows/bin/activate
export PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md
python Molecular_BG/training_diagnostics/oracle.py
python Molecular_BG/training_diagnostics/train_alkanes.py methane
```

Unvalidated constructions live below `candidates/`.  Only independently
reviewed candidates that pass the complete scientific gate are copied below
`bundles/` and entered in the private registry; they are never added to the
public `jflows_md` registry. Runtime loads the promoted explicit path. Immutable
raw runs live below `runs/` in this folder. Plotting and tables must consume
saved artifacts and must not rerun sampling.

The preregistered parameters, gates, seed roles, finite-`N` ESS qualification,
and outcome rules are authoritative in
`.aris/experiments/molecular_training_diagnostics/EXPERIMENT_PLAN.md`.

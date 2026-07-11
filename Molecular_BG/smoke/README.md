# jflows_md smoke tests

Run the complete pure-JAX suite from the repository root:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/.envs/jax/bin/python Molecular_BG/smoke/run_all.py
```

The tests verify bundle hashes and metadata, pure-JAX energies and forces
against stored OpenMM Reference results for all three molecules, BAT/chart
round trips and Jacobians, ADP L-only support, full small-molecule parity
support, source shapes, JIT compatibility, and a mixed-domain MALA step.
It also exercises a two-rung potential-space SMC bridge on a mixed-domain toy
target.

Rebuild the bundles only when their versioned model definition changes:

```bash
conda run -n jflows python Molecular_BG/bundles/build_molecular_bundles.py
```

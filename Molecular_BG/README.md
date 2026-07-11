# Molecular Boltzmann-generator benchmark

This directory contains the local `jflows_md` molecular potential milestone
and three frozen molecular targets. A user evaluating or training against a
target needs only `jflows/`, `jflows_md/`, and the selected directory under
`bundles/`. The `reference/` directory is evaluation data and is never read
by `Molecular_Potential`.

## Layout

```text
Molecular_BG/
├── README.md
├── JFLOWS_MD_PLAN.md
├── bundles/
│   ├── build_molecular_bundles.py
│   ├── fab_adp_ff96_obc1_v1/
│   ├── glycerol_gaff2_am1bcc_obc1_v1/
│   └── diethanolamine_neutral_gaff2_am1bcc_obc1_v1/
├── reference/
│   ├── adp_truth.png
│   ├── fab_train.h5
│   └── fab_train_phi_psi.npz
├── jflows/
├── jflows_md/
└── smoke/
```

The copied `jflows/` source is unmodified from
`/mnt/projects/jflows/jflows/` at commit
`fcee81c432d95cb4d841a14605e133b54d08c84d`. The molecular additions live
only in the parallel `jflows_md/` package.

## Runtime use

```python
import jax
jax.config.update("jax_enable_x64", True)

from jflows_md import Molecular_Potential

target = Molecular_Potential.from_bundle("fab_adp_ff96_obc1_v1")
q = target.source().samples(jax.random.key(0), 32)
energy = target(q)       # [32]
gradient = target.grad(q)  # [32, 60]
```

Run user programs with `Molecular_BG` on `PYTHONPATH`. The complete selected
bundle is the runtime model: it freezes the force field, implicit solvent,
coordinate chart, chirality support, validation frame, metadata, and integrity
hashes. No PDB, trajectory, asset directory, or FAB sample file is consulted
at runtime.

See `jflows_md/README.md` for the public API and the three model definitions.

## Bundle maintenance and validation

Bundles rebuild in place from canonical seed artifacts already stored inside
each bundle:

```bash
conda run -n jflows python Molecular_BG/bundles/build_molecular_bundles.py
```

The builder is a maintenance tool requiring OpenMM, ParmEd, and AmberTools. It
is not imported during JAX potential evaluation.

Run the complete accelerator-backed smoke suite with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/.envs/jax/bin/python Molecular_BG/smoke/run_all.py
```

## FAB ground-truth reference

`reference/fab_train.h5` is the FAB authors' one-million-configuration REMD
training split. `fab_train_phi_psi.npz` is its cached Ramachandran projection,
and `adp_truth.png` is the corresponding 100 x 100 unsmoothed histogram with
logarithmic color range `[1e-4, 1]`. These files support evaluation and
plotting comparisons but do not define the potential.

SHA-256:

```text
c0c9da5d4e5f9d7ed04385ef9fe8612d4754fcbdabd5f1867fbd503e3673d0fe  fab_train.h5
84635035228c7bcd90935a53923f140b91668344983141d9af18c6030baf827a  fab_train_phi_psi.npz
0d59591a3188da5a2069f4f7ae54c12c58d66d16510f954053c6f21f49a11a04  adp_truth.png
```

The bundle-driven potential, mixed-domain MALA, and potential-space SMC are
implemented. Full mixed-spline flow and Boltzmann-generator training remain the
next milestone.

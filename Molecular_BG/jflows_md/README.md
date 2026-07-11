# `jflows_md`

`jflows_md` is the molecular companion to the local `jflows` package. It keeps
the user-facing API small while storing force-field evaluation, internal
coordinates, chirality charts, validation, and bundle construction in
`jflows_md/core/`.

## Public API

Import ordinary user-facing objects directly from `jflows_md`:

```python
import jax
jax.config.update("jax_enable_x64", True)

from jflows_md import Molecular_Potential, available_bundles

print(available_bundles())
target = Molecular_Potential.from_bundle("fab_adp_ff96_obc1_v1")
q = target.reference_internal()[None, :]
energy = target(q)                    # shape (1,), dimensionless
gradient = target.grad(q)             # shape (1, 60)
cartesian_nm = target.cartesian(q)     # shape (1, 22, 3)
source = target.source()
```

The top-level exports are `Molecular_Potential`, `Molecular_Bundle`,
`Molecular_Source`, `Mixed_Identity`, `mixed_mala`,
`potential_space_smc`, and `available_bundles`. Modules below `core/` are
implementation details and are deliberately not re-exported.

Run from the project root with `PYTHONPATH=Molecular_BG`, or add
`Molecular_BG` to the active environment's development paths.

## Frozen molecular targets

| Bundle | Model | Mixed coordinate domain |
|---|---|---|
| `fab_adp_ff96_obc1_v1` | FAB-compatible Amber ff96/OBC1, L-ADP only | R^42 x T^18 (60D) |
| `glycerol_gaff2_am1bcc_obc1_v1` | GAFF2/AM1-BCC/OBC1, neutral | R^25 x T^11 (36D) |
| `diethanolamine_neutral_gaff2_am1bcc_obc1_v1` | GAFF2/AM1-BCC/OBC1, explicitly neutral | R^33 x T^15 (48D) |

Every target uses 300 K, mbondi2 radii, ACE nonpolar solvation, solvent and
solute dielectric constants 78.5 and 1.0, zero salt, `NoCutoff`, and no
constraints. A PDB is included for inspection but is not treated as a complete
Hamiltonian. Each immutable bundle also freezes the topology, parameters,
coordinate chart, model metadata, validation frames, and SHA-256 hashes.

The reduced internal-coordinate potential is

```text
U(q) = beta E_bundle(x(q)) - log |det(dx/dq)|.
```

No sharpening or clipped surrogate is part of this target. ADP uses a chiral
half-chart to exclude the unwanted enantiomer; glycerol and neutral
diethanolamine retain both signs of their diagnostic determinant because those
signs are not fixed stereocentres.

## Build and validate

The checked-in bundles are sufficient for JAX runtime use and require neither
OpenMM nor AmberTools. For maintenance, rebuild them in place from the
canonical seed artifacts already frozen inside each bundle:

```bash
conda run -n jflows python Molecular_BG/bundles/build_molecular_bundles.py
```

Run the complete accelerator-backed validation suite with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/.envs/jax/bin/python Molecular_BG/smoke/run_all.py
```

The suite verifies bundle integrity, coordinate round trips and Jacobians,
chirality support, finite and JIT-compatible values/gradients, pure-JAX energy
and force parity against stored OpenMM Reference results, and mixed-domain
MALA. Full mixed-spline BG training is the next milestone; `Mixed_Identity` is
only the safe baseline transform for this potential milestone.

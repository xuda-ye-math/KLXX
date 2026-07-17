# Ethane 18D e/r regularization test

This is the full-size ethane KLXX run using the two main molecular potential
regularization constants:

- `energy_threshold_kj_mol = 100.0` kJ/mol (`e`);
- `pair_distance_floor_nm = 0.1` nm (`r`).

Every model, optimizer, sampling, validation, and adaptive-bridge setting is
identical to the completed methane 9D run except `t_safe`, which is reduced
from `0.30` to `0.20`. The molecule identity, dimension, and physical bundle
necessarily change to ethane 18D.

The physical six-file bundle is copied byte-for-byte from
`Molecular_BG_jflows_v1_nor/ethane_18d_c50/bundle/`. `train.py` calls the
current `jflows_md` v0.3.0 interface and records every optimizer step's
pre-update KLXX batch ESS in `results/per_step_ess.csv` and
`artifacts/per_step_ess.npz`.

The completed replacement run on 2026-07-17 reached `t=1` in four accepted
stages at `t=0.20/0.50/0.95/1.00`. Full-validation selected ESS was
`0.826795/0.965812/0.960052/0.998610`; the trained flow was selected at every
stage. All 2,000 recorded per-step ESS values and all 200,000 final validation
particles are finite. The earlier same-session `t_safe=0.25` outputs were
removed and overwritten by this requested `t_safe=0.20` run.

To reproduce after deliberately removing the existing outputs, run from this
directory with:

```bash
source ~/.envs/jflows/bin/activate
PYTHONPATH=/data/projects/jflows:/data/projects/jflows_md \
  python train.py
```

The driver refuses to overwrite a nonempty `artifacts/` or `results/`
directory.

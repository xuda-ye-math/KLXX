# Methane 9D e/r regularization test

This workspace is reserved for the full-size methane KLXX test using the two
main molecular potential regularization constants:

- `energy_threshold_kj_mol = 100.0` kJ/mol (`e`);
- `pair_distance_floor_nm = 0.1` nm (`r`).

The physical six-file methane bundle is copied byte-for-byte from
`Molecular_BG_jflows_v1_nor/methane_9d_c50/bundle/`. The legacy c50 code is not
used: `train.py` calls the current `jflows_md` v0.3.0 interface directly.

The completed run used the full 200,000-particle validation population,
`mc_dt=1e-3`, and `mc_steps=100`. It writes every
optimizer step's pre-update KLXX batch ESS to `results/per_step_ess.csv`, an
exact NumPy copy to `artifacts/per_step_ess.npz`, and stage/attempt summaries to
`results/ess.md` and `artifacts/summary.json`.

The run completed on 2026-07-17 in four accepted stages. To reproduce it after
deliberately removing the existing outputs, run from this directory with:

```bash
source ~/.envs/jflows/bin/activate
PYTHONPATH=/data/projects/jflows:/data/projects/jflows_md \
  python train.py
```

The driver refuses to overwrite a nonempty `artifacts/` or `results/`
directory.

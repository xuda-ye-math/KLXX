# Methane KLXX with e/r regularization

- `energy_threshold_kj_mol` (e): `100` kJ/mol
- `pair_distance_floor_nm` (r): `0.1` nm
- Full validation population: `200000`
- Mixed MALA: `mc_dt=0.001`, `mc_steps=100`
- `per_step_ess.csv` records the pre-update KLXX batch ESS at every optimizer step.

## Full-validation stage ESS

| Level | t | Selected | Selected ESS | Trained ESS | Identity ESS |
|---:|---:|:---:|---:|---:|---:|
| 1 | 0.210000 | trained | 0.978385 | 0.978385 | 0.023208 |
| 2 | 0.525000 | trained | 0.994428 | 0.994428 | 0.617977 |
| 3 | 0.997500 | trained | 0.998197 | 0.998197 | 0.751080 |
| 4 | 1.000000 | identity | 0.999992 | 0.999796 | 0.999992 |

## Per-attempt optimizer ESS

| Level | Attempt | t | Status | Steps | First | Last | Minimum | Maximum | Mean | Updates |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.210000 | accepted | 500 | 0.022687 | 0.980645 | 0.022687 | 0.983127 | 0.880024 | 493 |
| 2 | 1 | 0.525000 | accepted | 500 | 0.612304 | 0.995313 | 0.612304 | 0.996641 | 0.981786 | 500 |
| 3 | 1 | 0.997500 | accepted | 500 | 0.752485 | 0.998717 | 0.752485 | 0.998963 | 0.991123 | 500 |
| 4 | 1 | 1.000000 | accepted | 500 | 0.999994 | 0.999838 | 0.997514 | 0.999994 | 0.999568 | 500 |

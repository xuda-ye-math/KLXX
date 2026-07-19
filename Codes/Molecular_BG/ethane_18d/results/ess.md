# Ethane 18D KLXX with e/r regularization

- `energy_threshold_kj_mol` (e): `100` kJ/mol
- `pair_distance_floor_nm` (r): `0.1` nm
- Full validation population: `200000`
- Mixed MALA: `mc_dt=0.001`, `mc_steps=100`
- `per_step_ess.csv` records the pre-update KLXX batch ESS at every optimizer step.

## Full-validation stage ESS

| Level | t | Selected | Selected ESS | Trained ESS | Identity ESS |
|---:|---:|:---:|---:|---:|---:|
| 1 | 0.200000 | trained | 0.826795 | 0.826795 | 0.000588 |
| 2 | 0.500000 | trained | 0.965812 | 0.965812 | 0.334034 |
| 3 | 0.950000 | trained | 0.960052 | 0.960052 | 0.484416 |
| 4 | 1.000000 | trained | 0.998610 | 0.998610 | 0.992609 |

## Per-attempt optimizer ESS

| Level | Attempt | t | Status | Steps | First | Last | Minimum | Maximum | Mean | Updates |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.200000 | accepted | 500 | 0.000681 | 0.827639 | 0.000461 | 0.837101 | 0.589154 | 491 |
| 2 | 1 | 0.500000 | accepted | 500 | 0.336564 | 0.968258 | 0.336564 | 0.968664 | 0.921107 | 500 |
| 3 | 1 | 0.950000 | accepted | 500 | 0.482684 | 0.960767 | 0.482684 | 0.964174 | 0.932486 | 500 |
| 4 | 1 | 1.000000 | accepted | 500 | 0.992630 | 0.998709 | 0.990258 | 0.999026 | 0.997667 | 500 |

# Propane 27D KLXX with e/r regularization

- `energy_threshold_kj_mol` (e): `50` kJ/mol
- `pair_distance_floor_nm` (r): `0.1` nm
- Full validation population: `400000`
- Mixed MALA: `mc_dt=0.001`, `mc_steps=100`
- `per_step_ess.csv` records the pre-update KLXX batch ESS at every optimizer step.

## Full-validation stage ESS

| Level | t | Selected | Selected ESS | Trained ESS | Identity ESS |
|---:|---:|:---:|---:|---:|---:|
| 1 | 0.100000 | trained | 0.716479 | 0.716479 | 0.120827 |
| 2 | 0.205000 | trained | 0.491296 | 0.491296 | 0.058591 |
| 3 | 0.362500 | trained | 0.485802 | 0.485802 | 0.155452 |
| 4 | 0.598750 | trained | 0.874668 | 0.874668 | 0.420977 |
| 5 | 0.953125 | trained | 0.939270 | 0.939270 | 0.484118 |
| 6 | 1.000000 | trained | 0.996787 | 0.996787 | 0.989253 |

## Per-attempt optimizer ESS

| Level | Attempt | t | Status | Steps | First | Last | Minimum | Maximum | Mean | Updates |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.100000 | accepted | 500 | 0.139816 | 0.720905 | 0.100332 | 0.726928 | 0.569923 | 465 |
| 2 | 1 | 0.250000 | rejected | 500 | 0.012053 | 0.395363 | 0.010579 | 0.417040 | 0.269106 | 499 |
| 2 | 2 | 0.205000 | accepted | 500 | 0.067600 | 0.506594 | 0.056655 | 0.514540 | 0.396649 | 495 |
| 3 | 1 | 0.362500 | accepted | 500 | 0.155555 | 0.486741 | 0.155450 | 0.491838 | 0.445938 | 500 |
| 4 | 1 | 0.598750 | accepted | 500 | 0.418732 | 0.873404 | 0.418732 | 0.879825 | 0.838095 | 499 |
| 5 | 1 | 0.953125 | accepted | 500 | 0.485880 | 0.941322 | 0.485880 | 0.942392 | 0.909249 | 500 |
| 6 | 1 | 1.000000 | accepted | 500 | 0.989317 | 0.997053 | 0.984737 | 0.997508 | 0.994952 | 500 |

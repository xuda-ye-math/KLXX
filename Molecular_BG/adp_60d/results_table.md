# adp (d=60, heteroatoms via alanine_dipeptide.prmtop) — KLXX vs forward KL

pool=160000 batch=12000 valid=960000 | NCSF bins=32 transforms=10 hidden=(256, 256) | compiled_inverse=True

*KL+X sharpening (`klxx_sharpen`): not run yet.*

*forward-KL sharpening (`kl_sharpen`): not run yet.*

*KL+X fixed-cap (`klxx_raw`): not run yet.*

*forward-KL fixed-cap (`kl_raw`): not run yet.*

## KL+X sharpening + delta-QT (`klxx_delta_sharpen`) — complete=True, K=11, **F=67.95**, full-target ESS=0.000

| stage | t_k | SMC ESS | validation ESS | sharpen ESS |
|---|---|---|---|---|
| 1 | 0.100 | 0.973 | **0.819** | 0.981 |
| 2 | 0.198 | 0.931 | **0.599** | 0.992 |
| 3 | 0.294 | 0.952 | **0.543** | 0.993 |
| 4 | 0.388 | 0.770 | **0.512** | 0.987 |
| 5 | 0.433 | 0.798 | **0.580** | 0.948 |
| 6 | 0.464 | 0.907 | **0.598** | 1.000 |
| 7 | 0.508 | 0.993 | **0.846** | 1.000 |
| 8 | 0.569 | 0.993 | **0.831** | 1.000 |
| 9 | 0.654 | 0.989 | **0.803** | 1.000 |
| 10 | 0.773 | 0.985 | **0.807** | 1.000 |
| 11 | 1.000 | 0.979 | **0.755** | 1.000 |

*KL+X fixed-cap + delta-QT (`klxx_delta_raw`): not run yet.*

## annealed-SMC baseline, identity flow (`asmc`) — complete=True, K=12, **F=129.70**

| stage | t_k | SMC ESS | validation ESS | sharpen ESS |
|---|---|---|---|---|
| 1 | 0.100 | 0.974 | **0.571** | 0.979 |
| 2 | 0.198 | 0.944 | **0.575** | 0.989 |
| 3 | 0.294 | 0.775 | **0.602** | 0.993 |
| 4 | 0.388 | 0.754 | **0.552** | 0.990 |
| 5 | 0.433 | 0.781 | **0.665** | 0.974 |
| 6 | 0.464 | 0.887 | **0.537** | 1.000 |
| 7 | 0.508 | 0.992 | **0.825** | 1.000 |
| 8 | 0.569 | 0.993 | **0.818** | 1.000 |
| 9 | 0.654 | 0.989 | **0.754** | 1.000 |
| 10 | 0.773 | 0.984 | **0.694** | 1.000 |
| 11 | 0.932 | 0.979 | **0.651** | 1.000 |
| 12 | 1.000 | 0.997 | **0.927** | 1.000 |

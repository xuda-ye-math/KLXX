# diethanolamine (d=48, heteroatoms via diethanolamine.prmtop) — KLXX vs forward KL

pool=150000 batch=15000 valid=750000 | NCSF bins=32 transforms=8 hidden=(256, 256) | compiled_inverse=True

*KL+X sharpening (`klxx_sharpen`): not run yet.*

## forward-KL sharpening (`kl_sharpen`) — complete=True, K=6, **F=92.62**

| stage | t_k | SMC ESS | validation ESS | sharpen ESS |
|---|---|---|---|---|
| 1 | 0.100 | 0.906 | **0.602** | 1.000 |
| 2 | 0.198 | 0.817 | **0.597** | 0.999 |
| 3 | 0.294 | 0.820 | **0.511** | 0.992 |
| 4 | 0.428 | 0.827 | **0.461** | 0.825 |
| 5 | 0.617 | 0.849 | **0.468** | 0.630 |
| 6 | 1.000 | 0.902 | **0.530** | 0.998 |

*KL+X fixed-cap (`klxx_raw`): not run yet.*

*forward-KL fixed-cap (`kl_raw`): not run yet.*

## KL+X sharpening + delta-QT (`klxx_delta_sharpen`) — complete=True, K=5, **F=35.90**

| stage | t_k | SMC ESS | validation ESS | sharpen ESS |
|---|---|---|---|---|
| 1 | 0.100 | 0.907 | **0.745** | 1.000 |
| 2 | 0.240 | 0.824 | **0.567** | 0.996 |
| 3 | 0.377 | 0.814 | **0.485** | 0.922 |
| 4 | 0.652 | 0.804 | **0.429** | 0.467 |
| 5 | 1.000 | 0.927 | **0.740** | 0.999 |

*KL+X fixed-cap + delta-QT (`klxx_delta_raw`): not run yet.*

## annealed-SMC baseline, identity flow (`asmc`) — complete=True, K=10, **F=1455.44**

| stage | t_k | SMC ESS | validation ESS | sharpen ESS |
|---|---|---|---|---|
| 1 | 0.049 | 0.906 | **0.597** | 1.000 |
| 2 | 0.118 | 0.953 | **0.472** | 1.000 |
| 3 | 0.185 | 0.911 | **0.489** | 1.000 |
| 4 | 0.251 | 0.911 | **0.464** | 0.999 |
| 5 | 0.315 | 0.912 | **0.472** | 0.994 |
| 6 | 0.379 | 0.922 | **0.577** | 0.981 |
| 7 | 0.467 | 0.948 | **0.528** | 0.860 |
| 8 | 0.591 | 0.936 | **0.442** | 0.747 |
| 9 | 0.765 | 0.943 | **0.524** | 0.981 |
| 10 | 1.000 | 0.974 | **0.524** | 1.000 |

*annealed-SMC baseline, identity flow, fixed-cap (`asmc_raw`): not run yet.*

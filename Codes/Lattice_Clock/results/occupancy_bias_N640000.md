# Occupancy bias across training batch sizes (L=8 clock, staged sampler, N=640000)

err = (1/6) sum_s |p_s - 1/6|; mean and SEM over 4 matched sampling seeds [60001, 60002, 60003, 60004].

## kl

| | B=2000 | B=1000 | B=500 | B=250 |
|---|---|---|---|---|
| mean occupancy bias | 0.00573 | 0.00888 | 0.00963 | 0.02760 |
| sem (over seeds) | 0.00101 | 0.00093 | 0.00138 | 0.00293 |
| seeds | 4 | 4 | 4 | 4 |

## klxx

| | B=2000 | B=1000 | B=500 | B=250 |
|---|---|---|---|---|
| mean occupancy bias | 0.00235 | 0.00517 | 0.01082 | 0.01506 |
| sem (over seeds) | 0.00041 | 0.00071 | 0.00282 | 0.00259 |
| seeds | 4 | 4 | 4 | 4 |

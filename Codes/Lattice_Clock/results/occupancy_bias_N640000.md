# Occupancy bias across training batch sizes (L=8 clock, staged sampler, N=640000)

err = (1/6) sum_s |p_s - 1/6|; mean and SEM over 4 matched sampling seeds [60001, 60002, 60003, 60004].

## klx

| | B=2000 | B=1500 | B=1000 | B=500 |
|---|---|---|---|---|
| mean occupancy bias | 0.00320 | 0.00526 | 0.00922 | 0.01130 |
| sem (over seeds) | 0.00040 | 0.00080 | 0.00218 | 0.00208 |
| seeds | 4 | 4 | 4 | 4 |

## klxx

| | B=2000 | B=1500 | B=1000 | B=500 |
|---|---|---|---|---|
| mean occupancy bias | 0.00222 | 0.00433 | 0.00564 | 0.00815 |
| sem (over seeds) | 0.00027 | 0.00061 | 0.00135 | 0.00116 |
| seeds | 4 | 4 | 4 | 4 |

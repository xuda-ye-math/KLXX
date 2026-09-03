# Occupancy bias across training batch sizes (L=8 clock, staged sampler, N=640000)

err = (1/6) sum_s |p_s - 1/6|; mean and SEM over 4 matched sampling seeds [60001, 60002, 60003, 60004].

## klx

| | B=2000 | B=1000 | B=500 | B=250 |
|---|---|---|---|---|
| mean occupancy bias | 0.00320 | 0.00922 | 0.01130 | 0.02479 |
| sem (over seeds) | 0.00040 | 0.00218 | 0.00208 | 0.00151 |
| seeds | 4 | 4 | 4 | 4 |

## klxx

| | B=2000 | B=1000 | B=500 | B=250 |
|---|---|---|---|---|
| mean occupancy bias | 0.00222 | 0.00564 | 0.00815 | 0.02918 |
| sem (over seeds) | 0.00027 | 0.00135 | 0.00116 | 0.00906 |
| seeds | 4 | 4 | 4 | 4 |

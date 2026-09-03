# Occupancy-bias Monte Carlo scaling (L=8 clock, staged sampler, B=2000, KL+X_pi and KLXX)

err = (1/6) sum_s |p_s - 1/6|; equal total work per row (10000*256 particles); 2^(8-k) independent tests at N=10000*2^k; k=0,...,6.

## klx

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 |
|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.02428 | 0.01819 | 0.01333 | 0.01039 | 0.00741 | 0.00464 | 0.00320 |
| sem (over tests) | 0.00052 | 0.00052 | 0.00063 | 0.00060 | 0.00063 | 0.00034 | 0.00040 |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.34', '1.36', '1.28', '1.40', '1.60', '1.45']

Log-log slope of bias vs N: -0.484 (Monte Carlo rate = -0.5)

## klxx

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 |
|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.02089 | 0.01522 | 0.01104 | 0.00722 | 0.00609 | 0.00375 | 0.00222 |
| sem (over tests) | 0.00048 | 0.00061 | 0.00046 | 0.00043 | 0.00043 | 0.00044 | 0.00027 |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.37', '1.38', '1.53', '1.19', '1.62', '1.69']

Log-log slope of bias vs N: -0.521 (Monte Carlo rate = -0.5)

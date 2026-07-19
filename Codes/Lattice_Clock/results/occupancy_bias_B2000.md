# Occupancy-bias Monte Carlo scaling (L=8 clock, staged sampler, B=2000, kl and klxx)

err = (1/6) sum_s |p_s - 1/6|; equal total work per row (10000*256 particles); 2^(8-k) independent tests at N=10000*2^k; k=0,...,6.

## kl

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 |
|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.03735 | 0.02641 | 0.01956 | 0.01527 | 0.01012 | 0.00751 | 0.00552 |
| sem (over tests) | 0.00090 | 0.00098 | 0.00095 | 0.00105 | 0.00107 | 0.00070 | 0.00148 |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.41', '1.35', '1.28', '1.51', '1.35', '1.36']

Log-log slope of bias vs N: -0.459 (Monte Carlo rate = -0.5)

## klxx

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 |
|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.02397 | 0.01728 | 0.01227 | 0.00871 | 0.00669 | 0.00386 | 0.00358 |
| sem (over tests) | 0.00049 | 0.00050 | 0.00052 | 0.00054 | 0.00042 | 0.00049 | 0.00033 |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.39', '1.41', '1.41', '1.30', '1.73', '1.08']

Log-log slope of bias vs N: -0.479 (Monte Carlo rate = -0.5)

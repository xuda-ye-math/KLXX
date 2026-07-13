# Occupancy-bias Monte Carlo scaling (L=8 clock, staged sampler, B=2000, kl and klxx)

err = (1/6) sum_s |p_s - 1/6|; equal total work per row (10000*256 particles); 2^(8-k) independent tests at N=10000*2^k.

## kl

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 | N=1e4*2^7 | N=1e4*2^8 |
|---|---|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.02930 | 0.02064 | 0.01598 | 0.01068 | 0.00665 | 0.00518 | 0.00360 | 0.00368 | 0.00161 |
| sem (over tests) | 0.00070 | 0.00083 | 0.00072 | 0.00050 | 0.00074 | 0.00045 | 0.00065 | 0.00031 | nan |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.42', '1.29', '1.50', '1.60', '1.28', '1.44', '0.98', '2.28']

Log-log slope of bias vs N: -0.492 (Monte Carlo rate = -0.5)

## klxx

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 | N=1e4*2^7 | N=1e4*2^8 |
|---|---|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.01851 | 0.01310 | 0.00921 | 0.00685 | 0.00464 | 0.00385 | 0.00246 | 0.00157 | 0.00106 |
| sem (over tests) | 0.00042 | 0.00041 | 0.00042 | 0.00040 | 0.00040 | 0.00046 | 0.00037 | 0.00048 | nan |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.41', '1.42', '1.34', '1.48', '1.21', '1.57', '1.56', '1.48']

Log-log slope of bias vs N: -0.505 (Monte Carlo rate = -0.5)

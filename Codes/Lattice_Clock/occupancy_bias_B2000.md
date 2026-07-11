# Occupancy-bias Monte Carlo scaling (L=8 clock, staged sampler, B=2000, kl and klxx)

err = (1/6) sum_s |p_s - 1/6|; equal total work per row (10000*256 particles); 2^(8-k) independent tests at N=10000*2^k.

## kl

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 | N=1e4*2^7 | N=1e4*2^8 |
|---|---|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.02857 | 0.02160 | 0.01586 | 0.01143 | 0.00826 | 0.00610 | 0.00447 | 0.00207 | 0.00265 |
| sem (over tests) | 0.00066 | 0.00069 | 0.00077 | 0.00060 | 0.00054 | 0.00085 | 0.00052 | 0.00019 | nan |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.32', '1.36', '1.39', '1.38', '1.35', '1.37', '2.16', '0.78']

Log-log slope of bias vs N: -0.474 (Monte Carlo rate = -0.5)

## klxx

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 | N=1e4*2^7 | N=1e4*2^8 |
|---|---|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.01867 | 0.01396 | 0.01011 | 0.00648 | 0.00452 | 0.00277 | 0.00220 | 0.00140 | 0.00064 |
| sem (over tests) | 0.00039 | 0.00042 | 0.00045 | 0.00043 | 0.00045 | 0.00032 | 0.00024 | 0.00027 | nan |
| tests | 256 | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.34', '1.38', '1.56', '1.43', '1.63', '1.26', '1.58', '2.17']

Log-log slope of bias vs N: -0.584 (Monte Carlo rate = -0.5)

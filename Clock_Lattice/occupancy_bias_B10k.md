# Occupancy-bias Monte Carlo scaling (L=8 clock, staged sampler, balance B=10k, 6 maps)

err = (1/6) sum_s |p_s - 1/6|; equal total work per row (10000*128 particles); 2^(7-k) independent tests at N=10000*2^k.

| | N=1e4*2^0 | N=1e4*2^1 | N=1e4*2^2 | N=1e4*2^3 | N=1e4*2^4 | N=1e4*2^5 | N=1e4*2^6 | N=1e4*2^7 |
|---|---|---|---|---|---|---|---|---|
| mean occupancy bias | 0.07506 | 0.06364 | 0.05140 | 0.04298 | 0.03641 | 0.01888 | 0.01890 | 0.01719 |
| sem (over tests) | 0.00266 | 0.00349 | 0.00351 | 0.00447 | 0.00687 | 0.00138 | 0.00327 | nan |
| tests | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |

Adjacent-row ratios (N^(-1/2) predicts sqrt(2)=1.41): ['1.18', '1.24', '1.20', '1.18', '1.93', '1.00', '1.10']

Log-log slope of bias vs N: -0.336 (Monte Carlo rate = -0.5)

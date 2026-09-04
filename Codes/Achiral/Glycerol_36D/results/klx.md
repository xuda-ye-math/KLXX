# Glycerol 36D KLX

- Stage policy: `{'t_safe': 0.1, 'shrink_factor': 0.7, 'enlarge_factor': 1.4, 'tau_valid': 0.4, 't_tol': 0.001, 'max_stages': 20, 'max_retry': 5}`
- Screen fraction: `0.0001`
- Complete: `True`
- Regularization path: `(50.0, 0.2)` to `(100.0, 0.1)`
- Factor F_hat = prod ESS^(-1/2): `3.76837`
- Total time: `30.76 min`
- Effective training time (accepted attempts): `24.13 min`

| Stage | t | rho | Selected | Validation ESS | Trained ESS | Identity ESS | Time (min) |
|---:|---:|:---:|:---:|---:|---:|---:|---:|
| 1 | 0.100000 | (55.0, 0.19) | trained | 0.827530 | 0.827530 | 0.223180 | 3.54 |
| 2 | 0.198000 | (59.9, 0.1802) | trained | 0.538040 | 0.538040 | 0.056562 | 6.73 |
| 3 | 0.294040 | (64.702, 0.17059600000000003) | trained | 0.482881 | 0.482881 | 0.180552 | 6.57 |
| 4 | 0.428496 | (71.42479999999999, 0.15715040000000002) | trained | 0.593582 | 0.593582 | 0.244138 | 3.45 |
| 5 | 0.616734 | (80.83671999999999, 0.13832656000000004) | trained | 0.687764 | 0.687764 | 0.393528 | 3.48 |
| 6 | 0.880268 | (94.01340799999997, 0.11197318400000006) | trained | 0.834289 | 0.834289 | 0.490858 | 3.51 |
| 7 | 1.000000 | (100.0, 0.1) | trained | 0.961648 | 0.961648 | 0.889961 | 3.48 |

# Glycerol 36D KLXX

- Stage policy: `{'t_safe': 0.1, 'shrink_factor': 0.7, 'enlarge_factor': 1.4, 'tau_valid': 0.4, 't_tol': 0.001, 'max_stages': 20, 'max_retry': 5}`
- Screen fraction: `0.0001`
- Complete: `True`
- Regularization path: `(50.0, 0.2)` to `(100.0, 0.1)`
- Factor F_hat = prod ESS^(-1/2): `3.27581`
- Total time: `30.92 min`
- Effective training time (accepted attempts): `26.55 min`

| Stage | t | rho | Selected | Validation ESS | Trained ESS | Identity ESS | Time (min) |
|---:|---:|:---:|:---:|---:|---:|---:|---:|
| 1 | 0.100000 | (55.0, 0.19) | trained | 0.869178 | 0.869178 | 0.223180 | 4.77 |
| 2 | 0.240000 | (62.0, 0.17600000000000002) | trained | 0.424535 | 0.424535 | 0.011620 | 4.30 |
| 3 | 0.377200 | (68.86, 0.16228) | trained | 0.490072 | 0.490072 | 0.135640 | 8.66 |
| 4 | 0.569280 | (78.464, 0.143072) | trained | 0.658767 | 0.658767 | 0.293088 | 4.35 |
| 5 | 0.838192 | (91.90960000000001, 0.1161808) | trained | 0.821073 | 0.821073 | 0.428160 | 4.37 |
| 6 | 1.000000 | (100.0, 0.1) | trained | 0.952723 | 0.952723 | 0.804737 | 4.47 |

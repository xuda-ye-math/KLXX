# NMA 30D KLXX

- Stage policy: `{'t_safe': 0.1, 'shrink_factor': 0.7, 'enlarge_factor': 1.4, 'tau_valid': 0.4, 't_tol': 0.001, 'max_stages': 20, 'max_retry': 5}`
- Screen fraction: `0.0001`
- Complete: `True`
- Regularization path: `(50.0, 0.2)` to `(100.0, 0.15)`
- Factor F_hat = prod ESS^(-1/2): `1.77129`
- Total time: `19.72 min`
- Effective training time (accepted attempts): `19.66 min`

| Stage | t | rho | Selected | Validation ESS | Trained ESS | Identity ESS | Time (min) |
|---:|---:|:---:|:---:|---:|---:|---:|---:|
| 1 | 0.100000 | (55.0, 0.195) | trained | 0.887608 | 0.887608 | 0.184827 | 4.37 |
| 2 | 0.240000 | (62.0, 0.188) | trained | 0.628620 | 0.628620 | 0.007848 | 3.67 |
| 3 | 0.436000 | (71.8, 0.1782) | trained | 0.598960 | 0.598960 | 0.173943 | 4.00 |
| 4 | 0.710400 | (85.52, 0.16448000000000002) | trained | 0.963280 | 0.963280 | 0.437022 | 3.90 |
| 5 | 1.000000 | (100.0, 0.15) | trained | 0.990065 | 0.990065 | 0.600939 | 3.78 |

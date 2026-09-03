# Ethane 18D KLX

- Stage policy: `{'t_safe': 1.0, 'shrink_factor': 0.7, 'enlarge_factor': 1.5, 'tau_valid': 0.4, 't_tol': 0.001, 'max_stages': 20, 'max_retry': 5}`
- Screen fraction: `0.0001`
- Complete: `True`
- Sharpening: `(100.0, 0.15)` to `(100.0, 0.15)`
- Factor F_hat = prod (ESS * sharpening ESS)^(-1/2): `1.1678`
- Total time: `2.23 min`

| Stage | t | Selected | Validation ESS | Trained ESS | Identity ESS | Sharpening ESS | Time (min) |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | trained | 0.733270 | 0.733270 | 0.000076 | 1.000000 | 2.23 |

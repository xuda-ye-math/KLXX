# NMA 30D KLX

- Stage policy: `{'t_safe': 0.1, 'shrink_factor': 0.7, 'enlarge_factor': 1.4, 'tau_valid': 0.4, 't_tol': 0.001, 'max_stages': 20, 'max_retry': 5}`
- Screen fraction: `0.0001`
- Complete: `True`
- Regularization path: `(50.0, 0.2)` to `(100.0, 0.15)`
- Factor F_hat = prod ESS^(-1/2): `2.0563`
- Total time: `20.18 min`
- Effective training time (accepted attempts): `17.36 min`

| Stage | t | rho | Selected | Validation ESS | Trained ESS | Identity ESS | Time (min) |
|---:|---:|:---:|:---:|---:|---:|---:|---:|
| 1 | 0.100000 | (55.0, 0.195) | trained | 0.849602 | 0.849602 | 0.184827 | 3.13 |
| 2 | 0.198000 | (59.9, 0.19010000000000002) | trained | 0.616851 | 0.616851 | 0.042768 | 5.61 |
| 3 | 0.335200 | (66.76, 0.18324000000000001) | trained | 0.519530 | 0.519530 | 0.111118 | 2.87 |
| 4 | 0.527280 | (76.364, 0.173636) | trained | 0.894446 | 0.894446 | 0.462993 | 2.85 |
| 5 | 0.796192 | (89.80959999999999, 0.1601904) | trained | 0.978368 | 0.978368 | 0.530219 | 2.86 |
| 6 | 1.000000 | (100.0, 0.15) | trained | 0.992579 | 0.992579 | 0.779430 | 2.84 |

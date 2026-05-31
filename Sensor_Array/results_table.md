# Sensor-array source localization (3 sources, 6 permutation modes)

- theta* = [5.0, 0.0, -5.0], ||theta*|| = 7.071
- QT pool modes found: 6/6, occ = [0.157, 0.178, 0.14, 0.18, 0.163, 0.182]
- steps = 2000

| loss | ESS | modes | k=5 cov | per-mode occupancy |
|------|-----|-------|---------|--------------------|
| forward KL | 0.904 | 2/6 | 0.128 | [0.0, 0.496, 0.0, 0.504, 0.0, 0.0] |
| forward KL + X_mu | 0.979 | 1/6 | 0.062 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] |
| forward KL + X_mu + X_hat_mu | 0.913 | 6/6 | 0.293 | [0.168, 0.174, 0.158, 0.171, 0.159, 0.169] |
| forward KL + X_mu + X_(hat_mu+bar_nu)/2 | 0.934 | 6/6 | 0.307 | [0.167, 0.168, 0.168, 0.16, 0.164, 0.173] |

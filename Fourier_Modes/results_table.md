# Fourier-field Bayesian inverse problem -- results

Run config: d=9, G=2, 4 sign-flip modes, AMP=2.5, sigma_obs=2.0, coarse 3x3 (train), fine 5x5 (eval), steps=2000, batch=500.

QT pool: modes_found=4/4, occupancy=[0.28, 0.25, 0.25, 0.21].

## Results (4 methods, honest fine-grid eval)

| loss | train-grid ESS | fine-grid ESS | modes_found/4 | Naeem k=5 cov | occupancy |
|---|---|---|---|---|---|
| forward KL | 0.9264 | 0.0003 | 1/4 | 0.2825 | [1.00, 0.00, 0.00, 0.00] |
| KL+X_mu | 0.8594 | 0.0004 | 1/4 | 0.2575 | [0.00, 1.00, 0.00, 0.00] |
| KL+X_mu+X_hat_mu | 0.8898 | 0.0003 | 4/4 | 1.0000 | [0.26, 0.25, 0.24, 0.25] |
| KL+X_mu+X_mix | 0.8856 | 0.0005 | 4/4 | 1.0000 | [0.24, 0.25, 0.24, 0.26] |


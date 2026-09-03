# p-state clock — per-stage ESS on a shared stage schedule

The final column is the propagation factor $\hat F_\Sigma = \sum_{k=0}^{K} \prod_{j=k+1}^{K} \mathrm{ESS}_j^{-1/2}$; smaller is better. It summarizes stagewise weight degeneracy and is not a full-chain ESS or endpoint error estimate. Every ESS is the selected-proposal validation ESS over the complete validation set.

<div align="center">

| stage $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | $\hat F_\Sigma$ |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| $t_k$ ($B = 2000$) | 0.250 | 0.625 | 0.754 | 0.889 | 1.000 | — | — | — |
| KL+$\mathrm{X}_\pi$ | 0.885 | 0.431 | 0.492 | 0.433 | **0.666** | — | — | 15.1 |
| KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$ | **0.937** | **0.553** | **0.554** | **0.456** | 0.662 | — | — | **13.2** |
| $t_k$ ($B = 1000$) | 0.250 | 0.512 | 0.648 | 0.747 | 0.851 | 1.000 | — | — |
| KL+$\mathrm{X}_\pi$ | 0.813 | 0.530 | 0.562 | 0.573 | 0.507 | **0.448** | — | 21.8 |
| KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$ | **0.888** | **0.636** | **0.627** | **0.617** | **0.528** | 0.441 | — | **19.1** |
| $t_k$ ($B = 500$) | 0.250 | 0.434 | 0.569 | 0.668 | 0.772 | 0.882 | 1.000 | — |
| KL+$\mathrm{X}_\pi$ | 0.715 | 0.607 | 0.566 | 0.593 | 0.490 | 0.443 | **0.529** | 30.6 |
| KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$ | **0.816** | **0.696** | **0.637** | **0.642** | **0.502** | **0.460** | 0.516 | **26.9** |

</div>

## Occupancy-bias amplitude

The staged sampler's occupancy bias over N = 10,000 to 640,000 particles follows the Monte Carlo law $\mathrm{err} \simeq C / \sqrt{N}$. $C$ is the geometric mean of $\sqrt{N}\,\mathrm{err}$ over the seven particle counts; the free-fit exponent is reported as a check on the assumed $-1/2$ power, and the last column is the largest relative departure of the fitted law from the measured mean.

<div align="center">

| loss | $C$ | free-fit exponent | max. deviation |
| :--- | :-: | :-: | :-: |
| KL+$\mathrm{X}_\pi$ | 2.67 | 0.484 | 10% |
| KL+$\mathrm{X}_\pi$+$\mathrm{X}_{(\hat\pi+\bar\nu)/2}$ | 2.11 | 0.521 | 19% |

</div>

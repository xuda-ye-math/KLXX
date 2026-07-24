# p-state clock — per-stage ESS on a shared stage schedule

The final column is the propagation factor $F = \prod_k \mathrm{ESS}_k^{-1}$; smaller is better. It summarizes stagewise weight degeneracy and is not a full-chain ESS or endpoint error estimate. Every ESS is the selected-proposal validation ESS over the complete validation set.

<div align="center">

| stage $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | $F$ |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| $t_k$ ($B = 2000$) | 0.250 | 0.495 | 0.663 | 0.779 | 0.887 | 1.000 | — | — | — |
| forward KL | 0.751 | 0.570 | 0.471 | 0.474 | 0.511 | 0.602 | — | — | 33.9 |
| KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ | **0.921** | **0.759** | **0.608** | **0.562** | **0.558** | **0.649** | — | — | **11.6** |
| $t_k$ ($B = 1000$) | 0.250 | 0.495 | 0.613 | 0.728 | 0.841 | 0.952 | 1.000 | — | — |
| forward KL | 0.672 | 0.479 | 0.580 | 0.463 | 0.427 | 0.520 | 0.830 | — | 63.0 |
| KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ | **0.866** | **0.642** | **0.700** | **0.545** | **0.462** | **0.541** | **0.859** | — | **22.0** |
| $t_k$ ($B = 500$) | 0.250 | 0.421 | 0.590 | 0.705 | 0.784 | 0.895 | 1.000 | — | — |
| forward KL | 0.580 | 0.549 | 0.408 | 0.432 | 0.525 | 0.417 | 0.530 | — | 153.5 |
| KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ | **0.778** | **0.695** | **0.506** | **0.500** | **0.595** | **0.429** | **0.566** | — | **50.7** |
| $t_k$ ($B = 250$) | 0.250 | 0.421 | 0.539 | 0.654 | 0.734 | 0.811 | 0.887 | 1.000 | — |
| forward KL | 0.468 | 0.448 | 0.495 | 0.428 | 0.506 | 0.484 | 0.514 | 0.424 | 421.2 |
| KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ | **0.659** | **0.586** | **0.601** | **0.480** | **0.563** | **0.536** | **0.564** | **0.464** | **113.8** |

</div>

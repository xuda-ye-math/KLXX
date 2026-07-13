# Six-state clock Boltzmann generator

The target is the six-state clock model on a periodic $8\times8$ lattice. We
compare forward KL with
$\mathrm{KL}+\mathrm{X}_\mu+\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ (KLXX) on
exactly the same accepted temperature levels at each batch size. Every value
below is the full-validation ESS of one accepted stage; minibatch monitors and
composed endpoint ESS are not used for this comparison.

## Schedule-matched validation ESS

The propagation factor is
$F=\prod_k\mathrm{ESS}_k^{-1}$, so smaller is better.

<div align="center">

| stage $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | $F$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| $t_k$ ($B=2000$) | 0.250 | 0.495 | 0.663 | 0.779 | 0.887 | 1.000 | — | — | — | — | — |
| forward KL | 0.749 | 0.584 | 0.502 | 0.503 | 0.533 | 0.631 | — | — | — | — | 26.9 |
| KLXX | **0.921** | **0.773** | **0.645** | **0.590** | **0.581** | **0.671** | — | — | — | — | **9.5** |
| $t_k$ ($B=1000$) | 0.250 | 0.495 | 0.663 | 0.779 | 0.887 | 1.000 | — | — | — | — | — |
| forward KL | 0.672 | 0.490 | 0.431 | 0.456 | 0.487 | 0.595 | — | — | — | — | 53.3 |
| KLXX | **0.862** | **0.661** | **0.531** | **0.525** | **0.521** | **0.604** | — | — | — | — | **20.0** |
| $t_k$ ($B=500$) | 0.250 | 0.421 | 0.590 | 0.705 | 0.784 | 0.895 | 1.000 | — | — | — | — |
| forward KL | 0.578 | 0.550 | 0.421 | 0.458 | 0.561 | 0.437 | 0.554 | — | — | — | 120.0 |
| KLXX | **0.780** | **0.708** | **0.530** | **0.541** | **0.617** | **0.457** | **0.588** | — | — | — | **38.0** |
| $t_k$ ($B=250$) | 0.250 | 0.421 | 0.539 | 0.654 | 0.734 | 0.811 | 0.920 | 1.000 | — | — | — |
| forward KL | 0.464 | 0.457 | 0.502 | 0.444 | 0.519 | 0.504 | 0.407 | 0.582 | — | — | 342.3 |
| KLXX | **0.656** | **0.602** | **0.628** | **0.512** | **0.582** | **0.554** | **0.423** | **0.637** | — | — | **90.7** |
| $t_k$ ($B=125$) | 0.175 | 0.346 | 0.464 | 0.579 | 0.659 | 0.736 | 0.812 | 0.886 | 0.966 | 1.000 | — |
| forward KL | 0.453 | 0.405 | 0.452 | 0.411 | 0.505 | 0.469 | 0.455 | 0.505 | 0.559 | **0.913** | 1058.4 |
| KLXX | **0.655** | **0.563** | **0.588** | **0.501** | **0.567** | **0.503** | **0.478** | **0.525** | **0.562** | 0.912 | **251.4** |

</div>

KLXX has higher validation ESS on 36 of the 37 shared levels. The sole
reversal is the final $B=125$ level, where 0.913 and 0.912 differ by 0.001.
Across batch sizes, KLXX reduces $F$ by factors 2.66--4.21.

## Fresh $B=2000$ staged samples

Two million fresh samples advanced through the trained KLXX ladder retain all
six global clock sectors. Nearest-neighbor angle differences are most
concentrated at zero; the alignment peak weakens with lattice separation while
the shoulders near $\pm\pi/3$ grow.

<p align="center"><img src="results/clock_marginals.png" alt="Clock angle marginals" width="700px"></p>

For the occupancy-scaling test, each row uses the same total particle work:
$N=10000\,2^k$ with $2^{8-k}$ independent rebuilds. The bias is
$\frac16\sum_{s=0}^{5}|p_s-1/6|$.

<div align="center">

| $N$ | rebuilds | forward KL | KLXX |
|---:|---:|---:|---:|
| 10000 | 256 | 0.02930 ± 0.00070 | **0.01851 ± 0.00042** |
| 20000 | 128 | 0.02064 ± 0.00083 | **0.01310 ± 0.00041** |
| 40000 | 64 | 0.01598 ± 0.00072 | **0.00921 ± 0.00042** |
| 80000 | 32 | 0.01068 ± 0.00050 | **0.00685 ± 0.00040** |
| 160000 | 16 | 0.00665 ± 0.00074 | **0.00464 ± 0.00040** |
| 320000 | 8 | 0.00518 ± 0.00045 | **0.00385 ± 0.00046** |
| 640000 | 4 | 0.00360 ± 0.00065 | **0.00246 ± 0.00037** |
| 1280000 | 2 | 0.00368 ± 0.00031 | **0.00157 ± 0.00048** |
| 2560000 | 1 | 0.00161 | **0.00106** |

</div>

*Values are means ± one standard error where more than one independent rebuild
is available. The largest particle count has one rebuild and therefore no
empirical error bar.*

<p align="center"><img src="results/occupancy_bias_B2000.png" alt="Clock-sector occupancy-bias scaling" width="700px"></p>

The log-log slopes are $-0.492$ for forward KL and $-0.505$ for KLXX, both
consistent with the $N^{-1/2}$ Monte Carlo rate rather than a persistent
sector imbalance. KLXX has lower bias at all nine particle counts.

## Verification summary

- Every KL/KLXX pair uses an exactly shared accepted $t$ history.
- All per-level values and $F$ factors were rebuilt from the live NPZ files.
- The marginal figure uses 2000000 fresh staged samples saved before plotting.
- Occupancy scaling was run as two independent 71-minute method processes;
  the merged archive retains every per-rebuild sector histogram and bias.
- Both final figures were regenerated from saved data and visually inspected.


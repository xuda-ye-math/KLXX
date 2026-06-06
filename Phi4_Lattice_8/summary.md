# phi^4 8x8: the showcase fake ESS case (paper headline together with 6x6; 3 seeds)

## 3-seed aggregate (referee p_+ = 0.139)

| method | ESS (3 seeds) | p+ per seed | verdict |
|---|---|---|---|
| KL               | 0.569 +- 0.090 | 0, 0, **1.000** | collapses every seed; seed 2 onto the **minority** vacuum (Delta F wrong sign at ESS 0.61) |
| KL+X_mu          | 0.809 +- 0.009 | 0, 0, **1.000** | collapses every seed at a precision-reproducible fake ESS; seed 2 also minority-vacuum |
| KL+X_mu+X_hat_mu | 0.664 +- 0.021 | 0.126, 0.127, 0.128 | repairs every seed |
| KL+X_mu+X_mix    | 0.676 +- 0.021 | 0.129, 0.127, 0.126 | repairs every seed |

Repaired weight 0.127 +- 0.001 vs referee 0.139: a small systematic underweight (~0.012), consistent
with finite-training self-normalized IS bias; well inside the +-0.03 acceptance band.

**Setup.** 2D phi^4 lattice, 8x8 periodic (D=64), broken Z2 phase: kappa=0.40, lambda=0.50,
h=0.0144 (pilot-frozen: barrier 10.19 kT at h=0, vacua at m=+-1.08; referee p(m>0)=0.139,
Delta F=-1.82 kT, 3687 PT roundtrips). Source: Gaussian sigma=0.5 at the barrier top.
Flow: NSF [-3,3]^64, 6 coupling transforms, 16 bins, hidden (256,256); batch 500, 2000 Adam
steps, lr 1e-3, M=1 one-step AIS surrogate; gradient-clip stability guard. Single-flow direct
training BY DESIGN: large lattices call for the adaptive-temperature Boltzmann generator
(Algorithm 4 / the clock experiment), which would mix the wells by annealing and dissolve the
failure under study -- this benchmark isolates the single-flow regime where the ESS diagnostic
is at its most misleading.

## Headline table

| method | final ESS | ESS(m<0) | ESS(m>0) | p(m>0) | Delta F (kT) |
|---|---|---|---|---|---|
| forward KL              | 0.443 | 0.443 | --- (0 samples) | **0.000** | -inf |
| KL+X_mu                 | **0.821** | 0.821 | --- (0 samples) | **0.000** | -inf |
| KL+X_mu+X_hat_mu        | 0.644 | 0.666 | 0.523 | **0.127** | -1.93 |
| KL+X_mu+X_mix           | 0.658 | 0.688 | 0.498 | **0.129** | -1.93 |
| **PT referee**          |  ---  |  ---  |  ---  | **0.139** | **-1.82** |

## Interpretation

The collapsed losses post the two most seductive ESS values on the board -- 0.44 and 0.82 --
while carrying ZERO samples in the minority phase: Delta F estimated at -infinity where the
truth is -1.82 kT. This is the purpose of the experiment: a fake ESS is only dangerous when it
is HIGH, and 0.82 is precisely the number a practitioner would celebrate. The oracle-driven
losses repair coverage at no meaningful ESS cost (0.64-0.66, with the minority well genuinely
well-fitted: per-mode ESS 0.50-0.52) and land the phase weight within 0.012 of the referee.
Training cleanliness: stability-guard skips are 0% for the collapsed losses, 25% for X_hat_mu,
and only 5% for the recommended mixture.

## Why 8x8 (with 6x6) is the reported regime

Larger lattices were measured and excluded deliberately: at 12x12 the collapsed ESS falls to
0.15 (no longer misleading -- it already looks bad) and the honest methods' ESS drops to
0.01-0.06 (correct physics, but sampling grows expensive); at 16x16 single-flow training and
mode-truncated proposals both hit measured ceilings (see .archive/Phi4_Lattice16/, .aris wiki). The
fake ESS showcase lives at 6x6-8x8, where the fake is high, training is clean, and the exact
PT referee is cheap. The 12x12/16x16 records remain on disk under .archive/ as the measured boundary of the
single-flow regime (appendix/referee-response material, not headline).

## Files

- `results_table.md` / `.csv`, `figures/fig_methods.png` (money plot), `fig_ess.png`,
  `fig_background.png`, `pilot_results.md`, `phi4_reference.pth`, `data.pth` (state_dicts),
  `train_status.log`. Background primer: `../Phi4_Lattice_6/BACKGROUND.md` (6x6, applies verbatim).

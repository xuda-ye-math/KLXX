# phi^4 6x6 phase-0: collapse confirmed, X repairs it (solidified: 3 seeds + skewed-oracle tests)

**Setup.** 2D phi^4 lattice, 6x6 periodic (D=36), broken Z2 phase: kappa=0.40, lambda=0.50, h=0.0257
(frozen by the PT-MALA pilot: barrier 7.98 kT at h=0, vacua at m = +-1.08, reference p(m>0)=0.128,
Delta F = -1.92 kT, 48,557 PT roundtrips). Source: Gaussian sigma=0.5 at the barrier top.
Flow: NSF [-3,3]^36, 6 coupling transforms, 16 bins, hidden (256,256); batch 500, 2000 Adam steps,
lr 1e-3. Staged protocol: bare forward KL first, then X variants with identical frozen parameters,
QT set (2000 samples, 45/55 across vacua), and validation batch. VRAM < 2 GB (cap 16 GB).

## Headline table

| method | final ESS | p(m>0) reweighted | Delta F (kT) |
|---|---|---|---|
| forward KL              | 0.720 | **0.000** | -inf (clipped -20.7) |
| KL+X_mu                 | 0.908 | **0.000** | -inf (clipped -20.7) |
| KL+X_mu+X_hat_mu        | 0.876 | **0.125** | -1.94 |
| KL+X_mu+X_mix           | 0.888 | **0.126** | -1.93 |
| **PT reference**        |  ---  | **0.128** | **-1.92** |

## Interpretation (one paragraph)

Bare forward KL and KL+X_mu collapse onto the favored (m<0) vacuum and report the *highest* ESS
of the four (0.72, 0.91) while carrying zero mass in the m>0 phase: the fake-ESS pitfall realized
in a genuine field theory, with the physical observable (the inter-vacuum free-energy difference)
infinitely wrong. The two oracle-driven losses recover the missing vacuum at p(m>0) = 0.125/0.126
versus the PT referee's 0.128 — Delta F correct to 0.02 kT — at an honest ESS of ~0.88. Training
the oracle-driven losses is harder: ~20% of steps tripped the non-finite stability guard
(gradient clip + skip, the same guard as the clock pipeline) because the hat_mu term forces the
flow to confront configurations ~100 kT apart in log-ratio from step one; all runs completed.

## Solidification round (3 seeds; skewed-oracle robustness)

| method | ESS (3 seeds) | p(m>0) per seed | verdict |
|---|---|---|---|
| KL               | 0.769 +- 0.038 | 0, 0, 0           | collapses every seed |
| KL+X_mu          | 0.927 +- 0.013 | 0, 0, **1.000**   | collapses every seed; seed 2 onto the **minority** vacuum: Delta F gets the wrong **sign** at ESS 0.94 |
| KL+X_mu+X_hat_mu | 0.881 +- 0.007 | 0.127, 0.126, 0.124 | repairs every seed |
| KL+X_mu+X_mix    | 0.891 +- 0.008 | 0.126, 0.122, 0.123 | repairs every seed |

PT reference p(m>0) = 0.128. Seed-to-seed spread of the repaired weight: +-0.002.

**Skewed-oracle test (mimicry refuted).** Training with QT sets deliberately subsampled to
m>0 fractions 0.05 and 0.25 (vs the natural 0.45): recovered p_+ = 0.124 (X_hat_mu, skew 0.05),
0.128 (mixture, skew 0.05), 0.126 (mixture, skew 0.25) — the recovered weight follows the action
through reweighting, not the oracle composition, across a 9x skew range. This is the
sample-robustness property of X_omega, demonstrated quantitatively; data_skew05/25.pth.

## Caveats

- Three seeds per method (solidification round above); spread of the repaired weight +-0.002.
- ESS of the collapsed runs is *restricted-support* ESS; the honest comparison is against
  p(m>0) and Delta F, which is the point of the experiment.
- Delta F for collapsed runs is reported as the 1e-9-clipped value -20.7; the true estimate is -inf.
- 6x6 is deliberately small; barrier grows with L (domain-wall cost), so collapse only sharpens
  at scale. Fourier-truncation axis not yet exercised (phase-0b candidate).

## Review response (3 independent agents, round 1; reviews in ../.aris/reviews/phi4-phase0/)

- **Circularity ("the QT set leaks the answer") — refuted by the data.** The QT oracle's mode
  ratio is 0.451; the recovered weight is 0.126 (true 0.128). Mimicry of the oracle would give
  ~0.45. The oracle supplies coverage only; the weight is set by the action through reweighting —
  the sample-robustness property, demonstrated rather than assumed.
- **Per-mode ESS (requested diagnostic):** collapsed methods carry ZERO samples at m>0; the
  oracle-driven flows hold ESS(m<0)/ESS(m>0) = 0.894/0.765 and 0.904/0.791 — both modes genuinely
  well-sampled, the recovered mode is not a reweighting artifact.
- **Referee barrier 7.98 -> 6.96 kT:** expected physics, not equilibration bias — the tilt raises
  the favored well, lowering the barrier seen from it; roundtrips are 48k-52k either way.
- **Acknowledged, queued next:** multi-seed error bars; skewed-QT robustness test (train with a
  deliberately 5/95 oracle: the framework predicts the recovered p_+ stays at the true value);
  diagnosis of the ~20% guard-skip rate (suspect: lr warmup absent while the hat_mu term sees
  ~100 kT log-ratio spreads at initialization).

## Files

- `BACKGROUND.md` — primer: what phi^4 is, why two vacua, why KL collapses, what PT certifies
- `pilot.py` / `pilot_results.md` / `phi4_reference.pth` — PT-MALA referee (kappa scan + frozen params)
- `train.py` (staged `--methods`), `core.py`, `parameters.py` — pipeline (state_dicts saved in data.pth)
- `plot_results.py` -> `figures/fig_background.png`, `fig_methods.png`, `fig_ess.png`
- `results_table.md` / `.csv` — headline numbers; `train_status.log`, `pilot_status.log` — live logs

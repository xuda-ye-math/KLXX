# Two-method comparison: `balance` (KL + X_mu + X_mix) vs `kl` (forward KL only)

p-state clock, P=6, J=1.0, H=0.5, periodic square lattice, torus domain.
Identical pipeline for both methods (Algorithm 4: NCSF, uniform-on-torus source,
adaptive ladder with ADAPIVE_TAU=0.7 / VALIDATION_TAU=0.3 / SHRINK=0.7, t_safe=0.25,
Gamma=2 warm start, M=4 shared SMC/AIS, STEPS=1000, N_BATCH=8000); the ONLY
difference is the training loss in step (iv).

## Headline

| | L=4 (D=16) balance | L=4 kl | L=8 (D=64) balance | L=8 kl |
|---|---|---|---|---|
| complete | yes | yes | yes | **NO (stuck at t=0.577)** |
| ladder | 0.25, 0.60, 1.00 | 0.25, 0.60, 1.00 | 0.25, 0.495, 0.663, 0.828, 1.00 | 0.25, 0.495, 0.577 |
| gate retries | 0 | 0 | 0 | 14 (2 @ stage 3, 12 @ stage 4) |
| val ESS per stage | 0.98 / 0.91 / 0.73 | 0.89 / 0.74 / 0.59 | 0.85 / 0.78 / 0.78 / 0.71 / 0.67 | 0.52 / 0.41 / 0.31, then 0.06-0.11 x12 |
| composed direct ESS | **0.519** | 0.263 | 0.0011 | 0.0001 (vs full target; not faithful) |
| sector TV | 0.078 | 0.092 | 0.092 | 0.244 |
| <E> rel. err (reweighted) | **0.44%** | 0.37% | 20% | n/a (incomplete) |
| wall | 5.5 min | 5.3 min | 65 min | **215 min, mostly failed retrains** |

## Findings

1. **d=16 does not discriminate**: both methods complete the same ladder with all
   6 sectors at near-uniform occupancy and sub-1% energy error. kl's composed ESS
   is 2x lower (0.263 vs 0.519) and its worst-sector in-sector ESS is 3.5x lower
   (s3: 0.081 vs 0.282), but both are usable. Consistent with the HD_Product
   pattern: the gap opens with dimension.
2. **d=64 separates the methods qualitatively.** balance: 5 stages, every gate
   passed first-try (0.67-0.85), full ladder in 65 min. kl: validation ESS decays
   monotonically with t (0.52 -> 0.41 -> 0.31 -> 0.11) and stage 4 fails 12/12
   attempts with ESS pinned at 0.06-0.11 across every step size from t=0.742 down
   to t=0.580 — shrinking Delta-t does not help, so the bottleneck is the loss,
   not the schedule. The run ends INCOMPLETE at t=0.577 after 3.6 h.
3. **Mode-seeking collapse is visible in-training**: in kl's last accepted stage
   and in all failed stage-4 attempts the per-step direct ESS *decreases* as
   training proceeds (e.g. 0.112 -> 0.095 -> 0.065 within one attempt): the pure
   forward KL objective concentrates the pushforward and degrades nu->mu_k
   overlap. The X-regularized loss shows the opposite signature (ESS climbs to a
   0.7-0.85 plateau every stage). See `figures/cmp_ess_steps_L8.png`.
4. **Honest caveat on L=8 balance**: although the ladder completes and all 6
   sectors are covered (TV 0.092), the one-shot composed generator is weak —
   direct ESS 0.0011 and reweighted-energy error 20% vs the classical-SMC
   reference. Per-stage residual bias compounds across K=5 maps. |m|=0.23 (vs
   0.73 at L=4) also suggests beta=1 at L=8 sits near the BKT window
   (near-critical, long correlation lengths). Candidate fixes for a later pass:
   more STEPS at d=64 (stage ESS still climbing at step 1000), stricter
   VALIDATION_TAU, or a colder target (larger J/beta).
5. The L=4 balance run was re-trained on the final pipeline (M-rung AIS
   surrogate + warm start) for metric-consistent histories; this improved its
   composed ESS from 0.292 to 0.519 at identical wall time. The pre-AIS run is
   archived in `archive_v1/data_L4_balance_preAIS.pth`.

Figures: `figures/cmp_ess_steps_L{4,8}.png`, `figures/cmp_sectors_L{4,8}.png`,
`figures/cmp_ladder_L{4,8}.png`. Raw data: `data_L{4,8}_{balance,kl}.pth`
(per-stage state_dicts, per-step train_ess_hist, per-attempt validation ESS).

**Review status.** Three-way independent review (Claude raw-data re-derivation,
OpenAI, Gemini): ACCEPTED with the headline narrowed to "ladder completes vs
stalls" — see `.aris/reviews/clock-results-cmp/verdict.md`. All numbers above
were re-derived exactly from the raw data by the Claude reviewer. Open items
before paper-grade: seed replication, kl more-STEPS/LR ablation, persisting
failed-attempt ESS histories.

# Staged sampler vs PT-MALA referee: total-variation distance of well weights
(CANONICAL results, 2026-06-07. Estimator: the staged sampler of Algorithm 4
step (v) -- per stage: map, per-stage reweight, resample, Langevin rejuvenation
-- replayed 3x per trained ladder from the saved per-stage state_dicts, plus the
per-step fine extension at the final temper. NO composed map anywhere.
Referee: 20-rung PT-MALA, 128 chains, roundtrips 7.3k-35k, 102k cold samples.)

| sigma_obs | kl TV (mean +- spread) | balance TV (mean +- spread) | kl ladder |
|---|---|---|---|
| 0.010 | 0.106 +- 0.039 | 0.042 +- 0.002 | INCOMPLETE (failed at t=1) |
| 0.015 | 0.077 +- 0.016 | 0.041 +- 0.004 | complete |
| 0.020 | 0.028 +- 0.010 | 0.020 +- 0.001 | complete |
| 0.025 | 0.050 +- 0.004 | 0.019 +- 0.003 | complete |

Headline: the X-regularized (balance) loss is closer to the certified weights in
every pair AND an order of magnitude more reproducible across replays (spread
<= 0.004 vs up to 0.04 for bare KL). Raw per-seed values: staged_tv.csv.
Tool: staged_census.py (per-stage replay). The earlier composed-pushforward
numbers (superseded, estimator did not match the procedure): kl 0.245/0.112/
0.079/0.147, balance 0.032/0.039/0.020/0.018 -- same qualitative ordering.

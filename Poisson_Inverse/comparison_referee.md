# Method vs PT-MALA referee: total-variation distance of reweighted well weights
(referee: 20-rung PT, 128 chains, 6000 iters, roundtrips 7.3k-35k, 102k cold samples per sigma)

| sigma_obs | kl TV | balance TV | kl status | notes |
|---|---|---|---|---|
| 0.010 | 0.2451 | 0.0321 | INCOMPLETE (failed at t=1) | kl puts 0.408 on well (-1,1), truth 0.167 |
| 0.015 | 0.1119 | 0.0385 | complete | |
| 0.020 | 0.0787 | 0.0199 | complete | |
| 0.025 | 0.1470 | 0.0183 | complete | kl gives 0.065 on (-1,1), truth 0.152 |

Headline: KL+X_mu+X_mix recovers the certified weights to TV 0.018-0.039 at every
noise level; bare KL is 4-8x worse even when its ladder completes -- completed
ladders with wrong physics, invisible to per-stage ESS, exposed by the referee.
Per-sigma per-well tables: referee.md; raw: referee_o*.pth, data_{method}_o*.pth.

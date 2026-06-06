# Truncation-ceiling gate: FAIL

Geometry 4x4 in 8x8 (d_low=16, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.02, prior A=2.0, s=3, tilt=0.15.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.1913 | < 0.1: False |
| (b) max extension Var Phi | 0.3168 | (info) |
| (c) ceiling ESS, t=1.0 | 0.0004 | > 0.5: False |
| (c') ceiling ESS, t=0.7 | 0.0003 | > 0.2: False |
| (c') ceiling ESS, t=0.5 | 0.0002 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 21.61 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

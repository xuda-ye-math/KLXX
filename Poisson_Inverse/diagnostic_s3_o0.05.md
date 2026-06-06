# Truncation-ceiling gate: FAIL

Geometry 4x4 in 8x8 (d_low=16, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.05, prior A=2.0, s=3, tilt=0.15.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.03061 | < 0.1: True |
| (b) max extension Var Phi | 0.01833 | (info) |
| (c) ceiling ESS, t=1.0 | 0.3029 | > 0.5: False |
| (c') ceiling ESS, t=0.7 | 0.2507 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.3115 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 3.457 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

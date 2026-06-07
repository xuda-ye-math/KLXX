# Truncation-ceiling gate: PASS

Geometry 6x6 in 8x8 (d_low=36, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.01, prior A=2.0, s=6, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 6.117e-05 | < 0.1: True |
| (b) max extension Var Phi | 2.342e-05 | (info) |
| (c) ceiling ESS, t=1.0 | 0.9950 | > 0.5: True |
| (c') ceiling ESS, t=0.7 | 0.9962 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.9985 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 3411 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

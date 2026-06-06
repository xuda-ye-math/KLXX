# Truncation-ceiling gate: PASS

Geometry 4x4 in 8x8 (d_low=16, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.05, prior A=2.0, s=4, tilt=0.15.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.01798 | < 0.1: True |
| (b) max extension Var Phi | 0.0301 | (info) |
| (c) ceiling ESS, t=1.0 | 0.7998 | > 0.5: True |
| (c') ceiling ESS, t=0.7 | 0.6465 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.6545 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 20.89 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

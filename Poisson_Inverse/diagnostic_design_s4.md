# Truncation-ceiling gate: PASS

Geometry 4x4 in 8x8 (d_low=16, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.02, prior A=2.0, s=4, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.05669 | < 0.1: True |
| (b) max extension Var Phi | 0.04888 | (info) |
| (c) ceiling ESS, t=1.0 | 0.5105 | > 0.5: True |
| (c') ceiling ESS, t=0.7 | 0.2698 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.1855 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 60.34 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

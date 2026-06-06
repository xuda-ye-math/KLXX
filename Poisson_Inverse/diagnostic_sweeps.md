# Truncation-gate sweep history (consolidated; individual files removed)

## diagnostic_design_s4.md

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

---

## diagnostic_design_s5.md

# Truncation-ceiling gate: PASS

Geometry 4x4 in 8x8 (d_low=16, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.02, prior A=2.0, s=5, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.01711 | < 0.1: True |
| (b) max extension Var Phi | 0.01333 | (info) |
| (c) ceiling ESS, t=1.0 | 0.8828 | > 0.5: True |
| (c') ceiling ESS, t=0.7 | 0.8186 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.7906 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 69.01 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_h_0.005_s5.md

# Truncation-ceiling gate: FAIL

Geometry 6x6 in 8x8 (d_low=36, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.005, prior A=2.0, s=5, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.002709 | < 0.1: True |
| (b) max extension Var Phi | 0.002962 | (info) |
| (c) ceiling ESS, t=1.0 | 0.0002 | > 0.5: False |
| (c') ceiling ESS, t=0.7 | 0.0007 | > 0.2: False |
| (c') ceiling ESS, t=0.5 | 0.0012 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 2485 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_h_0.005_s6.md

# Truncation-ceiling gate: PASS

Geometry 6x6 in 8x8 (d_low=36, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.005, prior A=2.0, s=6, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.0002447 | < 0.1: True |
| (b) max extension Var Phi | 9.315e-05 | (info) |
| (c) ceiling ESS, t=1.0 | 0.8777 | > 0.5: True |
| (c') ceiling ESS, t=0.7 | 0.8713 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.8795 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 1.364e+04 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_h_0.01_s4.md

# Truncation-ceiling gate: FAIL

Geometry 6x6 in 8x8 (d_low=36, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.01, prior A=2.0, s=4, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.02045 | < 0.1: True |
| (b) max extension Var Phi | 0.01314 | (info) |
| (c) ceiling ESS, t=1.0 | 0.0191 | > 0.5: False |
| (c') ceiling ESS, t=0.7 | 0.0063 | > 0.2: False |
| (c') ceiling ESS, t=0.5 | 0.2643 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 542.1 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_h_0.01_s5.md

# Truncation-ceiling gate: FAIL

Geometry 6x6 in 8x8 (d_low=36, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.01, prior A=2.0, s=5, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.0006771 | < 0.1: True |
| (b) max extension Var Phi | 0.0007654 | (info) |
| (c) ceiling ESS, t=1.0 | 0.2580 | > 0.5: False |
| (c') ceiling ESS, t=0.7 | 0.5250 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.7652 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 621.1 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_hard005.md

# Truncation-ceiling gate: FAIL

Geometry 6x6 in 8x8 (d_low=36, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.005, prior A=2.0, s=4, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.0818 | < 0.1: True |
| (b) max extension Var Phi | 0.05791 | (info) |
| (c) ceiling ESS, t=1.0 | 0.0011 | > 0.5: False |
| (c') ceiling ESS, t=0.7 | 0.0008 | > 0.2: False |
| (c') ceiling ESS, t=0.5 | 0.0004 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 2169 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_m6.md

# Truncation-ceiling gate: PASS

Geometry 6x6 in 8x8 (d_low=36, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.02, prior A=2.0, s=4, tilt=0.0.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.005113 | < 0.1: True |
| (b) max extension Var Phi | 0.003532 | (info) |
| (c) ceiling ESS, t=1.0 | 0.9566 | > 0.5: True |
| (c') ceiling ESS, t=0.7 | 0.9483 | > 0.2: True |
| (c') ceiling ESS, t=0.5 | 0.9315 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 135.5 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_s3_o0.02.md

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

---

## diagnostic_s3_o0.05.md

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

---

## diagnostic_s4_o0.02.md

# Truncation-ceiling gate: FAIL

Geometry 4x4 in 8x8 (d_low=16, d_full=64); grid 32; alpha=3.0, c^2=25.0, sigma_obs=0.02, prior A=2.0, s=4, tilt=0.15.

| check | value | gate |
|---|---|---|
| (a) max extension curvature ratio | 0.1124 | < 0.1: False |
| (b) max extension Var Phi | 0.2309 | (info) |
| (c) ceiling ESS, t=1.0 | 0.0003 | > 0.5: False |
| (c') ceiling ESS, t=0.7 | 0.0067 | > 0.2: False |
| (c') ceiling ESS, t=0.5 | 0.0003 | (info) |

Figure: figures/diagnostic.png. Low-block max curvature ratio 130.6 (should be >> extension: the data must constrain the trained block, else the posterior is trivial).

---

## diagnostic_s4_o0.05.md

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

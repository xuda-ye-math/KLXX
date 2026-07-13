# 2D benchmarks — forward KL with log-ratio variation

Four targets isolate accuracy and mode discovery under the same family of
objectives. Two-Moon stresses small-batch fitting on distorted ridges;
Three-Well, Himmelblau, and Sparse place progressively harder modes outside
the support reached by the flow's own annealing chain. Each ESS is computed on
the full fixed validation set. Coverage uses a quench-and-temper reference pool
with $\kappa=5$, exposing modes that a high support-local ESS can miss.

## Two-Moon

All methods cover both moons, while the
equal-weight mixture gives the best final ESS.

![Two-Moon samples](Two-Moon/results/samples.png)

## Three-Well

Forward KL and KL + X_μ retain high
ESS while missing the low-probability well. Both objectives supplied with the
quench-and-temper coverage pool recover that well, with the equal-weight
mixture giving the strongest coverage.

![Three-Well samples](Three-Well/results/samples.png)

## Himmelblau

The first two objectives cover only
the two right-hand wells despite high ESS. Both coverage-pool variants recover
all four wells. The X_μ̂-only pushforward retains visible low-density paths
between wells, while the equal-weight mixture is cleaner and better calibrated.

![Himmelblau samples](Himmelblau/results/samples.png)

## Sparse

Forward KL and KL + X_μ model only the
central pair. Both quench-and-temper variants reach the isolated corner modes;
the equal-weight mixture again suppresses much of the connecting leakage seen
in the X_μ̂-only pushforward.

![Sparse samples](Sparse/results/samples.png)

## Final ESS and coverage

<div align="center">

| Target | forward KL | KL + X_μ | KL + X_μ + X_μ̂ | KL + X_μ + X_(μ̂+ν̄)/2 |
|:---|---:|---:|---:|---:|
| Two-Moon | 0.799 (1.000) | 0.915 (1.000) | 0.918 (1.000) | **0.957** (1.000) |
| Three-Well | 0.988 (0.760) | 0.994 (0.760) | 0.973 (0.969) | **0.992** (0.977) |
| Himmelblau | 0.975 (0.516) | 0.963 (0.516) | 0.952 (1.000) | **0.978** (1.000) |
| Sparse | 0.949 (0.534) | 0.959 (0.534) | 0.883 (1.000) | **0.935** (1.000) |

</div>

*Final ESS, with $\kappa=5$ coverage in parentheses. Bold marks the equal-weight
mixture, the strongest overall setting because it combines essentially full
coverage with near-best calibration on every target.*

## Verification summary

- All four full training scripts exited successfully and ended with `DONE`.
- All 16 final ESS/coverage pairs are finite; every reported coverage lies in
  `[0, 1]`.
- The four presented sample figures were regenerated and visually inspected;
  no panel is blank, clipped, or malformed.
- The figures and paired metrics separate calibration from discovery: a high
  ESS on reached support does not certify that every mode was found.
- Per-run logs and temporary data live below each target's `artifacts/`
  directory. The figures above are the final outputs below `results/`.

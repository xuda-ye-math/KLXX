# 2D benchmarks — forward KL with log-ratio variation

Four targets isolate accuracy and mode discovery under the same family of
objectives. Two-Moon stresses small-batch fitting on distorted ridges;
Three-Well, Himmelblau, and Sparse place progressively harder modes outside
the support reached by the flow's own annealing procedure. Every method starts from
the same identity NSF and is trained directly through `jflows.train`. The runs
using QT set `pool_size=0`, so quench and temper acts on the complete fixed
validation set. Each ESS is computed on that complete validation set.
Coverage uses an independent quench and temper reference set with $\kappa=5$,
exposing modes that a high ESS on reached support can miss. As a k-NN estimate,
values close to one are consistent with the complete mode coverage visible in
the sample panels.

## Two-Moon

All methods cover both moons with measured coverage 1.000. The equal-weight
mixture gives the highest final ESS at 0.924, ahead of KL + X_μ at 0.911,
and remains cleaner than the X_μ̂-only pushforward samples.

![Two-Moon samples](Two-Moon/results/samples.png)

## Three-Well

Forward KL and KL + X_μ retain high ESS while missing the low-probability
well. Both quench and temper variants recover all three wells; their near-unity
coverage estimates agree with the complete coverage visible in the figures.
The equal-weight mixture retains ESS 0.989, nearly matching KL + X_μ at 0.990,
while recovering the missing well and producing the cleaner full-coverage
pushforward samples.

![Three-Well samples](Three-Well/results/samples.png)

## Himmelblau

The first two objectives cover only the two right-hand wells despite ESS as
high as 0.981. Both quench and temper variants recover all four wells and raise
measured coverage from 0.516 to 1.000. The equal-weight mixture is cleaner than
the X_μ̂-only pushforward samples and raises full-coverage ESS from 0.956 to 0.974.

![Himmelblau samples](Himmelblau/results/samples.png)

## Sparse

Forward KL and KL + X_μ model only the central pair. Both quench and temper
variants reach the isolated corner modes and raise measured coverage from
0.534 to 1.000. The equal-weight mixture suppresses much of the connecting
leakage seen in the X_μ̂-only pushforward samples and increases ESS from 0.826 to
0.956, the strongest result among all four methods.

![Sparse samples](Sparse/results/samples.png)

## Final ESS and coverage

<div align="center">

| Target | forward KL | KL + X_μ | KL + X_μ + X_μ̂ | KL + X_μ + X_(μ̂+ν̄)/2 |
|:---|---:|---:|---:|---:|
| Two-Moon | 0.836 (1.000) | 0.911 (1.000) | 0.835 (1.000) | **0.924** (1.000) |
| Three-Well | 0.988 (0.760) | 0.990 (0.760) | 0.961 (0.978) | **0.989** (0.977) |
| Himmelblau | 0.932 (0.516) | 0.981 (0.516) | 0.956 (1.000) | **0.974** (1.000) |
| Sparse | 0.912 (0.534) | 0.927 (0.534) | 0.826 (1.000) | **0.956** (1.000) |

</div>

*Final ESS, with $\kappa=5$ coverage in parentheses. Bold marks the highest-ESS
setting among methods whose sample panel covers every target mode. All methods
cover Two-Moon; for the other targets, a larger ESS on reached support does not
override visibly missing modes.*

The sample panels separate the two roles of the wide-coverage objective.
Although $\mathrm{X}_{\hat\mu}$ enables the flow to discover modes unseen by
the target-only objectives, the equal-weight
$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$ objective produces substantially cleaner
pushforward samples: the $\bar\nu$ component curbs spurious coverage of intermodal and
off-target regions without sacrificing the discovered modes. On Two-Moon,
Three-Well, and Himmelblau, the two quench and temper objectives remain in the
same high-ESS range; Sparse strengthens that comparison because the cleaner
mixture also has substantially higher ESS.

## Verification summary

- All four full training scripts exited successfully and ended with `DONE`.
- All eight calls to the KLXX trainer used `pool_size=0`; no separate training pool was
  resampled.
- All 16 final ESS/coverage pairs are finite; every reported coverage lies in
  `[0, 1]`.
- The four sample figures and four ESS-history figures were regenerated and
  visually inspected; no panel is blank or malformed. The figures confirm
  complete mode coverage for all runs using QT, including the Three-Well runs
  whose approximate scalar coverage is slightly below one.
- The figures and paired metrics separate calibration from discovery: a high
  ESS on reached support does not certify that every mode was found.
- Per-run logs and temporary data live below each target's `artifacts/`
  directory. The figures above are the final outputs below `results/`.

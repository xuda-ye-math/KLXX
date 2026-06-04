# Clock_Lattice summary — latest run `L4_balance`



**Target.** p-state clock, P=6, J=1.0, H=0.5, L=4 (D=16), periodic square lattice, torus domain.
**Method.** Algorithm 4: NCSF per stage, uniform-on-torus source, adaptive ladder (ADAPIVE_TAU=0.7, VALIDATION_TAU=0.3, SHRINK=0.7).

**Headline.** K=3 stages, ladder [0.25, 0.6, 1.0]; composed-generator direct ESS = **0.5192**; sectors found (pushforward) = **6/6** (TV from uniform 0.078, mean |m| 0.775); kNN coverage vs QT set = 0.922; wall 5.5 min.

Per-stage validation ESS (post-training acceptance gate): t=0.250: 0.982, t=0.600: 0.910, t=1.000: 0.727

**Interpretation.** All P sectors found at near-uniform occupancy with a healthy composed ESS means the ladder bridged the BKT regime without sector collapse; missing sectors with a high ESS would be the fake-ESS failure the X functionals are designed to prevent.

Files: `data_<tag>.pth` (per-stage state_dicts inside), `results_table.md/.csv`, `figures/ladder_<tag>.png`, `figures/sectors_<tag>.png`, `figures/magnetization_<tag>.png`, `train_status.log` (live).

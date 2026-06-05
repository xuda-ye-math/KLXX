# Clock_Lattice summary — latest run `L6_balance`



**Target.** p-state clock, P=6, J=1.0, H=0.5, L=6 (D=36), periodic square lattice, torus domain.
**Method.** Algorithm 4: NCSF per stage, uniform-on-torus source, adaptive ladder (ADAPIVE_TAU=0.7, VALIDATION_TAU=0.3, SHRINK=0.7).

**Headline.** K=4 stages, ladder [0.25, 0.6, 0.796, 1.0]; composed-generator direct ESS = **0.0351**; sectors found (pushforward) = **6/6** (TV from uniform 0.176, mean |m| 0.573); kNN coverage vs QT set = 0.996; wall 43.6 min.

Per-stage validation ESS (post-training acceptance gate): t=0.250: 0.858, t=0.600: 0.472, t=0.796: 0.465 (retries 2), t=1.000: 0.506

**Interpretation.** All P sectors found at near-uniform occupancy with a healthy composed ESS means the ladder bridged the BKT regime without sector collapse; missing sectors with a high ESS would be the fake-ESS failure the X functionals are designed to prevent.

Files: `data_<tag>.pth` (per-stage state_dicts inside), `results_table.md/.csv`, `figures/ladder_<tag>.png`, `figures/sectors_<tag>.png`, `figures/magnetization_<tag>.png`, `train_status.log` (live).

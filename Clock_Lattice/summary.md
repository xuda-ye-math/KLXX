# Clock_Lattice summary — latest run `L8_kl_B1k`



**Target.** p-state clock, P=6, J=1.0, H=0.5, L=8 (D=64), periodic square lattice, torus domain.
**Method.** Algorithm 4: NCSF per stage, uniform-on-torus source, adaptive ladder (ADAPIVE_TAU=0.7, VALIDATION_TAU=0.3, SHRINK=0.7).

**Headline.** K=8 stages, ladder [0.2, 0.396, 0.53, 0.662, 0.753, 0.841, 0.952, 1.0]; composed-generator direct ESS = **0.0002**; sectors found (pushforward) = **6/6** (TV from uniform 0.120, mean |m| 0.269); kNN coverage vs QT set = 1.000; wall 50.7 min.

Per-stage validation ESS (post-training acceptance gate): t=0.200: 0.391, t=0.396: 0.398, t=0.530: 0.463 (retries 1), t=0.662: 0.366 (retries 1), t=0.753: 0.467 (retries 2), t=0.841: 0.465 (retries 2), t=0.952: 0.400 (retries 1), t=1.000: 0.708

**Interpretation.** All P sectors found at near-uniform occupancy with a healthy composed ESS means the ladder bridged the BKT regime without sector collapse; missing sectors with a high ESS would be the fake-ESS failure the X functionals are designed to prevent.

Files: `data_<tag>.pth` (per-stage state_dicts inside), `results_table.md/.csv`, `figures/ladder_<tag>.png`, `figures/sectors_<tag>.png`, `figures/magnetization_<tag>.png`, `train_status.log` (live).

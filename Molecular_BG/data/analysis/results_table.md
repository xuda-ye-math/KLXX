# Final benchmark diagnostics

| Quantity | Chain 1 | Chain 2 |
|---|---:|---:|
| Total simulated time per slot | 20 ns | 20 ns |
| Frames after fixed 2 ns burn-in | 18,000 | 18,000 |
| Exchange acceptance, min--max | 0.543--0.882 | 0.544--0.879 |
| Minimum ladder round trips per walker | 461 | 471 |
| 36-bin JS to FAB train data | 0.01104 bits | 0.01172 bits |
| Positive-phi samples | 45 (0.250%) | 61 (0.339%) |

Cross-chain JS divergence is 0.00523, 0.01229, and 0.01944 bits at 24, 36,
and 48 bins. The pooled dataset contains 36,000 post-burn-in 300 K frames.

# p-state clock — Algorithm 4 results

| tag | D | P | method | K | complete | ladder | final_ess | sectors | tv | abs_m | knn | wall_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4_balance | 16 | 6 | balance | 3 | True | 0.250 0.600 1.000 | 0.5192 | 6/6 | 0.0780 | 0.7749 | 0.9220 | 5.4775 |
| L4_kl | 16 | 6 | kl | 3 | True | 0.250 0.600 1.000 | 0.2631 | 6/6 | 0.0924 | 0.7442 | 0.9570 | 5.2751 |
| L4_kl_smoke | 16 | 6 | kl | 2 | False | 0.250 0.495 | 0.0023 | 6/6 | 0.0947 | 0.2729 | 0.9990 | 0.4898 |
| L4_smoke | 16 | 6 | balance | 3 | True | 0.250 0.600 1.000 | 0.0041 | 6/6 | 0.1100 | 0.3198 | 1.0000 | 0.3086 |
| L8_balance | 64 | 6 | balance | 5 | True | 0.250 0.495 0.663 0.828 1.000 | 0.0011 | 6/6 | 0.0916 | 0.2288 | 1.0000 | 65.4940 |
| L8_kl | 64 | 6 | kl | 3 | False | 0.250 0.495 0.577 | 0.0001 | 6/6 | 0.2435 | 0.1920 | 0.9995 | 214.5727 |

**Energy check (L4_balance)**: reweighted <E> = -24.516 (Var 15.873) vs classical-SMC reference <E> = -24.409 (Var 15.896), relative error 0.0044. Per-sector ESS: s0: 0.658 (n=2532), s1: 0.672 (n=2876), s2: 0.549 (n=3627), s3: 0.282 (n=4289), s4: 0.527 (n=3645), s5: 0.643 (n=3031)

**Energy check (L4_kl)**: reweighted <E> = -24.499 (Var 15.848) vs classical-SMC reference <E> = -24.409 (Var 15.896), relative error 0.0037. Per-sector ESS: s0: 0.491 (n=2468), s1: 0.447 (n=2809), s2: 0.311 (n=3814), s3: 0.081 (n=4311), s4: 0.359 (n=3724), s5: 0.463 (n=2874)

**Energy check (L8_balance)**: reweighted <E> = -74.034 (Var 50.876) vs classical-SMC reference <E> = -93.066 (Var 80.870), relative error 0.2045. Per-sector ESS: s0: 0.002 (n=4112), s1: 0.002 (n=3919), s2: 0.001 (n=2889), s3: 0.003 (n=2740), s4: 0.001 (n=2540), s5: 0.001 (n=3800)

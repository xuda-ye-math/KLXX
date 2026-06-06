# Single-stage smoke results

| tag | d | t | method | steps | batch | train_ok | final direct ESS | validation ESS | wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| m4_t0.1_balance | 16 | 0.1 | balance | 400 | 1000 | True | 0.9268 | 0.9252 | 3.2 |
| m4_t0.01_balance | 16 | 0.01 | balance | 400 | 1000 | True | 0.9974 | 0.9972 | 0.4 |
| m4_t0.1_kl | 16 | 0.1 | kl | 400 | 1000 | True | 0.8284 | 0.8279 | 0.7 |
| m4_t0.3_balance | 16 | 0.3 | balance | 400 | 1000 | True | 0.7244 | 0.7121 | 0.5 |
| m4_t0.2_balance | 16 | 0.2 | balance | 400 | 1000 | True | 0.8117 | 0.8188 | 0.8106 | 1.7 |
| m4_t0.2_kl | 16 | 0.2 | kl | 400 | 1000 | True | 0.7365 | 0.7172 | 0.7070 | 0.6 |
| m6_t0.2_balance | 36 | 0.2 | balance | 400 | 1000 | False | 0.0064 | 0.0049 | 0.0066 | 3.2 |
| m6_t0.05_balance | 36 | 0.05 | balance | 400 | 1000 | True | 0.0419 | 0.1193 | 0.0873 | 1.2 |
| m6_t0.02_balance | 36 | 0.02 | balance | 400 | 1000 | True | 0.3250 | 0.2675 | 0.1491 | 1.2 |

# Ac-Pro-NHMe regularization screen

This standalone screen reuses the audited chiral-molecule regularization
protocol without changing its frozen three-molecule configuration. The target
is the 26-atom, 72-dimensional `ACE--L-PRO--NME` ff96/OBC1 bundle one directory
above this folder.

The candidates are `(100, 0.10)` and `(150, 0.10)`. At the hottest 400 K
replica, the finite-second-moment tail exponents are 60.136 and 90.204,
respectively. The required threshold is `d + 2 = 74`, so `(100, 0.10)` is
rejected analytically and only `(150, 0.10)` is eligible for trajectory-based
selection.

Typical pilot commands are:

```bash
python run.py sanity --molecule ac_pro_nhme --condition raw
python run.py sanity --molecule ac_pro_nhme --condition candidate --e 150 --r 0.10
python run.py pilot --molecule ac_pro_nhme --condition raw
python run.py pilot --molecule ac_pro_nhme --condition candidate --e 150 --r 0.10
python analyze.py evaluate pilot --molecule ac_pro_nhme --e 150 --r 0.10
```

Candidate-to-raw RESS is the primary sharpening-fidelity metric. Reverse RESS
is retained only as an audit diagnostic because raw samples need not represent
the broader regularized support efficiently.

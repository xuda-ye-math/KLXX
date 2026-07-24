# Chiral-molecule OpenMM regularization study

This standalone folder selects and verifies regularization parameters for
`(S)`-2-butanol, `(2R,3R)`-2,3-butanediol, and L-alanine dipeptide. The default
is `(100, 0.10)` and every candidate is at least as strict. ADP starts at
`(125, 0.10)` because the default misses the preregistered finite-second-moment
tail margin at the 400 K replica.

The frozen protocol is `.aris/EXPERIMENT_PLAN.md`. `run.py` writes atomic,
metadata-validated trajectory artifacts with saved-frame stereochemistry
checks and walker identities. `analyze.py` binds each metric to the exact NPZ
bytes by SHA-256, evaluates candidate-to-raw RESS, retains the reverse
direction as an audit diagnostic, selects the first eligible candidate, and
generates the final report. The final provenance file records the current
runner hash post hoc; it was not embedded in the NPZ metadata at generation.

Run one GPU job at a time. Typical commands are:

```bash
python run.py sanity --molecule s_2_butanol --condition raw
python run.py sanity --molecule s_2_butanol --condition candidate --e 100 --r 0.10
python run.py pilot --molecule s_2_butanol --condition raw
python run.py pilot --molecule s_2_butanol --condition candidate --e 100 --r 0.10
python analyze.py evaluate pilot --molecule s_2_butanol --e 100 --r 0.10
python analyze.py select
```

After pilot selection, run both verification seeds for the raw and selected
candidate conditions, evaluate them, then use `python analyze.py finalize`.
All selection metrics use post-burn-in 300 K frames. Low reverse ESS or failed
mixing is reported without being misclassified as failed sharpening fidelity.

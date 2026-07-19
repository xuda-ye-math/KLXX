# Alkane-family OpenMM regularization study

This directory selects one molecular regularization pair `(e, r)` for the
GAFF2/AM1-BCC/OBC1 n-alkane family from methane through n-hexane (54D).
Sampling, energy evaluation, and force evaluation use only the native OpenMM
backend exposed by `jflows_md.openmm`.

The frozen design and decision gates are in
`.aris/EXPERIMENT_PLAN.md`. The final evidence-backed recommendation is written
to `REPORT.md`. Raw trajectories are saved after each completed target and
seed, so an interrupted sweep can continue without repeating valid files.

Commands are run with the project environment:

```bash
/home/xuda/.envs/jflows/bin/python build_hexane.py
/home/xuda/.envs/jflows/bin/python run.py sanity
/home/xuda/.envs/jflows/bin/python run.py pilot
/home/xuda/.envs/jflows/bin/python analyze.py pilot
```

One user-directed refinement can be added without changing the frozen sampling
configuration:

```bash
/home/xuda/.envs/jflows/bin/python run.py candidate --e 100 --r 0.12
```

The all-family verification command is selected only after the n-hexane pilot
has been analyzed.

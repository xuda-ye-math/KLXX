# Run manifest

- Experiment/run ID: `calibration_20260711`
- Attempt/status: 1 / complete
- Environment: `.aris/artifacts/ENV_SNAPSHOT.md`
- Git revision: `ecc2371dc00a2fa5882767db1223dc1b91f97f1f` plus dirty molecular worktree
- Exact command: `/home/xuda/miniconda3/envs/jflows/bin/python Molecular_BG/openmm_remd.py --output Molecular_BG/data/runs/calibration_20260711/remd.npz --ns 0.1 --seed 20260711`
- Seed: 20260711
- Inputs: `Molecular_BG/system/alanine_dipeptide.pdb`, `Molecular_BG/system/l_minimum_positions_nm.npy`
- Output: `Molecular_BG/data/runs/calibration_20260711/remd.npz`
- Protocol: 21 linear slots, 300--1300 K; constraints=None; 1 fs; exchange 200 steps; sample 1000 steps
- Checkpoint/resume: schema-v2 native OpenMM checkpoints + exchange RNG/walker state
- Result: 0.1 ns completed in 65.1 s; 100 target samples; edge acceptance 0.520--0.920; all walkers visited both ladder ends; 34 ensemble round trips; no non-finite state.

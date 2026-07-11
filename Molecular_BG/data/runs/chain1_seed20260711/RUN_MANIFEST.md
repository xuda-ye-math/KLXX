# Run manifest

- Experiment/run ID: `chain1_seed20260711`
- Attempt/status: 1 / complete at 20 ns
- Start: 2026-07-11 America/New_York
- Environment: `.aris/artifacts/ENV_SNAPSHOT.md`
- Git revision: `ecc2371dc00a2fa5882767db1223dc1b91f97f1f` plus dirty molecular worktree
- Exact command: `/home/xuda/miniconda3/envs/jflows/bin/python Molecular_BG/openmm_remd.py --output Molecular_BG/data/runs/chain1_seed20260711/remd.npz --ns 10 --seed 20260711`
- Seed: 20260711
- Inputs: `Molecular_BG/system/alanine_dipeptide.pdb`, `Molecular_BG/system/l_minimum_positions_nm.npy`
- Output: `Molecular_BG/data/runs/chain1_seed20260711/remd.npz`
- Protocol: 21 linear slots, 300--1300 K; constraints=None; 1 fs; exchange 200 steps; sample 1000 steps
- Burn-in policy: first 2 ns excluded during analysis
- Checkpoint/resume: schema-v2 native OpenMM checkpoints + exchange RNG/walker state
- Result: 10 ns completed in 6488.5 s; 10,000 total and 8,000 strictly post-burn-in frames; edge acceptance 0.542--0.884; all walkers visited all 21 slots; 5,173 ensemble round trips; 36 x 36 JS to exact FAB train data 0.01783 bits.
- Extension command: `/home/xuda/miniconda3/envs/jflows/bin/python Molecular_BG/openmm_remd.py --output Molecular_BG/data/runs/chain1_seed20260711/remd.npz --ns 20 --seed 20260711 --resume`
- Extension result: reached 20 ns; second 10 ns segment completed in 6435.1 s; final edge acceptance 0.543--0.882.

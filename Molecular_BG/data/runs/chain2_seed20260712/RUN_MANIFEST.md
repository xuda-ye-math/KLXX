# Run manifest

- Experiment/run ID: `chain2_seed20260712`
- Attempt/status: 1 / complete at 20 ns
- Start: 2026-07-11 America/New_York
- Environment: `.aris/artifacts/ENV_SNAPSHOT.md`
- Exact planned command: `/home/xuda/miniconda3/envs/jflows/bin/python Molecular_BG/openmm_remd.py --output Molecular_BG/data/runs/chain2_seed20260712/remd.npz --ns 10 --seed 20260712`
- Seed: 20260712
- Protocol/burn-in: identical to chain 1; first 2 ns excluded
- Output: `Molecular_BG/data/runs/chain2_seed20260712/remd.npz`
- Result: 10 ns completed in 6465.5 s; 10,000 total and 8,000 strictly post-burn-in frames; edge acceptance 0.545--0.878; all walkers visited all 21 slots; 5,251 ensemble round trips; 36 x 36 JS to exact FAB train data 0.01873 bits.
- Extension command: `/home/xuda/miniconda3/envs/jflows/bin/python Molecular_BG/openmm_remd.py --output Molecular_BG/data/runs/chain2_seed20260712/remd.npz --ns 20 --seed 20260712 --resume`
- Extension result: reached 20 ns; second 10 ns segment completed in 6434.0 s; final edge acceptance 0.544--0.879.

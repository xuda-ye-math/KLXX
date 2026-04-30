"""Fit a Gaussian-mixture prior to a target distribution.

Usage:
    python prior.py --name HB
"""
from parameters import *  # noqa: F401  (provides `name` from --name)
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

MAX_MODES = 20

np.random.seed(0)
torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# Load target-distributed samples via importance resampling.                  #
# --------------------------------------------------------------------------- #
data = torch.load(f"data_{name}_old.pt", weights_only=False)
samples = data['samples'].numpy()                # [N, 2]
weights = data['weights'].numpy()                # [N]
probs = weights / weights.sum()
N = samples.shape[0]
idx = np.random.choice(N, size=N, replace=True, p=probs)
x = samples[idx]
D = x.shape[1]
print(f"Loaded {N} target samples for '{name}', dim={D}")

# --------------------------------------------------------------------------- #
# Fit GMM, auto-selecting K in [1, max_modes] by BIC.                         #
# --------------------------------------------------------------------------- #
print(f"Auto-selecting K in [1, {MAX_MODES}] by BIC...")
best_score = float('inf')
K = 1
w = np.array([1.0])
μ = x.mean(axis=0, keepdims=True)
σ = np.array([x.std()])
for k in range(1, MAX_MODES + 1):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='spherical',
        max_iter=300,
        random_state=0,
    ).fit(x)
    score = gmm.bic(x)
    ll = gmm.score(x) * N
    print(f"  K={k:2d}  log-lik={ll:10.2f}  BIC={score:10.2f}")
    if score < best_score:
        best_score = score
        K = k
        w = gmm.weights_
        μ = gmm.means_
        σ = np.sqrt(gmm.covariances_)
print(f"Selected K={K}")

# --------------------------------------------------------------------------- #
# Save as torch tensors.                                                      #
# --------------------------------------------------------------------------- #
w = np.asarray(w)
μ = np.asarray(μ)
σ = np.asarray(σ)
w_t = torch.tensor(w, dtype=torch.float32)
μ_t = torch.tensor(μ, dtype=torch.float32)
σ_t = torch.tensor(σ, dtype=torch.float32)
print("w_list:", w_t)
print("μ_list:\n", μ_t)
print("σ_list:", σ_t)

out_pt = f'prior_{name}.pt'
torch.save({'w_list': w_t, 'μ_list': μ_t, 'σ_list': σ_t}, out_pt)
print(f"Saved {out_pt}")


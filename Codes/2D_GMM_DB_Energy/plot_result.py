import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from parameters import *

# load inference data: samples are x = F(y), y ~ mu_0
data    = torch.load(f"data_{name}.pt", weights_only=False)
samples = data['samples']   # [N, 2]  x = F(y) ~ F_# mu_0
weights = data['weights']   # [N]     normalized importance weights toward mu
ess     = data['ess']

# importance-resample x to recover target mu
log_w = torch.log(weights + 1e-30)
x_rs  = resample(samples, log_w)

# plot
bound = target.BOUND
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(x_rs[:, 0].numpy(), x_rs[:, 1].numpy(), s=0.5, c='darkred', alpha=0.3)
axes[0].set_title(r'target $\mu$')
axes[0].set_xlim(-bound, bound); axes[0].set_ylim(-bound, bound)
axes[0].set_aspect('equal')

axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=0.5, c='darkblue', alpha=0.3)
μ_np = gmm.μ_list.numpy()
σ_np = gmm.σ_list.numpy()
θ = np.linspace(0, 2 * np.pi, 200)
for k in range(μ_np.shape[0]):
    axes[1].plot(μ_np[k, 0] + 2 * σ_np[k] * np.cos(θ),
                 μ_np[k, 1] + 2 * σ_np[k] * np.sin(θ),
                 'b-', lw=1.0, alpha=0.8)
    axes[1].plot(μ_np[k, 0], μ_np[k, 1], 'b+', markersize=10)
axes[1].set_title(r'mapped $F_{\#}\mu_0$ with prior centers')
axes[1].set_xlim(-bound, bound); axes[1].set_ylim(-bound, bound)
axes[1].set_aspect('equal')

fig.suptitle(f'DB-Energy flow samples for {target} (ESS: {ess:.1%})', fontsize=14)
plt.savefig(f'result_{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print(f"Saved result_{name}.png")

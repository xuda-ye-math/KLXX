import torch
import matplotlib.pyplot as plt
from utilities import *
from parameters import *

# load inference data
data    = torch.load(f"data_{name}.pt", weights_only=False)
samples = data['samples']   # [N, 2]  x ~ approx mu
weights = data['weights']   # [N]     normalized importance weights
ess     = data['ess']

# load trained flow to compute T_# mu = F(x)
flow.load_state_dict(torch.load(f"flow_{name}.pt", weights_only=True))
flow.eval()
with torch.no_grad():
    y, _ = flow.t().call_and_ladj(samples)  # [N, 2]  F(x)

# resample with importance weights so scatter reflects true density
log_w = torch.log(weights + 1e-30)
x_rs  = resample(samples, log_w)
y_rs  = resample(y,       log_w)

# plot
bound = target.BOUND
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(x_rs[:, 0].numpy(), x_rs[:, 1].numpy(), s=0.5, c='red', alpha=0.3)
axes[0].set_title(r'target $\mu$')
axes[0].set_xlim(-bound, bound); axes[0].set_ylim(-bound, bound)
axes[0].set_aspect('equal')

axes[1].scatter(y_rs[:, 0].numpy(), y_rs[:, 1].numpy(), s=0.5, c='blue', alpha=0.3)
axes[1].set_title(r'mapped $F_{\#}\mu$')
axes[1].set_xlim(-bound, bound); axes[1].set_ylim(-bound, bound)
axes[1].set_aspect('equal')

fig.suptitle(f'DBNF samples for {target} (ESS: {ess:.1%})', fontsize=14)
plt.savefig(f'result_{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print(f"Saved result_{name}.png")

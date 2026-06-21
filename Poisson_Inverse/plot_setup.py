# pyright: reportArgumentType=false
"""Setup figure for the paper's Poisson subsection: (a) true source field,
(b) PDE solution with the sensor ring, (c) posterior well lattice from the
PT referee cold samples at sigma_obs = 0.01. -> figures/fig_setup.png"""
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import parameters as p
import potential as pot

p.SIGMA_OBS = 0.01
p.N_SENSORS = 54
device = 'cpu'
torch.manual_seed(0)
B = pot.build(p, device)
d_full, std_all = B['d_full'], B['std_all']
th = B['theta_truth']

n = 128                                    # fine display grid
pts = pot.cell_grid(n)
Phi = pot.eval_basis(B['modes'], pts)
v = (Phi @ th).view(n, n)
g = p.G0 * (1.0 + p.DELTA * torch.cos(p.ALPHA * v))
G = torch.fft.fft2(g.view(1, n, n))
inv = pot.inv_helmholtz_mult(n, p.C2, torch.get_default_dtype(), 'cpu')
u = torch.fft.ifft2(G * inv.unsqueeze(0)).real[0]

R = torch.load('referee_o0.01.pth', weights_only=False, map_location='cpu')
cold = R['cold']
th1 = cold[:, 0] * std_all[0]
th2 = cold[:, 1] * std_all[1]
n_lab = torch.round(th1 * p.ALPHA / (2.0 * math.pi)).long()

plt.rcParams['axes.titlepad'] = 10          # extra gap between subplot titles and axes
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
ax = axes[0]
im = ax.imshow(g.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
ax.set_title(r'(a) source $G_0(1+\delta\cos(\alpha v(x;\theta^\dagger)))$')
ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
plt.colorbar(im, ax=ax, fraction=0.046)
ax = axes[1]
im = ax.imshow(u.T, origin='lower', extent=[0, 1, 0, 1], cmap='coolwarm')
ths = 2 * math.pi * torch.arange(p.N_SENSORS) / p.N_SENSORS
ax.scatter(0.5 + 0.3 * torch.cos(ths), 0.5 + 0.3 * torch.sin(ths),
           s=14, c='black', edgecolors='white', linewidths=0.5, zorder=3)
ax.set_title(r'(b) solution $u$ and the $54$ sensors')
ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
plt.colorbar(im, ax=ax, fraction=0.046)
ax = axes[2]
sel = torch.randperm(cold.shape[0])[:8000]
sc = ax.scatter(th1[sel], th2[sel], s=2, c=n_lab[sel], cmap='tab10',
                alpha=0.35, vmin=-3, vmax=6)
ax.set_xlabel(r'$\theta_{(0,0)}$')
ax.set_ylabel(r'$\theta_{(0,1)}$')
ax.set_title(r'(c) marginal posterior, referee at $\sigma_{\mathrm{obs}}=0.01$')
plt.tight_layout()
plt.savefig('figures/fig_setup.png', dpi=400, bbox_inches='tight', pad_inches=0.02)
print('wrote figures/fig_setup.png')

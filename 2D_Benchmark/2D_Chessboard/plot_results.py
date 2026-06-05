from pathlib import Path
import torch
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'mathtext.fontset': 'cm', 'font.family': 'serif',
})
from matplotlib.colors import LinearSegmentedColormap

from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import importance_weights, resample, langevin

from core import Chessboard, coverage
from parameters import SIGMA, NSF_LIM, PLT_LIM, BINS, TRANSFORMS, HIDDEN_FEATURES, STEPS

HERE = Path(__file__).resolve().parent

METHODS = (
    'KL',
    'KL+X_mu',
    'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix',
)
METHOD_LABEL = {
    'KL':                       'forward KL',
    'KL+X_mu':                  r'forward KL+$\mathrm{X}_\mu$',
    'KL+X_mu+X_hat_mu':         r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$',
    'KL+X_mu+X_mix':            r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{(\hat\mu+\bar\nu)/2}$',
}
METHOD_COLOR = {
    'KL':                       "#1F77B4A0",   # tab:blue
    'KL+X_mu':                  "#2CA02CA0",   # tab:green
    'KL+X_mu+X_hat_mu':         "#D62728A0",   # tab:red
    'KL+X_mu+X_mix':            "#9467BDA0",   # tab:purple
}

cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

u1 = Chessboard().to(device)
u1.enable_grad()


def compute_pipeline(state_dict, x_unif):
    flow = NSF(
        a=[-NSF_LIM, -NSF_LIM], b=[+NSF_LIM, +NSF_LIM],
        bins=BINS, transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES,
    ).to(device)
    flow.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    u0 = Gaussian(mean=[0.0]*2, variance=[SIGMA**2]*2).to(device)
    x = x_unif.to(device)
    with torch.no_grad():
        G = flow.t()
        y, _ = G.inv.call_and_ladj(x)
        w = importance_weights(x, u0, u1, G.inv)
    y_resampled = resample(y, w)
    y_mu = langevin(y_resampled, u1, step=2e-3, iters=50)
    return y.cpu(), y_resampled.cpu(), y_mu.cpu()


# Background contour data shared across all sample figures
xlim = (-PLT_LIM, PLT_LIM)
ylim = (-PLT_LIM, PLT_LIM)
n = 400
xs = torch.linspace(xlim[0], xlim[1], n)
ys = torch.linspace(ylim[0], ylim[1], n)
X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(device)
with torch.no_grad():
    U_grid = u1(grid).reshape(*X1.shape).cpu().numpy()
# Narrow level range around the chessboard mode tops (U ~ -1 at centers, ~20+ between cells);
# extend='max' clips the high-U gaps to the top color.
levels = torch.linspace(-2.0, 20.0, 50).tolist()

STAGES = (
    ('pushforward', 'samples', r'$y = G^{-1}(x)$'),
    ('resampled',   'resample', r'resample$(y, w)$'),
)

# Gaussian prior samples for visual context (mean 0, std SIGMA)
torch.manual_seed(42)
prior_np = (torch.randn(5000, 2) * SIGMA).numpy()


data_path = HERE / 'data.pth'
if not data_path.exists():
    raise SystemExit(f"{data_path} not found. Run train.py first.")
data = torch.load(data_path, weights_only=False)
runs_data = data['runs']
y_hat_mu = data.get('y_hat_mu')
x_unif = data['x_unif']

# final ESS of each method's raw pushforward -- always available
for m in METHODS:
    print(f"[{m:<24}] final ESS = {runs_data[m]['final_ess']:.4f}")

if y_hat_mu is not None:
    for m in METHODS:
        cov = coverage(runs_data[m]['samples'], y_hat_mu, k=5)
        runs_data[m]['coverage'] = cov
        print(f"[{m:<24}] ESS = {runs_data[m]['final_ess']:.4f}   coverage_k=5 vs y_hat_mu = {cov:.4f}")

pipeline = {}
for m in METHODS:
    y, y_resampled, _ = compute_pipeline(runs_data[m]['state_dict'], x_unif)
    pipeline[m] = {'pushforward': y, 'resampled': y_resampled}

# ESS figure
fig_ess, ax_ess = plt.subplots(1, 1, figsize=(4.5, 3.5))
for m in METHODS:
    ax_ess.plot(runs_data[m]['ess_history'], color=METHOD_COLOR[m], label=METHOD_LABEL[m], linewidth=0.6)
ax_ess.set_xlabel('step')
ax_ess.set_ylabel('ESS')
ax_ess.set_xlim(0, STEPS)
ax_ess.set_ylim(0, 1)
ax_ess.legend(loc='lower right')
plt.tight_layout()
ess_path = HERE / 'ESS.png'
plt.savefig(ess_path, dpi=400, bbox_inches='tight')
plt.close(fig_ess)
print(f"Saved {ess_path}")

# samples + resample figures: 1 x 4
for key, suffix, stage_label in STAGES:
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for idx, m in enumerate(METHODS):
        samples_np = pipeline[m][key].numpy()
        ax = axes[idx]
        ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
        ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
        ax.scatter(prior_np[:, 0], prior_np[:, 1], s=0.04, alpha=0.3, color='gray', zorder=5)
        ax.scatter(samples_np[:, 0], samples_np[:, 1], s=0.04, alpha=0.35, color=METHOD_COLOR[m], zorder=10)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        title = METHOD_LABEL[m]
        if key == 'pushforward':
            title += f"\nESS = $\\mathbf{{{runs_data[m]['final_ess']:.2f}}}$"
            if 'coverage' in runs_data[m]:
                title += f", cvrg = $\\mathbf{{{runs_data[m]['coverage']:.2f}}}$"
        else:
            title += f"\n{stage_label}"
        ax.set_title(title)
    plt.tight_layout()
    out_path = HERE / f"{suffix}.png"
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

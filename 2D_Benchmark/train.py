# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false

from pathlib import Path
import torch
from zuko.transforms import ComposedTransform
from zflows.flow import NSF
from zflows.potential import Potential, Gaussian
from zflows.utils import compute_ESS, importance_weights, resample, langevin

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# boundary of the domain
SIGMA = 2.0 # deviation of Gaussian prior
PLT_LIM = 6.0
NSF_LIM = 6.0

# source: Gaussian U0
u0 = Gaussian(mean=[0.0]*2, variance=[SIGMA**2]*2).to(device)

# target: Himmelblau potential U1
class U1(Potential):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1.square() + x2 - 11).square()
        term2 = (x1 + x2.square() - 7).square()
        return term1 + term2
u1 = U1().to(device)
u1.enable_grad() # enable_grad for Langevin rejuvenation

# training parameters
N_TRAIN: int = 50000   # number of training samples
N_VALID: int = 50000   # number of validation samples
LR = {100: 1e-3, 1000: 3e-3}   # learning rate per batch (sqrt scaling: BATCH=1000 uses sqrt(10)x ~3.16x)
STEPS = {100: 1000, 1000: 500}    # BATCH=1000 trained 5x beyond compute parity (parity at step=100)

# forward KL loss (KL(mu_1 || G^{-1}_# u_0) up to a G-independent constant)
def forward_KL(y: torch.Tensor, source: Potential, G: ComposedTransform):
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    return (source(x) - ladj).mean()

# forward XX loss
def forward_XX(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    N = y.shape[0]
    perm = torch.randperm(N, device=y.device)
    y_ = y[perm]
    x,  ladj  = G.call_and_ladj(y)  # x  = G(y),  ladj  = log|det J_G(y)|
    x_, ladj_ = G.call_and_ladj(y_) # x_ = G(y_), ladj_ = log|det J_G(y_)|
    A  = source(x)  - target(y)  - ladj
    A_ = source(x_) - target(y_) - ladj_
    return (A - A_).abs().mean()

def new_flow():
    flow = NSF(a=[-NSF_LIM, -NSF_LIM], b=[+NSF_LIM, +NSF_LIM], bins=32, transforms=6, hidden_features=(128, 128)).to(device)
    flow.zeros()
    return flow

torch.manual_seed(0)
x_pool = u0.samples(N_TRAIN)                       # x ~ u_0 (shared across all runs)

def train(method: str, BATCH: int, steps: int):
    torch.manual_seed(0)
    flow = new_flow()
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR[BATCH])
    ess_history = []
    for step in range(steps):
        idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
        x = x_pool[idx]

        with torch.no_grad():
            G_now = flow.t()
            y, _ = G_now.inv.call_and_ladj(x)
            w = importance_weights(x, u0, u1, G_now.inv)
            ess_history.append(compute_ESS(w).item())

            if method in ('KL', 'KL++'):
                # one-step IS -> approximate mu_1 samples
                y_mu1 = resample(y, w)
                y_mu1 = langevin(y_mu1, u1, step=1e-2, iters=10)
            elif method == 'FAB':
                # squared IS -> g samples
                y_g = resample(y, w * w)
            elif method == 'FAB++':
                # two-step IS: mu_1 samples then g samples by reweighting mu_1
                y_mu1 = resample(y, w)
                y_mu1 = langevin(y_mu1, u1, step=1e-2, iters=10)
                x_back, ladj = G_now.call_and_ladj(y_mu1)
                log_w2 = -u1(y_mu1) + u0(x_back) - ladj
                w2 = (log_w2 - log_w2.max()).exp()
                y_g = resample(y_mu1, w2)

        G = flow.t()
        if method == 'KL':
            loss = forward_KL(y_mu1, u0, G)
        elif method == 'KL++':
            loss = forward_KL(y_mu1, u0, G) + forward_XX(y_mu1, u0, u1, G)
        elif method == 'FAB':
            loss = forward_KL(y_g, u0, G)
        elif method == 'FAB++':
            loss = forward_KL(y_g, u0, G) + forward_XX(y_mu1, u0, u1, G)
        else:
            raise ValueError(method)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[{method:<6} B={BATCH:>4}] step {step+1:>4}/{steps}   loss = {loss.item():.4e}")

    print(f"[{method:<6} B={BATCH:>4}] last training ESS = {ess_history[-1]:.4f}")
    return flow, ess_history

DATA_PATH = HERE / 'data.pth'

if not DATA_PATH.exists():
    # Run all 4 methods × 2 batch sizes
    results = {}
    for B in (100, 1000):
        for m in ('KL', 'KL++', 'FAB', 'FAB++'):
            print(f"=== {m}, BATCH={B} ===")
            results[(m, B)] = train(m, B, STEPS[B])

    # Compute final samples and ESS for each run
    x_unif = u0.samples(N_VALID)
    runs_data = {}
    for B in (100, 1000):
        runs_data[B] = {}
        for m in ('KL', 'KL++', 'FAB', 'FAB++'):
            flow, ess_history = results[(m, B)]
            with torch.no_grad():
                G = flow.t()
                y_push, _ = G.inv.call_and_ladj(x_unif)
                w = importance_weights(x_unif, u0, u1, G.inv)
                final_ess = compute_ESS(w).item()
            runs_data[B][m] = {
                'state_dict': {k: v.cpu() for k, v in flow.state_dict().items()},
                'ess_history': ess_history,
                'samples': y_push.cpu(),
                'final_ess': final_ess,
            }
            print(f"[{m:<6} B={B:>4}] final ESS = {final_ess:.4f}")

    torch.save({
        'config': {
            'STEPS': STEPS,
            'LR': LR,
            'N_TRAIN': N_TRAIN,
            'N_VALID': N_VALID,
            'SIGMA': SIGMA,
            'NSF_LIM': NSF_LIM,
            'PLT_LIM': PLT_LIM,
            'batch_sizes': [100, 1000],
            'methods': ['KL', 'KL++', 'FAB', 'FAB++'],
        },
        'x_unif': x_unif.cpu(),
        'runs': runs_data,
    }, DATA_PATH)
    print(f"Saved all data to {DATA_PATH}")
else:
    print(f"Found {DATA_PATH}; skipping training")

# Always load from disk so plotting is a single source of truth
data = torch.load(DATA_PATH, weights_only=False)
runs_data = data['runs']
config = data['config']

# --- Plotting ---
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])

# Color scheme:
#   BATCH=100:  dark blue / dark red  (sample-limited regime — emphasized)
#   BATCH=1000: regular blue / regular red
COLORS = {
    100:  {'base': "#00008B", 'pp': "#8B0000"},
    1000: {'base': "#00008B", 'pp': "#8B0000"},
}

# ESS figure: 2x2  (rows = batch sizes 100/1000, cols = KL/FAB families)
fig_ess, axes_ess = plt.subplots(2, 2, figsize=(8, 8))
for row, B in enumerate((100, 1000)):
    lw = 0.75 if B == 100 else 1.0
    for col, fam in enumerate(('KL', 'FAB')):
        ax = axes_ess[row, col]
        ax.plot(runs_data[B][fam]['ess_history'],         color=COLORS[B]['base'], label=fam,        linewidth=lw)
        ax.plot(runs_data[B][f'{fam}++']['ess_history'],  color=COLORS[B]['pp'],   label=f'{fam}++', linewidth=lw)
        if B == 1000:
            # compute-parity marker: BATCH=1000 * step=100 matches BATCH=100 * step=1000
            ax.axvline(x=100, linestyle='--', color='black', linewidth=0.8)
        ax.set_xlabel('step')
        ax.set_ylabel('ESS')
        ax.set_xlim(0, config['STEPS'][B])
        ax.set_ylim(0, 1)
        ax.set_title(f'{fam}, BATCH={B}')
        ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(HERE / "ESS.png", dpi=300)

# Samples figure: 1x2  (only KL++ and FAB++ at BATCH=1000)
xlim = (-config['PLT_LIM'], config['PLT_LIM'])
ylim = (-config['PLT_LIM'], config['PLT_LIM'])
n = 300
xs = torch.linspace(xlim[0], xlim[1], n)
ys = torch.linspace(ylim[0], ylim[1], n)
X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(device)
with torch.no_grad():
    U_grid = u1(grid).reshape(*X1.shape).cpu().numpy()
levels = torch.linspace(0.0, 30.0, 50).tolist()

fig_samples, axes_samples = plt.subplots(1, 2, figsize=(8, 4))
for col, fam in enumerate(('KL', 'FAB')):
    name = f'{fam}++'
    entry = runs_data[100][name]
    samples_np = entry['samples'].numpy()
    ess = entry['final_ess']

    ax = axes_samples[col]
    ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
    ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
    ax.scatter(samples_np[:, 0], samples_np[:, 1], s=0.1, alpha=0.5, color=COLORS[100]['pp'], zorder=10)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(f'{name}: ESS={ess:.4f}')

plt.tight_layout()
plt.savefig(HERE / "samples.png", dpi=300)

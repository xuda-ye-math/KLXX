# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false

from pathlib import Path
import torch
from zuko.transforms import ComposedTransform
from zflows.flow import NSF
from zflows.potential import Potential, Uniform, Gaussian
from zflows.utils import compute_ESS, importance_weights, resample, langevin

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# boundary of the domain (123 rule)
SIGMA = 3.0 # deviation of Gaussian prior
PLT_LIM = 6.0
NSF_LIM = 9.0

# source: uniform distribution U0
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
        return 0.2 * (term1 + term2)
u1 = U1().to(device)
u1.enable_grad() # enable_grad for Langevin rejuvenation

# training parameters
N_TRAIN: int = 40000  # number of training samples
N_VALID: int = 40000 # number of validation samples
LR: float = 1e-3 # learning rate
BATCH: int = 100 # batch size
EPOCH: int = 500 # number of epochs

# reverse KL loss
def reverse_KL(x: torch.Tensor, target: Potential, G: ComposedTransform):
    y, ladj_inv = G.inv.call_and_ladj(x) # y = G^{-1}(x), ladj_inv = log|det J_{G^{-1}}(x)|
    return (target(y) - ladj_inv).mean()

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
    return 0.5 * (A - A_).abs().mean()

torch.manual_seed(0)
x_pool = u0.samples(N_TRAIN)                       # x ~ u_0 (shared)

# --- Run 1: FAB (surrogate only, squared-IS g-samples) ---
torch.manual_seed(0)
flow_1 = NSF(a=[-NSF_LIM, -NSF_LIM], b=[+NSF_LIM, +NSF_LIM], bins=32, transforms=6, hidden_features=(128, 128)).to(device)
flow_1.zeros()
optimizer = torch.optim.Adam(flow_1.parameters(), lr=LR)
print("=== Run 1: FAB ===")
ess_history_1 = []
for step in range(EPOCH):
    idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
    x = x_pool[idx]

    # squared IS resample -> approximate samples from g ~ mu_1^2 / G^{-1}_# u_0
    with torch.no_grad():
        G_now = flow_1.t()
        y, _ = G_now.inv.call_and_ladj(x)
        w = importance_weights(x, u0, u1, G_now.inv)
        ess_history_1.append(compute_ESS(w).item())
        y_g = resample(y, w * w)

    G = flow_1.t()
    loss = forward_KL(y_g, u0, G)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 10 == 0 or step == 0:
        print(f"[Run 1] step {step+1:>3}/{EPOCH}   loss = {loss.item():.4e}")

# --- Run 2: FAB++ (FAB surrogate + XX on mu_1 samples; two-step IS) ---
torch.manual_seed(0)
flow_2 = NSF(a=[-NSF_LIM, -NSF_LIM], b=[+NSF_LIM, +NSF_LIM], bins=32, transforms=6, hidden_features=(128, 128)).to(device)
flow_2.zeros()
optimizer = torch.optim.Adam(flow_2.parameters(), lr=LR)
print("=== Run 2: FAB++ ===")
ess_history_2 = []
for step in range(EPOCH):
    idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
    x = x_pool[idx]

    # two-step IS: mu_1 samples via IS+Langevin, then g samples by reweighting mu_1
    with torch.no_grad():
        G_now = flow_2.t()
        y, _ = G_now.inv.call_and_ladj(x)
        w = importance_weights(x, u0, u1, G_now.inv)
        ess_history_2.append(compute_ESS(w).item())
        y_mu1 = resample(y, w)
        y_mu1 = langevin(y_mu1, u1, step=1e-2, iters=10)
        x_back, ladj = G_now.call_and_ladj(y_mu1)
        log_w2 = -u1(y_mu1) + u0(x_back) - ladj
        w2 = (log_w2 - log_w2.max()).exp()
        y_g = resample(y_mu1, w2)

    G = flow_2.t()
    loss = forward_KL(y_g, u0, G) + forward_XX(y_mu1, u0, u1, G)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 10 == 0 or step == 0:
        print(f"[Run 2] step {step+1:>3}/{EPOCH}   loss = {loss.item():.4e}")

# Plot pushforward samples for each method
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])

xlim = (-PLT_LIM, PLT_LIM)
ylim = (-PLT_LIM, PLT_LIM)
n = 300
xs = torch.linspace(xlim[0], xlim[1], n)
ys = torch.linspace(ylim[0], ylim[1], n)
X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(device)
with torch.no_grad():
    U_grid = u1(grid).reshape(*X1.shape).cpu().numpy()
levels = torch.linspace(0.0, 30.0, 50).tolist()

x_unif = u0.samples(N_VALID)                        # shared x ~ u_0 for ESS and plotting

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
panels = [('FAB',   flow_1, "#00008B"),
          ('FAB++', flow_2, "#8B0000")]
for col, (name, flow, color) in enumerate(panels):
    with torch.no_grad():
        G = flow.t()
        y_push, _ = G.inv.call_and_ladj(x_unif)          # raw pushforward y = G^{-1}(x), x ~ u_0
        w = importance_weights(x_unif, u0, u1, G.inv)    # weights for (G_# mu_1) / mu_0 = mu_1 / G^{-1}_# u_0
        ess = compute_ESS(w).item()
    print(f"[{name}] ESS = {ess:.4f}")

    # G^{-1}_# u_0 pushforward on target side
    ax = axes[col]
    ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
    ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
    y_np = y_push.cpu().numpy()
    ax.scatter(y_np[:, 0], y_np[:, 1], s=0.1, alpha=0.5, color=color, zorder=10)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(f'{name}: $G^{{-1}}_{{\\#}} \\mu_0$,  ESS = {ess:.4f}')

plt.tight_layout()
plt.savefig(HERE / "samples.png", dpi=300)

# ESS curves over training
fig_ess, ax_ess = plt.subplots(figsize=(8, 4))
ax_ess.plot(ess_history_1, color="#00008B", label='FAB')
ax_ess.plot(ess_history_2, color="#8B0000", label='FAB++')
ax_ess.set_xlabel('step')
ax_ess.set_ylabel('ESS')
ax_ess.set_ylim(0, 1)
ax_ess.legend()
plt.tight_layout()
plt.savefig(HERE / "ESS.png", dpi=300)

plt.show()

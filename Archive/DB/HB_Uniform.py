# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false

"""
Himmelblau: reverse KL, forward DB, and 0.5/0.5 mix.

    KL  : KL(mu_0 || G_# mu_1)  = E_{x ~ mu_0}[U_1(G^{-1}(x)) - log|det J_{G^{-1}}(x)|] + const
    DB  : DB(G_# mu_1 || mu_0)  = E_{y ~ mu_1} | grad U_0(G(y))
                                                - J_G(y)^{-T} (grad U_1(y) + grad log|det J_G(y)|) |
          estimated by self-normalized IS with y ~ mu_0 (uniform proposal for mu_1).
    Mix : 0.5 KL + 0.5 DB
"""

from pathlib import Path
import torch
from zuko.transforms import ComposedTransform
from zflows.flow import NSF
from zflows.potential import Potential, Uniform
from zflows.utils import compute_ESS, importance_weights

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BOUND = 6.0
_a = [-BOUND, -BOUND]
_b = [ BOUND,  BOUND]

u0 = Uniform(a=_a, b=_b).to(device)

class U1(Potential):
    def __init__(self):
        super().__init__()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1.square() + x2 - 11).square()
        term2 = (x1 + x2.square() - 7).square()
        return 0.5 * (term1 + term2)
u1 = U1().to(device)

def reverse_KL(x: torch.Tensor, target: Potential, G: ComposedTransform):
    y, ladj_inv = G.inv.call_and_ladj(x)      # y = G^{-1}(x), ladj_inv = log|det J_{G^{-1}}(x)|
    return (target(y) - ladj_inv).mean()

def forward_DB(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    y = y.detach().requires_grad_(True)
    x, ladj = G.call_and_ladj(y)              # x = G(y), ladj = log|det J_G(y)|

    grad_U1,   = torch.autograd.grad(target(y).sum(), y, create_graph=True)
    grad_ladj, = torch.autograd.grad(ladj.sum(),       y, create_graph=True)

    d = y.shape[-1]
    rows = [torch.autograd.grad(x[:, i].sum(), y, create_graph=True, retain_graph=True)[0]
            for i in range(d)]
    J_G = torch.stack(rows, dim=-2)            # [N, d, d]

    U0_x = source(x)
    if U0_x.requires_grad:
        grad_U0_x, = torch.autograd.grad(U0_x.sum(), x, create_graph=True, allow_unused=True)
        if grad_U0_x is None:
            grad_U0_x = torch.zeros_like(x)
    else:
        grad_U0_x = torch.zeros_like(x)

    v = grad_U1 + grad_ladj
    J_G_inv_T_v = torch.linalg.solve(J_G.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)

    integrand = (grad_U0_x - J_G_inv_T_v).norm(dim=-1)

    with torch.no_grad():
        log_w = -target(y) + source(y)         # log[mu_1(y)/mu_0(y)]
        w = (log_w - log_w.max()).exp()
        w = w / w.sum()

    return (w * integrand).sum()

N_TRAIN, BATCH, LR, EPOCH = 20000, 2000, 1e-3, 400
N_VALID = 20000

torch.manual_seed(1)
x_pool = u0.samples(N_TRAIN)                       # x ~ u_0 (shared)

# --- KL ---
torch.manual_seed(1)
flow_KL = NSF(a=_a, b=_b, bins=8, transforms=6, hidden_features=(128, 128)).to(device)
flow_KL.zeros()
optimizer = torch.optim.Adam(flow_KL.parameters(), lr=LR)
print("=== Training: KL ===")
for step in range(EPOCH):
    idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
    x_batch = x_pool[idx]
    G = flow_KL.t()
    loss = reverse_KL(x_batch, u1, G)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 10 == 0 or step == 0:
        print(f"[KL] step {step+1:>3}/{EPOCH}   loss = {loss.item():.4e}")

# --- DB ---
torch.manual_seed(1)
flow_DB = NSF(a=_a, b=_b, bins=8, transforms=6, hidden_features=(128, 128)).to(device)
flow_DB.zeros()
optimizer = torch.optim.Adam(flow_DB.parameters(), lr=LR)
print("=== Training: DB ===")
for step in range(EPOCH):
    idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
    x_batch = x_pool[idx]
    G = flow_DB.t()
    loss = forward_DB(x_batch, u0, u1, G)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 10 == 0 or step == 0:
        print(f"[DB] step {step+1:>3}/{EPOCH}   loss = {loss.item():.4e}")

# --- Mix ---
torch.manual_seed(0)
flow_Mix = NSF(a=_a, b=_b, bins=8, transforms=6, hidden_features=(128, 128)).to(device)
flow_Mix.zeros()
optimizer = torch.optim.Adam(flow_Mix.parameters(), lr=LR)
print("=== Training: Mix ===")
for step in range(EPOCH):
    idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
    x_batch = x_pool[idx]
    G = flow_Mix.t()
    loss = 0.5 * reverse_KL(x_batch, u1, G) + 0.5 * forward_DB(x_batch, u0, u1, G)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 10 == 0 or step == 0:
        print(f"[Mix] step {step+1:>3}/{EPOCH}   loss = {loss.item():.4e}")

# Plot raw G^{-1}_# u_0 pushforward samples for each method
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])

xlim = ylim = (-BOUND, BOUND)
n = 300
xs = torch.linspace(xlim[0], xlim[1], n)
ys = torch.linspace(ylim[0], ylim[1], n)
X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(device)
with torch.no_grad():
    U_grid = u1(grid).reshape(*X1.shape).cpu().numpy()
levels = torch.linspace(0.0, 30.0, 50).tolist()

x_unif = u0.samples(N_VALID)                        # shared x ~ u_0 for ESS and plotting

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
panels = [('KL',              flow_KL,  'darkblue'),
          ('DB',              flow_DB,  'darkred'),
          ('0.5 KL + 0.5 DB', flow_Mix, 'darkgreen')]
for ax, (name, flow, color) in zip(axes, panels):
    with torch.no_grad():
        G = flow.t()
        y_push, _ = G.inv.call_and_ladj(x_unif)          # raw pushforward y = G^{-1}(x), x ~ u_0
        w = importance_weights(x_unif, u0, u1, G.inv)    # mu_1 / G^{-1}_# u_0
        ess = compute_ESS(w).item()
    print(f"[{name}] ESS = {ess:.4f}")
    ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
    ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
    y_np = y_push.cpu().numpy()
    ax.scatter(y_np[:, 0], y_np[:, 1], s=0.1, alpha=0.5, color=color, zorder=10)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(f'{name},  ESS = {ess:.4f}')

plt.tight_layout()
plt.savefig(HERE / "HB_Uniform.png", dpi=300)
plt.show()

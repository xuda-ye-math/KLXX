import torch
from zflows.potential import Potential, Gaussian_Mixture
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin

# Sparse Gaussian-mixture potential on [-4, 4]^2.
#   Four modes: a dense pair sits near the source center (where mu_0 =
#   N(0, SIGMA^2 I) has essentially all of its mass) and is easy to hit; two
#   isolated modes sit in opposite corners, MANY source-standard-deviations away
#   (with the small SIGMA in parameters.py the source tail never reaches them).
#   Plain forward KL / forward KL+X_mu get no training signal there and collapse;
#   only the methods that draw on QT-discovered samples (X_{hat_mu}, X_mix)
#   recover the isolated modes.
class Sparse(Gaussian_Mixture):
    def __init__(self, sigma: float = 0.08):
        means = [
            [ 0.0,  0.0],    # dense pair, at the source center          (easy)
            [ 0.9,  0.9],    # dense pair, close companion of mode 1     (easy)
            [-2.5,  2.5],    # isolated mode, top-left corner    (collapse-prone)
            [ 2.5, -2.5],    # isolated mode, bottom-right corner (collapse-prone)
        ]
        K = len(means)  # 4
        weights = [1.0] * K
        variance = [[sigma**2, sigma**2]] * K
        super().__init__(weights=weights, mean=means, variance=variance)
        self.sigma = sigma

# forward KL
def loss_KL(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    return z.mean()

# X functional
def loss_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return (z - z[perm]).abs().mean() # autograd graph of z

# forward KL + X functional
def loss_KL_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform, lambda_: float = 1.0):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return z.mean() + lambda_ * (z - z[perm]).abs().mean()

# Quench and Temper (QT) algorithm for mode discovery
#   requires target.enable_grad() and target.enable_eval() to be called beforehand
def quench_and_temper(x: torch.Tensor, target: Potential, sigma: float, opt_step, opt_iters, mc_step, mc_iters):
    x = x + sigma * torch.randn_like(x)                                     # diffusion (melt):     scatter samples across R^d
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)       # optimization (quench): drive each sample to a mode center of target
    x = langevin(x, target, step=mc_step, iters=mc_iters)                   # rejuvenation (temper): spread samples around each mode
    return x

# coverage metric (Naeem et al. 2020): fraction of reference points x_i whose k-NN
# ball (within x) contains at least one candidate y_j
def coverage(y: torch.Tensor, x: torch.Tensor, k: int = 5) -> float:
    dxx = torch.cdist(x, x)
    dxx.fill_diagonal_(float('inf'))
    nnd_k = dxx.topk(k, dim=1, largest=False).values[:, -1] # [P]: distance to k-th nearest neighbor in x
    dxy = torch.cdist(x, y)                                 # [P, N]
    return (dxy < nnd_k.unsqueeze(1)).any(dim=1).float().mean().item()


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    HERE = Path(__file__).resolve().parent
    target = Sparse(sigma=0.1)

    LIM = 4.0
    n = 400
    xs = torch.linspace(-LIM, LIM, n)
    ys = torch.linspace(-LIM, LIM, n)
    X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    with torch.no_grad():
        U_grid = target(grid).reshape(*X1.shape).numpy()

    cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])
    levels = torch.linspace(-2.0, 20.0, 50).tolist()

    torch.manual_seed(0)
    samples = target.samples(5000).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
    ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
    ax.scatter(samples[:, 0], samples[:, 1], s=0.5, alpha=0.5, color="#8B0000", zorder=10)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    plt.tight_layout()
    plt.savefig(HERE / "core.png", dpi=300)
    plt.close(fig)
    print(f"Saved {HERE / 'core.png'}")

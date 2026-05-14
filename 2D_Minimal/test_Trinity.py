from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from zflows.potential import Gaussian

from core import Himmelblau, Trinity, coverage

HERE = Path(__file__).resolve().parent

def test_Trinity():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Himmelblau target: 4 known modes at the corners of (~3, ~3)
    target = Himmelblau().to(device)
    target.enable_grad() # required by lbfgs and langevin
    target.enable_eval() # required by lbfgs(armijo=True)

    modes = torch.tensor([ # known Himmelblau mode centers
        [ 3.000000,  2.000000],
        [-2.805118,  3.131312],
        [-3.779310, -3.283186],
        [ 3.584428, -1.848126],
    ], device=device)

    # start from a unimodal Gaussian at origin: without diffusion, the outer modes would be unreachable
    source = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
    x = source.samples(2048)

    x_out = Trinity(x, target, sigma=2.0, opt_step=0.5, opt_iters=200, mc_step=1e-2, mc_iters=100)
    assert torch.isfinite(x_out).all(), "Trinity produced non-finite samples"

    # each output should land near one of the 4 modes
    d = torch.cdist(x_out, modes)        # [N, 4]
    nearest_dist = d.min(dim=1).values   # [N]
    median_dist = nearest_dist.median().item()
    print(f"median distance to nearest Himmelblau mode = {median_dist:.4f}")
    assert median_dist < 0.5, f"Trinity samples failed to reach modes (median dist = {median_dist:.4f})"

    # all 4 modes should be discovered
    modes_hit = d.argmin(dim=1).unique().numel()
    print(f"modes covered: {modes_hit}/4")
    assert modes_hit == 4, f"Trinity covered only {modes_hit}/4 modes"

    # quantified coverage: ground-truth reference = 4 tight Gaussians at the known modes
    P = 512
    ref = modes[torch.randint(0, 4, (P,), device=device)] + 0.1 * torch.randn(P, 2, device=device)
    cov = coverage(x_out, ref, k=5)
    print(f"coverage_k=5 of Himmelblau reference by Trinity output = {cov:.4f}")
    assert cov > 0.8, f"coverage too low: {cov:.4f}"
    print("Trinity sanity: OK")

    # --- Plotting ---
    cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])
    PLT_LIM = 6.0
    xlim = (-PLT_LIM, PLT_LIM)
    ylim = (-PLT_LIM, PLT_LIM)
    n = 300
    xs = torch.linspace(xlim[0], xlim[1], n)
    ys = torch.linspace(ylim[0], ylim[1], n)
    X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(device)
    with torch.no_grad():
        U_grid = target(grid).reshape(*X1.shape).cpu().numpy()
    levels = torch.linspace(0.0, 30.0, 50).tolist()

    x_np     = x.cpu().numpy()
    x_out_np = x_out.cpu().numpy()

    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
    ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
    ax.scatter(x_np[:, 0],     x_np[:, 1],     s=0.5, alpha=0.5, color="#00008B", zorder=10, label='Gaussian samples')
    ax.scatter(x_out_np[:, 0], x_out_np[:, 1], s=0.5, alpha=0.5, color="#8B0000", zorder=11, label='Trinity samples')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Himmelblau: source vs Trinity')
    ax.legend(loc='upper right', markerscale=8)
    plt.tight_layout()
    plt.savefig(HERE / "Trinity.png", dpi=300)

if __name__ == '__main__':
    test_Trinity()

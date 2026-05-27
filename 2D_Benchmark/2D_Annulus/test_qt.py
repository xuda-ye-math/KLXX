from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from zflows.potential import Gaussian

from core import Annulus, quench_and_temper
from parameters import SIGMA, PLT_LIM, BATCH

HERE = Path(__file__).resolve().parent

cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])

def test_qt():
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Annulus target: minima at r = 0 and on the ring r = 2
    target = Annulus().to(device)
    target.enable_grad() # required by lbfgs and langevin
    target.enable_eval() # required by lbfgs(armijo=True)

    # start from a tight Gaussian at origin
    source = Gaussian(mean=[0.0, 0.0], variance=[SIGMA**2, SIGMA**2]).to(device)
    x = source.samples(BATCH)

    x_out = quench_and_temper(x, target, sigma=2.0, opt_step=0.5, opt_iters=200, mc_step=2e-3, mc_iters=1000)
    assert torch.isfinite(x_out).all(), "QT produced non-finite samples"
    print("QT sanity: OK")

    # --- Plotting (style aligned with plot_results.py) ---
    xlim = (-PLT_LIM, PLT_LIM)
    ylim = (-PLT_LIM, PLT_LIM)
    n = 300
    xs = torch.linspace(xlim[0], xlim[1], n)
    ys = torch.linspace(ylim[0], ylim[1], n)
    X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(device)
    with torch.no_grad():
        U_grid = target(grid).reshape(*X1.shape).cpu().numpy()
    # Narrow level range around the annulus features (min 20, barrier ~44, ring 20);
    # extend='max' clips higher values outside the ring to the top color.
    levels = torch.linspace(20.0, 60.0, 50).tolist()

    x_np     = x.cpu().numpy()
    x_out_np = x_out.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
    ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
    ax.scatter(x_np[:, 0],     x_np[:, 1],     s=0.5, alpha=0.3, color='gray',     zorder=5,  label='source')
    ax.scatter(x_out_np[:, 0], x_out_np[:, 1], s=0.5, alpha=0.6, color="#8B0000", zorder=10, label='QT')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.legend(loc='upper right', markerscale=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / "qt.png", dpi=400)
    plt.close(fig)

if __name__ == '__main__':
    test_qt()

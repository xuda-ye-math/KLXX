from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from core import Himmelblau, coverage

HERE = Path(__file__).resolve().parent

METHODS = ('KL', 'KL+X_mu', 'KL+X_mu+X_hat_mu')
METHOD_LABEL = {
    'KL':                'forward KL',
    'KL+X_mu':           r'forward KL+$\mathrm{X}_\mu$',
    'KL+X_mu+X_hat_mu':  r'forward KL+$\mathrm{X}_\mu$+$\mathrm{X}_{\hat\mu}$',
}
METHOD_COLOR = {
    'KL':                "#00008B",   # dark blue
    'KL+X_mu':           "#006400",   # dark green
    'KL+X_mu+X_hat_mu':  "#8B0000",   # dark red
}

cmap = LinearSegmentedColormap.from_list('light_yellow_red', ["#fffefa", "#ffe5e5"])

u1 = Himmelblau()

def plot_run(tag: str):
    data_path = HERE / f'data_{tag}.pth'
    if not data_path.exists():
        raise SystemExit(f"{data_path} not found. Run train_{tag}.py first.")
    data = torch.load(data_path, weights_only=False)
    runs_data = data['runs']
    config = data['config']
    y_hat_mu = data.get('y_hat_mu')

    # coverage of each method's pushforward samples against the QT-generated y_hat_mu
    if y_hat_mu is not None:
        for m in METHODS:
            cov = coverage(runs_data[m]['samples'], y_hat_mu, k=5)
            runs_data[m]['coverage'] = cov
            print(f"[tag={tag}, {m:<16}] coverage_k=5 vs y_hat_mu = {cov:.4f}")

    # ESS figure
    fig_ess, ax_ess = plt.subplots(1, 1, figsize=(5, 4))
    for m in METHODS:
        ax_ess.plot(runs_data[m]['ess_history'], color=METHOD_COLOR[m], label=METHOD_LABEL[m], linewidth=1.0)
    ax_ess.set_xlabel('step')
    ax_ess.set_ylabel('ESS')
    ax_ess.set_xlim(0, config['STEPS'])
    ax_ess.set_ylim(0, 1)
    ax_ess.legend(loc='lower right')
    plt.tight_layout()
    ess_path = HERE / f"ESS_{tag}.png"
    plt.savefig(ess_path, dpi=300)
    plt.close(fig_ess)
    print(f"Saved {ess_path}")

    # Samples figure: 1x3 (cols = KL, KL+X_mu, KL+X_mu+X_hat_mu)
    xlim = (-config['PLT_LIM'], config['PLT_LIM'])
    ylim = (-config['PLT_LIM'], config['PLT_LIM'])
    n = 300
    xs = torch.linspace(xlim[0], xlim[1], n)
    ys = torch.linspace(ylim[0], ylim[1], n)
    X1, X2 = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    with torch.no_grad():
        U_grid = u1(grid).reshape(*X1.shape).numpy()
    levels = torch.linspace(0.0, 30.0, 50).tolist()

    fig_samples, axes_samples = plt.subplots(1, 3, figsize=(8, 3))
    for col, m in enumerate(METHODS):
        entry = runs_data[m]
        samples_np = entry['samples'].numpy()
        ess = entry['final_ess']

        ax = axes_samples[col]
        ax.contourf(X1.numpy(), X2.numpy(), U_grid, levels=levels, cmap=cmap.reversed(), extend='max')
        ax.contour (X1.numpy(), X2.numpy(), U_grid, levels=levels, colors='gray', linewidths=0.2, alpha=0.2)
        ax.scatter(samples_np[:, 0], samples_np[:, 1], s=0.05, alpha=0.5, color=METHOD_COLOR[m], zorder=10)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
        ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
        title = METHOD_LABEL[m] + f'\nESS = $\\mathbf{{{ess:.2f}}}$'
        if 'coverage' in entry:
            title += f', Cov = $\\mathbf{{{entry["coverage"]:.2f}}}$'
        ax.set_title(title)

    plt.tight_layout()
    samples_path = HERE / f"samples_{tag}.png"
    plt.savefig(samples_path, dpi=400)
    plt.close(fig_samples)
    print(f"Saved {samples_path}")

plot_run('1')
plot_run('1.5')

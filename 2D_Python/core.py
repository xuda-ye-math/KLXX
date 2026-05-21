"""Python-logo potential built by tracing the actual python.jpg centerline.

We skeletonize the binary mask of python.jpg (so the centerline of every
stroke is a 1-pixel-wide curve), then place K_STROKE Gaussian centers evenly
along that curve via k-means. Stroke width is the parametric SIGMA_STROKE
(not fit from data), and two larger Gaussians mark the eyes.

Validation render overlays the centers on python.jpg to confirm they trace
the logo exactly.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
from zflows.potential import Potential, Gaussian_Mixture
from zflows.flow import ComposedTransform
from zflows.utils import lbfgs, langevin

HERE = Path(__file__).resolve().parent
IMG_PATH = HERE / "python.jpg"


def _extract_skeleton_points(img_path: Path, lim: float, margin: float,
                             threshold: int = 200) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Skeletonize the dark strokes of img_path and return their (x, y) coords in the domain.
    Returns (skel_xy, raw_arr, inner_x, inner_y) where raw_arr is the original L-mode image."""
    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    H, W = arr.shape
    binary = (255 - arr) >= threshold                            # foreground mask
    skel = skeletonize(binary)                                   # 1-pixel-wide centerline

    inner   = lim - margin
    scale   = inner / max(H, W)
    inner_x = scale * W
    inner_y = scale * H
    ys, xs = np.where(skel)                                      # row, col indices of skeleton pixels
    px = -inner_x + (xs / max(W - 1, 1)) * 2 * inner_x           # column -> +x
    py = +inner_y - (ys / max(H - 1, 1)) * 2 * inner_y           # row    -> -y (flip)
    return np.column_stack([px, py]), arr, inner_x, inner_y


class Python(Gaussian_Mixture):
    def __init__(self,
                 LIM:          float = 8.0,
                 MARGIN:       float = 0.5,
                 K_STROKE:     int   = 1024,
                 SIGMA_STROKE: float = 0.04,
                 SIGMA_EYE:    float = 0.30,
                 EYE_TOP:      tuple = (-1.50,  4.80),
                 EYE_BOT:      tuple = ( 1.50, -4.80),
                 EYE_WEIGHT:   float = 0.05,
                 SEED:         int   = 0):
        skel_pts, _, _, _ = _extract_skeleton_points(IMG_PATH, LIM, MARGIN)
        # K-means on the skeleton to get K_STROKE roughly-equidistant centers on the curve.
        # This is curve discretization (each cluster center sits on the 1-pixel centerline),
        # not statistical fitting -- we set the variance below from SIGMA_STROKE, not from data.
        km = KMeans(n_clusters=K_STROKE, init="k-means++", n_init=4, random_state=SEED).fit(skel_pts)
        stroke_means = km.cluster_centers_.astype(np.float64)

        eye_means = np.array([EYE_TOP, EYE_BOT], dtype=np.float64)
        means = np.concatenate([stroke_means, eye_means], axis=0)

        n_stroke   = len(stroke_means)
        stroke_var = np.full((n_stroke, 2), SIGMA_STROKE ** 2, dtype=np.float64)
        eye_var    = np.full((2, 2),        SIGMA_EYE    ** 2, dtype=np.float64)
        variance   = np.concatenate([stroke_var, eye_var], axis=0)

        stroke_w = np.full(n_stroke, (1.0 - 2 * EYE_WEIGHT) / n_stroke, dtype=np.float64)
        eye_w    = np.full(2, EYE_WEIGHT, dtype=np.float64)
        weights  = np.concatenate([stroke_w, eye_w])
        weights  = weights / weights.sum()

        super().__init__(
            weights=torch.from_numpy(weights),
            mean=torch.from_numpy(means),
            variance=torch.from_numpy(variance),
            device="cpu",
        )
        self.K   = means.shape[0]
        self.LIM = LIM


# forward KL
def loss_KL(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    return z.mean()

# X functional
def loss_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return (z - z[perm]).abs().mean()

# forward KL + X functional
def loss_KL_X(y: torch.Tensor, source: Potential, target: Potential, G: ComposedTransform, lambda_: float = 1.0):
    N = y.shape[0]
    x, ladj = G.call_and_ladj(y)
    z = source(x) - target(y) - ladj
    perm = torch.randperm(N, device=y.device)
    return z.mean() + lambda_ * (z - z[perm]).abs().mean()

# Quench and Temper for mode discovery (requires target.enable_grad / enable_eval)
def quench_and_temper(x: torch.Tensor, target: Potential, sigma: float, opt_step, opt_iters, mc_step, mc_iters):
    x = x + sigma * torch.randn_like(x)
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True)
    x = langevin(x, target, step=mc_step, iters=mc_iters)
    return x

# coverage metric (Naeem et al. 2020)
def coverage(y: torch.Tensor, x: torch.Tensor, k: int = 5) -> float:
    dxx = torch.cdist(x, x)
    dxx.fill_diagonal_(float('inf'))
    nnd_k = dxx.topk(k, dim=1, largest=False).values[:, -1]
    dxy = torch.cdist(x, y)
    return (dxy < nnd_k.unsqueeze(1)).any(dim=1).float().mean().item()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    target = Python()
    LIM = target.LIM
    print(f"Python: K={target.K} components on [-{LIM}, +{LIM}]^2")

    # show centers overlaid on the source image to verify exact placement
    _, raw_arr, inner_x, inner_y = _extract_skeleton_points(IMG_PATH, LIM, 0.5)
    means_np = target.mean.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(raw_arr, cmap='gray', extent=(-inner_x, +inner_x, -inner_y, +inner_y))
    ax.scatter(means_np[:, 0], means_np[:, 1], s=1.5, color="#FF3030", alpha=0.9, zorder=10)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(f'Python: {target.K} Gaussian centers on python.jpg')
    plt.tight_layout()
    plt.savefig(HERE / "core.png", dpi=200)
    plt.close(fig)
    print(f"Saved {HERE / 'core.png'}")

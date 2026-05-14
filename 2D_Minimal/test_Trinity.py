import torch
from zflows.potential import Gaussian

from core import Himmelblau, Trinity, coverage

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
    assert cov > 0.9, f"coverage too low: {cov:.4f}"
    print("Trinity sanity: OK")

if __name__ == '__main__':
    test_Trinity()

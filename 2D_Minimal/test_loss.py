import torch
from zflows.potential import Gaussian
from zflows.flow import NSF

from core import loss_KL, loss_X, loss_KL_X

def test_loss():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2D Gaussians with different means/variances so z = U_0(G(y)) - U_1(y) - ladj is non-trivial
    source = Gaussian(mean=[0.0, 0.0], variance=[4.0, 4.0]).to(device)
    target = Gaussian(mean=[1.0, 1.0], variance=[1.0, 1.0]).to(device)

    flow = NSF(a=[-6.0, -6.0], b=[6.0, 6.0], bins=16, transforms=4, hidden_features=(64, 64)).to(device)
    flow.zeros() # initialize G as identity
    G = flow.t()

    y = target.samples(1024)

    L_KL  = loss_KL(y, source, target, G)
    L_X   = loss_X(y, source, target, G)
    L_KLX = loss_KL_X(y, source, target, G, lambda_=1.0)
    print(f"loss_KL    = {L_KL.item():.4f}")
    print(f"loss_X     = {L_X.item():.4f}")
    print(f"loss_KL_X  = {L_KLX.item():.4f}  (should be ~ loss_KL + loss_X up to permutation noise)")

    assert torch.isfinite(L_KL) and torch.isfinite(L_X) and torch.isfinite(L_KLX), "loss values must be finite"
    assert L_X.item() >= 0, "X functional should be non-negative"

    # gradient flows through the flow parameters
    L_KLX.backward()
    grads = [p.grad for p in flow.parameters() if p.grad is not None]
    assert grads, "no gradient flowed to flow parameters"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient encountered"
    print("gradient sanity: OK")

if __name__ == '__main__':
    test_loss()

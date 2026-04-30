import torch
from functools import partial
from zuko.transforms import MonotonicRQSTransform, ComposedTransform
from zuko.flows.autoregressive import MAF
from potential import Potential

class NSF(MAF):
    """Neural Spline Flow on via MAF with MonotonicRQS."""

    def __init__(self, bound: float, bins: int, hidden_features: tuple[int, ...], transforms: int):
        super().__init__(
            features=2,
            context=0,
            univariate=partial(MonotonicRQSTransform, bound=bound),
            shapes=[(bins,), (bins,), (bins - 1,)],
            hidden_features=hidden_features,
            transforms=transforms,
            activation=torch.nn.SiLU,
        )
        self.bound = bound

    def t(self): # return ComposedTransform
        return self().transform  

def gradient_ratio_v(source: Potential, target: Potential, x: torch.Tensor, flow_t: ComposedTransform, epsilon: float = 1e-3):
    """Compute v^T ∇_y R_F(x) via autograd, where
        R_F(x) = U_0(F(x)) - U(x) - log|det J_F(x)|,
    and v is drawn uniformly on S^1 per sample.
    Input:
        source: source potential U_0
        target: target potential U
        x [N, 2]: batch input positions
        flow_t:   ComposedTransform, forward map F
    Output:
        grv [N]:  scalar v^T ∇_y R_F(x)
    """
    x = x.detach()
    x.requires_grad_(True)

    y, ladj = flow_t.call_and_ladj(x)  # [N, 2], [N]

    target_x    = target(x)  # [N]
    grad_target = torch.autograd.grad(target_x.sum(), x, create_graph=True)[0]  # [N, 2]
    grad_ladj   = torch.autograd.grad(ladj.sum(),     x, create_graph=True)[0]  # [N, 2]

    v = torch.randn_like(x)
    v = v / torch.norm(v, dim=-1, keepdim=True)  # [N, 2]

    # Central finite differences at y = F(x) along direction v
    y_1, y_2 = y + epsilon * v, y - epsilon * v

    # v^T ∇source(y) ≈ [source(y_1) - source(y_2)] / (2ε)
    dv_source = (source(y_1) - source(y_2)) / (2 * epsilon)  # [N]

    # J_F^{-1} v ≈ [F^{-1}(y_1) - F^{-1}(y_2)] / (2ε)
    x_1, x_2 = flow_t.inv(y_1), flow_t.inv(y_2)
    assert x_1 is not None and x_2 is not None
    dv_Jinv = (x_1 - x_2) / (2 * epsilon)  # [N, 2]

    return dv_source - (dv_Jinv * (grad_target + grad_ladj)).sum(dim=-1)  # [N]

def detailed_balance_loss(source: Potential, source_frozen: Potential, target: Potential, x: torch.Tensor, flow_t: ComposedTransform, epsilon: float = 1e-3):
    """DB loss E_{x~bar_mu_0}[ mu(x)/bar_mu_0(x) * |v^T ∇_y R_F(x)| ].
    Input:
        source:        adaptive prior potential  U_0
        source_frozen: fixed importance sampling potential bar U_0 (~ bar mu_0)
        target:        target potential U
        x [N, 2]:      samples drawn from bar mu_0
        flow_t:        ComposedTransform, forward map F
    Output:
        loss: scalar
    """
    grv     = gradient_ratio_v(source, target, x, flow_t, epsilon)  # [N]
    weights = torch.exp(source_frozen(x) - target(x))               # mu(x) / bar_mu_0(x)  [N]
    return (grv.abs() * weights).mean()

def compute_ESS(log_weights: torch.Tensor) -> float:
    """Compute the ESS from log-weights."""
    weights            = torch.exp(log_weights - torch.max(log_weights))   # stabilize
    normalized_weights = weights / weights.sum()                           # normalize
    N = log_weights.shape[0]
    return 1.0 / (N * (normalized_weights ** 2).sum().item())              # ESS in [0, 1]

def resample(x: torch.Tensor, log_weights: torch.Tensor) -> torch.Tensor:
    """Resample x according to log_weights using multinomial resampling."""
    weights            = torch.exp(log_weights - torch.max(log_weights))   # stabilize
    normalized_weights = weights / weights.sum()                           # normalize
    indices = torch.multinomial(normalized_weights, num_samples=x.shape[0], replacement=True)
    return x[indices]
    
if __name__ == "__main__":
    # define the potential function
    from potential import Uniform, Himmelblau
    target = Himmelblau() # get potential
    BOUND = target.BOUND

    # generate test samples
    uniform = Uniform(BOUND=BOUND) # uniform distribution
    uniform_frozen = Uniform(BOUND=BOUND) # another instance of uniform distribution
    N = 10 # number of samples
    x = uniform.samples(N) # generate uniform samples

    # initialize the flow
    flow = NSF(bound=BOUND, bins=8, hidden_features=(64, 64), transforms=4)
    flow_t = flow.t() # define the NSF

    # compute gradient_ratio_v
    grv = gradient_ratio_v(source=uniform, target=target, x=x, flow_t=flow_t) 
    print(grv)

    # compute detailed balance loss
    loss = detailed_balance_loss(source_frozen=uniform_frozen, source=uniform, target=target, x=x, flow_t=flow_t)
    print(loss)

    # one AdamW step on flow parameters
    optim = torch.optim.AdamW(flow.parameters(), lr=1e-3)
    optim.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=float('inf'))
    print(f"grad norm: {grad_norm.item():.4e}")
    optim.step()

    # re-evaluate after step (fresh flow_t since params changed)
    flow_t = flow.t()
    loss_after = detailed_balance_loss(source_frozen=uniform_frozen, source=uniform, target=target, x=x, flow_t=flow_t)
    print(f"loss before: {loss.item():.6e}, after: {loss_after.item():.6e}")

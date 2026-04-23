import torch
from utilities import *
from parameters import *

# load trained flow
flow.load_state_dict(torch.load(f"flow_{name}.pt", weights_only=True))
flow.eval()

with torch.no_grad():
    flow_t = flow.t()

    # draw y ~ mu_0 (source prior)
    y = uniform.samples(N=N_VALID)  # [N, 2]

    # x = F^{-1}(y)
    x = flow_t.inv(y)
    assert x is not None

    # log-likelihood ratio R_F(x) = U_0(y) - U(x) - log|det J_F(x)|
    _, ladj     = flow_t.call_and_ladj(x)
    log_weights = uniform(y) - target(x) - ladj  # [N]

    # effective sample size and normalized weights
    ess     = compute_ESS(log_weights)
    weights = torch.exp(log_weights - log_weights.max())
    weights = weights / weights.sum()

torch.save({
    'samples': x,
    'weights': weights,
    'ess':     ess,
}, f"data_{name}.pt")
print(f"Saved data_{name}.pt  (N={N_VALID}, ESS={ess:.1%})")

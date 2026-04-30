import torch
from utilities import *
from parameters import *

# load trained flow
flow.load_state_dict(torch.load(f"flow_{name}.pt", weights_only=True))
flow.eval()

with torch.no_grad():
    flow_t = flow.t()

    # direct importance sampling: y ~ ν, push forward x = F(y)
    y = gmm.samples(N=N_VALID)                   # [N, 2]
    x, ladj = flow_t.call_and_ladj(y)            # [N, 2], [N]

    # log w = log ν(y) − log F_# ν(x) + log π(x)  =  gmm(y) + ladj − target(x)
    log_weights = gmm(y) + ladj - target(x)      # [N]

    ess     = compute_ESS(log_weights)
    weights = torch.exp(log_weights - log_weights.max())
    weights = weights / weights.sum()

torch.save({
    'samples': x,
    'weights': weights,
    'ess':     ess,
}, f"data_{name}.pt")
print(f"Saved data_{name}.pt  (N={N_VALID}, ESS={ess:.1%})")

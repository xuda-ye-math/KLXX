# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false

from pathlib import Path
import torch
from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import compute_ESS, importance_weights, resample, langevin

from core import Python, loss_KL, loss_X, quench_and_temper
from parameters import SIGMA, PLT_LIM, NSF_LIM, BINS, TRANSFORMS, HIDDEN_FEATURES, N_TRAIN, N_VALID, BATCH, STEPS, LR

import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1") # cleaner logs

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# source: Gaussian U0
u0 = Gaussian(mean=[0.0]*2, variance=[SIGMA**2]*2).to(device)

# target: Python potential U1
u1 = Python().to(device)
u1.enable_grad() # enable_grad for Langevin rejuvenation
u1.enable_eval() # enable_eval for QT's lbfgs(armijo=True)

METHODS = (
    'KL',
    'KL+X_mu',
    'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix',
)

def new_flow():
    flow = NSF(a=[-NSF_LIM, -NSF_LIM], b=[+NSF_LIM, +NSF_LIM], bins=BINS, transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES).to(device)
    flow.zeros()
    return flow

torch.manual_seed(0)
x_pool = u0.samples(N_TRAIN)                       # x ~ u_0 (shared across all runs)

def train(method: str, y_hat_mu: torch.Tensor):
    terms = set(method.split('+'))
    use_x_mix    = 'X_mix' in terms        # X functional weighted by (hat_mu + nu) / 2
    use_x_mu     = 'X_mu' in terms
    use_x_hat_mu = 'X_hat_mu' in terms
    needs_hat_mu = use_x_hat_mu or use_x_mix

    torch.manual_seed(0)
    flow = new_flow()
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    ess_history = []
    half = BATCH // 2
    for step in range(STEPS):
        idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
        x = x_pool[idx]

        with torch.no_grad():
            G_now = flow.t()
            y, _ = G_now.inv.call_and_ladj(x)
            w = importance_weights(x, u0, u1, G_now.inv)
            ess_history.append(compute_ESS(w).item())

            # one-step IS -> approximate mu_1 samples
            y_mu = resample(y, w)
            y_mu = langevin(y_mu, u1, step=2e-3, iters=50)

            # rejuvenate hat_mu samples across iterations (used by X_hat_mu and X_mix)
            if needs_hat_mu:
                y_hat_mu = langevin(y_hat_mu, u1, step=2e-3, iters=50)

        G = flow.t()
        loss = loss_KL(y_mu, u0, u1, G)
        if use_x_mix:
            # X functional under weight (hat_mu + nu) / 2: half from y_hat_mu, half from y (pushforward)
            y_mix = torch.cat([y_hat_mu[:half], y[:half]], dim=0)
            loss = loss + loss_X(y_mix, u0, u1, G)
        if use_x_mu:
            loss = loss + loss_X(y_mu, u0, u1, G)
        if use_x_hat_mu:
            loss = loss + loss_X(y_hat_mu, u0, u1, G)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[{method:<24}] step {step+1:>4}/{STEPS}   loss = {loss.item():.4e}")

    print(f"[{method:<24}] last training ESS = {ess_history[-1]:.4f}")
    return flow, ess_history

DATA_PATH = HERE / 'data.pth'

if not DATA_PATH.exists():
    # one-shot QT to build hat_mu samples used by X_{hat_mu}; same set is reused every step
    torch.manual_seed(1)
    x_qt = u0.samples(BATCH)
    y_hat_mu = quench_and_temper(x_qt, u1, sigma=2.0, opt_step=0.5, opt_iters=200, mc_step=2e-3, mc_iters=1000)
    print(f"Generated {y_hat_mu.shape[0]} hat_mu samples via QT")

    results = {}
    for m in METHODS:
        print(f"=== {m} ===")
        results[m] = train(m, y_hat_mu)

    # Compute final samples and ESS for each run
    x_unif = u0.samples(N_VALID)
    runs_data = {}
    for m in METHODS:
        flow, ess_history = results[m]
        with torch.no_grad():
            G = flow.t()
            y_push, _ = G.inv.call_and_ladj(x_unif)
            w = importance_weights(x_unif, u0, u1, G.inv)
            final_ess = compute_ESS(w).item()
        runs_data[m] = {
            'state_dict': {k: v.cpu() for k, v in flow.state_dict().items()},
            'ess_history': ess_history,
            'samples': y_push.cpu(),
            'final_ess': final_ess,
        }
        print(f"[{m:<24}] final ESS = {final_ess:.4f}")

    torch.save({
        'config': {
            'STEPS': STEPS,
            'LR': LR,
            'BATCH': BATCH,
            'N_TRAIN': N_TRAIN,
            'N_VALID': N_VALID,
            'SIGMA': SIGMA,
            'NSF_LIM': NSF_LIM,
            'PLT_LIM': PLT_LIM,
            'methods': list(METHODS),
        },
        'x_unif': x_unif.cpu(),
        'y_hat_mu': y_hat_mu.cpu(),
        'runs': runs_data,
    }, DATA_PATH)
    print(f"Saved all data to {DATA_PATH}")
else:
    print(f"Found {DATA_PATH}; skipping training")

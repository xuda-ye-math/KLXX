# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false

import time
from pathlib import Path
import torch
from zflows.flow import NSF
from zflows.potential import Gaussian
from zflows.utils import compute_ESS, importance_weights, resample, langevin

from core import Phi4, magnetization, loss_KL, loss_X, quench_and_temper
from parameters import D, SIGMA, NSF_LIM, BINS, TRANSFORMS, HIDDEN_FEATURES, N_TRAIN, N_VALID, BATCH, STEPS, LR

import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1") # cleaner logs

HERE = Path(__file__).resolve().parent
STATUS = HERE / 'train_status.log'

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, 'a') as f:
        f.write(line + '\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# source: Gaussian U0
u0 = Gaussian(mean=[0.0]*D, variance=[SIGMA**2]*D).to(device)

# target: phi^4 lattice action U1
u1 = Phi4().to(device)
u1.enable_grad() # enable_grad for Langevin rejuvenation
u1.enable_eval() # enable_eval for QT's lbfgs(armijo=True)

METHODS = (
    'KL',
    'KL+X_mu',
    'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix',
)

def new_flow():
    flow = NSF(a=[-NSF_LIM]*D, b=[+NSF_LIM]*D, bins=BINS, transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES).to(device)
    flow.zeros()
    return flow

torch.manual_seed(0)
x_pool = u0.samples(N_TRAIN)                       # x ~ u_0 (shared across all runs)

def train(method: str, y_hat_mu: torch.Tensor, seed: int = 0, steps: int = STEPS):
    terms = set(method.split('+'))
    use_x_mix    = 'X_mix' in terms          # X functional weighted by (hat_mu + bar_nu) / 2
    use_x_mu     = 'X_mu' in terms
    use_x_hat_mu = 'X_hat_mu' in terms
    needs_hat_mu = use_x_hat_mu or use_x_mix

    torch.manual_seed(seed)
    flow = new_flow()
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    ess_history = []
    half = BATCH // 2
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randperm(N_TRAIN, device=device)[:BATCH]
        x = x_pool[idx]

        with torch.no_grad():
            G_now = flow.t()
            y, _ = G_now.inv.call_and_ladj(x)
            w = importance_weights(x, u0, u1, G_now.inv)
            if not torch.isfinite(w).all() or w.sum() <= 0:   # stability guard
                log(f"[{method:<18}] step {step+1}: non-finite weights, batch skipped")
                ess_history.append(float('nan'))
                continue
            ess_history.append(compute_ESS(w).item())

            # one-step IS -> approximate mu samples
            y_mu = resample(y, w)
            y_mu = langevin(y_mu, u1, step=2e-3, iters=50)

            # rejuvenate hat_mu samples across iterations (used by X_hat_mu and X_mix)
            if needs_hat_mu:
                y_hat_mu = langevin(y_hat_mu, u1, step=2e-3, iters=50)

        G = flow.t()
        loss = loss_KL(y_mu, u0, u1, G)
        if use_x_mix:
            # X functional under weight (hat_mu + bar_nu) / 2: half from y_hat_mu, half from y (pushforward)
            perm_h = torch.randperm(y_hat_mu.shape[0], device=device)[:half]
            y_mix = torch.cat([y_hat_mu[perm_h], y[:half]], dim=0)
            loss = loss + loss_X(y_mix, u0, u1, G)
        if use_x_mu:
            loss = loss + loss_X(y_mu, u0, u1, G)
        if use_x_hat_mu:
            perm_h = torch.randperm(y_hat_mu.shape[0], device=device)[:BATCH]
            loss = loss + loss_X(y_hat_mu[perm_h], u0, u1, G)

        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        if not (torch.isfinite(loss) and torch.isfinite(gnorm)):  # stability guard
            log(f"[{method:<18}] step {step+1}: non-finite loss/grad, step skipped")
            optimizer.zero_grad()
            continue
        optimizer.step()
        if (step + 1) % 100 == 0 or step == 0:
            dt = (time.perf_counter() - t0) / (step + 1) * 1000
            log(f"[{method:<18}] step {step+1:>4}/{steps}  loss={loss.item():.4e}  "
                f"ESS={ess_history[-1]:.3f}  {dt:.0f} ms/step")

    log(f"[{method:<18}] last training ESS = {ess_history[-1]:.4f}")
    return flow, ess_history

DATA_PATH = HERE / 'data.pth'

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', type=str, default=','.join(METHODS),
                    help='comma list; staged runs append to the data file')
    ap.add_argument('--seed', type=int, default=0, help='training seed (flow init, batches, AIS)')
    ap.add_argument('--qt-skew', type=float, default=None,
                    help='subsample frozen QT set to this m>0 fraction (robustness test)')
    ap.add_argument('--steps', type=int, default=STEPS)
    args = ap.parse_args()
    requested = [m.strip() for m in args.methods.split(',') if m.strip()]

    tag = ''
    if args.seed != 0:
        tag += f'_seed{args.seed}'
    if args.qt_skew is not None:
        tag += f'_skew{int(round(100 * args.qt_skew)):02d}'
    DATA_PATH = HERE / f'data{tag}.pth'

    master = HERE / 'data.pth'
    if DATA_PATH.exists():               # append mode: reuse frozen shared state
        data = torch.load(DATA_PATH, weights_only=False, map_location=device)
        runs_data = data['runs']
        y_hat_mu = data['y_hat_mu'].to(device)
        x_unif = data['x_unif'].to(device)
        log(f"append mode [{DATA_PATH.name}]: existing runs {list(runs_data.keys())}")
    elif tag and master.exists():        # variant file: inherit frozen QT set + x_unif
        data = torch.load(master, weights_only=False, map_location=device)
        y_hat_mu = data['y_hat_mu'].to(device)
        x_unif = data['x_unif'].to(device)
        runs_data = {}
        if args.qt_skew is not None:     # subsample oracle to the requested m>0 fraction
            mq = magnetization(y_hat_mu)
            pos, neg = y_hat_mu[mq > 0], y_hat_mu[mq <= 0]
            n_pos = max(1, int(args.qt_skew / (1 - args.qt_skew) * neg.shape[0]))
            y_hat_mu = torch.cat([pos[:n_pos], neg], dim=0)
            log(f"skewed QT set: {n_pos} of {pos.shape[0]} m>0 kept; "
                f"m>0 fraction = {(magnetization(y_hat_mu) > 0).float().mean().item():.3f}")
        log(f"variant [{DATA_PATH.name}]: frozen QT/x_unif inherited from data.pth")
    else:
        # one-shot QT to build hat_mu samples used by X_{hat_mu}; frozen across stages
        torch.manual_seed(1)
        x_qt = u0.samples(2000)
        y_hat_mu = quench_and_temper(x_qt, u1, sigma=2.0, opt_step=0.5, opt_iters=200, mc_step=2e-3, mc_iters=1000)
        m_qt = magnetization(y_hat_mu)
        log(f"QT: {y_hat_mu.shape[0]} hat_mu samples, m>0 fraction = {(m_qt > 0).float().mean().item():.3f}")
        torch.manual_seed(2)
        x_unif = u0.samples(N_VALID)
        runs_data = {}

    todo = [m for m in requested if m not in runs_data]
    log(f"training: {todo}")
    for m in todo:
        log(f"=== {m}{tag} ===")
        flow, ess_history = train(m, y_hat_mu, seed=args.seed, steps=args.steps)
        with torch.no_grad():
            G = flow.t()
            y_push, _ = G.inv.call_and_ladj(x_unif)
            w = importance_weights(x_unif, u0, u1, G.inv)
            final_ess = compute_ESS(w).item()
        mag = magnetization(resample(y_push, w))
        occ = (mag > 0).float().mean().item()
        runs_data[m] = {
            'state_dict': {k: v.cpu() for k, v in flow.state_dict().items()},
            'ess_history': ess_history,
            'samples': y_push.cpu(),
            'weights': w.cpu(),
            'final_ess': final_ess,
            'occ_plus': occ,
        }
        log(f"[{m:<18}] final ESS = {final_ess:.4f}  reweighted p(m>0) = {occ:.3f}")
        torch.save({
            'config': {
                'STEPS': STEPS, 'LR': LR, 'BATCH': BATCH, 'N_TRAIN': N_TRAIN,
                'N_VALID': N_VALID, 'SIGMA': SIGMA, 'NSF_LIM': NSF_LIM,
                'methods': list(METHODS),
            },
            'x_unif': x_unif.cpu(),
            'y_hat_mu': y_hat_mu.cpu(),
            'runs': runs_data,
        }, DATA_PATH)
        log(f"saved {DATA_PATH} ({len(runs_data)} runs)")

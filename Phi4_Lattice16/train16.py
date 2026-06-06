# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""16x16 phi^4 sampled by a TRUNCATED flow: NSF on the N0 lowest normal modes,
exact Gaussians on the rest, complete action only in the importance weights.

Coordinates: xi = E^T phi, with E the orthonormal eigenbasis of the action
Hessian at the uniform vacuum (eigenvalues omega_k^2 ascending; mode 0 is the
constant/zero-momentum mode carrying the vacuum at +-v*sqrt(D)). The flow acts
on xi_{0:N0-1}; modes N0..D-1 keep their physics-informed Gaussians N(0, 1/omega_k^2)
through an identity map, so the proposal density is exact there.
"""
import time
from pathlib import Path
import torch
from zflows.flow import NSF
from zflows.potential import Gaussian, Potential
from zflows.utils import compute_ESS, importance_weights, resample, langevin

from core import loss_KL, loss_X, quench_and_temper
from parameters import L, D, KAPPA, LAMBDA, H, SIGMA, BINS, TRANSFORMS, HIDDEN_FEATURES, N_TRAIN, N_VALID, BATCH, STEPS, LR

import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

HERE = Path(__file__).resolve().parent
STATUS = HERE / 'train_status.log'
N0 = 16                                  # flow dimension: lowest-omega modes

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, 'a') as f:
        f.write(line + '\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- lattice action in site coordinates ----------------
def action_site(phi):
    p = phi.view(-1, L, L)
    hop = p * (torch.roll(p, 1, dims=1) + torch.roll(p, 1, dims=2))
    return (-2.0 * KAPPA * hop + p.square()
            + LAMBDA * (p.square() - 1.0).square() + H * p).sum(dim=(1, 2))

# ---------------- mode basis from the numerical Hessian at the vacuum ----------------
def build_modes():
    # uniform vacuum value v: minimize S over the constant configuration
    v = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([v], max_iter=100)
    def closure():
        opt.zero_grad()
        s = action_site(v.expand(1, D)).sum()
        s.backward()
        return s
    opt.step(closure)
    v0 = float(v.detach())
    # Hessian of S at the vacuum (D x D), exact via autograd
    phi0 = torch.full((D,), v0, dtype=torch.float64)
    def S_flat(p):
        return action_site(p.unsqueeze(0).to(torch.float32)).squeeze(0).to(torch.float64)
    Hmat = torch.autograd.functional.hessian(S_flat, phi0)
    evals, evecs = torch.linalg.eigh(Hmat)        # ascending; evecs orthonormal columns
    return float(abs(v0)), evals.to(torch.float32), evecs.to(torch.float32)

V0, OMEGA2, E = build_modes()
assert (OMEGA2 > 0).all(), "Hessian not positive definite at the vacuum"
E = E.to(device)
OMEGA2 = OMEGA2.to(device)
log(f"modes: v={V0:.3f}, omega2 range [{OMEGA2[0]:.2f}, {OMEGA2[-1]:.2f}], "
    f"zero-mode |xi_0| at vacuum = {V0 * D**0.5:.1f}")

# target potential in mode coordinates: S_tilde(xi) = S(E xi)
class Phi4Modes(Potential):
    def forward(self, xi):
        return action_site(xi @ E.T)

def magnetization(xi: torch.Tensor) -> torch.Tensor:
    return (xi @ E.T).mean(dim=1)

# ---------------- hybrid flow: NSF on first N0 modes + RealNVP conditioning ----------------
# G (target -> source): y --RealNVP--> z --[NSF on z_{0:N0}, identity above]--> x.
# The RealNVP (zflows-native, closed-form O(d) inverse, zeros() = identity start)
# learns the low<->high mode conditioning that the ceiling diagnostic showed is
# missing; all nonlinear multimodal shaping stays in the N0-dim NSF block.
from zflows.flow import RealNVP

class BlockTransform:
    """Duck-typed ComposedTransform: inner transform on the first n0 coords, identity above."""
    _for_ladj_fn = None                  # no compiled fast path
    def __init__(self, inner, n0):
        self.inner, self.n0 = inner, n0
    def call_and_ladj(self, y):
        x0, ladj = self.inner.call_and_ladj(y[:, :self.n0])
        return torch.cat([x0, y[:, self.n0:]], dim=1), ladj
    def __call__(self, y):
        return self.call_and_ladj(y)[0]
    @property
    def inv(self):
        return BlockTransform(self.inner.inv, self.n0)

class ChainTransform:
    """Duck-typed composition: t2 after t1 (forward = t2(t1(y)))."""
    _for_ladj_fn = None
    def __init__(self, t1, t2):
        self.t1, self.t2 = t1, t2
    def call_and_ladj(self, y):
        z, l1 = self.t1.call_and_ladj(y)
        x, l2 = self.t2.call_and_ladj(z)
        return x, l1 + l2
    def __call__(self, y):
        return self.call_and_ladj(y)[0]
    @property
    def inv(self):
        return ChainTransform(self.t2.inv, self.t1.inv)

class HybridFlow(torch.nn.Module):
    def __init__(self, n0):
        super().__init__()
        lim = [1.7 * V0 * D**0.5] + [6.0 / float(OMEGA2[k].sqrt()) for k in range(1, n0)]
        lim = [max(li, 3.0) for li in lim]            # per-mode NSF boxes
        self.nsf = NSF(a=[-li for li in lim], b=[+li for li in lim],
                       bins=BINS, transforms=TRANSFORMS,
                       hidden_features=HIDDEN_FEATURES)
        self.nsf.zeros()
        self.nvp = RealNVP(dimension=D, transforms=4, randmask=True,
                           hidden_features=HIDDEN_FEATURES)
        self.nvp.zeros()                              # identity start
        self.n0 = n0
    def t(self):
        return ChainTransform(self.nvp.t(), BlockTransform(self.nsf.t(), self.n0))

# ---------------- source: iso sigma on low modes, physics Gaussians above ----------------
var = [SIGMA**2] * N0 + [float(1.0 / OMEGA2[k]) for k in range(N0, D)]
u0 = Gaussian(mean=[0.0] * D, variance=var).to(device)
u1 = Phi4Modes().to(device)
u1.enable_grad()
u1.enable_eval()

METHODS = (
    'KL',
    'KL+X_mu',
    'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix',
)

torch.manual_seed(0)
x_pool = u0.samples(N_TRAIN)

def train(method: str, y_hat_mu: torch.Tensor, seed: int = 0, steps: int = STEPS):
    terms = set(method.split('+'))
    use_x_mix    = 'X_mix' in terms
    use_x_mu     = 'X_mu' in terms
    use_x_hat_mu = 'X_hat_mu' in terms
    needs_hat_mu = use_x_hat_mu or use_x_mix

    torch.manual_seed(seed)
    flow = HybridFlow(N0).to(device)
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

            y_mu = resample(y, w)
            y_mu = langevin(y_mu, u1, step=2e-3, iters=50)

            if needs_hat_mu:
                y_hat_mu = langevin(y_hat_mu, u1, step=2e-3, iters=50)

        G = flow.t()
        loss = loss_KL(y_mu, u0, u1, G)
        if use_x_mix:
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


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', type=str, default=','.join(METHODS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=STEPS)
    args = ap.parse_args()
    requested = [m.strip() for m in args.methods.split(',') if m.strip()]

    tag = f'_seed{args.seed}' if args.seed != 0 else ''
    DATA_PATH = HERE / f'data{tag}.pth'
    master = HERE / 'data.pth'

    if DATA_PATH.exists():
        data = torch.load(DATA_PATH, weights_only=False, map_location=device)
        runs_data = data['runs']
        y_hat_mu = data['y_hat_mu'].to(device)
        x_unif = data['x_unif'].to(device)
        log(f"append mode [{DATA_PATH.name}]: existing runs {list(runs_data.keys())}")
    elif tag and master.exists():
        data = torch.load(master, weights_only=False, map_location=device)
        y_hat_mu = data['y_hat_mu'].to(device)
        x_unif = data['x_unif'].to(device)
        runs_data = {}
        log(f"variant [{DATA_PATH.name}]: frozen QT/x_unif inherited from data.pth")
    else:
        torch.manual_seed(1)
        x_qt = u0.samples(2000)
        y_hat_mu = quench_and_temper(x_qt, u1, sigma=2.0, opt_step=0.5, opt_iters=200, mc_step=2e-3, mc_iters=1000)
        m_qt = magnetization(y_hat_mu)
        log(f"QT: {y_hat_mu.shape[0]} hat_mu samples, m>0 fraction = {(m_qt > 0).float().mean().item():.3f}")
        torch.manual_seed(2)
        x_unif = u0.samples(N_VALID)
        runs_data = {}

    todo = [m for m in requested if m not in runs_data]
    log(f"training (N0={N0}): {todo}")
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
                'L': L, 'N0': N0, 'KAPPA': KAPPA, 'LAMBDA': LAMBDA, 'H': H,
                'STEPS': STEPS, 'LR': LR, 'BATCH': BATCH,
                'N_TRAIN': N_TRAIN, 'N_VALID': N_VALID, 'SIGMA': SIGMA,
                'V0': V0, 'methods': list(METHODS),
            },
            'E': E.cpu(), 'OMEGA2': OMEGA2.cpu(),
            'x_unif': x_unif.cpu(),
            'y_hat_mu': y_hat_mu.cpu(),
            'runs': runs_data,
        }, DATA_PATH)
        log(f"saved {DATA_PATH} ({len(runs_data)} runs)")

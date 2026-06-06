"""Re-certify the L=16 reference at 4x budget (roundtrip gate >= 50)."""
import torch, numpy as np
from pilot import pt_run, barrier_from_hist, log, HERE
from parameters import KAPPA, H, LAMBDA

device = 'cuda'
m2, samples_t1, rt2 = pt_run(KAPPA, H, 64000, 256, device, seed=30)
mm = m2.ravel()
p_plus = (mm > 0).float().mean().item()
dF = float(np.log(max(p_plus, 1e-9) / max(1 - p_plus, 1e-9)))
b2, v2, _ = barrier_from_hist(m2)
log(f"[refine] h={H}  p(+)={p_plus:.4f}  DeltaF={dF:.2f} kT  barrier={b2:.2f}  roundtrips={rt2}")
torch.save(dict(kappa=KAPPA, lam=LAMBDA, h=H, v=v2, barrier=b2,
                delta_F=dF, p_plus=p_plus, roundtrips=rt2,
                m_trace=m2, samples_t1=samples_t1),
           HERE / 'phi4_reference.pth')
log("[refine] reference overwritten with certified run")

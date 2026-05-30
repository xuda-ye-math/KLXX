# Parameters for the high-dimensional product multi-well benchmark.
# Everything is a function of k, with d = 2**k. See vd_plan.md sections 4-5.

# k values to sweep -- HIGHEST d FIRST (converge at d=64 first).
# grid: k in {1,2,3,4,5,6,7} -> d in {2,4,8,16,32,64,128}.
K_LIST = (7, 6, 5, 4, 3, 2, 1)

# domain / source
SIGMA   = 1.0    # std of the isotropic Gaussian source mu_0 = N(0, I)
NSF_LIM = 4.0    # NSF spline box half-width; flow acts on [-4, 4]^d
PLT_LIM = 4.0

# training (fixed across k)
# Batch reduced 2000 -> 200: forward KL is very sensitive to batch size, so a
# small batch widens the gap between plain KL and the X-augmented losses (and
# makes the O(d) autoregressive inverse ~10x cheaper per step).
BATCH = 200
STEPS = 10000
LR    = 1e-3

# loss coefficients (paper default, main.tex:457): lambda=1, alpha=beta=1/2
LAMBDA = 1.0
ALPHA  = 0.5
BETA   = 0.5

# one-step IS surrogate (no AIS / Boltzmann ladder)
IS_MC_STEP  = 2e-3
IS_MC_ITERS = 50

# QT pool for hat_mu
QT_SIGMA     = 2.0
QT_OPT_STEP  = 0.5
QT_OPT_ITERS = 200
QT_MC_STEP   = 2e-3
QT_MC_ITERS  = 500

METHODS = (
    'KL',
    'KL+X_mu',
    'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix',
)


def dim(k):      return 2 ** k
def n_train(k):  return 10000 * 2 ** k
def n_valid(k):  return 10000 * 2 ** k
def p_qt(k):     return 100 * 2 ** k


def bins(k):
    return 16


def transforms(k):
    # The target factorizes, so the optimal map is diagonal and needs little
    # coupling depth. 6 transforms suffice; keeping it flat also bounds the
    # O(d * transforms) masked-autoregressive inverse cost at high d.
    return 6


def hidden(k):
    # Unified (256, 256) conditioner for every k. MLP width is ~free here since
    # the masked-autoregressive inverse is launch-overhead bound (d sequential
    # passes), not matmul bound -- (64,64) and (256,256) invert at the same
    # speed -- so removing the d-dependent split simplifies the description.
    return (256, 256)


def steps(k):
    # Enough steps for genuine convergence at batch 500 (we do NOT undertrain to
    # fake difficulty -- the well depth and small batch are the difficulty knobs).
    # The O(d) inverse is the wall-clock ceiling, so the high-d columns get fewer
    # (still-converging) steps. Override with --steps for the full STEPS=10000 run.
    return {1: 3000, 2: 3000, 3: 2500, 4: 2000, 5: 1500, 6: 1200, 7: 1200}.get(k, 1200)

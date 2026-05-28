# Parameters for the Fourier-field Bayesian inverse problem (idea_C).
# d = 9 Fourier coeffs, G = 2 groups -> 4 sign-flip posterior modes.

# ---- model / data (pinned per idea_C.md) ----
DIM        = 9       # Fourier coefficients (|k|^2 <= 2 on the 2D torus)
GROUPS     = 2       # mode groups -> 2^G = 4 posterior modes
AMP        = 2.5     # signal amplitude: theta* = AMP * sqrt(prior_var) * z, z~N(0,I)
SIGMA_OBS  = 2.0     # observation noise std
COARSE_N   = 3       # 3x3 = 9 coarse-grid obs points (TRAINING likelihood)
FINE_N     = 5       # 5x5 = 25 fine-grid points (data gen + post-training eval).
                     # reduced from 8 so the fine/coarse posterior-width gap is mild
                     # enough that the fine-grid ESS also discriminates the 4 losses.
DATA_SEED  = 42      # fixed seed for theta* and the synthetic observation noise

# ---- source mu_0 ----
# The source is the centered Gaussian smoothness prior itself,
# N(0, diag(prior_var)) with prior_var_m = GAMMA2 / (1 + |k|^2)^s (set in core.py).

# ---- NSF flow architecture (the paper's 2D architecture) ----
NSF_LIM   = 12.0     # spline box half-width: flow acts on [-12, 12]^9
BINS      = 16
TRANSFORMS = 6
HIDDEN    = (128, 128)

# ---- training (one-step IS, M = 1) ----
BATCH = 500          # neutral batch size (NOT 200)
STEPS = 2000
LR    = 1e-3
LAMBDA = 1.0         # X_mu coefficient

# ---- one-step IS surrogate ----
IS_MC_STEP  = 2e-3
IS_MC_ITERS = 50

# ---- pools ----
N_TRAIN = 20000      # source pool for training batches
N_VALID = 20000      # fresh source pool for the honest fine-grid eval
P_QT    = 400        # 100 * 4 QT particles for the hat_mu pool

# ---- QT (quench-and-temper) ----
# u^2 quartic needs a SMALL armijo LBFGS step; default 1.0 diverges to NaN.
QT_SIGMA     = 2.5
QT_OPT_STEP  = 0.05
QT_OPT_ITERS = 200
QT_MC_STEP   = 5e-4
QT_MC_ITERS  = 300

# ---- mode coverage ----
MODE_FRAC = 0.05     # a mode counts as found if it holds >= 5% of samples
KNN_K     = 5        # Naeem coverage neighbourhood

METHODS = (
    'KL',
    'KL+X_mu',
    'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix',
)

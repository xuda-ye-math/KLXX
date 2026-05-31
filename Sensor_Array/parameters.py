# Parameters for the sensor-array source-localization Bayesian inverse problem.
# d = N_SRC identical sources on a line; the permutation-invariant likelihood
# gives exactly N_SRC! degenerate, well-separated posterior modes.
# With N_SRC = 3 there are 3! = 6 modes. The sources are spread widely so the
# modes sit several prior-std from the origin; the prior tail then cannot reach
# them and bare forward KL collapses onto a subset while QT/X_mix recover all.

# ---- model / data ----
N_SRC      = 3        # number of identical sources -> d = N_SRC, NMODES = N_SRC!
N_SENSORS  = 25       # sensors equispaced on [-SENSOR_LIM, SENSOR_LIM]
SENSOR_LIM = 6.0      # sensor-line half-width (comfortably brackets the sources)
ELL        = 0.8      # Gaussian sensor kernel width (source "footprint")
SIGMA_OBS  = 0.3      # observation-noise std (small -> sharp, deep modes)
THETA_STAR = (5.0, 0.0, -5.0)    # true source positions; widely spread so the 6 permutation modes are far apart
DATA_SEED  = 42       # fixed seed for the synthetic observation noise

# ---- source mu_0 ----
# Centered isotropic Gaussian prior N(0, SIGMA_PRIOR^2 I); it sits on the common
# permutation-symmetry saddle (the origin). The modes sit ~4.4 prior-std out, so
# the prior tail does not reach them and bare forward KL collapses onto one mode.
SIGMA_PRIOR = 1.6     # prior std (modes sit ~4.4 prior-std from the origin)

# ---- NSF flow architecture (the paper's 2D architecture) ----
NSF_LIM    = 11.0     # spline box half-width: flow acts on [-11, 11]^3
BINS       = 16
TRANSFORMS = 6
HIDDEN     = (128, 128)

# ---- training (one-step IS, M = 1) ----
BATCH  = 500
STEPS  = 2000
LR     = 1e-3
LAMBDA = 1.0          # X_mu coefficient

# ---- one-step IS surrogate ----
IS_MC_STEP  = 1e-3
IS_MC_ITERS = 50

# ---- pools ----
N_TRAIN = 20000       # source pool for training batches
N_VALID = 20000       # fresh source pool for the honest eval
P_QT    = 600         # 100 * 6 QT particles for the hat_mu pool

# ---- QT (quench-and-temper) ----
# Sharp modes (small SIGMA_OBS) -> small armijo LBFGS step to avoid overshoot.
QT_SIGMA     = 3.0
QT_OPT_STEP  = 0.05
QT_OPT_ITERS = 200
QT_MC_STEP   = 2e-4
QT_MC_ITERS  = 300

# ---- mode coverage ----
MODE_FRAC = 0.05      # a mode counts as found if it holds >= 5% of samples
KNN_K     = 5         # Naeem coverage neighbourhood

METHODS = (
    'KL',
    'KL+X_mu',
    'KL+X_mu+X_hat_mu',
    'KL+X_mu+X_mix',
)

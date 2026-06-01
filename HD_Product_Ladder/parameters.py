# Parameters for the AIS-ladder study on the product multi-well benchmark.
# Dimension is FIXED at d = 2**K = 32; the swept variable is the annealed-
# importance-sampling ladder length M. We train a single loss -- KL+X_mu+X_mix,
# the best loss in HD_Product -- for every M and record its ESS history, to see
# how the ladder length affects convergence and the final ESS. The mu-surrogate
# is produced by zflows.utils.annealed_importance_sampling_G, annealing the flow
# proposal nu -> the posterior mu over M geometric rungs (fixed iters per rung).
#
# d = 256 (HD_Product's hardest column): in high d the one-step IS surrogate (M=1)
# is weak (ESS ~0.38), so the AIS ladder length M actually matters here -- which is
# the whole point of the sweep. The compiled inv_ladj/for_ladj fast paths make the
# 256-deep inverse tractable (one-time compile per run, then fast steps).

# ---- fixed dimension / architecture ----
K       = 8          # d = 2**K = 256
SIGMA   = 1.0        # std of the isotropic Gaussian source mu_0 = N(0, I)
NSF_LIM = 4.0        # NSF spline box half-width; flow acts on [-4, 4]^d
BINS       = 16
TRANSFORMS = 6
HIDDEN     = (256, 256)

# ---- swept variable: AIS ladder length M ----
M_LIST = (1, 2, 4, 8, 16, 32)

# ---- training (fixed across M) ----
BATCH = 200
STEPS = 1000
LR    = 1e-3

# loss coefficients (paper default): lambda=1, alpha=beta=1/2 -> KL+X_mu+X_mix
LAMBDA = 1.0
ALPHA  = 0.5
BETA   = 0.5

# ---- AIS mu-surrogate (the swept knob lives here) ----
# annealed_importance_sampling(x, source=u0, target=u1, ladder=M, step, iters).
# FIXED iters per rung: every rung runs AIS_ITERS Langevin steps, so the total
# MCMC budget is M * AIS_ITERS and grows with the ladder length.
AIS_STEP  = 2e-3
AIS_ITERS = 20       # Langevin steps PER RUNG (not divided by M)
AIS_CHUNK = 1

# ---- hat_mu refresh (X_mix's QT half) ----
IS_MC_STEP  = 2e-3
IS_MC_ITERS = 50

# ---- pools ----
def dim():      return 2 ** K
def n_train():  return 10000 * 2 ** K     # source pool for training batches
def n_valid():  return 10000 * 2 ** K     # fresh source pool for honest eval
def p_qt():     return 100 * 2 ** K       # QT pool size for hat_mu

# ---- QT (quench-and-temper) for hat_mu, built once (target is fixed) ----
QT_SIGMA     = 2.0
QT_OPT_STEP  = 0.5
QT_OPT_ITERS = 200
QT_MC_STEP   = 2e-3
QT_MC_ITERS  = 500

# ---- mode coverage ----
MODE_FRAC = 0.01

# Single loss for the whole sweep (the best loss in HD_Product).
METHOD = 'KL+X_mu+X_mix'

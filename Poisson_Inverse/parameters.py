# Parameters for the screened-Poisson Bayesian-inversion Boltzmann generator.
# Canonical names per ../.aris/parameters.txt; potential-specific extras at the end.
# ALL dynamics run on WHITENED coordinates xi (prior = N(0, I)); the prior scaling
# Sigma^{1/2} lives inside the potential only (user hard rule: Langevin/QT steps
# are isotropic, never use raw Fourier coefficients).

# ---- mode geometry (smoke defaults; train scripts override via --m-low/--m-full) ----
M_LOW = 4              # trained block: modes with max(|k1|,|k2|) <= M_LOW/2 -> M_LOW^2 coeffs
M_FULL = 8             # full set for inference/referee -> M_FULL^2 coeffs
D = M_LOW * M_LOW      # flow dimension d_low (16 smoke, 36 headline)
N_GRID = 32            # spectral grid for the forward solve (>= 4*M_FULL/2 dealiased)

# ---- domain & flow parameters ----
SIGMA = 1.0            # source Gaussian width (whitened prior = N(0, I))
NSF_LIM = 5.0          # NSF box half-width on whitened coords (prior is N(0,1) per mode)
BINS = 16              # spline bins per coordinate
TRANSFORMS = 6         # coupling transforms in the flow
HIDDEN = (256, 256)    # hidden widths of the coupling networks

# ---- basic training parameters ----
N_VALID = 100000       # validation set size (no N_TRAIN)
N_POOL = 20000         # pool size P: QT pool and adaptive-selection particles
N_BATCH = 2000         # batch size B: per-gradient-step batch for mu and hat_mu draws
STEPS = 1000           # gradient steps per stage
LR = 1e-3              # Adam learning rate

# ---- optimization and rejuvenation parameters ----
OPT_STEP = 1e-2        # QT quench step (L-BFGS, armijo)
OPT_ITERS = 100        # QT quench iterations
MC_STEP = 1e-3         # Langevin rejuvenation step (whitened space, isotropic)
MC_ITERS = 100         # Langevin rejuvenation iterations
SMC_RUNG_ITERS = 20    # Langevin iters per rung -- shared by SMC (Alg. 3) and AIS
SMC_RUNGS = 4          # ladder rungs M = 4 (user-fixed, as clock)
T_SAFE = 0.2           # safe start (user: 0.1 too easy at smoke, revised to 0.2)

# ---- empirical parameters ----
SHRINK_FACTOR = 0.7    # step shrink factor gamma on abort
ADAPIVE_TAU = 0.7      # SMC ESS floor tau (adaptive temperature selection)
VALIDATION_TAU = 0.3   # validation ESS floor tau_v (acceptance gate)

# ---- loss ----
LAMBDA = 1.0           # X_mu weight in the fused stage loss

# ---- screened-Poisson inverse problem (potential-specific) ----
C2 = 25.0              # screening c^2 in (-Lap + c^2): adds smoothing, kills k=0 blowup
G0 = 3.0               # source amplitude
DELTA = 0.5            # cosine contrast (FROZEN: small delta keeps fold-wells shallow, symmetry wells deep -- Darcy precedent)
ALPHA = 3.0            # cosine frequency: well lattice spacing 2*pi/ALPHA in v
EPS_TILT = 0.0         # FROZEN 0: prior theta_0 term tilts the shift lattice, truth draw tilts signs (Darcy precedent); explicit tilt unneeded
PRIOR_AMP = 2.0        # sigma_pr^2 prefactor of the prior variance
PRIOR_S = 4            # prior decay (FROZEN by gate sweep: s=4 + sigma_obs=0.05 PASS)
SENSOR_RING_CENTER = (0.50, 0.50)
SENSOR_RING_RADIUS = 0.30
N_SENSORS = 24         # MUST overdetermine: >= 1.5 * d_low (24 at 4x4; use 48 at 6x6 headline)
SIGMA_OBS = 0.02       # observation noise (FROZEN: well barrier ~46 kT; gate re-passed at s=4)
SEED_TRUTH = 42        # theta_truth draw
SEED_NOISE = 43        # observation noise draw

# ---- safety guards / diagnostics ----
MAX_STAGES = 30
MAX_RETRY = 12
GRAD_CLIP = 1e3
MAX_SKIP = 10
MODE_FRAC = 0.01

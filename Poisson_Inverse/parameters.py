# Parameters for the screened-Poisson Bayesian-inversion Boltzmann generator.
# Canonical names per ../.aris/parameters.txt; potential-specific extras at the end.
# ALL dynamics run on WHITENED coordinates xi (prior = N(0, I)); the prior scaling
# Sigma^{1/2} lives inside the potential only (user hard rule: Langevin/QT steps
# are isotropic, never use raw Fourier coefficients).

# ---- mode geometry (smoke defaults; train scripts override via --m-low/--m-full) ----
M_LOW = 6              # trained block (project default: 6x6 in 8x8)
M_FULL = 8
D = M_LOW * M_LOW      # flow dimension d_low (16 smoke, 36 headline)
N_GRID = 32            # spectral grid for the forward solve (>= 4*M_FULL/2 dealiased)

# ---- domain & flow parameters ----
SIGMA = 1.0            # source Gaussian width (whitened prior = N(0, I))
NSF_LIM = 5.0          # NSF box half-width on whitened coords (prior is N(0,1) per mode)
BINS = 16              # spline bins per coordinate
TRANSFORMS = 6         # coupling transforms in the flow
HIDDEN = (256, 256)    # hidden widths of the coupling networks

# ---- basic training parameters ----
N_VALID = 160000        # validation set size (reduced from 100k: VRAM headroom, user)
N_POOL = 40000         # pool size P: QT pool and adaptive-selection particles
N_BATCH = 4000         # B=5k (user: 10k too slow per step)
STEPS = 1000           # gradient steps per stage
LR = 1e-3              # halved for the hardened regime (Adam runaway at stage-5 sharpness)

# ---- optimization and rejuvenation parameters ----
OPT_STEP = 1e-2        # QT quench step (L-BFGS, armijo)
OPT_ITERS = 200        # QT quench iterations (doubled for the hardened sharp misfit)
MC_STEP = 1e-4         # Langevin step (scaled down for sigma_obs=0.005: ULA stability)
MC_ITERS = 400         # more iters to keep iters*step useful at the smaller step
SMC_RUNG_ITERS = 40    # Langevin iters per rung -- shared by SMC (Alg. 3) and AIS
SMC_RUNGS = 6          # ladder rungs M = 6 (user, hardened regime)
T_SAFE = 0.1           # safe start (hardened regime: 0.2 over-shrinks from ESS 0.004)

# ---- empirical parameters ----
SHRINK_FACTOR = 0.7    # step shrink factor gamma on abort
ADAPIVE_TAU = 0.5      # SMC ESS floor (user: 0.7 too conservative in the hardened regime)
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
PRIOR_S = 6            # prior decay (hardened point: s=6 + sigma_obs=0.005, gate PASS 0.88)
SENSOR_RING_CENTER = (0.50, 0.50)
SENSOR_RING_RADIUS = 0.30
N_SENSORS = 54         # overdetermination rule: >= 1.5 * d_low
SIGMA_OBS = 0.01       # softened from 0.005 (user: collapse too severe even for KLXX); gate PASS 0.995
SEED_TRUTH = 42        # theta_truth draw
SEED_NOISE = 43        # observation noise draw

# ---- safety guards / diagnostics ----
MAX_STAGES = 30
MAX_RETRY = 10         # consecutive failed attempts per stage -> stop, claim failure (raised from 5, user)
GRAD_CLIP = 200      # clip 100 (user: 10 too tight, 1e3 allowed the runaway)
MAX_SKIP = 10
MODE_FRAC = 0.01

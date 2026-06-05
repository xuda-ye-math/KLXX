# Parameters for the p-state clock Boltzmann generator benchmark (Algorithm 4).
# Canonical names per ../parameters.txt; potential-specific extras at the end.
# Source distribution is UNIFORM on the torus [-pi, pi)^D (so no SIGMA here).

import math

# ---- domain & flow parameters ----
L = 8                  # lattice side (train.py --L overrides)
D = L * L              # dimension: one angle per site
NSF_LIM = math.pi      # box half-width: NCSF acts on the torus [-pi, pi)^D
BINS = 16              # spline bins per coordinate
TRANSFORMS = 6         # coupling transforms in the flow
HIDDEN = (256, 256)    # hidden widths of the coupling networks

# ---- basic training parameters ----
N_VALID = 1000000      # validation set size (no N_TRAIN)
N_POOL = 200000        # pool size P: QT pool and adaptive-selection particles
N_BATCH = 5000         # batch size B: per-gradient-step batch for both mu and hat_mu draws
STEPS = 500            # gradient steps per stage
LR = 1e-3              # Adam learning rate

# ---- optimization and rejuvenation parameters ----
OPT_STEP = 1e-2        # QT quench step (L-BFGS, armijo)
OPT_ITERS = 100        # QT quench iterations
MC_STEP = 1e-3         # Langevin rejuvenation step
MC_ITERS = 100         # Langevin rejuvenation iterations
SMC_RUNG_ITERS = 20    # Langevin iters per rung -- SHARED by SMC (Alg. 3, G=identity)
                       # and the during-training AIS (G=flow): same algorithm,
                       # exactly the same parameters
SMC_RUNGS = 4          # ladder rungs M, shared by SMC and the AIS surrogate
T_SAFE = 0.25          # safe start: stage-1 initial guess t_init (Algorithm 4)

# ---- empirical parameters (should be not very sensitive) ----
SHRINK_FACTOR = 0.7    # step shrink factor gamma on abort
ADAPIVE_TAU = 0.7      # SMC ESS floor tau (adaptive temperature selection)
VALIDATION_TAU = 0.3   # validation ESS floor tau_v (acceptance gate)

# ---- loss ----
LAMBDA = 1.0           # X_mu weight in the fused stage loss

# ---- clock model (potential-specific) ----
P = 6                  # number of clock states -> P symmetry-broken sectors
J = 1.0                # nearest-neighbor coupling
H = 0.5                # Z_p anisotropy strength

# ---- safety guards / diagnostics ----
MAX_STAGES = 30        # hard cap on ladder stages
MAX_RETRY = 12         # hard cap on abort-and-shrink retries per stage
MODE_FRAC = 0.01       # a sector counts as found if it holds >= MODE_FRAC/P mass

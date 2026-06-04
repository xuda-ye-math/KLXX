# Parameters for the p-state clock Boltzmann generator benchmark (Algorithm 4).
# Canonical names per ../parameters.txt; potential-specific extras at the end.
# Source distribution is UNIFORM on the torus [-pi, pi)^D (so no SIGMA here).

import math

# ---- domain & flow parameters ----
L = 4                  # lattice side (train.py --L overrides); first 4, then 8
D = L * L              # dimension: one angle per site
NSF_LIM = math.pi      # box half-width: NCSF acts on the torus [-pi, pi)^D
BINS = 16
TRANSFORMS = 6
HIDDEN = (256, 256)

# ---- basic training parameters ----
# Aggressive VRAM profile (16 GB fully free, tty mode): large batch + pools.
N_VALID = 80000        # validation set size (no N_TRAIN)
N_POOL = 20000         # QT set and adaptive-temperature-selection particles
N_BATCH = 8000         # per-gradient-step batch for both mu and hat_mu draws
STEPS = 1000           # gradient steps per stage (halved: 4x batch needs fewer steps)
LR = 1e-3

# ---- optimization and rejuvenation parameters ----
OPT_STEP = 1e-2        # QT quench (L-BFGS, armijo)
OPT_ITERS = 100
MC_STEP = 1e-3         # Langevin rejuvenation
MC_ITERS = 100
SMC_RUNG_ITERS = 20    # Langevin iters per rung -- SHARED by SMC (Alg. 3, G=identity)
                       # and the during-training AIS (G=flow): same algorithm,
                       # exactly the same parameters
SMC_RUNGS = 4          # SMC ladder rungs M (fixed M=4 for both L=4 and L=8)
T_SAFE = 0.25          # safe start: stage-1 initial guess t_init (Algorithm 4)

# ---- empirical parameters (should be not very sensitive) ----
SHRINK_FACTOR = 0.7
ADAPIVE_TAU = 0.7      # experiment: raised from the spec default 0.5 (the SMC
                       # gate was non-binding at 0.5: ESS_min 0.93+ at the full jump)
VALIDATION_TAU = 0.3

# ---- loss ----
LAMBDA = 1.0           # X_mu weight in the fused stage loss

# ---- clock model (potential-specific) ----
P = 6                  # number of clock states -> P symmetry-broken sectors
J = 1.0                # nearest-neighbor coupling
H = 0.5                # Z_p anisotropy strength

# ---- safety guards / diagnostics ----
MAX_STAGES = 30
MAX_RETRY = 12         # 0.7^12 ~ 1.4% of the jump: the gate passes long before
MODE_FRAC = 0.01       # a sector counts as found if it holds >= MODE_FRAC/P mass

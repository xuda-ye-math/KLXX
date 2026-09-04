"""Diethanolamine Boltzmann generator parameters.

The settings of the old NMA run (Codes/Molecular_BG/achiral/nma_30d) carry over,
with four changes: the regularization follows the diagonal path from RG_PARAM_0
to RG_PARAM_1 instead of a separate sharpening sweep, the quench-and-temper
coefficient is COEFF_QT, the MALA steps on the intermediate ladder levels are
MC_STEPS_1, and the training length is TRAIN_STEPS.
"""

MOLECULE = "diethanolamine"
FORMULA = "C4H11NO2"
DIMENSION = 48
TEMPERATURE_KELVIN = 300.0
SEED = 0

# Flow: as in the old run.
NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

# Training.
VALID_SIZE = 200000
POOL_SIZE = 0          # 0: the complete validation set is the QT pool
BATCH_SIZE = 10000
TRAIN_STEPS = 500
LR = 1e-3
LR_WARMUP = 25
U_CLIP = 1e3
G_CLIP = 1e2
CHUNKS = 32
CHECKPOINT = True
SCREEN_FRACTION = 1e-4

# SMC surrogate and rejuvenation.
LADDER = 8
MC_DT = 1e-3
MC_STEPS_1 = 20        # MALA steps on the intermediate ladder levels
MC_STEPS_2 = 100       # MALA steps of every other rejuvenation
MC_IMAGE_RADIUS = 3

# Quench and temper.
MELT = 1.0
OPT_ALPHA = 1e-2
OPT_STEPS = 100
COEFF_QT = 0.2

# Regularization path rho_s = (1-s) rho_0 + s rho_1, applied on the diagonal.
RG_PARAM_0 = (50.0, 0.2)
RG_PARAM_1 = (100.0, 0.1)

INITIALIZE_FROM_IDENTITY = True

# Adaptive stage schedule: as in the old run.
BG_PARAM = {
    "t_safe": 0.1,
    "shrink_factor": 0.6,
    "enlarge_factor": 1.4,
    "tau_valid": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

# The monitor prints every tenth step; the driver keeps the ESS of every step.
MONITOR_EVERY = 10

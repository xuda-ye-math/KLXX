"""Full-size n-butane Boltzmann generator parameters."""

MOLECULE = "n_butane"
FORMULA = "C4H10"
DIMENSION = 36
TEMPERATURE_KELVIN = 300.0
SEED = 0

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 8
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

# Full n-butane configuration.
VALID_SIZE = 200000
POOL_SIZE = 0
BATCH_SIZE = 10000
TRAIN_STEPS = 1500
LR = 2e-3
LR_WARMUP = 25
U_CLIP = 1e3
G_CLIP = 1e2

LADDER = 8
MC_DT = 1e-3
MC_STEPS_1 = 10
MC_STEPS_2 = 100
MC_IMAGE_RADIUS = 3
CHUNKS = 32

MELT = 1.0
# Energy-weighted resampling exponent of the tempered QT pool (then rejuvenated again).
COEFF_QT = 0.5
OPT_ALPHA = 1e-2
OPT_STEPS = 100
CHECKPOINT = True

# Regularization path rho = (e [kJ/mol], r [nm]); None uses the
# potential as given, equal values fix the regularization.
RG_PARAM_0 = (50.0, 0.25)
RG_PARAM_1 = (100.0, 0.15)
INITIALIZE_FROM_IDENTITY = True

# Fraction of the largest log weights screened from the loss, the ESS, and
# the resampling weights (the jflows_md default).
SCREEN_FRACTION = 1e-4

# Adaptive stage schedule: the first proposed endpoint is t = 1, one stage.
BG_PARAM = {
    "t_safe": 0.15,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.5,
    "tau_valid": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

# The monitor prints every tenth step; the driver saves ESS for every step.
MONITOR_EVERY = 10

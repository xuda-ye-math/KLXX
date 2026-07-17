"""Full-size n-butane KLXX parameters for the e/r-regularized target."""

MOLECULE = "n_butane"
FORMULA = "C4H10"
DIMENSION = 36
TEMPERATURE_KELVIN = 300.0
SEED = 0

# Main molecular potential regularization constants.
ENERGY_THRESHOLD_KJ_MOL = 50.0
PAIR_DISTANCE_FLOOR_NM = 0.15

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

# Match the completed methane 9D configuration.
N_VALID = 400000
POOL_SIZE = 0
BATCH_SIZE = 20000
TRAIN_STEPS = 500
LR = 1e-3
LR_WARMUP = 25

LADDER = 8
MC_DT = 1e-3
MC_STEPS = 100
MC_IMAGE_RADIUS = 3
CHUNKS = 32

MELT = 1.0
OPT_ALPHA = 1e-2
OPT_STEPS = 200
COEFF_LAMBDA = 1.0
COEFF_ALPHA = 0.5
COEFF_BETA = 0.5

# Optimizer safeguards are separate from the e/r potential regularization.
E_CLIP = 1e3
G_CLIP = 1e2
CHECKPOINT = True
INITIALIZE_FROM_IDENTITY = True

BG_PARAM = {
    # Match the completed propane 27D bridge start exactly.
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.3,
    "tau_smc": 0.8,
    "tau_ess": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

# The monitor prints every tenth step; the driver saves ESS for every step.
MONITOR_EVERY = 10

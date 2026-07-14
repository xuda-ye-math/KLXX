"""Parameters for the propane adaptive KLXX c50 test."""

MOLECULE = "propane"
FORMULA = "C3H8"
DIMENSION = 27
SEED = 200

ENERGY_CUT_KJ_MOL = 50.0
ENERGY_SCALE_KJ_MOL = 50.0
TAIL_FRACTION = 0.01

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

N_VALID = 200000
POOL_SIZE = 100000
BATCH_SIZE = 50000
TRAIN_STEPS = 500
LR = 1e-3
LR_WARMUP = 25

LADDER = 8
MC_DT = 1e-2
MC_STEPS = 50
MC_IMAGE_RADIUS = 3
CHUNKS = 64

MELT = 1.0
OPT_ALPHA = 1e-2
OPT_STEPS = 200
COEFF_LAMBDA = 1.0
COEFF_ALPHA = 0.5
COEFF_BETA = 0.5
E_CLIP = 1e3
G_CLIP = 1e2
CHECKPOINT = True

BG_PARAM = {
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.5,
    "tau_smc": 0.5,
    "tau_ess": 0.4,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

MONITOR_EVERY = 10

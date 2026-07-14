"""Parameters for the n-butane adaptive forward-KL c50 control."""

MOLECULE = "n_butane"
FORMULA = "C4H10"
DIMENSION = 36
SEED = 300
OBJECTIVE = "kl"

ENERGY_CUT_KJ_MOL = 50.0
ENERGY_SCALE_KJ_MOL = 50.0
TAIL_FRACTION = 0.01

NSF_LIM = 8.0
BINS = 32
TRANSFORMS = 6
HIDDEN_FEATURES = (256, 256)
SLOPE = 1e-3

N_VALID = 400000
POOL_SIZE = 0
BATCH_SIZE = 40000
TRAIN_STEPS = 400
LR = 1e-4
LR_WARMUP = 30

LADDER = 10
MC_DT = 1e-3
MC_STEPS = 50
MC_IMAGE_RADIUS = 3
CHUNKS = 64

E_CLIP = 1e3
G_CLIP = 1e2
CHECKPOINT = True

BG_PARAM = {
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.5,
    "tau_smc": 0.6,
    "tau_ess": 0.3,
    "t_tol": 1e-3,
    "max_stages": 20,
    "max_retry": 5,
}

MONITOR_EVERY = 10

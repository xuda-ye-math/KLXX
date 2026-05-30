# Sweep T001 — beta=0.01. Bisecting between T000 (trivial) and T010 (stuck).
# If T001 climbs to high ESS, the pipeline handles slightly nontrivial targets.
BETA = 0.01                 # 1/10 of T010's tempering
N_GRID    = 16
G0        = 3.0
DELTA     = 0.5
ALPHA     = 3.0
SENSOR_RING_CENTER = (0.50, 0.50)
SENSOR_RING_RADIUS = 0.30
N_SENSORS          = 12
SIGMA_OBS = 0.001
A_TRUTH    = 1.0
SEED_TRUTH = 42
SEED_NOISE = 43

NSF_LIM    = 8.0
BINS       = 16
TRANSFORMS = 6
HIDDEN     = (256, 256)

BATCH      = 20000        # was 4000; previous tests used ~1GB / 16GB VRAM. Big batch -> low-variance gradient
STEPS      = 800          # was 2500; bigger batch carries more info per step, so fewer steps
LR         = 3e-3         # was 1e-3; bigger batch makes higher LR safe

LAMBDA     = 1.0

IS_M        = 1
IS_MC_STEP  = 3e-4
IS_MC_ITERS = 30

USE_PRECONDITIONED = False

FINE_N_GRID = 32

QT_SIGMA     = 0.3
QT_OPT_STEP  = 0.005
QT_OPT_ITERS = 200
QT_MC_STEP   = 3e-4
QT_MC_ITERS  = 40
QT_POOL      = 800

SOURCE_POOL  = 200000     # bigger pool so BATCH=20000 has 10x diversity
N_VALID = 20000

METHODS = ('KL', 'KL+X_mu', 'KL+X_mu+X_hat_mu', 'KL+X_mu+X_mix')

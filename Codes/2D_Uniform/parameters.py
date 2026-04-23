import argparse
from potential import *
from utilities import NSF

# parameters of NSF
BINS = 8 # number of bins for RQS transform
HIDDEN_FEATURES = (64, 64) # hidden layer sizes
TRANSFORMS = 4 # number of flow transforms
LR = 1e-3 # learning rate
BATCH = 4000 # batch size
EPOCH = 200 # number of training epochs

# parameters for training
N_TRAIN =  40000 # number of training samples
N_VALID = 160000 # number of validation samples

# available potentials
potential_dict = {'HB': Himmelblau, 'AN': Annulus, 'TW': Three_Well, 'PW': Periodic_Well}

# choice of target potential via command line
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='HB', choices=potential_dict.keys())
args, _ = parser.parse_known_args()
name = args.name

# initialize potential
target = potential_dict[name]()
BOUND = target.BOUND
uniform = Uniform(BOUND=BOUND)
uniform_frozen = Uniform(BOUND=BOUND)

# initialize flow
flow = NSF(bound=BOUND, bins=BINS, hidden_features=HIDDEN_FEATURES, transforms=TRANSFORMS)
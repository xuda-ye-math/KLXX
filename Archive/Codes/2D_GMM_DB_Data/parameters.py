import argparse
import os
from potential import *
from utilities import NSF

# parameters of NSF
BINS = 8 # number of bins for RQS transform
HIDDEN_FEATURES = (64, 64) # hidden layer sizes
TRANSFORMS = 4 # number of flow transforms
LR = 2.5e-4 # learning rate
BATCH = 4000 # batch size
EPOCH = 400 # number of training epochs

# parameters for training
N_TRAIN =  40000 # number of training samples
N_VALID = 160000 # number of validation samples

# available potentials
potential_dict = {'HB': Himmelblau, 'AN': Annulus, 'TW': Three_Well, 'RB': Rosenbrock}

# choice of target potential via command line
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='HB', choices=potential_dict.keys())
args, _ = parser.parse_known_args()
name = args.name

# initialize potential
target = potential_dict[name]()
BOUND = target.BOUND
gmm = GMM(); gmm_frozen = GMM()
if os.path.exists(f"prior_{name}.pt"):
    gmm.load(name=name); gmm_frozen.load(name=name)

# initialize flow
flow = NSF(bound=BOUND, bins=BINS, hidden_features=HIDDEN_FEATURES, transforms=TRANSFORMS)
"""Run the matched n-butane vacuum forward-KL c50 control."""

import parameters as P
from alkane_bg import main as train_kl


if __name__ == "__main__":
    train_kl(P)

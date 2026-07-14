#!/usr/bin/env python
"""Train the 36D n-butane c20 target with adaptive forward KL only."""

from alkane_bg import main as train_kl
import parameters as P


if __name__ == "__main__":
    train_kl(P)

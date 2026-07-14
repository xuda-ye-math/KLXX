#!/usr/bin/env python
"""Train the 36D n-butane c20 target with full-pool adaptive KLXX."""

from alkane_bg import main as train_klxx
import parameters as P


if __name__ == "__main__":
    train_klxx(P)

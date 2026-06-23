"""pytest config for the zflows_md smoke tests (style: ../zflows_md/tests).

Puts the LOCAL, in-development package (X-regularization/zflows_md) first on
sys.path so `import zflows_md` resolves here, not to any conda-installed copy.
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../X-regularization
sys.path.insert(0, REPO)

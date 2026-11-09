"""
The paper's ablations, control group: the full solver, but with 60s per attempt instead of 30s, so the search can
run all its evolution rounds (at 30s, about a quarter of ARC-1 Eval levels hit the cap). Every compression_*.py and
search_*.py ablation starts from this, so each one differs from the control in exactly one way.
(60s is also what the Kaggle config gives, there as 2x time for Kaggle's slower CPUs)
"""

from .default import *

TIME_LIMIT_PER_ATTEMPT = 60.0

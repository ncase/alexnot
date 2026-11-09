"""
The Kaggle config: same as default, but every wall-clock time limit gets 2x, for safety.
Kaggle's CPUs are hyperthreads on a 2.2 GHz Xeon, ~1.2x slower per puzzle than our MacBook (from a practice run),
so the same seconds buy less search there. Used by the Kaggle notebook (ALEXNOT_CONFIG=kaggle).
"""

from .default import *

KAGGLE_EXTRA_TIME_CONSTANT = 2.0 # 1.5 was cutting it close: a level right at the attempt cap could flip on Kaggle

############################################################
# Main Solver (main_solver.py)
############################################################

# Same single parse attempt as default (TOTAL_ATTEMPTS = 1), so Kaggle scores stay comparable to our local blasts
TIME_LIMIT_PER_ATTEMPT = TIME_LIMIT_PER_ATTEMPT * KAGGLE_EXTRA_TIME_CONSTANT
TIME_LIMIT_PER_PROGRAM = TIME_LIMIT_PER_PROGRAM * KAGGLE_EXTRA_TIME_CONSTANT

############################################################
# Evolutionary Search (search_transform.py)
############################################################

CLOCK_BUDGET_PER_MONKEY = CLOCK_BUDGET_PER_MONKEY * KAGGLE_EXTRA_TIME_CONSTANT
CLOCK_BUDGET_PER_BATCHER_MONKEY = CLOCK_BUDGET_PER_BATCHER_MONKEY * KAGGLE_EXTRA_TIME_CONSTANT

"""
Same as default, but every search wall-clock time budget gets 2x time.
"""

from .default import *

############################################################
# Main Solver (main_solver.py)
############################################################

TIME_LIMIT_PER_ATTEMPT = TIME_LIMIT_PER_ATTEMPT * 2 # 2x default's, whatever it is (so this never goes stale)

############################################################
# Evolutionary Search (search_transform.py)
############################################################

CLOCK_BUDGET_PER_MONKEY = CLOCK_BUDGET_PER_MONKEY * 2 # 2x default's, whatever it is
CLOCK_BUDGET_PER_BATCHER_MONKEY = CLOCK_BUDGET_PER_BATCHER_MONKEY * 2 # 2x default's, whatever it is

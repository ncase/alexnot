"""
Ablation: the control with 2x every search wall-clock budget (per attempt, and per monkey per round).
"""

from .ablation_control import *

TIME_LIMIT_PER_ATTEMPT = TIME_LIMIT_PER_ATTEMPT * 2
CLOCK_BUDGET_PER_MONKEY = CLOCK_BUDGET_PER_MONKEY * 2
CLOCK_BUDGET_PER_BATCHER_MONKEY = CLOCK_BUDGET_PER_BATCHER_MONKEY * 2

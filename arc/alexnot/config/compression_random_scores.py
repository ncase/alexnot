"""
Ablation: the dumbest possible selection. Every program's scores (total, shape-diff, each demo pair's) are random,
so the winner, every parent, and attempt 2 are random picks from whatever the monkeys proposed.
"""

from .ablation_control import *

ABLATION_STUDY_MODE = "RANDOM SCORES"

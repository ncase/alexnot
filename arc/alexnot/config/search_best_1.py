"""
Ablation: one parent per round, just the best by total score. No shape-diff island, no demo islands.
"""

from .ablation_control import *

PARENTS_TOP_K_BY_SCORE = 1
PARENTS_TOP_K_BY_SHAPE_DIFF = 0
ABLATION_STUDY_MODE = "NO DEMO ISLANDS"

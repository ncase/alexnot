"""
Ablation: score programs by wrong pixels ONLY (across all demo pairs). No length, no unique concepts/lines,
no edit cost, so every demo-perfect program ties, and the winner is a random one of them.
(The shape-diff score still picks its parents: that's a search island, ablated in search_no_shape_diff_island.py)
"""

from .ablation_control import *

SCORE_COEFF_TOKEN_COST = 0
SCORE_COEFF_UNIQUE_TOKEN_COST = 0
SCORE_COEFF_UNIQUE_LINE_COST = 0
SCORE_COEFF_EDIT_COST = 0

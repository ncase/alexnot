"""
Ablation: the full scoring, minus the cost of more edits/actions. (This also drops the penalty for a one() that
picks randomly among 2+ shapes, since that's charged as edits too)
"""

from .ablation_control import *

SCORE_COEFF_EDIT_COST = 0

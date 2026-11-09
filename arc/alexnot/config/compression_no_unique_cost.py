"""
Ablation: the full scoring, minus the cost of more unique concepts (tokens) and unique lines.
"""

from .ablation_control import *

SCORE_COEFF_UNIQUE_TOKEN_COST = 0
SCORE_COEFF_UNIQUE_LINE_COST = 0

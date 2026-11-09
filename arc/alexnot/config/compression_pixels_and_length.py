"""
Ablation: wrong pixels + plain program length only. Every token costs 1 (no word or value prices),
and no unique concepts/lines or edit cost.
"""

from .ablation_control import *

SCORE_COEFF_UNIQUE_TOKEN_COST = 0
SCORE_COEFF_UNIQUE_LINE_COST = 0
SCORE_COEFF_EDIT_COST = 0

# Every token the same price: words, values, and "magic numbers" alike
TOKEN_BIASES = {}
VALUE_BIASES = {}
VALUE_DEFAULT_BIAS = 1

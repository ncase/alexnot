"""
Ablation: at most 5 rounds of evolution (the control allows 10, and stops early after 2 rounds with no improvement).
Fills in the rounds curve between 3 and 10.
"""

from .ablation_control import *

EVO_MAX_GENERATIONS = 5

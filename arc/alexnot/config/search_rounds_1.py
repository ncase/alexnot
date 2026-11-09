"""
Ablation: at most 1 round(s) of evolution (the control allows 10, and stops early after 2 rounds with no improvement).
"""

from .ablation_control import *

EVO_MAX_GENERATIONS = 1

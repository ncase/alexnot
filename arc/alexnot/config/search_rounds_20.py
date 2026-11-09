"""
Ablation: at most 20 rounds of evolution (the control allows 10, and stops early after 2 rounds with no improvement).
About a quarter of ARC-1 Eval levels use all 10 rounds in the control, so this asks whether they'd gain from more.
"""

from .ablation_control import *

EVO_MAX_GENERATIONS = 20

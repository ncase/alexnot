"""
Ablation: the full scoring, minus "naturalness" and the hard-coded-value penalty: every token costs the same
(no cheap common words or subitizable numbers, and a "magic number" costs no more than any other token).
Unlike compression_pixels_and_length.py, the unique concepts/lines and edit costs stay on.
"""

from .ablation_control import *

TOKEN_BIASES = {}
VALUE_BIASES = {}
VALUE_DEFAULT_BIAS = 1

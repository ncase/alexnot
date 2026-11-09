"""
Ablation: the top 5 by total score as parents, no other islands. The full solver has about as many parents
(the best, the best by shape diff, and the best on each demo pair), so this tests islands vs just more parents.
"""

from .ablation_control import *

PARENTS_TOP_K_BY_SCORE = 5
PARENTS_TOP_K_BY_SHAPE_DIFF = 0
ABLATION_STUDY_MODE = "NO DEMO ISLANDS"

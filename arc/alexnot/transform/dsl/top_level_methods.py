from ...utils import _select_shape_datas
from typing import Dict, List, Any, Union
import random
import numpy as np

from .shape import Shape
from ...config import ONE_AMBIGUOUS_EDITS_COST # What a one() with 2+ matching shapes costs

DSL_CONFIG = {}

def all(**kwargs) -> List[Shape]:
    scene = DSL_CONFIG['scene']
    shape_datas:List[Dict] = _select_shape_datas( scene['shapes'], kwargs )
    return [Shape(data,scene) for data in shape_datas]

def one(**kwargs) -> Shape:
    # Get all, if none give a MISSING shape (graceful fail), else pick random
    selected = all(**kwargs)
    if len(selected)==0:
        return Shape() # blank data
    # Several matches = a dice roll, so this call costs extra (each call, so one() in a loop pays once per run)
    if len(selected) >= 2:
        scene = DSL_CONFIG['scene']
        scene['_edits_cost'] = scene.get('_edits_cost', 0) + ONE_AMBIGUOUS_EDITS_COST
    return random.choice(selected) # random instead of one -- forces explicit selectors

# Find-and-replace on the raw output pixels. `old` and `new` are 2D grids of the same size,
# colors 0-9, with -1 meaning "don't care" in `old` and "leave that pixel alone" in `new`. The scene
# stays shape-based: this only RECORDS the replacement on the scene, and reverse_parse applies all of
# them at the very end, on the rendered pixels (no matter where in the program this call sits).
# An option we may try later: re-parse the edited pixels back into shapes right here, so later
# lines of the program could select & act on them.
def replace_pixels(old:list[list[int]], new:list[list[int]]) -> None:
    old_grid:np.ndarray = np.array(old, dtype=np.int8)
    new_grid:np.ndarray = np.array(new, dtype=np.int8)
    # A malformed call (ragged, 1D, or mismatched sizes) is a broken program: raise, so exec fails gracefully
    if old_grid.ndim != 2 or old_grid.shape != new_grid.shape:
        raise ValueError(f"replace_pixels: old {old_grid.shape} and new {new_grid.shape} must be 2D grids of the same size")
    scene = DSL_CONFIG['scene']
    if '_pixel_replacements' not in scene:
        scene['_pixel_replacements'] = [] # lives on this program run's scene copy only, never the parsed original
    scene['_pixel_replacements'].append((old_grid, new_grid))

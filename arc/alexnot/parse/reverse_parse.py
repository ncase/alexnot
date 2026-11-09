from typing import List, Dict
import numpy as np
from ..grid import _draw_pixels, _replace_pixels

def reverse_parse(scenes:List[Dict]) -> List[np.ndarray]:
    return [reverse_parse_one(scene) for scene in scenes]

def reverse_parse_one(scene:Dict) -> np.ndarray:

    # Make a 2D array of that background color
    w = int( scene['width'] ) # force-cast to int. uh oh, is something making it float?
    h = int( scene['height'] )
    bg = int( scene['background'][1:] ) # CAST BACK TO INT, WHATEVER
    pixels = np.full((h, w), max(bg, 0), dtype=np.int8) # if bg=-1 (no_bg mode), just fill with 0

    # Now DRAW ALL SHAPES
    shapes = [ s for s in scene['shapes'] if s.get('hidden',False)==False ]
    # Back to front: lowest stacking order (z) first, then most recently-moved last (on top).
    # (Shapes without a z, i.e. every parse that doesn't autocomplete, count as 0, so only _moves decides)
    shapes = sorted(shapes, key=lambda x: (x.get('z', 0), x.get('_moves', 0)))
    for shape in shapes:
        _draw_pixels( pixels, shape['_pixels'], shape['left'], shape['top'] )

    # Last of all, apply any replace_pixels() calls the program recorded, in program order,
    # on the rendered pixels. (Scenes that never called it have no such key: .get gives an empty list.)
    # Every spot a replacement matches is 1 edit, like painting 1 shape. They're ADDED to the scene's own
    # `_edits_cost` (shapes keep theirs), since other things can add to that same cost too.
    replacement_edits_cost:int = 0
    for old_grid, new_grid in scene.get('_pixel_replacements', []):
        pixels, matched_spot_count = _replace_pixels(pixels, old_grid, new_grid)
        replacement_edits_cost += matched_spot_count
    # Rendering the same scene twice mustn't double-count: so remember what we added last time,
    # and swap it out for this time's amount (same pixels in, same amount out, so it just stays put)
    previous_replacement_edits_cost:int = scene.get('_replace_pixels_edits_cost', 0)
    if replacement_edits_cost != previous_replacement_edits_cost:
        scene['_edits_cost'] = scene.get('_edits_cost', 0) - previous_replacement_edits_cost + replacement_edits_cost
        scene['_replace_pixels_edits_cost'] = replacement_edits_cost

    # Gimme
    return pixels
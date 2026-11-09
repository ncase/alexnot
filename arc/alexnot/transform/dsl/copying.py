# Copying! shape.copy_to(**target selector) stamps NEW copies of a shape wherever its pixels land on as many
# same-colored target pixels as possible (like a stamp lining up its colored marks with marker dots).
# The original stays where it is. Same correlate2d scoring idea as attraction (physics_attract.py), but greedy:
# a copy can be made any number of times, so there's no placement order to search over.
from typing import Dict, List
import random # Ties between equally good spots: picked randomly, like everywhere else in the DSL
import numpy as np
import scipy.signal # correlate2d: slide a shape over a grid, count overlaps at every spot in one go
from ...utils import _select_shape_datas


def copy_to_targets(shape_data:Dict, scene, target_selector:Dict) -> None:
    """Add copies of this shape to the scene, each one where it matches the most still-uncovered target pixels."""

    # The targets: every visible shape matching the selector, except me (I'd always match myself perfectly)
    targets:List[Dict] = [target for target in _select_shape_datas(scene['shapes'], target_selector)
                          if target is not shape_data and not target.get('hidden')]
    if not targets:
        return

    shadow_mask:np.ndarray = shape_data['_shadow'] == 1
    pixels:np.ndarray = shape_data['_pixels']
    shape_height, shape_width = shadow_mask.shape

    # The "canvas": the targets' combined bounding box, padded by my size on every side, so every spot where
    # I overlap a target fits on it. (Copies may hang off the scene's edge, like attraction's spots)
    canvas_top:int = min(int(target['top']) for target in targets) - (shape_height - 1)
    canvas_left:int = min(int(target['left']) for target in targets) - (shape_width - 1)
    canvas_bottom:int = max(int(target['top']) + target['_shadow'].shape[0] for target in targets) + (shape_height - 1)
    canvas_right:int = max(int(target['left']) + target['_shadow'].shape[1] for target in targets) + (shape_width - 1)

    # Every target pixel's color on that canvas (-1 = no target there)
    target_colors:np.ndarray = np.full((canvas_bottom - canvas_top, canvas_right - canvas_left), -1, dtype=np.int32)
    for target in targets:
        target_shadow_mask:np.ndarray = target['_shadow'] == 1
        target_top:int = int(target['top']) - canvas_top
        target_left:int = int(target['left']) - canvas_left
        target_window:np.ndarray = target_colors[target_top:target_top + target_shadow_mask.shape[0],
                                                 target_left:target_left + target_shadow_mask.shape[1]]
        target_window[target_shadow_mask] = target['_pixels'][target_shadow_mask]

    # My pixels, one mask per color (correlate2d 'valid' mode: entry [row, column] = my top-left corner at canvas (row, column))
    my_color_masks:Dict[int, np.ndarray] = {int(color): (shadow_mask & (pixels == color)).astype(np.int32)
                                            for color in np.unique(pixels[shadow_mask])}

    # Ruled-out spots: any of my pixels landing on a target pixel of a DIFFERENT color.
    # (How many of my pixels hit any target pixel, vs. how many hit a same-colored one)
    overlap_counts:np.ndarray = scipy.signal.correlate2d((target_colors >= 0).astype(np.int32), shadow_mask.astype(np.int32), mode='valid')
    matching_counts:np.ndarray = np.zeros_like(overlap_counts)
    for color, my_color_mask in my_color_masks.items():
        matching_counts += scipy.signal.correlate2d((target_colors == color).astype(np.int32), my_color_mask, mode='valid')
    has_a_mismatch:np.ndarray = overlap_counts > matching_counts

    # Greedy: the copy that covers the most still-uncovered matching target pixels goes first, and so on,
    # until no spot covers anything new. (Each copy covers at least 1 new pixel, so this always ends)
    uncovered:np.ndarray = target_colors >= 0
    while True:
        # How many NEW target pixels would a copy cover, at every spot?
        new_matches:np.ndarray = np.zeros_like(overlap_counts)
        for color, my_color_mask in my_color_masks.items():
            new_matches += scipy.signal.correlate2d(((target_colors == color) & uncovered).astype(np.int32), my_color_mask, mode='valid')
        new_matches[has_a_mismatch] = 0
        best_match_count:int = int(new_matches.max())
        if best_match_count <= 0:
            return

        # The best spot (random tie-break), and a copy there
        best_rows, best_columns = np.nonzero(new_matches == best_match_count)
        spot_index:int = random.randrange(len(best_rows))
        spot_row:int = int(best_rows[spot_index])
        spot_column:int = int(best_columns[spot_index])
        scene['shapes'].append(_copy_shape_at(shape_data, top=spot_row + canvas_top, left=spot_column + canvas_left))

        # Those target pixels are taken now
        uncovered[spot_row:spot_row + shape_height, spot_column:spot_column + shape_width] &= ~shadow_mask


def _copy_shape_at(shape_data:Dict, top:int, left:int) -> Dict:
    """A fresh copy of a shape (its own grids), moved so its top-left corner is at (top, left)."""
    copy:Dict = shape_data.copy()
    copy['_pixels'] = shape_data['_pixels'].copy()
    copy['_shadow'] = shape_data['_shadow'].copy()
    copy['_hole_grids'] = [hole_grid.copy() for hole_grid in shape_data['_hole_grids']]

    # Same box props _set_position keeps in sync (right/bottom, centers)
    copy['top'], copy['left'] = top, left
    copy['bottom'], copy['right'] = top + copy['height'], left + copy['width']
    copy['center_y'], copy['center_x'] = top + copy['height'] / 2, left + copy['width'] / 2

    # Making a copy is one action = 1 edit (like making a ray). The copy counts as edited, the original doesn't
    copy['_edits_cost'] = 1
    copy['edited'] = True
    return copy

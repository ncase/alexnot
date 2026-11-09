# The pathfind monkey: a shape that moved along a bent path (not a straight line) and ended up touching,
# or on top of, a shape it wasn't touching before, probably wound its way there through the gaps.
# Propose pathfind() towards those shapes, with passthru= set to the color it ended up on top of (if any).
import numpy as np
from scipy.ndimage import binary_dilation

from .monkey_helpers import ShapeActionBatcher
from .shape_change import _get_moved_shape_pairs
from ..dsl.physics_pathfind import _visible_pixels_in_scene

# Touching includes corners (unlike the touching_* annotations), so grow a footprint by all 8 neighbors
_EIGHT_NEIGHBORS:np.ndarray = np.ones((3, 3), dtype=bool)


def _footprint_mask(shadow:np.ndarray, top:int, left:int, scene_height:int, scene_width:int) -> np.ndarray:
    """A scene-sized mask of where a shadow sits with its corner at (top, left), minus whatever hangs off the canvas."""
    local_ys, local_xs = np.nonzero(shadow)
    scene_ys:np.ndarray = local_ys + top
    scene_xs:np.ndarray = local_xs + left
    on_canvas:np.ndarray = (scene_ys >= 0) & (scene_ys < scene_height) & (scene_xs >= 0) & (scene_xs < scene_width)
    mask:np.ndarray = np.zeros((scene_height, scene_width), dtype=bool)
    mask[scene_ys[on_canvas], scene_xs[on_canvas]] = True
    return mask


def shapes_pathfind(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that took a bent path to touch (or land on) another shape, propose pathfind towards it
    """
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]

            # Pathfinding never resizes the grid, so skip pairs where the sizes differ
            if predicted_scene['_raw'].shape != pair['output']['_raw'].shape:
                continue
            scene_height, scene_width = predicted_scene['_raw'].shape

            for before, after in _get_moved_shape_pairs(predicted_scene, pair['output']):

                # A straight line (one axis, or a perfect diagonal) is move/fall's job, not pathfind's
                dx:int = after['left'] - before['left']
                dy:int = after['top'] - before['top']
                if dx == 0 or dy == 0 or abs(dx) == abs(dy):
                    continue

                # Where the shape sits at the end, and which pixels it touches there and at the start (corners count)
                end_footprint:np.ndarray = _footprint_mask(before['_shadow'], after['top'], after['left'], scene_height, scene_width)
                end_touching:np.ndarray = binary_dilation(end_footprint, structure=_EIGHT_NEIGHBORS)
                start_touching:np.ndarray = binary_dilation(
                    _footprint_mask(before['_shadow'], before['top'], before['left'], scene_height, scene_width), structure=_EIGHT_NEIGHBORS)

                for other in predicted_scene['shapes']:
                    if other is before or other.get('hidden', False):
                        continue
                    ys, xs, colors = _visible_pixels_in_scene(other, scene_height, scene_width)

                    # Only shapes it moved TO: touching at the end, but not already at the start.
                    # (A shape that slides along a wall the whole way didn't head for the wall)
                    if not end_touching[ys, xs].any() or start_touching[ys, xs].any():
                        continue

                    # Ended up on top of some of its pixels? Then it must be allowed to walk onto that color
                    colors_walked_onto:set[int] = set(colors[end_footprint[ys, xs]].tolist())
                    if not colors_walked_onto:
                        shape_actions.add(before, 'pathfind()', other, pair_index=pair_index)
                    for color_digit in colors_walked_onto:
                        shape_actions.add(before, f'pathfind(passthru="c{color_digit}")', other, pair_index=pair_index)

        # Flush, append all suggestions!
        for code in shape_actions.flush():
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates

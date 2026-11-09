# pathfind(): a shape winds its way through the gaps between other shapes, towards the nearest target
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

# The 8 king's-move steps (diagonals allowed), as (dy, dx)
_KING_STEPS:Tuple[Tuple[int, int], ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _visible_pixels_in_scene(shape:Dict, scene_height:int, scene_width:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A visible shape's on-canvas pixels: their scene ys, scene xs, and color digits."""
    local_ys, local_xs = np.nonzero(shape['_shadow'])
    scene_ys:np.ndarray = local_ys + int(shape['top'])
    scene_xs:np.ndarray = local_xs + int(shape['left'])
    colors:np.ndarray = shape['_pixels'][local_ys, local_xs]
    # A shape can hang off the canvas, so drop whatever's off it
    on_canvas:np.ndarray = (scene_ys >= 0) & (scene_ys < scene_height) & (scene_xs >= 0) & (scene_xs < scene_width)
    return scene_ys[on_canvas], scene_xs[on_canvas], colors[on_canvas]


def pathfind_offset(mover:Dict, targets:List[Dict], scene:Dict, passthru_digit:Optional[int]) -> Tuple[int, int]:
    """How far (dx, dy) the mover slides along the gaps to get as close as it can to the nearest target."""
    scene_height:int = int(scene['height'])
    scene_width:int = int(scene['width'])

    # Which pixels are walls? Every other visible shape's pixels, except the passthru color.
    # (The background is simply wherever no shape is, so no color is special here)
    blocked:np.ndarray = np.zeros((scene_height, scene_width), dtype=bool)
    for shape in scene['shapes']:
        if shape is mover or shape.get('hidden', False):
            continue
        ys, xs, colors = _visible_pixels_in_scene(shape, scene_height, scene_width)
        solid:np.ndarray = colors != passthru_digit if passthru_digit is not None else np.ones(len(ys), dtype=bool)
        blocked[ys[solid], xs[solid]] = True

    # How many steps from each free pixel to the nearest target, for a 1-pixel walker? (BFS out from every target pixel)
    unreachable:int = scene_height * scene_width + 1
    distance:np.ndarray = np.full((scene_height, scene_width), unreachable, dtype=np.int32)
    frontier:deque = deque()
    for target in targets:
        ys, xs, _ = _visible_pixels_in_scene(target, scene_height, scene_width)
        distance[ys, xs] = 0
        frontier.extend(zip(ys.tolist(), xs.tolist()))
    while frontier:
        y, x = frontier.popleft()
        for step_y, step_x in _KING_STEPS:
            next_y, next_x = y + step_y, x + step_x
            if (0 <= next_y < scene_height and 0 <= next_x < scene_width and not blocked[next_y, next_x]
                    and distance[next_y, next_x] == unreachable):
                distance[next_y, next_x] = distance[y, x] + 1
                frontier.append((next_y, next_x))

    # Now slide the WHOLE shape around (BFS over its top-left spots, 8 directions), wherever all its pixels fit on free pixels
    shadow:np.ndarray = mover['_shadow'] == 1
    shape_height, shape_width = shadow.shape
    start_top:int = int(mover['top'])
    start_left:int = int(mover['left'])

    def closeness(top:int, left:int) -> Optional[float]:
        """The shape's total walking distance to the target at this spot (lower = closer), or None if it doesn't fit."""
        if top < 0 or left < 0 or top + shape_height > scene_height or left + shape_width > scene_width:
            return None
        if blocked[top:top + shape_height, left:left + shape_width][shadow].any():
            return None
        # Adding up every pixel's distance is like measuring from the shape's center.
        # (A sum, not an average: same pixel count at every spot, so same ranking, and cheaper)
        return float(distance[top:top + shape_height, left:left + shape_width][shadow].sum())

    # The start spot always counts (even if it hangs off the canvas), so a stuck shape just stays put
    best_spot:Tuple[int, int] = (start_top, start_left)
    start_closeness:Optional[float] = closeness(start_top, start_left)
    best_closeness:float = start_closeness if start_closeness is not None else float('inf')
    visited:set = {best_spot}
    spots:deque = deque([best_spot])
    # (A total of 0 = sitting right on the targets, nothing can beat it, so stop there.
    # BFS order means the first 0 found is already the fewest-steps one)
    while spots and best_closeness > 0:
        top, left = spots.popleft()
        for step_y, step_x in _KING_STEPS:
            next_spot:Tuple[int, int] = (top + step_y, left + step_x)
            if next_spot in visited:
                continue
            visited.add(next_spot)
            next_closeness:Optional[float] = closeness(*next_spot)
            if next_closeness is None:
                continue
            # Strictly closer only, so ties go to the spot the fewest steps away
            if next_closeness < best_closeness:
                best_spot, best_closeness = next_spot, next_closeness
            spots.append(next_spot)

    return best_spot[1] - start_left, best_spot[0] - start_top

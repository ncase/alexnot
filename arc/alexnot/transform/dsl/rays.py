# Rays! The helpers behind shape.add_ray() and shape.add_ray_towards(): trace a 1-pixel-wide ray from a pixel
# in one of 8 directions until it bumps into a (non-ray) shape or the scene's edge, then turn it into a brand-new shape.
# (connect_with_lines() in lines.py borrows make_line_shape, the new-shape builder, too)
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ...parse.apply_parse import shape_coords_to_shape_dict # The same shape-dict builder every parsed shape comes from

# Each direction's (dx, dy) step, in grid coords (y goes DOWN)
RAY_DIRECTION_STEPS:Dict[str, Tuple[int, int]] = {
    'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0),
    'ul': (-1, -1), 'ur': (1, -1), 'dl': (-1, 1), 'dr': (1, 1),
}

# Shortcuts for several directions at once (each one still makes its own ray shape)
RAY_DIRECTION_SHORTCUTS:Dict[str, List[str]] = {
    'lr': ['l', 'r'],
    'ud': ['u', 'd'],
    '+': ['u', 'd', 'l', 'r'],
    'x': ['ul', 'ur', 'dl', 'dr'],
    'all': ['u', 'd', 'l', 'r', 'ul', 'ur', 'dl', 'dr'],
}

# add_ray_towards checks these in order: all 4 straight ones first, THEN the diagonals
STRAIGHT_DIRECTIONS:List[str] = ['u', 'd', 'l', 'r']
DIAGONAL_DIRECTIONS:List[str] = ['ul', 'ur', 'dl', 'dr']


def ray_directions(direction:str) -> List[str]:
    """A direction (or a shortcut like "+") -> the list of single directions it means. Unknown = none."""
    if direction in RAY_DIRECTION_SHORTCUTS:
        return RAY_DIRECTION_SHORTCUTS[direction]
    if direction in RAY_DIRECTION_STEPS:
        return [direction]
    return []


# Takes the origin shape too, so it can leave it out (see below)
def blocker_index_grid(scene, origin_shape_data:Optional[Dict] = None) -> np.ndarray:
    """Scene-sized grid: which shape (its index in scene['shapes']) covers each pixel, -1 = nobody. Rays don't count."""
    scene_height:int = int(scene['height'])
    scene_width:int = int(scene['width'])
    blocker_grid:np.ndarray = np.full((scene_height, scene_width), -1, dtype=np.int32)
    for shape_index, shape in enumerate(scene['shapes']):
        # Rays pass right through each other, and hidden shapes aren't drawn, so neither one blocks anything
        if shape.get('type') == 'ray' or shape.get('hidden', False):
            continue
        # The origin doesn't block its own rays either (Nicky's spec: rays "ignore the shape that called it").
        # Leading-edge pixels never shoot INTO their own shape anyway, but a ray from a concave (e.g. U-shaped)
        # origin can cross back over a part of it, and then it just passes over it instead of stopping
        if shape is origin_shape_data:
            continue
        # Clip the shape's shadow to the scene (a shape can hang off the canvas)
        top:int = int(shape['top'])
        left:int = int(shape['left'])
        shadow:np.ndarray = shape['_shadow']
        source_top:int = max(0, -top)
        source_left:int = max(0, -left)
        source_bottom:int = min(shadow.shape[0], scene_height - top)
        source_right:int = min(shadow.shape[1], scene_width - left)
        if source_top >= source_bottom or source_left >= source_right:
            continue # completely off-canvas
        # Stamp this shape's index onto every pixel it fills
        filled:np.ndarray = shadow[source_top:source_bottom, source_left:source_right] == 1
        blocker_grid[top + source_top:top + source_bottom, left + source_left:left + source_right][filled] = shape_index
    return blocker_grid


def trace_ray(blocker_grid:np.ndarray, start_x:int, start_y:int, direction:str, cut_corners:bool) -> Tuple[List[int], List[int], Set[int]]:
    """Walk from (start_x, start_y) one step at a time. Returns the ray's xs & ys, and which shapes stopped it."""
    dx, dy = RAY_DIRECTION_STEPS[direction]
    scene_height, scene_width = blocker_grid.shape
    ray_xs:List[int] = []
    ray_ys:List[int] = []
    # A start pixel that's OFF the scene (shapes can hang off the canvas, e.g. after a focus() crop)
    # shoots no ray at all. (Without this, the corner check below could index off the grid, or wrap around)
    if not (0 <= start_x < scene_width and 0 <= start_y < scene_height):
        return [], [], set()
    x:int = start_x
    y:int = start_y
    while True:
        next_x:int = x + dx
        next_y:int = y + dy
        # Fell off the scene's edge? The ray's done (nobody stopped it)
        if not (0 <= next_x < scene_width and 0 <= next_y < scene_height):
            return ray_xs, ray_ys, set()
        # Who's in the way? The next pixel itself...
        blocking_shapes:Set[int] = set()
        if blocker_grid[next_y, next_x] >= 0:
            blocking_shapes.add(int(blocker_grid[next_y, next_x]))
        # ...and for a diagonal step WITHOUT cut_corners, the two pixels it squeezes between, too
        # (e.g. moving up-right: the pixel above AND the pixel to the right must both be free).
        # (Both are in-bounds whenever the diagonal pixel is, so no edge check needed)
        if dx != 0 and dy != 0 and not cut_corners:
            for corner_x, corner_y in ((next_x, y), (x, next_y)):
                if blocker_grid[corner_y, corner_x] >= 0:
                    blocking_shapes.add(int(blocker_grid[corner_y, corner_x]))
        if blocking_shapes:
            return ray_xs, ray_ys, blocking_shapes
        # All clear: the ray grows by one pixel
        ray_xs.append(next_x)
        ray_ys.append(next_y)
        x, y = next_x, next_y


def leading_edge_pixels(shape_data:Dict, direction:str) -> List[Tuple[int, int, int]]:
    """The origin's pixels whose next pixel in that direction ISN'T part of it: (scene x, scene y, color digit) each."""
    dx, dy = RAY_DIRECTION_STEPS[direction]
    shadow:np.ndarray = shape_data['_shadow']
    shadow_height, shadow_width = shadow.shape
    # Pad the shadow by 1 so "the next pixel" always exists (outside the bounding box = not part of the shape)
    padded_shadow:np.ndarray = np.pad(shadow, 1)
    next_pixel_filled:np.ndarray = padded_shadow[1 + dy:1 + dy + shadow_height, 1 + dx:1 + dx + shadow_width]
    # Filled here, but the next pixel over is empty = leading edge
    edge_ys, edge_xs = np.nonzero((shadow == 1) & (next_pixel_filled != 1))
    # Each one's own color (for multicolor shapes), and where it sits in the scene
    pixel_colors:np.ndarray = shape_data['_pixels'][edge_ys, edge_xs]
    top:int = int(shape_data['top'])
    left:int = int(shape_data['left'])
    return [(left + int(x), top + int(y), int(color)) for x, y, color in zip(edge_xs, edge_ys, pixel_colors)]


def center_edge_pixel(edge_pixels:List[Tuple[int, int, int]], shape_data:Dict, direction:str) -> List[Tuple[int, int, int]]:
    """Of these leading-edge pixels (scene coords), just the outermost one in line with the shape's center (or none)."""
    dx, dy = RAY_DIRECTION_STEPS[direction]
    box_height, box_width = shape_data['_shadow'].shape
    top:int = int(shape_data['top'])
    left:int = int(shape_data['left'])
    # Diagonals only line up with the center of a SQUARE bounding box
    if dx != 0 and dy != 0 and box_width != box_height:
        return []

    # Is this pixel on the line through the center? (Doubled coords, so an even box's half-pixel center stays an integer.
    # A straight ray from an even-length side has no center pixel at all, so it fires none)
    def in_line_with_center(x:int, y:int) -> bool:
        local_x:int = x - left
        local_y:int = y - top
        if dx == 0:
            return 2 * local_x == box_width - 1
        if dy == 0:
            return 2 * local_y == box_height - 1
        # Up-left & down-right go along the main diagonal, up-right & down-left along the other one
        if dx == dy:
            return local_x == local_y
        return local_x + local_y == box_width - 1
    centered_pixels:List[Tuple[int, int, int]] = [pixel for pixel in edge_pixels if in_line_with_center(pixel[0], pixel[1])]
    if not centered_pixels:
        return []
    # A hollow shape can have several on that line: only the outermost one (furthest along the ray) fires
    return [max(centered_pixels, key=lambda pixel: pixel[0] * dx + pixel[1] * dy)]


def make_line_shape(scene, line_xs:List[int], line_ys:List[int], color_digit:int) -> Optional[Dict]:
    """Some pixels (scene coords) -> a brand-new one-color shape dict, typed like any parsed shape (or None if no pixels)."""
    if not line_xs:
        return None # e.g. a ray blocked right away: no pixels isn't a shape
    # Build it just like a parsed shape (top/left/width/height/area/color/type, _pixels, _shadow, _original_shadow,
    # _hole_grids, _edits_cost, _moves...), from a scene-sized grid that's all the line's color
    scene_grid:np.ndarray = np.full((int(scene['height']), int(scene['width'])), color_digit, dtype=np.int8)
    line_shape:Dict = shape_coords_to_shape_dict(scene_grid, np.array(line_ys), np.array(line_xs))
    line_shape['_edits_cost'] = 1 # Every action = 1 edit (making a new line is one)
    return line_shape


def make_ray_shape(scene, ray_xs:List[int], ray_ys:List[int], color_digit:int) -> Optional[Dict]:
    """The traced pixels -> a new shape dict with type="ray" (or None if the ray has no pixels at all)."""
    ray_shape:Optional[Dict] = make_line_shape(scene, ray_xs, ray_ys, color_digit)
    if ray_shape is not None:
        ray_shape['type'] = 'ray' # Rays get their own type: they never block other rays (see blocker_index_grid)
    return ray_shape

# The ray monkey: probe every direction from every shape's leading-edge pixels (the same pixels add_ray shoots from),
# and whenever the output has that ray drawn in, propose add_ray (and add_ray_towards whatever stopped it).
# When the edge pixel in line with the shape's center fired too, also propose add_ray(from_center=True).
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .monkey_helpers import ShapeActionBatcher
from ..dsl.rays import (blocker_index_grid, trace_ray, leading_edge_pixels, center_edge_pixel, RAY_DIRECTION_STEPS,
                        RAY_DIRECTION_SHORTCUTS, DIAGONAL_DIRECTIONS)

# The cheap probe: a direction "fired" from a pixel if the next 5 pixels that way (or fewer, up to the first
# pixel that's off the grid or not empty in the prediction) are all empty in the prediction and that pixel's color in the output.
# (Full raycasts for every shape & direction were too slow, so this probe decides what fired)
_PROBE_LENGTH:int = 5


# Why it stops early: a strict "all 5 must match" probe never fires on rays between nearby walls (e.g. a dot 4 pixels
# from the wall on both sides). So the probe stops at the first pixel that isn't empty in the prediction (or is off
# the grid), like a ray would. Still all 5 steps at once per direction (no per-pixel loop), and "blocked" is just
# "not empty in the prediction" (no blocker grid). That needs the steps in ORDER, which a correlate count can't give,
# so this uses shifted comparisons
def _probe_fires_grids(probe_matches:np.ndarray, empty_pixels:np.ndarray) -> Dict[str, np.ndarray]:
    """Per direction, a boolean grid: from each pixel, do the next ≤5 pixels (up to the first blocked one) all match?"""
    grid_height, grid_width = probe_matches.shape
    # Pad by 5 with "not empty, not matching", so stepping off the grid = blocked
    padded_matches:np.ndarray = np.pad(probe_matches, _PROBE_LENGTH, constant_values=False)
    padded_empty:np.ndarray = np.pad(empty_pixels, _PROBE_LENGTH, constant_values=False)
    fires_grids:Dict[str, np.ndarray] = {}
    for direction, (dx, dy) in RAY_DIRECTION_STEPS.items():
        # Every pixel's probe pixel at this step = the padded grid, shifted by step * (dx, dy)
        def at_step(padded_grid:np.ndarray, step:int) -> np.ndarray:
            top:int = _PROBE_LENGTH + dy * step
            left:int = _PROBE_LENGTH + dx * step
            return padded_grid[top:top + grid_height, left:left + grid_width]
        # The first step has to match (blocked right away = no ray = doesn't fire)
        still_going:np.ndarray = at_step(padded_matches, 1).copy()
        failed:np.ndarray = np.zeros_like(still_going)
        for step in range(2, _PROBE_LENGTH + 1):
            # An empty pixel that's the wrong color in the output = this ray isn't drawn. (A blocked one just ends it)
            failed |= still_going & at_step(padded_empty, step) & ~at_step(padded_matches, step)
            still_going &= at_step(padded_matches, step)
        fires_grids[direction] = at_step(padded_matches, 1) & ~failed
    return fires_grids


# Cache for leading_edge_pixels: the same shape gets asked the same question for every parent, every round.
# Stored in the shape's OWN coords (top-left = 0, 0), keyed by everything the answer depends on: the direction and the
# shadow & pixels (their bytes, shape and dtype). Not keyed on the shape dict itself, since parents copy & change those.
# The scene offset (left, top) gets added back after, so a moved shape still hits the cache
_LOCAL_LEADING_EDGES_CACHE:Dict[tuple, List[Tuple[int, int, int]]] = {}
# Judgment call: a hardcoded cap, so a blast worker solving many levels doesn't grow it forever.
# When full, it just starts over empty (slower for a moment, never wrong)
_MAX_CACHED_LEADING_EDGES:int = 50_000


def _cached_leading_edge_pixels(shape:dict, direction:str) -> List[Tuple[int, int, int]]:
    """Exactly rays.leading_edge_pixels(shape, direction), but computed once per shadow/pixels/direction."""
    shadow:np.ndarray = shape['_shadow']
    pixels:np.ndarray = shape['_pixels']
    cache_key:tuple = (direction, shadow.shape, shadow.dtype.str, shadow.tobytes(),
                       pixels.shape, pixels.dtype.str, pixels.tobytes())
    local_edge_pixels:Optional[List[Tuple[int, int, int]]] = _LOCAL_LEADING_EDGES_CACHE.get(cache_key)
    if local_edge_pixels is None:
        # Not seen yet: ask the real function, with the shape moved to (0, 0) so the answer is in its own coords
        if len(_LOCAL_LEADING_EDGES_CACHE) >= _MAX_CACHED_LEADING_EDGES:
            _LOCAL_LEADING_EDGES_CACHE.clear()
        local_edge_pixels = leading_edge_pixels({'_shadow': shadow, '_pixels': pixels, 'top': 0, 'left': 0}, direction)
        _LOCAL_LEADING_EDGES_CACHE[cache_key] = local_edge_pixels
    # Shift back to where the shape sits in the scene (same int() as leading_edge_pixels)
    top:int = int(shape['top'])
    left:int = int(shape['left'])
    return [(left + local_x, top + local_y, color_digit) for local_x, local_y, color_digit in local_edge_pixels]


def shapes_add_ray(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice rays drawn in the output (straight out of a shape, in its color), propose add_ray / add_ray_towards
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes.
        # One batch per fired direction SET, so shapes that fired the same set (in any demo) get one selector together
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            predicted_grid:np.ndarray = predicted_scene['_raw']
            output_grid:np.ndarray = pair['output']['_raw']

            # Rays can't resize the grid, so skip pairs where the sizes differ
            if predicted_grid.shape != output_grid.shape:
                continue

            # Where's everyone? (Same blocker grid add_ray builds: rays & hidden shapes don't block.)
            # Only add_ray_towards needs it, so it's built lazily, the first time a ray fires
            scene_blocker_grid:Optional[np.ndarray] = None

            # "Empty" = the predicted grid's background color there.
            # (Judgment call: so pixels covered by ANY shape, rays included, block the probe)
            scene_height, scene_width = predicted_grid.shape
            empty_pixels:np.ndarray = predicted_grid == int(predicted_scene['background'][1:])

            # The probe's fires grids, per origin color (a ray has to be its start pixel's color in the output).
            # Built the first time a color shows up, so at most 10 per pair
            fires_by_color:Dict[int, Dict[str, np.ndarray]] = {}

            # Every shape is a possible ray origin, except rays themselves & hidden shapes
            # (speed: they're never the origin we're looking for)
            for shape_index, shape in enumerate(predicted_scene["shapes"]):
                if shape.get('type') == 'ray' or shape.get('hidden', False):
                    continue

                # Which directions fired? The probe can't tell cut_corners apart, so a fired diagonal goes in
                # BOTH lists (the scorer picks). Same for the directions that were blocked right away
                # (an empty ray draws nothing, so it fits ANY direction set)
                fired_directions:Dict[bool, List[str]] = {False: [], True: []}
                empty_directions:Dict[bool, List[str]] = {False: [], True: []}
                # The directions where the pixel in line with the center fired (for add_ray(from_center=True)).
                # A dot's only pixel IS its center, so for dots that'd just repeat the plain proposals
                is_dot:bool = shape['_shadow'].size == 1
                center_fired_directions:List[str] = []
                # This origin's blocker grid (itself erased), built only if one of its directions fires
                blocker_grid:Optional[np.ndarray] = None
                for direction in RAY_DIRECTION_STEPS:
                    dx, dy = RAY_DIRECTION_STEPS[direction]
                    direction_fired:bool = False
                    # Empty = every leading-edge pixel is blocked right away (its first step is off-grid or not empty)
                    direction_empty:bool = True
                    # Same answer as leading_edge_pixels, from the cache (see _cached_leading_edge_pixels)
                    edge_pixels:List[Tuple[int, int, int]] = _cached_leading_edge_pixels(shape, direction)
                    for start_x, start_y, pixel_color_digit in edge_pixels:
                        # An off-canvas start pixel shoots no ray at all (like trace_ray)
                        if not (0 <= start_x < scene_width and 0 <= start_y < scene_height):
                            continue
                        # Does a ray of THIS pixel's color fire from here? (Just a lookup in that color's fires grid)
                        if pixel_color_digit not in fires_by_color:
                            probe_matches:np.ndarray = empty_pixels & (output_grid == pixel_color_digit)
                            fires_by_color[pixel_color_digit] = _probe_fires_grids(probe_matches, empty_pixels)
                        if fires_by_color[pixel_color_digit][direction][start_y, start_x]:
                            direction_fired = True
                        # Is its first step free?
                        next_x:int = start_x + dx
                        next_y:int = start_y + dy
                        if 0 <= next_x < scene_width and 0 <= next_y < scene_height and empty_pixels[next_y, next_x]:
                            direction_empty = False

                    # Did the centered pixel fire too? (It's one of the edge pixels, so its color's fires grid is built.
                    # Only worth asking if some pixel fired at all)
                    if direction_fired and not is_dot:
                        for start_x, start_y, pixel_color_digit in center_edge_pixel(edge_pixels, shape, direction):
                            if 0 <= start_x < scene_width and 0 <= start_y < scene_height \
                                    and fires_by_color[pixel_color_digit][direction][start_y, start_x]:
                                center_fired_directions.append(direction)

                    # Judgment call: the direction fires if ANY of its leading-edge pixels fires
                    # (a wide bar's rays can be partly blocked close by)
                    is_diagonal:bool = direction in DIAGONAL_DIRECTIONS
                    if direction_fired:
                        fired_directions[False].append(direction)
                        if is_diagonal:
                            fired_directions[True].append(direction)
                    elif direction_empty:
                        empty_directions[False].append(direction)
                        if is_diagonal:
                            empty_directions[True].append(direction)
                    if not direction_fired:
                        continue

                    # ALSO: shoot towards whoever stopped this ray. Only for the fired directions, so these
                    # real raycasts are few. (The batcher finds what those targets have in common)
                    if scene_blocker_grid is None:
                        scene_blocker_grid = blocker_index_grid(predicted_scene)
                    if blocker_grid is None:
                        # The origin doesn't block its own rays (like add_ray), so erase it from a copy.
                        # (Cheaper than rebuilding the grid per shape. Only differs from add_ray where two shapes overlap:
                        # there the other shape's pixels get erased too, if this one was stamped on top)
                        blocker_grid = scene_blocker_grid.copy()
                        blocker_grid[blocker_grid == shape_index] = -1
                    # Diagonals get both cut_corners versions, straight ones just the plain one
                    for cut_corners in ((False, True) if is_diagonal else (False,)):
                        _, _, _, blocking_shapes = _trace_all_rays(blocker_grid, shape, direction, cut_corners)
                        towards_action:str = 'add_ray_towards(cut_corners=True)' if cut_corners else 'add_ray_towards()'
                        for blocking_shape_index in blocking_shapes:
                            shape_actions.add(shape, towards_action, predicted_scene['shapes'][blocking_shape_index])

                # Propose the plain version (cut_corners=False), if anything fired.
                # (Via _add_ray_proposals, so the "+ the empty ones" set gets proposed too)
                if fired_directions[False]:
                    _add_ray_proposals(shape_actions, shape, fired_directions[False], empty_directions[False], cut_corners=False)

                # And if any diagonal fired with cut_corners=True, that version too (even if it's the same set:
                # we want the runner-up for attempt 2)
                if fired_directions[True]:
                    straight_fired:List[str] = [direction for direction in fired_directions[False] if direction not in DIAGONAL_DIRECTIONS]
                    straight_empty:List[str] = [direction for direction in empty_directions[False] if direction not in DIAGONAL_DIRECTIONS]
                    _add_ray_proposals(shape_actions, shape, straight_fired + fired_directions[True],
                                       straight_empty + empty_directions[True], cut_corners=True)

                # And the from-center version: one ray per direction, from the pixel in line with the center.
                # (Plain cut_corners only, to keep the proposal count down)
                if center_fired_directions:
                    _add_ray_proposals(shape_actions, shape, center_fired_directions, empty_directions[False],
                                       cut_corners=False, from_center=True)

        # Flush, append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates


# A non-dot origin shoots one ray per leading-edge pixel (rays.leading_edge_pixels, like add_ray_towards),
# so a direction's "ray" is all of those rays together. (Each ray shape costs 1 edit, so a 5-wide bar's cost 5)
def _trace_all_rays(blocker_grid:np.ndarray, shape:dict, direction:str, cut_corners:bool) -> Tuple[List[int], List[int], List[int], Set[int]]:
    """Trace every leading-edge pixel's ray in one direction: all their xs, ys, colors, and who stopped them."""
    all_xs:List[int] = []
    all_ys:List[int] = []
    all_color_digits:List[int] = []
    all_blocking_shapes:Set[int] = set()
    # Same answer as leading_edge_pixels, from the cache (see _cached_leading_edge_pixels)
    for start_x, start_y, pixel_color_digit in _cached_leading_edge_pixels(shape, direction):
        # (Off-canvas start pixels just give an empty ray, thanks to the guard in trace_ray)
        ray_xs, ray_ys, blocking_shapes = trace_ray(blocker_grid, start_x, start_y, direction, cut_corners)
        all_xs += ray_xs
        all_ys += ray_ys
        # Each ray pixel should be the color of the pixel it shot out of
        all_color_digits += [pixel_color_digit] * len(ray_xs)
        all_blocking_shapes |= blocking_shapes
    return all_xs, all_ys, all_color_digits, all_blocking_shapes


# E.g. a dot right up against a wall can't shoot towards it, so its fired set is "u, l, r", not "+".
# On its own, it'd never share a batch (= a selector) with the dots that fired all 4. So ALSO propose the fired set
# plus the empty directions that complete the smallest shortcut: still exactly what this shape needs, since empty rays draw nothing.
# (Not "fired + ALL empty directions": that also grabs empty diagonals, since a wall blocks those too,
# so {u, l, r} would become {u, d, l, r, dl, dr}, which still isn't "+")
def _add_ray_proposals(shape_actions:ShapeActionBatcher, shape:dict, fired:List[str], empty:List[str], cut_corners:bool,
                       from_center:bool = False) -> None:
    """Batch add_ray for the fired directions, and (if empty rays complete a shortcut) for that shortcut too."""
    shape_actions.add(shape, _add_ray_action(fired, cut_corners, from_center))
    # Add only the empty directions that complete the SMALLEST shortcut containing the fired set
    # (e.g. {u, l, r} + an empty d = "+"). If the fired set already IS a shortcut, there's nothing to add
    completed_shortcut:Optional[List[str]] = _smallest_completing_shortcut(fired, empty)
    if completed_shortcut is not None and set(completed_shortcut) != set(fired):
        shape_actions.add(shape, _add_ray_action(completed_shortcut, cut_corners, from_center))


def _smallest_completing_shortcut(fired:List[str], empty:List[str]) -> Optional[List[str]]:
    """The smallest shortcut that contains every fired direction, where every extra direction was empty (or None)."""
    for shortcut_directions in sorted(RAY_DIRECTION_SHORTCUTS.values(), key=len):
        extra_directions:Set[str] = set(shortcut_directions) - set(fired)
        if set(fired) <= set(shortcut_directions) and extra_directions <= set(empty):
            return shortcut_directions
    return None


def _add_ray_action(directions:List[str], cut_corners:bool, from_center:bool = False) -> str:
    """The fired directions -> add_ray action text: one shortcut if they're exactly one, else one line per direction."""
    cut_corners_text:str = ', cut_corners=True' if cut_corners else ''
    from_center_text:str = ', from_center=True' if from_center else ''

    # Exactly a shortcut's set? Then one line
    for shortcut, shortcut_directions in RAY_DIRECTION_SHORTCUTS.items():
        if set(directions) == set(shortcut_directions):
            # (cut_corners only matters for diagonals, so don't spend tokens on it for "lr" / "ud" / "+")
            has_diagonal:bool = any(direction in DIAGONAL_DIRECTIONS for direction in shortcut_directions)
            # ("all" always shoots from the center, so it doesn't need from_center either)
            return f'add_ray(direction="{shortcut}"{cut_corners_text if has_diagonal else ""}{"" if shortcut == "all" else from_center_text})'

    # Otherwise, one add_ray line per direction (the batcher's template puts "shape." on the first one only,
    # so the rest get their own "\n\tshape.")
    lines:List[str] = []
    for direction in directions:
        line_cut_corners_text:str = cut_corners_text if direction in DIAGONAL_DIRECTIONS else ''
        lines.append(f'add_ray(direction="{direction}"{line_cut_corners_text}{from_center_text})')
    return '\n\tshape.'.join(lines)

# Small helpers shared by the physics files, plus the placement search (shared by fitting & attraction)
from typing import Any, Callable, Dict, List, Tuple
import random # The placement search's random tie-breaks (and _direction_away's, _closest_target's)
import numpy as np
from ...grid import _shapes_hit_test # _direction_away: are we touching the target?
from ...config import FIT_SEARCH_MAX_WIDTH # The placement search's width caps
from ...config import FIT_SEARCH_WIDE_DEPTH, FIT_SEARCH_NARROW_WIDTH
from ...utils.deadline import check_deadline # The placement search is the heavy loop



####################################################################
## HELPERS
####################################################################


# _pixel_gap prunes pixels first only when the shapes have more pixel pairs than this (measured: below it, pruning is slower)
_PIXEL_GAP_PRUNE_MIN_PAIRS:int = 5000

def _bounds_gap(a:Dict, b:Dict) -> int:
    """Chebyshev gap between two shapes' bounding boxes (0 = overlapping, 1 = touching): a cheap lower bound on _pixel_gap."""
    # right/bottom are one past the last pixel, hence the +1s
    gap_x:int = max(b['left'] - a['right'] + 1, a['left'] - b['right'] + 1, 0)
    gap_y:int = max(b['top'] - a['bottom'] + 1, a['top'] - b['bottom'] + 1, 0)
    return max(gap_x, gap_y)

def _pixel_gap(a:Dict, b:Dict) -> int:
    """
    Smallest Chebyshev distance from any of a's pixels to any of b's (0 = overlapping, 1 = touching).
    (Taxicab = rook's diamond circles, Chebyshev = king's square circles)
    So a long line counts as close when its END is close, even if its center is far away.
    """
    # Every pixel of each shape, both in b's local coords (so only a needs shifting)
    a_ys, a_xs = np.nonzero(a['_shadow'])
    b_ys, b_xs = np.nonzero(b['_shadow'])
    a_ys = a_ys + (int(a['top']) - int(b['top']))
    a_xs = a_xs + (int(a['left']) - int(b['left']))

    # Big shapes would make a huge all-pairs grid, so first prune to the pixels that could possibly be closest.
    # (Small ones skip this: pruning costs ~16us, more than all-pairs below ~5000 pairs)
    if a_ys.size * b_ys.size > _PIXEL_GAP_PRUNE_MIN_PAIRS:
        # How far is each of a's pixels from b's bounding box? (Never more than its distance to any of b's pixels)
        b_height, b_width = b['_shadow'].shape
        a_to_b_box:np.ndarray = np.maximum(np.maximum(-a_ys, a_ys - (b_height - 1)), np.maximum(np.maximum(-a_xs, a_xs - (b_width - 1)), 0))
        # An upper bound on the answer: a's pixel nearest b's box, measured to all of b's real pixels
        nearest:int = int(a_to_b_box.argmin())
        upper_bound:int = int(np.maximum(np.abs(b_ys - a_ys[nearest]), np.abs(b_xs - a_xs[nearest])).min())
        # Keep only a's pixels within that bound of b's box, and b's pixels within that bound of a's (kept pixels') box
        keep_a:np.ndarray = a_to_b_box <= upper_bound
        a_ys, a_xs = a_ys[keep_a], a_xs[keep_a]
        keep_b:np.ndarray = ((b_ys >= a_ys.min() - upper_bound) & (b_ys <= a_ys.max() + upper_bound) &
                             (b_xs >= a_xs.min() - upper_bound) & (b_xs <= a_xs.max() + upper_bound))
        b_ys, b_xs = b_ys[keep_b], b_xs[keep_b]

    # Every remaining pixel pair's Chebyshev distance, keep the smallest
    distances:np.ndarray = np.maximum(np.abs(a_ys[:, None] - b_ys[None, :]), np.abs(a_xs[:, None] - b_xs[None, :]))
    return int(distances.min())

def _direction_from_bounds(shape:Dict, target:Dict) -> Tuple[int, int]:
    """Gridworld direction from shape to target, based on bounds. Returns (dx, dy) in {-1,0,1}."""
    dx:int = 0
    dy:int = 0
    if shape['right'] < target['left']:
        dx += 1
    if target['right'] < shape['left']:
        dx -= 1
    if shape['bottom'] < target['top']:
        dy += 1
    if target['bottom'] < shape['top']:
        dy -= 1
    return dx, dy

def _direction_towards(shape:Dict, target:Dict) -> Tuple[int, int]:
    """Gridworld direction (dx, dy) from shape towards target, comparing their edges. Works even when they touch."""
    # Head l/r/u/d based on whether the shape's bounds stick out past the target's on that side
    dx:int = 0
    dy:int = 0
    if shape['left'] < target['left']:
        dx += 1
    if target['right'] < shape['right']:
        dx -= 1
    if shape['top'] < target['top']:
        dy += 1
    if target['bottom'] < shape['bottom']:
        dy -= 1
    return dx, dy

def _direction_away(shape:Dict, targets:List[Dict]) -> Tuple[int, int]:
    """Which way (dx, dy) is "away from the targets"? Away from the sides we touch them on, else away from the closest."""
    # Touching targets straight on (one step left/right/up/down would bump into them)? Add up one "push" per
    # (target, side) pair touching us, so a side touched by 3 dots pushes 3 times as hard as a side touched by 1.
    # (Diagonal-only contact doesn't count, like 4-neighbors)
    push_x:int = 0
    push_y:int = 0
    touched_any:bool = False
    for step_x, step_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        # A copy of our bounds, one step over (the hit test only reads the bounds & shadow)
        stepped:Dict = {**shape, 'left': shape['left'] + step_x, 'right': shape['right'] + step_x,
                        'top': shape['top'] + step_y, 'bottom': shape['bottom'] + step_y}
        touched_count:int = len(_shapes_hit_test(stepped, targets))
        touched_any = touched_any or touched_count > 0
        # Pushed the opposite way from the side we touch
        push_x -= step_x * touched_count
        push_y -= step_y * touched_count

    # Touching something: round the total push to the nearest of l/r/u/d (a straight step, never diagonal).
    # A dead-even diagonal push picks one of its two sides randomly.
    if push_x != 0 or push_y != 0:
        if abs(push_x) > abs(push_y) or (abs(push_x) == abs(push_y) and random.random() < 0.5):
            return (1 if push_x > 0 else -1), 0
        return 0, (1 if push_y > 0 else -1)

    # Touched, but the pushes cancel out (e.g. squeezed evenly from left AND right)? Stay put.
    # (Hasn't happened on any level yet)
    if touched_any:
        return 0, 0

    # Not touching anything straight on: head the opposite way from where the closest target's bounds are
    towards_x, towards_y = _direction_from_bounds(shape, _closest_target(shape, targets))
    return -towards_x, -towards_y

def _closest_target(shape:Dict, targets:List[Dict]) -> Dict:
    """The closest of these targets by pixel gap (random among equally close ones)."""
    return random.choice(_closest_targets(shape, targets))

def _closest_targets(shape:Dict, targets:List[Dict]) -> List[Dict]:
    """All the targets tied for the smallest pixel gap to the shape."""
    # Nearest bounding boxes first: the pixel gap is never smaller than the bounds gap, so once a target's
    # bounds gap is past the best pixel gap so far, it (and everyone after it) can't win, and we skip its pixels
    bounds_gaps:List[Tuple[int, Dict]] = sorted(((_bounds_gap(shape, target), target) for target in targets), key=lambda pair: pair[0])
    closest_distance:float = float('inf')
    closest:List[Dict] = []
    for bounds_gap, target in bounds_gaps:
        if bounds_gap > closest_distance:
            break
        distance:int = _pixel_gap(shape, target)
        if distance < closest_distance:
            closest_distance, closest = distance, [target]
        elif distance == closest_distance:
            closest.append(target)
    return closest


# Add a shape's shadow to a scene-sized grid (clipping whatever's off the canvas)
def _stamp_shadow(grid:np.ndarray, shape:Dict) -> None:
    grid_height, grid_width = grid.shape
    shadow:np.ndarray = shape['_shadow']
    top:int = int(shape['top'])
    left:int = int(shape['left'])

    # The part of the shape that's actually on the canvas, in grid coords...
    clip_top:int = max(top, 0)
    clip_left:int = max(left, 0)
    clip_bottom:int = min(top + shadow.shape[0], grid_height)
    clip_right:int = min(left + shadow.shape[1], grid_width)
    if clip_bottom <= clip_top or clip_right <= clip_left:
        return # totally off-canvas

    # ...and the same part, in the shape's own local coords
    grid[clip_top:clip_bottom, clip_left:clip_right] += shadow[clip_top - top:clip_bottom - top, clip_left - left:clip_right - left]



####################################################################
## THE PLACEMENT SEARCH (shared by apply_fit & apply_attract)
####################################################################


# Fitting and attraction share this search. What differs gets passed in: how to score every spot (score_spots),
# which shapes count as interchangeable (same_shape_key: same shadow for fitting, same pixels for attraction),
# the memo, and the margin. score_spots returns (moving shape, scores, top of scores[0, 0], left of scores[0, 0])
# for each moving shape.
# It's a depth-first search over placement orders. At each step: score every moving shape at every spot, try the
# (shape, spot)s within the margin of the step's best (and scoring >= min_score), recurse with that shape as a
# new target, and keep whichever whole sequence has the highest TOTAL score.
# Capped: at most FIT_SEARCH_MAX_WIDTH options per step, best score first, ties in RANDOM order.
# depth = how many shapes this search has placed so far, for the wide-then-narrow width cap.
# (The memo doesn't need it in its key: within one search, the shapes left to place already pin down the depth)
# min_score = the lowest score that still counts as a fit. Fitting's pinch counts are whole-ish numbers, so 1.
# Attract's heatmap scores are fractions, so it passes a tiny number.
def _search_placement_sequence(moving_shapes:List, targets:List, scene_height:int, scene_width:int,
                               score_spots:Callable, same_shape_key:Callable, memo:Dict, margin:float,
                               depth:int=0, min_score:float=1) -> Tuple[float, List[Tuple[Any, int, int]]]:
    """Best (total score, [(moving shape, top, left), ...]) for placing these moving shapes against these targets."""

    # Out of time for this program? Raises ProgramTimedOut, so the whole program just fails (nothing gets
    # cached either: the exception skips apply_fit's cache write)
    check_deadline()

    # Been at this exact layout before (same shapes left to place, every target at the same spot)? Same answer.
    # (Ids are fine here: apply_fit empties the memo before & after each search, and all these shapes live that long)
    memo_key:tuple = (frozenset(id(moving_shape) for moving_shape in moving_shapes),
                      frozenset((id(target), target.data['top'], target.data['left']) for target in targets))
    if memo_key in memo:
        return memo[memo_key]

    # Everyone's scores, this step. Nobody can pinch anywhere? Then the rest stay put, and we're done
    scored_spots:List[Tuple[Any, np.ndarray, int, int]] = score_spots(moving_shapes, targets, scene_height, scene_width)
    best_score:float = max((float(scores.max()) for _, scores, _, _ in scored_spots), default=0.0) # Float, for the edge pinches
    if best_score < min_score:
        memo[memo_key] = (0, []) # Dead ends get remembered too (their scoring isn't free)
        return 0, []

    # Which (shape, spot) pairs are worth trying? Everything within the margin of this step's best (but at least min_score)
    worth_trying_score:float = max(best_score - margin, min_score)

    best_total:float = -1 # Float, since scores can be fractions
    best_sequence:List[Tuple[Any, int, int]] = []
    # Same-shape skip: unplaced moving shapes aren't obstacles, so moving shapes with the same same_shape_key
    # are interchangeable here. Only try the FIRST of each per step, so "which of the 4 identical dots goes first"
    # doesn't multiply the search by 4. (Fitting only looks at shadows; attraction looks at colors too, so it
    # needs the same PIXELS)
    shapes_tried:set = set()
    # First just COLLECT every option that passes the filters, as (score, moving shape, top, left)...
    options:List[Tuple[float, Any, int, int]] = []
    for moving_shape, scores, scores_top, scores_left in scored_spots:
        shape_key:tuple = same_shape_key(moving_shape)
        if shape_key in shapes_tried:
            continue
        shapes_tried.add(shape_key)

        # np.nonzero goes row by row, so this is scan order (top-to-bottom, then left-to-right)
        # scores_top/scores_left: where the score grid starts (0, 0 for fitting, the canvas corner for attraction)
        for spot_row, spot_column in zip(*np.nonzero(scores >= worth_trying_score)):
            # float(), not int(): int() would silently drop the 0.1 edge pinches
            options.append((float(scores[spot_row, spot_column]), moving_shape,
                            int(spot_row) + scores_top, int(spot_column) + scores_left))

    # ...then keep only the best FIT_SEARCH_MAX_WIDTH of them: best score first, ties in random order
    # (shuffle, then a stable sort by score). Random on purpose (Nicky): if a level only solves with one lucky
    # ordering, it should fail loudly across seeds, not pass silently because of scan order
    random.shuffle(options)
    options.sort(key=lambda option: -option[0])
    # Wide for the first few placements (the big early choices), then narrower
    max_width:int = FIT_SEARCH_MAX_WIDTH if depth < FIT_SEARCH_WIDE_DEPTH else FIT_SEARCH_NARROW_WIDTH
    options = options[:max_width]

    for spot_score, moving_shape, spot_top, spot_left in options:
        # Try it: put the shape there (just its position, no edits counted: it's only a "what if"), and it's a
        # target now. Then the best way to fit everyone else, from there
        old_left, old_top = moving_shape.data['left'], moving_shape.data['top']
        moving_shape._set_position(left=spot_left, top=spot_top)
        rest_total, rest_sequence = _search_placement_sequence(
            [other for other in moving_shapes if other is not moving_shape], targets + [moving_shape],
            scene_height, scene_width, score_spots, same_shape_key, memo, margin,
            depth + 1, min_score) # One more placed (and the same min_score all the way down)
        moving_shape._set_position(left=old_left, top=old_top) # ...and undo the "what if"

        # Best whole sequence so far? (Strictly better, so the first one found wins ties, and options are
        # in random order among ties)
        total:float = spot_score + rest_total # Score saved with the option (scores is from the loop above)
        if total > best_total:
            best_total = total
            best_sequence = [(moving_shape, spot_top, spot_left)] + rest_sequence

    # Remember it for this layout, in case another placement order lands here too
    memo[memo_key] = (best_total, best_sequence)
    return best_total, best_sequence


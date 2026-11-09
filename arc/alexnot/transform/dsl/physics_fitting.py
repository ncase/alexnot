# Jigsaw fitting: apply_fit, its cache & DFS memo, and its spot scoring
from typing import Dict, List, Tuple
import numpy as np
import scipy.signal # correlate2d: slide a shape over a grid, count overlaps at every spot in one go
from .top_level_methods import DSL_CONFIG, all
from .shape import Shape
from ...config import FIT_SEARCH_MARGIN # How far below each step's best apply_fit's search still looks
from ...config import FIT_EDGE_PINCH_WEIGHT, FIT_ALIGNMENT_BONUS, FIT_ALIGNMENT_MIN_AREA # Edge pinches & always-on alignment
from .physics_utils import _stamp_shadow
from .physics_utils import _search_placement_sequence # The DFS, shared with apply_attract


# Jigsaw fitting! Every moving shape (flagged with shape.fit()) slides into the GAPS of the target shapes (all the others).
# Shapes flagged with fit() are "moving", everything they fit against is a "target".
# Each spot gets scored by CORNER PINCHES, counted both ways (target pinches moving + moving pinches target),
# plus an alignment bonus and a small edge tie-breaker (see _score_fit_spots). Then a depth-first search over WHOLE
# placement orders picks the order with the highest TOTAL score, and each placed shape becomes a target for the next.
# (Translation only for now.)
# TODO: rotations/flips eventually, for pieces that need turning to fit
# The alignment bonus is always on, but only for moving shapes with area >= FIT_ALIGNMENT_MIN_AREA (that's what
# stops identical 1x1 pieces from stacking into lines). As an opt-in parameter instead, plain apply_fit() could
# pass a level's demos on lucky seeds, then win on simplicity and get the test wrong.
def apply_fit():
    """
    Example:

    for shape in all(rank_area_GTE=1):
        shape.fit() # data stored; shape['_moving_fit'] = True
    apply_fit() # NOW they all fit into the biggest shape's gaps, like a jigsaw
    """

    scene:Dict = DSL_CONFIG['scene']
    scene_width:int = int(scene['width'])
    scene_height:int = int(scene['height'])

    # Who's moving, and who's the target being fit against? (Hidden shapes aren't on the canvas, so they're neither)
    moving_shapes:List[Shape] = []
    targets:List[Shape] = []
    for shape in all():
        is_moving:bool = shape.data.pop('_moving_fit', False) # consume the flag
        if shape.data.get('hidden'):
            continue
        if is_moving:
            moving_shapes.append(shape)
        else:
            targets.append(shape)

    # Nobody's moving, or nothing to fit against? Quit!
    if not moving_shapes or not targets:
        return

    # Not greedy: a depth-first search over WHOLE placement orders. Each round, try every (moving shape, spot)
    # within FIT_SEARCH_MARGIN points of that round's best, and keep the order with the highest TOTAL score.
    # (Greedy got stuck on ties: e.g. yellow taking red's turn, when yellow only fits against red)

    # Cached! Lots of candidate programs pick the SAME moving shapes in the same scene (just with
    # different selectors), and the search only depends on where everyone is & what shape they are. So: same
    # moving shapes + same targets (position & shadow, in the same order, since order breaks ties) = same answer.
    fit_cache_key:tuple = (scene_height, scene_width,
                           tuple(_fit_cache_shape_key(moving_shape) for moving_shape in moving_shapes),
                           tuple(_fit_cache_shape_key(target) for target in targets))
    if fit_cache_key not in _FIT_CACHE:
        # The DFS memo's keys are object ids, which Python reuses once objects are gone, so it's ONLY valid
        # inside this one search: fresh before, emptied after (even if the deadline cuts the search short)
        _FIT_MEMO.clear()
        try:
            best_total, best_sequence = _search_fit_sequence(moving_shapes, targets, scene_height, scene_width)
        finally:
            _FIT_MEMO.clear()
        # Store it by INDEX into moving_shapes, not the Shape objects: every program runs on its own copy of the scene
        _FIT_CACHE[fit_cache_key] = [(moving_shapes.index(moving_shape), spot_top, spot_left)
                                     for moving_shape, spot_top, spot_left in best_sequence]

    # Now actually move them, in that order (move() adds the +1 edit, like a fall, if it actually moved).
    # Anyone not in the sequence never found a spot with a pinch, so they stay put.
    for moving_index, spot_top, spot_left in _FIT_CACHE[fit_cache_key]:
        moving_shape:Shape = moving_shapes[moving_index]
        moving_shape.move(x=spot_left - moving_shape.data['left'], y=spot_top - moving_shape.data['top'])


# apply_fit's cache: (scene size, moving shapes, targets) -> [(moving index, top, left), ...].
# search_transform clears it every round of evolution, so it never grows forever.
_FIT_CACHE:Dict[tuple, List[Tuple[int, int, int]]] = {}

def clear_fit_cache() -> None:
    """Forget every cached apply_fit answer."""
    _FIT_CACHE.clear()

# The DFS memo, inside ONE search: (which moving shapes are left, where every target is) -> the best
# (total, sequence) for placing the rest. Different placement orders often land on the same layout (on one
# big jigsaw test, up to ~96% of DFS visits were repeats), and the best way to fit the rest only depends on that layout.
# Heads up: with the random tie-breaks, a cached answer freezes ONE random draw per layout for the rest of the search.
_FIT_MEMO:Dict[tuple, Tuple[float, list]] = {}

def _fit_cache_shape_key(shape:Shape) -> tuple:
    """Everything about a shape that apply_fit's search looks at: where it is, and its shadow."""
    shadow:np.ndarray = shape.data['_shadow']
    return (int(shape.data['top']), int(shape.data['left']), shadow.shape, shadow.tobytes())


# One round of apply_fit's scoring, for the search to call: every moving shape's score at every (top, left)
# spot, given the current targets. (Score 0 = overlaps, or no pinch.) Scores are FLOATS: edge pinches are worth
# FIT_EDGE_PINCH_WEIGHT each. Each entry is (moving shape, scores, 0, 0): the shared search wants to know where
# the score grid starts.
def _score_fit_spots(moving_shapes:List[Shape], targets:List[Shape], scene_height:int,
                     scene_width:int) -> List[Tuple[Shape, np.ndarray, int, int]]:


    # Where are the target shapes' actual pixels? A fit spot must not overlap ANY of these.
    # (Still-moving shapes don't count as obstacles: they're about to move anyway)
    target_pixels:np.ndarray = np.zeros((scene_height, scene_width), dtype=np.int32)
    for target in targets:
        _stamp_shadow(target_pixels, target.data)

    # CORNER PINCHES. Which pixels count as target? Any target shape's pixel. A 1-px border around the scene =
    # the off-canvas neighbors. (A moving shape always lands inside the scene, so its neighbors are never more
    # than 1 pixel off the canvas)
    # The scene edge does NOT count as target here (border of 0s, not 1s): with the search, a layout stacking
    # pieces against the scene edge could otherwise outscore the right one. (It comes back below, as a weak tie-breaker)
    target_with_border:np.ndarray = np.zeros((scene_height + 2, scene_width + 2), dtype=np.int32)
    target_with_border[1:-1, 1:-1] = (target_pixels > 0)

    # Target pinches moving, step 1: for every in-scene pixel, how many of its 4 corners are pinched by
    # target? (Left AND top = +1, top AND right = +1, right AND bottom = +1, bottom AND left = +1, separately.)
    # This doesn't depend on the moving shape, so it's computed once per round. (Shifted copies of the bordered grid)
    # (Named *_pixels, since the alignment loop below uses "target_left"/"target_right" as ints for each target's box)
    target_left_pixels:np.ndarray = target_with_border[1:-1, :-2]
    target_right_pixels:np.ndarray = target_with_border[1:-1, 2:]
    target_above:np.ndarray = target_with_border[:-2, 1:-1]
    target_below:np.ndarray = target_with_border[2:, 1:-1]
    target_corner_counts:np.ndarray = ((target_left_pixels & target_above) + (target_above & target_right_pixels)
                                       + (target_right_pixels & target_below) + (target_below & target_left_pixels))

    # The scene edge is back, as a weak tie-breaker: count the corners again with the edge as wall too (border
    # of 1s), and the difference is the corners the edge helps pinch. Only for "target pinches moving": a moving shape
    # can never pinch the edge back (that'd need it on two perpendicular sides of an off-canvas pixel)
    wall_with_border:np.ndarray = np.ones((scene_height + 2, scene_width + 2), dtype=np.int32)
    wall_with_border[1:-1, 1:-1] = (target_pixels > 0)
    wall_left:np.ndarray = wall_with_border[1:-1, :-2]
    wall_right:np.ndarray = wall_with_border[1:-1, 2:]
    wall_above:np.ndarray = wall_with_border[:-2, 1:-1]
    wall_below:np.ndarray = wall_with_border[2:, 1:-1]
    edge_corner_counts:np.ndarray = ((wall_left & wall_above) + (wall_above & wall_right)
                                     + (wall_right & wall_below) + (wall_below & wall_left)) - target_corner_counts

    # Scan every moving shape at every spot, and keep each one's whole grid of scores (0 = not allowed there)
    # A spot has to score at least 1 pinch, else the moving shape stays put
    scored_spots:List[Tuple[Shape, np.ndarray, int, int]] = []
    for moving_shape in moving_shapes:
        moving_shadow:np.ndarray = moving_shape.data['_shadow'].astype(np.int32)
        moving_height, moving_width = moving_shadow.shape

        # Moving shapes bigger than the scene can't fit anywhere in it
        if moving_height > scene_height or moving_width > scene_width:
            continue

        # Judgment call: a piece has to land fully INSIDE the scene (like gravity stops at the border).
        # The spec didn't say; letting pieces hang off the canvas would let their pixels silently vanish.
        # How many of the moving shape's pixels would overlap a target pixel, at every (top, left) inside the scene?
        # (correlate2d 'valid' mode: entry [top, left] = overlap if the moving shape's top-left corner sits there)
        overlap_counts:np.ndarray = scipy.signal.correlate2d(target_pixels, moving_shadow, mode='valid')

        # Target pinches moving, step 2: add up the target corner counts under the moving shape's pixels,
        # at every spot. (At no-overlap spots those pixels are all empty, so it's exactly the moving shape's pinches)
        target_pinches_moving:np.ndarray = scipy.signal.correlate2d(target_corner_counts, moving_shadow, mode='valid')

        # Moving pinches target, step 1: for every pixel in the moving shape's bounding box, how many of
        # its 4 corners are pinched by the moving shape? (Same 4 pairs. Pad by 1 with 0s: nothing moving out there)
        moving_with_border:np.ndarray = np.pad(moving_shadow, 1)
        moving_left:np.ndarray = moving_with_border[1:-1, :-2]
        moving_right:np.ndarray = moving_with_border[1:-1, 2:]
        moving_above:np.ndarray = moving_with_border[:-2, 1:-1]
        moving_below:np.ndarray = moving_with_border[2:, 1:-1]
        moving_corner_counts:np.ndarray = ((moving_left & moving_above) + (moving_above & moving_right)
                                           + (moving_right & moving_below) + (moving_below & moving_left))

        # Moving pinches target, step 2: add up those corner counts over the TARGET pixels inside the box,
        # at every spot. (A target pixel outside the box can never have the moving shape on two perpendicular
        # sides, so the box is enough. The moving shape's own pixels never sit on target at no-overlap spots)
        moving_pinches_target:np.ndarray = scipy.signal.correlate2d(target_with_border[1:-1, 1:-1], moving_corner_counts, mode='valid')

        # Both halves, equal weight
        pinch_counts:np.ndarray = target_pinches_moving + moving_pinches_target

        # Alignment bonus, on top of the pinches: +FIT_ALIGNMENT_BONUS for every TOUCHING target whose top & bottom
        # rows exactly match the moving shape's, and another if its left & right columns do.
        # (Both can happen, e.g. two chessboard-style pieces sharing a bounding box. The scene edge doesn't count.)
        # (Nicky) "Touching" is a cheap bounding-box check, no ring: the moving shape's box grown by 1 on
        # every side overlaps the target's box. Can give false positives (e.g. diagonal neighbors), but that's fine:
        # the bonus only ever counts at spots that already have a real pinch. ~30% faster than a 4-way ring check.
        # All the spots' tops & lefts, to check every spot at once:
        spot_tops:np.ndarray = np.arange(pinch_counts.shape[0])
        spot_lefts:np.ndarray = np.arange(pinch_counts.shape[1])
        alignment_bonus:np.ndarray = np.zeros_like(pinch_counts)
        # Only for moving shapes with area >= FIT_ALIGNMENT_MIN_AREA (no trivial little lines)
        is_big_enough:bool = int(moving_shadow.sum()) >= FIT_ALIGNMENT_MIN_AREA
        for target in (targets if is_big_enough else []):
            target_top:int = int(target.data['top'])
            target_left:int = int(target.data['left'])
            rows_match:bool = int(target.data['height']) == moving_height and 0 <= target_top < pinch_counts.shape[0]
            columns_match:bool = int(target.data['width']) == moving_width and 0 <= target_left < pinch_counts.shape[1]
            if not (rows_match or columns_match):
                continue

            # Bounding-box touch. Only the matching row (or column) of spots ever gets the bonus, so
            # only check those: along the row, does the grown box overlap the target's box left-to-right? Along the
            # column, top-to-bottom? (The other direction already matches. Bottom/right from height/width, so they
            # stay right while the search moves shapes around.)
            target_bottom:int = target_top + int(target.data['height'])
            target_right:int = target_left + int(target.data['width'])
            touches_along_row:np.ndarray = (spot_lefts - 1 < target_right) & (spot_lefts + moving_width + 1 > target_left)
            touches_along_column:np.ndarray = (spot_tops - 1 < target_bottom) & (spot_tops + moving_height + 1 > target_top)

            # Same top (and same height) = that whole row of spots. Same left (and same width) = that column.
            # (The bonus is a config knob: at +2, a staircase of pieces once tied exactly between "aligned next to"
            # (right) and "nested under" (wrong), so random tie-breaks got the test wrong on some seeds)
            if rows_match:
                alignment_bonus[target_top, :] += FIT_ALIGNMENT_BONUS * touches_along_row
            if columns_match:
                alignment_bonus[:, target_left] += FIT_ALIGNMENT_BONUS * touches_along_column

        # The edge's pinches at every spot (same trick as target_pinches_moving), at FIT_EDGE_PINCH_WEIGHT each
        edge_pinches_moving:np.ndarray = scipy.signal.correlate2d(edge_corner_counts, moving_shadow, mode='valid')

        # The bonus only counts at spots with at least 1 real pinch: alignment alone isn't "fitting into" anything
        # (else a shape could move just to sit flush next to something, with nothing actually holding it)
        # Same for the edge pinches: a tie-breaker between real pinch spots, never a reason to move on its own
        pinch_counts = np.where(pinch_counts >= 1,
                                pinch_counts + alignment_bonus + FIT_EDGE_PINCH_WEIGHT * edge_pinches_moving, 0)

        # Only no-overlap spots count. (The search picks from every shape's full score grid, best first,
        # random tie-breaks)
        pinch_counts[overlap_counts > 0] = 0
        # Where the score grid starts, for the shared search: (0, 0), since entry [top, left] IS the spot
        scored_spots.append((moving_shape, pinch_counts, 0, 0))

    return scored_spots


# Fitting's settings for the shared search in physics_utils.py (_search_placement_sequence): pinch scores,
# same-SHADOW skip, its own memo & margin.
def _search_fit_sequence(moving_shapes:List[Shape], targets:List[Shape], scene_height:int,
                         scene_width:int) -> Tuple[float, List[Tuple[Shape, int, int]]]:
    """Best (total score, [(moving shape, top, left), ...]) for fitting these moving shapes against these targets."""
    return _search_placement_sequence(moving_shapes, targets, scene_height, scene_width,
                                      score_spots=_score_fit_spots, same_shape_key=_shadow_key,
                                      memo=_FIT_MEMO, margin=FIT_SEARCH_MARGIN)

def _shadow_key(shape:Shape) -> tuple:
    """A shape's shadow as something hashable (fitting only looks at shadows, so same shadow = interchangeable)."""
    shadow:np.ndarray = shape.data['_shadow']
    return (shadow.shape, shadow.tobytes())

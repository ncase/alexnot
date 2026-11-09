# Attraction! Shapes flagged with shape.attract() slide over onto their targets, so their pixels OVERLAP the
# targets' pixels, color-matched (like magnets snapping red dot onto red dot, or puzzle pieces sharing an edge).
# Same flag-now-act-later pattern as fit() & apply_fit, and the same DFS search over placement orders (physics_utils.py).
# The only real difference from fitting is how each spot gets scored.
from typing import Dict, List, Tuple
import random # With no targets at all, one of the biggest movers (picked randomly) becomes the first target (solve() seeds it)
import numpy as np
import scipy.signal # correlate2d: slide a shape over a grid, count overlaps at every spot in one go
from .top_level_methods import DSL_CONFIG, all
from .shape import Shape
from ...config import ATTRACT_SEARCH_MARGIN # How far below each step's best the search still looks
from .physics_utils import _search_placement_sequence


# Every moving shape (flagged with shape.attract()) gets attracted TO the targets, which stay put.
# One at a time: the best (moving shape, spot) moves there, and becomes a target for the next round.
def apply_attract() -> None:
    """
    Example:

    for shape in all(width_GTE=5):
        shape.attract(overlap=True) # data stored; shape['_moving_attract'] = {} (the target selector)
    apply_attract() # NOW they all snap onto the other shapes, matching colors where they overlap
    """
    # TODO: overlap=False (attract until touching, no overlap). For now attract(overlap=False) returns
    # right away at its very top, without flagging the shape, so there's nothing for apply_attract to do then.

    scene:Dict = DSL_CONFIG['scene']
    scene_width:int = int(scene['width'])
    scene_height:int = int(scene['height'])

    # Who's moving (and attracted to which selector)? Who's unmarked? (Hidden shapes aren't on the canvas: neither)
    moving_shapes:List[Shape] = []
    target_selectors:List[Dict] = []
    unmarked_shapes:List[Shape] = []
    for shape in all():
        target_selector:Dict | None = shape.data.pop('_moving_attract', None) # consume the flag
        if shape.data.get('hidden'):
            continue
        if target_selector is not None:
            moving_shapes.append(shape)
            if target_selector not in target_selectors:
                target_selectors.append(target_selector)
        else:
            unmarked_shapes.append(shape)

    # Nobody's moving? Quit!
    if not moving_shapes:
        return

    # The targets: unmarked shapes that match the target selector. (No selector = all() = every unmarked shape)
    # Judgment call: if different movers asked for different selectors (e.g. color=shape.color), the
    # targets are everyone matching ANY of them, since the search shares one list of targets.
    target_data_ids:set = set()
    for target_selector in target_selectors:
        target_data_ids.update(id(shape.data) for shape in all(**target_selector))
    targets:List[Shape] = [shape for shape in unmarked_shapes if id(shape.data) in target_data_ids]

    # No targets at all (e.g. all() marked everybody)? Then the BIGGEST mover stays put as the first target:
    # the biggest piece is the natural anchor everything else snaps onto. (Ties broken randomly, on purpose: if a
    # level only works with one lucky first target, it should fail loudly across seeds)
    if not targets:
        biggest_area:int = max(int(moving_shape.data['area']) for moving_shape in moving_shapes)
        first_target:Shape = random.choice([moving_shape for moving_shape in moving_shapes if int(moving_shape.data['area']) == biggest_area])
        moving_shapes.remove(first_target)
        targets.append(first_target)
        if not moving_shapes:
            return

    # Every shape's "affinity" heatmap, computed ONCE up front: how much each of its pixels
    # matters for connecting (see _affinity_heatmap). Rare colors, like red connector dots, count the most.
    raw_grid:np.ndarray = np.asarray(scene['_raw'])
    color_counts:Dict[int, int] = dict(zip(*[values.tolist() for values in np.unique(raw_grid, return_counts=True)]))
    for shape in moving_shapes + targets:
        shape.data['_attract_heat'] = _affinity_heatmap(shape, color_counts)

    # Gaps (any non-filled space) count as one more "color" for the gap-on-gap tie-breaker: their heat is their
    # global rarity too, 1 / (background pixels in the raw grid), or 1 / (the most common color's pixels) if the parse
    # has no background. Tiny, but computed the same way as every other heat (no hard-coded epsilon)
    global _GAP_HEAT
    background_color:int = int(str(scene['background'])[1:]) # "c0" -> 0 (-1 for no background)
    gap_pixel_count:int = color_counts.get(background_color, 0) if background_color >= 0 else 0
    _GAP_HEAT = 1 / (gap_pixel_count or max(color_counts.values()))

    # Cached, like apply_fit: same moving shapes + same targets (position & pixels, in the same order) = same answer.
    # (Pixels, not just shadows: here the colors matter. And the raw grid: the heatmaps' rarity part depends on its color counts)
    attract_cache_key:tuple = (scene_height, scene_width, raw_grid.shape, raw_grid.tobytes(),
                               tuple(_attract_cache_shape_key(moving_shape) for moving_shape in moving_shapes),
                               tuple(_attract_cache_shape_key(target) for target in targets))
    if attract_cache_key not in _ATTRACT_CACHE:
        # The DFS memo's keys are object ids, so it's ONLY valid inside this one search: fresh before, emptied after
        _ATTRACT_MEMO.clear()
        try:
            best_total, best_sequence = _search_placement_sequence(
                moving_shapes, targets, scene_height, scene_width,
                score_spots=_score_attract_spots, same_shape_key=_pixels_key,
                memo=_ATTRACT_MEMO, margin=ATTRACT_SEARCH_MARGIN,
                min_score=1e-9) # Heatmap scores are fractions: anything above 0 is a fit
        finally:
            _ATTRACT_MEMO.clear()
        # Store it by INDEX into moving_shapes, not the Shape objects: every program runs on its own copy of the scene
        _ATTRACT_CACHE[attract_cache_key] = [(moving_shapes.index(moving_shape), spot_top, spot_left)
                                             for moving_shape, spot_top, spot_left in best_sequence]

    # Now actually move them, in that order (move() adds the +1 edit). Anyone not in the sequence never found a
    # spot that overlaps a target with only matching colors, so they stay put.
    for moving_index, spot_top, spot_left in _ATTRACT_CACHE[attract_cache_key]:
        moving_shape:Shape = moving_shapes[moving_index]
        moving_shape.move(x=spot_left - moving_shape.data['left'], y=spot_top - moving_shape.data['top'])

    # The heatmaps were just for this search: tidy them away
    for shape in moving_shapes + targets:
        shape.data.pop('_attract_heat', None)


# How much does each pixel of a shape matter for connecting?
# We did a LOT of experimenting to land here (snap-together puzzles, over many random seeds):
# plain match counts, "nub-ness" (pixels sticking out), rarity scaled various ways, rarity within each shape,
# color RANKS (globally and per shape), adding vs multiplying, squared ranks... What worked, reliably across seeds:
#   heat = global rarity x (local rank)^2
#   - global rarity = 1 / (pixels of this color in the whole raw input grid)
#   - local rank = 1 / (rank of how uncommon this color is WITHIN this shape: rarest = 1, next = 1/2, ...; ties share)
# ...with a matched pixel pair scoring moving heat x target heat (so rare-on-rare counts a LOT more than rare-on-common),
# plus the "no 3 shapes on one pixel" rule and the gap-on-gap tie-breaker in _score_attract_spots,
# and the biggest piece as the first target in apply_attract. (Nubs always made things worse; local RARITY, 1/count in
# a shape, backfired: a piece with more connector dots got less per dot, so a full multi-dot join didn't add up.)
def _affinity_heatmap(shape:Shape, color_counts:Dict[int, int]) -> np.ndarray:
    """A float grid the size of the shape's bounding box: each shape pixel's global rarity x local rank^2 (0 outside the shape)."""
    shadow_mask:np.ndarray = shape.data['_shadow'] == 1
    pixels:np.ndarray = shape.data['_pixels']

    # This shape's colors, and how many pixels of each. Ranks: rarest count = rank 1, tied counts share a rank
    colors, color_pixel_counts = np.unique(pixels[shadow_mask], return_counts=True)
    distinct_counts_in_shape:list = sorted(set(color_pixel_counts.tolist()))

    heat:np.ndarray = np.zeros(shadow_mask.shape, dtype=np.float64)
    for color, color_pixel_count in zip(colors.tolist(), color_pixel_counts.tolist()):
        # (max(..., 1) just in case a color isn't in the raw grid, e.g. an off-canvas chunk's guessed color)
        global_rarity:float = 1 / max(color_counts.get(int(color), 0), 1)
        local_rank:float = 1 / (distinct_counts_in_shape.index(color_pixel_count) + 1)
        heat[shadow_mask & (pixels == color)] = global_rarity * local_rank ** 2
    return heat


# apply_attract's cache: (scene size, raw grid, moving shapes, targets) -> [(moving index, top, left), ...].
# search_transform clears it every round of evolution (with apply_fit's), so it never grows forever.
_ATTRACT_CACHE:Dict[tuple, List[Tuple[int, int, int]]] = {}

def clear_attract_cache() -> None:
    """Forget every cached apply_attract answer."""
    _ATTRACT_CACHE.clear()

# The gap "color"'s heat, set by apply_attract before each search (for the gap-on-gap tie-breaker)
_GAP_HEAT:float = 0.0

# The DFS memo, inside ONE search (same idea as apply_fit's _FIT_MEMO)
_ATTRACT_MEMO:Dict[tuple, Tuple[float, list]] = {}

def _pixels_key(shape:Shape) -> tuple:
    """A shape's exact pixels (colors + shadow), as something hashable."""
    pixels:np.ndarray = shape.data['_pixels']
    return (pixels.shape, pixels.tobytes())

def _attract_cache_shape_key(shape:Shape) -> tuple:
    """Everything about a shape that apply_attract's search looks at: where it is, and its pixels."""
    return (int(shape.data['top']), int(shape.data['left'])) + _pixels_key(shape)


# One round of apply_attract's scoring: every moving shape's score at every spot that could overlap a target.
# Each same-color overlapping pixel pair adds moving heat x target heat, -inf if ANY overlapping pixel has a
# different color (or lands on an already-glued pixel), 0 = no overlap. Plus a small gap-on-gap tie-breaker.
# Returns (moving shape, scores, top of scores[0, 0], left of scores[0, 0]) for each moving shape.
def _score_attract_spots(moving_shapes:List[Shape], targets:List[Shape], scene_height:int,
                         scene_width:int) -> List[Tuple[Shape, np.ndarray, int, int]]:

    # Everybody's placed already (the search's last step)? Nothing to score
    if not moving_shapes:
        return []

    # Judgment call: spots may hang off the canvas (unlike apply_fit), since targets can sit near the edge
    # and a chain of attracted shapes can grow past it (e.g. a focus() afterwards crops to everyone).
    # So the "canvas" here is the targets' combined bounding box, padded by the biggest moving shape on every side:
    # any spot where a moving shape overlaps a target fits inside that. (So scene size isn't needed here)
    targets_top:int = min(int(target.data['top']) for target in targets)
    targets_left:int = min(int(target.data['left']) for target in targets)
    targets_bottom:int = max(int(target.data['top']) + target.data['_shadow'].shape[0] for target in targets)
    targets_right:int = max(int(target.data['left']) + target.data['_shadow'].shape[1] for target in targets)
    padding_height:int = max(moving_shape.data['_shadow'].shape[0] for moving_shape in moving_shapes) - 1
    padding_width:int = max(moving_shape.data['_shadow'].shape[1] for moving_shape in moving_shapes) - 1
    canvas_top:int = targets_top - padding_height
    canvas_left:int = targets_left - padding_width
    canvas_height:int = (targets_bottom - targets_top) + 2 * padding_height
    canvas_width:int = (targets_right - targets_left) + 2 * padding_width

    # Every target pixel's color on that canvas (-1 = no target there). Targets overlapping each other with
    # different colors: the later one wins (that pixel can only match one color anyway)
    target_colors:np.ndarray = np.full((canvas_height, canvas_width), -1, dtype=np.int32)
    for target in targets:
        target_shadow_mask:np.ndarray = target.data['_shadow'] == 1
        target_top:int = int(target.data['top']) - canvas_top
        target_left:int = int(target.data['left']) - canvas_left
        target_window:np.ndarray = target_colors[target_top:target_top + target_shadow_mask.shape[0],
                                                 target_left:target_left + target_shadow_mask.shape[1]]
        target_window[target_shadow_mask] = target.data['_pixels'][target_shadow_mask]
    target_pixels:np.ndarray = (target_colors >= 0).astype(np.int32)

    # For the gap-on-gap tie-breaker, EACH TARGET ON ITS OWN: its shadow, and its own gaps (cells inside its
    # OWN bounding box that aren't its pixels, even if another target fills them), on the canvas
    per_target_shadow_and_gaps:List[Tuple[np.ndarray, np.ndarray]] = []
    for target in targets:
        target_shadow_mask = target.data['_shadow'] == 1
        box_top:int = int(target.data['top']) - canvas_top
        box_left:int = int(target.data['left']) - canvas_left
        box_height, box_width = target_shadow_mask.shape
        this_target_shadow:np.ndarray = np.zeros((canvas_height, canvas_width), dtype=np.int32)
        this_target_gaps:np.ndarray = np.zeros((canvas_height, canvas_width), dtype=np.int32)
        this_target_shadow[box_top:box_top + box_height, box_left:box_left + box_width] = target_shadow_mask
        this_target_gaps[box_top:box_top + box_height, box_left:box_left + box_width] = ~target_shadow_mask
        per_target_shadow_and_gaps.append((this_target_shadow, this_target_gaps))

    # No 3 shapes on one pixel: how many targets cover each canvas pixel? A pixel 2 shapes already share (a joint
    # that's already glued, like a used red connector) can't take a THIRD shape on top
    target_coverage:np.ndarray = np.zeros((canvas_height, canvas_width), dtype=np.int32)
    for target in targets:
        target_shadow_mask = target.data['_shadow'] == 1
        target_top = int(target.data['top']) - canvas_top
        target_left = int(target.data['left']) - canvas_left
        target_coverage[target_top:target_top + target_shadow_mask.shape[0],
                        target_left:target_left + target_shadow_mask.shape[1]] += target_shadow_mask
    already_double:np.ndarray = (target_coverage >= 2).astype(np.int32)

    # ...and every target pixel's heat, on the same canvas (same "later one wins" rule)
    target_heat:np.ndarray = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    for target in targets:
        target_shadow_mask = target.data['_shadow'] == 1
        target_top = int(target.data['top']) - canvas_top
        target_left = int(target.data['left']) - canvas_left
        heat_window:np.ndarray = target_heat[target_top:target_top + target_shadow_mask.shape[0],
                                             target_left:target_left + target_shadow_mask.shape[1]]
        heat_window[target_shadow_mask] = target.data['_attract_heat'][target_shadow_mask]

    scored_spots:List[Tuple[Shape, np.ndarray, int, int]] = []
    for moving_shape in moving_shapes:
        moving_shadow_mask:np.ndarray = moving_shape.data['_shadow'] == 1
        moving_pixels:np.ndarray = moving_shape.data['_pixels']

        # How many of the moving shape's pixels overlap ANY target pixel, at every spot?
        # (correlate2d 'valid' mode: entry [row, column] = overlap if its top-left corner sits at canvas (row, column))
        overlap_counts:np.ndarray = scipy.signal.correlate2d(target_pixels, moving_shadow_mask.astype(np.int32), mode='valid')

        # ...and how many overlap a target pixel of the SAME color? One color at a time
        matching_counts:np.ndarray = np.zeros_like(overlap_counts)
        # ...and how much heat those matches are worth: each matching pixel pair scores the moving pixel's
        # heat x the target pixel's heat (so rare-on-rare counts a LOT more than rare-on-common)
        moving_heat:np.ndarray = moving_shape.data['_attract_heat']
        matching_heat:np.ndarray = np.zeros(overlap_counts.shape, dtype=np.float64)
        for color in np.unique(moving_pixels[moving_shadow_mask]):
            moving_color_mask:np.ndarray = (moving_shadow_mask & (moving_pixels == color)).astype(np.int32)
            target_color_mask:np.ndarray = (target_colors == color).astype(np.int32)
            matching_counts += scipy.signal.correlate2d(target_color_mask, moving_color_mask, mode='valid')
            matching_heat += scipy.signal.correlate2d(target_color_mask * target_heat, moving_color_mask * moving_heat, mode='valid')

        # Score = the matches' heat, but ANY mismatched overlap rules the spot out. (No overlap = 0, not a fit)
        scores:np.ndarray = matching_heat
        scores[overlap_counts > matching_counts] = -np.inf
        # Gap-on-gap tie-breaker, per TOUCHED target (>= 1 real pixel overlap): count the moving shape's gaps
        # (inside its own box) landing on that target's gaps (inside the target's box), times gap heat x gap heat.
        # Only for real fits: gaps alone aren't a fit. (Only touched targets, so it's about the joint, and a layout's total
        # doesn't depend on placement order.)
        moving_gaps:np.ndarray = (~moving_shadow_mask).astype(np.int32)
        gap_on_gap_heat:np.ndarray = np.zeros(scores.shape, dtype=np.float64)
        for this_target_shadow, this_target_gaps in per_target_shadow_and_gaps:
            touches_this_target:np.ndarray = scipy.signal.correlate2d(this_target_shadow, moving_shadow_mask.astype(np.int32), mode='valid') > 0
            gap_on_gap_counts:np.ndarray = scipy.signal.correlate2d(this_target_gaps, moving_gaps, mode='valid')
            gap_on_gap_heat += np.where(touches_this_target, gap_on_gap_counts, 0) * _GAP_HEAT * _GAP_HEAT
        is_real_fit:np.ndarray = (matching_counts > 0) & (overlap_counts == matching_counts)
        scores[is_real_fit] += gap_on_gap_heat[is_real_fit]

        # No third shape on an already-glued pixel
        scores[scipy.signal.correlate2d(already_double, moving_shadow_mask.astype(np.int32), mode='valid') > 0] = -np.inf
        scored_spots.append((moving_shape, scores, canvas_top, canvas_left))
    return scored_spots

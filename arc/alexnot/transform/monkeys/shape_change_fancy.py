# from __future__ import annotations
import numpy as np
from collections import Counter
from ...grid import _shapes_hitbox_test, _shapes_hit_test, _chebyshev_dilate

from .monkey_helpers import ShapeActionBatcher
from .monkey_helpers import _original_footprint
from ...utils.random_ties import _random_most_common # most_common, with random tie-breaks

###################################################
### FANCY SHAPE CHANGES: OUTLINE, INSET, FILL #####
###################################################

def _shape_sig(s:dict) -> tuple:
    """A shape's 'identity': WHERE it is + its filled-mask. (Color-agnostic, so a recolored
    shape still counts as 'the same shape'. _shadow is a numpy array, so tuple-ify it to be hashable.)"""
    return (s['left'], s['top'], tuple(tuple(row) for row in s['_shadow']))


def _shape_sig_with_colors(s:dict) -> tuple:
    """Like _shape_sig, but color-AWARE: where it is + its filled-mask + its pixels' colors."""
    # (_pixels is -1 outside the shadow, so it already carries the mask; the shadow's in there just to be explicit)
    return (s['left'], s['top'], tuple(tuple(row) for row in s['_shadow']), tuple(tuple(row) for row in s['_pixels']))


def _one_pixel_ring_color(shape:dict, predicted_grid:np.ndarray, output_grid:np.ndarray) -> str|None:
    """The output's color on the 1 px ring around a shape, if that ring is all one color AND new vs the prediction."""

    # Different grid sizes? Then the pixels don't line up, nothing to compare
    if predicted_grid.shape != output_grid.shape:
        return None

    # The ring outline(line_width=1) would paint: empty cells touching the shape, corners included.
    # (Pad by 1 first, so the ring has room to grow past the shape's box)
    filled:np.ndarray = np.pad(shape['_shadow'] != 0, 1, constant_values=False)
    ring_mask:np.ndarray = _chebyshev_dilate(filled, 1) & ~filled

    # Where does that padded box sit in the grid? Then cut off whatever hangs past the grid's edges
    grid_height, grid_width = output_grid.shape
    top:int = shape['top'] - 1
    left:int = shape['left'] - 1
    ring_ys, ring_xs = np.nonzero(ring_mask)
    ring_ys, ring_xs = ring_ys + top, ring_xs + left
    inside_grid:np.ndarray = (ring_ys >= 0) & (ring_ys < grid_height) & (ring_xs >= 0) & (ring_xs < grid_width)
    ring_ys, ring_xs = ring_ys[inside_grid], ring_xs[inside_grid]
    if len(ring_ys) == 0:
        return None

    # The output's ring has to be ONE color...
    output_ring:np.ndarray = output_grid[ring_ys, ring_xs]
    ring_color:int = int(output_ring[0])
    if not np.all(output_ring == ring_color):
        return None

    # ...that the prediction doesn't already have all around (else there's nothing for an outline to fix)
    if np.all(predicted_grid[ring_ys, ring_xs] == ring_color):
        return None
    return f"c{ring_color}"


def shapes_outline(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Propose OUTLINE-ing a shape.
    The tell: a shape survives UNCHANGED (same spot, same mask -- color may differ), but it's
    now HITBOX-hugging a brand-new shape that wasn't there before (and NOT exact-hitting it,
    'coz the outline ring is hollow in the middle where our shape sits!). That new shape is
    the outline ring -> propose outlining in ITS color.
    Strict on purpose: "unchanged" includes the colors, my box has to be fully INSIDE the new shape's box,
    and we propose just ONE outline per shape: in its TIGHTEST hugging new shape's color, with line_width =
    the biggest side gap between the two boxes (max 5).
    Plus a pixel tell: if the output's 1 px ring around the unchanged shape is all one new color, propose a
    1 px outline in that color too (the output parse may glue touching rings into one blob, hiding the hug).
    """
    candidates:set = set()

    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # TODO: LIMIT TO THE BEST-PREDICTED & SECOND-BEST TRAIN PAIRS ?

        for pair_index, pair in enumerate(train_pairs):
            before_shapes:list[dict] = program_result['predicted_scenes'][pair_index]['shapes'] # PREDICTED = before
            after_shapes:list[dict]  = pair['output']['shapes']
            if not after_shapes:
                continue
            # Raw pixels, for the 1 px ring check
            predicted_grid:np.ndarray = program_result['predicted_scenes'][pair_index]['_raw']
            output_grid:np.ndarray = pair['output']['_raw']

            # Who's "new" in the output? (No before-shape sharing its spot + mask AND colors.
            # So a recolored shape doesn't count as "unchanged" (the repaint monkey has to go first, on purpose),
            # and a recolored one counts as "new")
            before_sigs:set = { _shape_sig_with_colors(b) for b in before_shapes }
            after_sigs:set  = { _shape_sig_with_colors(a) for a in after_shapes }
            new_shapes:list[dict] = [a for a in after_shapes if _shape_sig_with_colors(a) not in before_sigs]
            if not new_shapes:
                continue # nothing new appeared, so nothing got outlined

            for before_shape in before_shapes:
                if before_shape.get('hidden'):
                    continue

                # My shape's gotta SURVIVE unchanged (outline doesn't touch the interior)
                # TODO: Be more lenient? Maybe it's obscured.
                if _shape_sig_with_colors(before_shape) not in after_sigs: # (Colors have to match too)
                    continue

                # Pixel check first: is the output's 1 px ring around me all one new color? Then propose a thin
                # outline in it. (Reads the pixels, so it still works when the output parse glued rings into blobs)
                ring_color:str|None = _one_pixel_ring_color(before_shape, predicted_grid, output_grid)
                if ring_color is not None:
                    shape_actions.add(before_shape, f'outline(color="{ring_color}", line_width=1)')

                # Which new shapes do I HITBOX-hug? (Not necessarily exact-hit.) My bounding box has to sit
                # FULLY INSIDE the new shape's box (edges may touch, 'coz the outline can get cropped off at the scene's edge)
                hits:list[dict] = [new_shape for new_shape in new_shapes
                                   if new_shape['left'] <= before_shape['left'] and new_shape['right'] >= before_shape['right']
                                   and new_shape['top'] <= before_shape['top'] and new_shape['bottom'] >= before_shape['bottom']]

                # Keep only my TIGHTEST hug: the nearest ring around me is the one most likely made by outlining me.
                # (Rings that touch get glued into one big shape, whose box would make me guess a way-too-thick line)
                tightest_max_gap:int = 0
                tightest_hit:dict|None = None
                for hit in hits:
                    # line_width guess: ONE width = the biggest of the 4 side gaps between the boxes,
                    # since that's how far the outline has to reach. (Smaller gaps may just be the ring cropped by the scene edge.)
                    side_gaps:list[int] = [before_shape['left'] - hit['left'], before_shape['top'] - hit['top'],
                                           hit['right'] - before_shape['right'], hit['bottom'] - before_shape['bottom']]
                    max_gap:int = max(side_gaps)
                    # Same box as mine on all 4 sides? Then there's no ring to explain, skip it
                    if max_gap <= 0:
                        continue
                    # Tighter = smaller max gap. A tie goes to the smaller box
                    if tightest_hit is None or max_gap < tightest_max_gap or (
                            max_gap == tightest_max_gap and hit['width'] * hit['height'] < tightest_hit['width'] * tightest_hit['height']):
                        tightest_max_gap, tightest_hit = max_gap, hit
                if tightest_hit is None:
                    continue

                color:str = tightest_hit['color']
                line_width:int = min(5, tightest_max_gap) # MAXIMUM 5, ELSE DSL GETS TOO SLUGGISH
                # (Skip it if the pixel check already proposed this exact outline, so my pixels don't count twice in the batcher)
                if (color, line_width) != (ring_color, 1):
                    shape_actions.add(before_shape, f'outline(color="{color}", line_width={line_width})')
                # ...and the same outline, but without painting into the shape's holes
                shape_actions.add(before_shape, f'outline(color="{color}", line_width={line_width}, ignore_holes=True)')
                # MAYBE TODO: only propose ignore_holes=True when the shape HAS holes (else it's a no-op that eats batcher slots). Not worth it yet

        # Flush, append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    return candidates


# shapes_inline DISABLED for now, like shape.inline() (we'll deal with in-lining later). (NOT ACTUALLY CURRENTLY USED; WIP)
# def shapes_inline(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
#     """
#     Propose IN-lining a shape (recolor its border INWARD, see shape.inline()).
#     The tell: a new output shape X (not in the before, by spot + mask) has the EXACT same box as a before-shape Y,
#     and some output shape Z sits fully inside X's box. So X is the inline ring and Z is what's left of Y.
#     """
#     candidates:set = set()
#
#     for program_result in parent_program_results:
#
#         # Execution-guided progsys: batch actions against *predicted* shapes
#         shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )
#
#         for pair_index, pair in enumerate(train_pairs):
#             before_shapes:list[dict] = program_result['predicted_scenes'][pair_index]['shapes'] # PREDICTED = before
#             after_shapes:list[dict]  = pair['output']['shapes']
#             if not after_shapes:
#                 continue
#
#             # a) Who's "new" in the output? (no before-shape sharing its spot+mask, color-agnostic)
#             before_sigs:set = { _shape_sig(b) for b in before_shapes }
#             new_shapes:list[dict] = [a for a in after_shapes if _shape_sig(a) not in before_sigs]
#             if not new_shapes:
#                 continue # nothing new appeared, so nothing got in-lined
#
#             for before_shape in before_shapes:
#                 if before_shape.get('hidden'):
#                     continue
#
#                 # b) A new shape with EXACT same box as mine = the inline ring
#                 for ring_shape in new_shapes:
#                     same_box:bool = (ring_shape['left'] == before_shape['left'] and ring_shape['top'] == before_shape['top']
#                                      and ring_shape['right'] == before_shape['right'] and ring_shape['bottom'] == before_shape['bottom'])
#                     if not same_box:
#                         continue
#
#                     # c) Output shapes fully inside the ring's box (edges may touch) = what's left of me.
#                     # How far in did the ring reach? The biggest of the 4 side gaps between my box and the leftover's box.
#                     # Judgment call: several shapes can sit inside the ring's box (e.g. other same-colored shapes
#                     # too), so take the TIGHTEST fit (smallest biggest-gap), to still propose just ONE width.
#                     tightest_gap:int|None = None
#                     for inner_shape in after_shapes:
#                         if inner_shape is ring_shape:
#                             continue
#                         fully_inside:bool = (inner_shape['left'] >= ring_shape['left'] and inner_shape['right'] <= ring_shape['right']
#                                              and inner_shape['top'] >= ring_shape['top'] and inner_shape['bottom'] <= ring_shape['bottom'])
#                         if not fully_inside:
#                             continue
#                         side_gaps:list[int] = [inner_shape['left'] - before_shape['left'], inner_shape['top'] - before_shape['top'],
#                                                before_shape['right'] - inner_shape['right'], before_shape['bottom'] - inner_shape['bottom']]
#                         max_gap:int = max(side_gaps)
#                         # Same box as mine on all 4 sides? Then there's no ring between us, skip it
#                         if max_gap <= 0:
#                             continue
#                         if tightest_gap is None or max_gap < tightest_gap:
#                             tightest_gap = max_gap
#                     if tightest_gap is None:
#                         continue # nothing inside => it was just a repaint, not an inline
#
#                     color:str = ring_shape['color']
#                     line_width:int = min(5, tightest_gap) # MAXIMUM 5, ELSE DSL GETS TOO SLUGGISH
#                     shape_actions.add(before_shape, f'inline(color="{color}", line_width={line_width})')
#                     # ...and the same inline, but without tracing the edges of the shape's holes
#                     shape_actions.add(before_shape, f'inline(color="{color}", line_width={line_width}, ignore_holes=True)')
#                     # MAYBE TODO: only propose ignore_holes=True when the shape HAS holes (else it's a no-op that eats batcher slots). Not worth it yet
#                     break # found my ring, done with this shape
#
#         # Flush, append all suggestions!
#         code_blocks:list[str] = shape_actions.flush()
#         for code in code_blocks:
#             candidates.add( program_result['program'] + '\n' + code.strip() )
#
#     return candidates


def shapes_fill_holes(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Propose FILL-HOLES-ing a shape.
    The tell: a shape is in before & after, and it has holes,
    and the most common color in the holes changed.
    """
    candidates:set = set()

    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        for pair_index, pair in enumerate(train_pairs):
            before_shapes:list[dict] = program_result['predicted_scenes'][pair_index]['shapes']  # PREDICTED = before
            after_shapes:list[dict] = pair['output']['shapes']
            predicted_grid:np.ndarray = program_result['predicted_scenes'][pair_index]['_raw'] # *RAW*
            output_grid:np.ndarray = pair['output']['_raw']
            for before_shape in before_shapes:

                # Not hidden
                if before_shape.get('hidden'): 
                    continue

                # Actually has holes:
                if before_shape['holes'] == 0:
                    continue

                # First filter out those without the same holed-shape in output.
                # Match on the shape's ORIGINAL pixels (ignoring e.g. an outline ring it's grown since),
                # so fill_holes can still be proposed after an outline
                original_left, original_top, original_shadow = _original_footprint(before_shape)
                same_shadow_after_shapes:list[dict] = [after_shape for after_shape in after_shapes
                                                       if np.array_equal(original_shadow, after_shape['_shadow'])]
                if not same_shadow_after_shapes:
                    continue

                # Merge all the holes into one mask, then get the plurality color of the
                # raw pixels in the holes BEFORE (in the predicted grid, at the shape's spot)
                all_holes_mask:np.ndarray = np.any([hole_grid == 1 for hole_grid in before_shape['_hole_grids']], axis=0)
                before_hole_color:int|None = _plurality_hole_color(all_holes_mask, predicted_grid, before_shape['left'], before_shape['top'])
                if before_hole_color is None:
                    continue

                # Now, get the AFTER holes' color, suggest filling IF DIFFERENT
                for after_shape in same_shadow_after_shapes:
                    # The hole mask is in the before-shape's CURRENT local coords (padded by any outline), so shift
                    # the after-shape's corner back by however far the original pixels sit inside that padded grid
                    mask_left:int = after_shape['left'] - (original_left - before_shape['left'])
                    mask_top:int = after_shape['top'] - (original_top - before_shape['top'])
                    after_hole_color:int|None = _plurality_hole_color(all_holes_mask, output_grid, mask_left, mask_top)
                    if after_hole_color is None or after_hole_color == before_hole_color:
                        continue
                    # Yeah fill them holes
                    shape_actions.add(before_shape, f'fill_holes(color="c{after_hole_color}")')

        # Flush, append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    return candidates


def _plurality_hole_color(holes_mask:np.ndarray, grid:np.ndarray, left:int, top:int) -> int|None:
    """The most common (plurality, not majority) raw color in a shape's holes."""

    # Where are the hole pixels? (Relative to the shape's top-left corner)
    hole_ys, hole_xs = np.nonzero(holes_mask)

    # Shift them to where the shape actually sits in the grid
    grid_ys:np.ndarray = hole_ys + top
    grid_xs:np.ndarray = hole_xs + left

    # Skip hole pixels that land off the grid (e.g. shapes pushed off-scene after a crop)
    grid_height, grid_width = grid.shape
    is_on_grid:np.ndarray = (grid_ys >= 0) & (grid_ys < grid_height) & (grid_xs >= 0) & (grid_xs < grid_width)
    if not is_on_grid.any():
        return None # No hole pixels on the grid at all? Then there's no color to report

    # Read the raw colors in the holes, and return the most common one
    hole_colors:np.ndarray = grid[grid_ys[is_on_grid], grid_xs[is_on_grid]]
    # Tied hole colors? Pick a random one, not the first one seen
    return int(_random_most_common(Counter(hole_colors.tolist()))[0])

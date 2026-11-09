from collections import Counter
import numpy as np
from ..grid import _flip_grid, _rotate_grid, _chebyshev_dilate
from ..utils.random_ties import _random_most_common # most_common, with random tie-breaks

# CONFIG KNOBS
from ..config import (COEFF_CONFIG_COMPLEXITY, POINTS_SHAPE_MOVE, POINTS_SHAPE_RECOLOR, POINTS_SHAPE_TRANSFORMED,
                      POINTS_SHAPE_MOVE_WHEN_RESIZED, # the move price when the scene changes size
                      POINTS_SHAPE_DELETED, POINTS_SHAPE_ADDED, POINTS_SHAPE_OUTLINE_OR_RAY, POINTS_SHAPE_FILLS_HOLE, POINTS_SCENE_SIZE_CHANGED, POINTS_SCENE_BG_CHANGED,
                      POINTS_GREEDY_INPUT_SHAPES_STATIC, POINTS_GREEDY_INPUT_SHAPES_MOVING,
                      POINTS_DENOISE, # de-noising shapes costs this
                      POINTS_AUTOCOMPLETE_RECTS, # ...and completing hidden rectangles on top costs this more
                      POINTS_PATTERN_BACKGROUND, # a pattern background behind everything costs this
                      POINTS_PATTERN_BACKGROUND_EVERY_PAIR_REWARD, # ...but earns this back if it's behind every demo pair (input or output)
                      POINTS_PARSE_CHUNK_PER_PIXEL, # a chunk costs this much per pixel of its area
                      PARSE_OUTLINE_WIDTH, PARSE_OUTLINE_COVERED_FRACTION, # when a new shape counts as an outline
                      PARSE_RAY_LENGTH, # how long a ray a new shape must cover to count as a ray
                      DEBUG_PARSE_SCORE_BREAKDOWN, # print a per-demo breakdown of every score (off by default)
                      )

# _score_change only checks input shapes with the SAME match key as each output shape (or one of its flips/rotations).
# That's only exact while a move or recolor can't be a reward: then a differently-shaped input shape always costs at least
# the de-novo price, so it could never win a claim anyway. A negative price breaks that, so fail loudly instead of silently.
assert min(POINTS_SHAPE_MOVE, POINTS_SHAPE_MOVE_WHEN_RESIZED, POINTS_SHAPE_RECOLOR) >= 0, (
   "_score_change's same-key shortcut assumes move/recolor prices are never negative. "
   "If you want a negative one (e.g. POINTS_SHAPE_MOVE_WHEN_RESIZED as a reward), switch its loop back to "
   "walking every input shape (the commented-out loop in _score_change).")

############################################################
# Now, actually scoring a Parse Config:
# How costly is *the diff* between the scenes? (Relative Algo Complexity)
############################################################

# free_color: the puzzle's free color doesn't depend on the config, so search_parse finds it ONCE (with
# _find_free_color below) and hands it to every config. (So all configs in one search agree on it, even on 50/50 ties)
# change_cost_cache: optional, for search_parse's bg-mode twins, which share the very same parsed pairs, so their demo
# pairs' change cost only gets worked out once. (Each config's own price still does, so bg modes still cost differently)
def score_parse(parse_config:dict, parsed_pairs:list[dict], free_color:str|None, change_cost_cache:dict|None=None) -> float:
   """
   We're scoring a parse config, by the "complexity" of the resulting parsed pairs.
   Lower = simpler = better parse.
   """
   score:float = 0.0

   # The "free" bg color isn't just the naive most common pixel (one big test grid could decide that).
   # There's a free color only if at least HALF the grids have the SAME color covering at least HALF their pixels,
   # else None. (So a color stays free even if another color wins a few grids.) The caller finds it once per search.
   most_common_puzzle_color:str|None = free_color

   # Every chunk's shadow key, in all 8 D4 orientations, so a new output shape shaped like a chunk is cheap.
   # TODO: match up to *scaling* too, later (a 2x-bigger chunk is still that chunk), not just D4
   chunk_shadow_keys:set[tuple] = set()
   for chunk in parse_config.get('chunks', []):
      chunk_shadow_keys.update(_shadow_key(transformed) for transformed in [chunk] + _non_identity_d4_transforms(chunk))

   # Rows-only or columns-only parse? Then no outline or ray prices: outlines never show up in ARC as strips, and
   # otherwise a new strip would count as the "outline" (or "ray") of its neighbor strip
   allow_outlines_and_rays:bool = parse_config['rows'] == parse_config['columns']
   # Rows-only or columns-only parse? Then no hole-fill price either: a 1-px strip can't have holes.
   # (Only this gate: no_bg and resized levels can still fill holes)
   allow_hole_fills:bool = parse_config['rows'] == parse_config['columns']
   # No background? Then no outline or ray prices either: the whole background region would be one giant "outline"
   allow_outlines_and_rays = allow_outlines_and_rays and parse_config['bg_mode'] != 'no_bg'
   # Any demo pair change grid size? Then no outline or ray prices anywhere in this level: a resized scene's new
   # shapes aren't drawn around the old ones in place, so a big blob would pass as an "outline" too easily
   allow_outlines_and_rays = allow_outlines_and_rays and all(
      (pair['input']['width'], pair['input']['height']) == (pair['output']['width'], pair['output']['height'])
      for pair in parsed_pairs if pair.get('output') is not None)

   # Debug switch on? Then each demo pair fills in a breakdown dict (else None = no extra work)
   pair_breakdowns:list[dict] = []

   # The very same parsed pairs (same list) as an earlier config, with the same settings the change cost reads? Then
   # it's the same change cost, so reuse it. (Keyed by id: the caller keeps every scored list alive, so ids can't get
   # reused. Chunk-free only, so the chunks don't need to go in the key. Not with the debug switch on: it needs every breakdown)
   change_cost_key:tuple = (id(parsed_pairs), allow_outlines_and_rays, allow_hole_fills)
   use_change_cost_cache:bool = change_cost_cache is not None and not chunk_shadow_keys and not DEBUG_PARSE_SCORE_BREAKDOWN
   # (The "is not None" again is just for the type checker: use_change_cost_cache already implies it)
   if use_change_cost_cache and change_cost_cache is not None and change_cost_key in change_cost_cache:
      score += change_cost_cache[change_cost_key]
   else:
      # Every demo pair's cost
      for pair in parsed_pairs:
         if pair.get('output') is not None: # skip no-output Tests
            # (+ the chunks, for cheap chunk-shaped new shapes, + a breakdown dict, only when the debug switch is on)
            pair_breakdown:dict|None = {} if DEBUG_PARSE_SCORE_BREAKDOWN else None
            score += _score_change(pair['input'], pair['output'], chunk_shadow_keys, allow_outlines_and_rays, allow_hole_fills, pair_breakdown)
            if pair_breakdown is not None:
               pair_breakdowns.append(pair_breakdown)
      # (score is just the change cost so far: the config's own price comes next)
      if use_change_cost_cache and change_cost_cache is not None:
         change_cost_cache[change_cost_key] = score

   # Penalize the Parse Config's complexity itself
   # (+ which settings the points came from, only when the debug switch is on)
   complexity_parts:dict|None = {} if DEBUG_PARSE_SCORE_BREAKDOWN else None
   config_complexity_points:float = _score_config_complexity(parse_config, most_common_puzzle_color, complexity_parts)
   # A pattern background found behind EVERY demo pair (on its input, its output, or both) is the puzzle's real
   # background: reward it. (One side is enough: crop / clean-up puzzles have the pattern on the input only.)
   # (Demos only, like the toggle itself: a test input never decides it)
   if parse_config.get('pattern_background', False) and all(
         any(shape.get('_pattern_background') for side in ('input', 'output') for shape in pair[side]['shapes'])
         for pair in parsed_pairs if pair.get('output') is not None):
      config_complexity_points += POINTS_PATTERN_BACKGROUND_EVERY_PAIR_REWARD
      if complexity_parts is not None: complexity_parts['pattern_background_every_pair'] = POINTS_PATTERN_BACKGROUND_EVERY_PAIR_REWARD # debug breakdown
   score += config_complexity_points * COEFF_CONFIG_COMPLEXITY

   # Debug switch on? Print the whole breakdown (and check its parts add up to the real score)
   if DEBUG_PARSE_SCORE_BREAKDOWN:
      _print_score_breakdown(parse_config, score, complexity_parts, config_complexity_points, pair_breakdowns)

   return score


# (most_common_puzzle_color can be None = no free color: the bg check below handles that fine)
def _score_config_complexity(parse_config:dict, most_common_puzzle_color:str|None,
                             complexity_parts:dict|None = None) -> float: # complexity_parts: where the points came from (debug only)
   """
   Count how "complex" this parse config is. More points = more complex.
   We want to regularize towards the simpler/default settings.

   Parsing connected shapes:
   - multicolor=True costs 1 point (default False is simpler)
   - Connectivity: 8-connected (the default) costs 0 points, 4-connected (no diagonals) costs 1 point,
     rows-only or columns-only costs 3 points
   - radius > 1 costs 1 point (default 1 is simpler)
   - denoise=True costs POINTS_DENOISE (default False is simpler); autocomplete_rects=True (always with denoise)
     costs POINTS_AUTOCOMPLETE_RECTS more on top
   - pattern_background=True costs POINTS_PATTERN_BACKGROUND (default False is simpler)
   - greedy_input_shapes_static=True costs POINTS_GREEDY_INPUT_SHAPES_STATIC (default False is simpler)
   - greedy_input_shapes_moving=True costs POINTS_GREEDY_INPUT_SHAPES_MOVING, on top
   - each chunk in 'chunks' costs POINTS_PARSE_CHUNK_PER_PIXEL x its pixel area
     (pixels that actually exist in it, not width x height) (default: no chunks)

   Parsing background:
   - bg_mode='most_common_input_color' costs 0 points (the simplest: ONE bg per pair, the input's)
   - bg_mode='most_common_scene_color' costs 1.1 points (each grid picks its own bg; the .1 breaks exact ties towards input-color)
   - bg_mode='most_common_edge_color' costs 1 point (the input's most common border color, for its output too)
   - bg_mode='no_bg' costs 2 points
   - bg_mode='cN' (hardcoded color) costs 6 points
     ...UNLESS it's the puzzle's most common color (over every grid): then it costs 1 point
   """
   
   complexity_points:float = 0.0

   # Multicolor is the non-default, more complex option
   if parse_config['multicolor']:
      complexity_points += 1.0
      if complexity_parts is not None: complexity_parts['multicolor'] = 1.0 # debug breakdown

   # Connectivity: one price per case. All 8 directions is the default, so it's free
   diagonals:bool = parse_config['diagonals']
   columns:bool = parse_config['columns']
   rows:bool = parse_config['rows']
   connectivity_name:str
   connectivity_points:float
   if diagonals and columns and rows:
      connectivity_name, connectivity_points = '8-connected', 0.0
   elif columns and rows:
      # 4-connected: no diagonals, but still both straight directions
      connectivity_name, connectivity_points = '4-connected', 1.0
   elif columns != rows:
      # Rows-only or columns-only: shapes only connect along ONE direction, a strong claim
      connectivity_name, connectivity_points = '2-connected', 3.0
   else:
      # Anything else (only hand-written test configs, search_parse never makes these): price it like the strangest case
      connectivity_name, connectivity_points = 'other connectivity', 3.0
   complexity_points += connectivity_points
   if complexity_parts is not None and connectivity_points > 0: complexity_parts[connectivity_name] = connectivity_points # debug breakdown
   # A connection radius > 1 (shapes connect across small gaps) is non-default too: +1 point, whatever the radius.
   # (.get() since configs without the key, e.g. with PARSE_CONNECTION_RADIUS_ENABLED off, are radius 1)
   if parse_config.get('radius', 1) > 1:
      complexity_points += 1.0
      if complexity_parts is not None: complexity_parts['radius'] = 1.0 # debug breakdown
   # De-noising shapes is non-default too
   if parse_config.get('denoise', False):
      complexity_points += POINTS_DENOISE
      if complexity_parts is not None: complexity_parts['denoise'] = POINTS_DENOISE # debug breakdown
   # ...and completing hidden rectangles on top of it is one more claim about the scene
   if parse_config.get('autocomplete_rects', False):
      complexity_points += POINTS_AUTOCOMPLETE_RECTS
      if complexity_parts is not None: complexity_parts['autocomplete_rects'] = POINTS_AUTOCOMPLETE_RECTS # debug breakdown
   # ...and so is a pattern background behind everything
   if parse_config.get('pattern_background', False):
      complexity_points += POINTS_PATTERN_BACKGROUND
      if complexity_parts is not None: complexity_parts['pattern_background'] = POINTS_PATTERN_BACKGROUND # debug breakdown

   # Greedily carrying input shapes over into the output parse is non-default, too.
   # (.get() so hand-written configs in the engine test scripts, which don't have this key, still score)
   if parse_config.get('greedy_input_shapes_static', False):
      complexity_points += POINTS_GREEDY_INPUT_SHAPES_STATIC
      if complexity_parts is not None: complexity_parts['greedy_input_shapes_static'] = POINTS_GREEDY_INPUT_SHAPES_STATIC # debug breakdown
   # ...and grabbing input shapes that MOVED in the output costs its own point, too
   if parse_config.get('greedy_input_shapes_moving', False):
      complexity_points += POINTS_GREEDY_INPUT_SHAPES_MOVING
      if complexity_parts is not None: complexity_parts['greedy_input_shapes_moving'] = POINTS_GREEDY_INPUT_SHAPES_MOVING # debug breakdown

   # Every repeated chunk the parser has to remember costs a bit, too (base configs have none).
   # By its pixel area: every pixel that actually exists in its shadow (NOT width x height), since that's
   # how much the parser has to "remember" to know the chunk. (Times POINTS_PARSE_CHUNK_PER_PIXEL)
   for chunk in parse_config.get('chunks', []):
      chunk_points:float = POINTS_PARSE_CHUNK_PER_PIXEL * np.count_nonzero(chunk)
      complexity_points += chunk_points
      if complexity_parts is not None: complexity_parts['chunks'] = complexity_parts.get('chunks', 0.0) + chunk_points # debug breakdown

   # Background: one TOTAL price per bg mode
   bg_mode:str = parse_config['bg_mode']
   bg_points:float
   if bg_mode == 'most_common_input_color':
      # The simplest (MDL-wise): the input and output share ONE bg, instead of each grid picking its own
      bg_points = 0.0
   elif bg_mode == 'most_common_scene_color':
      # Each grid picks its own bg: its most common color.
      # The extra .1 is a tie-breaker: on an exact tie, the simpler one-bg-per-pair input-color parse wins
      bg_points = 1.1
   elif bg_mode == 'most_common_edge_color':
      # The input's most common border color (its output shares it)
      bg_points = 1.0
   elif bg_mode == 'no_bg':
      # No background at all: more complex than the majority-color modes, but simpler than a hardcoded color
      bg_points = 2.0
   elif bg_mode == most_common_puzzle_color:
      # Hardcoding the puzzle's most common color is the "obvious" background, so it's cheap
      bg_points = 1.0
   else:
      # Any other hardcoded color ('c0', 'c3', ...) is a strong claim ("THIS color is the bg, everywhere"),
      # and weird bg colors (even ones missing from most grids) kept winning by lumping the whole grid into a few big shapes
      bg_points = 6.0
   complexity_points += bg_points
   if complexity_parts is not None and bg_points > 0: complexity_parts[f'bg {bg_mode}'] = bg_points # debug breakdown
   return complexity_points


############################################################
# The estimated cost of change between one before/after scene pair
############################################################

# Reads the RAW pairs (same grids as the parsed scenes' '_raw'), so search_parse can call it ONCE,
# before any config gets parsed
def _find_free_color(raw_pairs:list[dict]) -> str|None:
   """The color that covers >= 50% of its grid, in >= 50% of the puzzle's grids (train in/out + test in). Or None."""
   majority_votes:Counter = Counter()
   grid_count:int = 0
   for raw_pair in raw_pairs:
      for grid in (raw_pair['input'], raw_pair.get('output')):
         if grid is None: # skip no-output Tests
            continue
         grid_count += 1
         # This grid's top color: does it cover at least half the grid? Then it gets a vote
         # (A 50/50 grid votes for a random one of its two colors, not the first one in raster order)
         top_color, top_count = _random_most_common(Counter(grid.ravel().tolist()))
         if top_count >= 0.5 * grid.size:
            majority_votes[top_color] += 1

   # Any color voted for by at least half the grids? (Only one color can be, unless it's an exact 50/50 split:
   # then it's a random one of the tied colors)
   if majority_votes:
      voted_color, vote_count = _random_most_common(majority_votes)
      if vote_count >= 0.5 * grid_count:
         return 'c' + str(voted_color) # same 'cN' format as bg_mode
   return None


# chunk_shadow_keys: every chunk's key, in all D4 orientations
def _score_change(input_scene:dict, output_scene:dict, chunk_shadow_keys:set[tuple],
                  allow_outlines_and_rays:bool, # False = no outline or ray prices (see score_parse for when)
                  allow_hole_fills:bool, # False = no hole-fill price (rows-only or columns-only parses)
                  breakdown:dict|None = None) -> float: # breakdown: a dict to fill in with the cost's parts (debug only, else None)
   """
   How many "points" does it cost to explain the change from input scene to output scene?
   Lower = cheaper change = better parse.

   Each output shape "claims" the unclaimed input shape it's cheapest to edit into
   (see _edit_distance), or, if nothing's cheaper, it's made de-novo. De-novo prices, cheapest first:
   - its pixels are exactly one or more whole input holes (see _fills_holes), and it isn't an input shape left in
     place (same shadow, same spot): POINTS_SHAPE_FILLS_HOLE.
     Only for one-color shapes (fill_holes paints one color), and never in a rows-only or columns-only parse.
     Priced before the claims, so a hole fill never claims (recolors) some same-shaped input shape instead
   - an outline of another shape (see _is_outline): POINTS_SHAPE_OUTLINE_OR_RAY (outline width + color).
   - a ray shot from a shape it touches (see _is_ray): POINTS_SHAPE_OUTLINE_OR_RAY (ray direction + color).
     Only for one-color shapes, and never in a rows-only, columns-only or no_bg parse,
     or a level where any demo pair changes grid size: those pay a price below
   - shadow is one of the config's chunks (any D4 orientation, any color): POINTS_SHAPE_ADDED + 1,
     as cheap as a new dot (the chunk's pixels were already paid for in the config's cost)
   - a one-color filled rectangle (shadow fills its bbox, lines too): POINTS_SHAPE_ADDED + 2 (corner, color, width & height),
     or the chunk/area price below if that's cheaper (so a dot stays POINTS_SHAPE_ADDED + 1)
   - a piece cut straight out of the input's pattern background (see _is_cut_from_pattern): POINTS_SHAPE_ADDED + 2 too
   - anything else (multicolor too): POINTS_SHAPE_ADDED + area (the pixels actually drawn)
   Edits of a claimed input shape (see _edit_distance): 0 if unchanged, + a move (POINTS_SHAPE_MOVE,
   or POINTS_SHAPE_MOVE_WHEN_RESIZED if the scene changed size), + POINTS_SHAPE_RECOLOR, + POINTS_SHAPE_TRANSFORMED (a flip/rotate).
   Each input shape that nobody claimed got deleted: POINTS_SHAPE_DELETED if the scene kept its size.
   If it resized, a deletion is free (cropped out), unless the shape's bbox is bigger than the whole output grid.
   The scene: + POINTS_SCENE_SIZE_CHANGED if its size changed, + POINTS_SCENE_BG_CHANGED if its background color changed.
   """

   points:float = 0.0
   input_shapes:list[dict] = input_scene['shapes']
   output_shapes:list[dict] = output_scene['shapes']

   # Efficiency: Precompute every input shape's match key once (instead of once per output shape)
   input_keys:list[tuple] = [_shadow_key(_match_grid(input_shape)) for input_shape in input_shapes]
   # Speedup: which input shapes (by index, in list order) have each match key? Only same-key (or flipped/rotated)
   # input shapes can ever beat the de-novo price (see the assert at the top), so those are the only ones worth checking
   input_indices_by_key:dict[tuple, list[int]] = {}
   for input_index, input_key in enumerate(input_keys):
      input_indices_by_key.setdefault(input_key, []).append(input_index)
   # Which input shapes (by index) have already been claimed by an output shape.
   # (A local set instead of marking the shape dicts, so we never modify the scenes = no deep copy needed)
   claimed_input_indices:set[int] = set()

   # Same size? The move price depends on it: if the scene changed size (a crop shifts everyone),
   # moves get their own price (POINTS_SHAPE_MOVE_WHEN_RESIZED), so reusing shapes can still win
   is_same_size:bool = (input_scene['width'] == output_scene['width']
                        and input_scene['height'] == output_scene['height'])
   move_points:float = POINTS_SHAPE_MOVE if is_same_size else POINTS_SHAPE_MOVE_WHEN_RESIZED
   # The cheapest any claim can possibly be: 0, or one move if moves are a reward (for the early stop)
   cheapest_possible_distance:float = min(0.0, move_points)

   # Every input hole, as a mask over the whole input scene, so a new shape that's exactly some holes is a cheap check.
   # (Empty = no hole-fill prices: rows/columns-only parse, or no input shape has a hole)
   input_hole_masks:list[np.ndarray] = _hole_masks(input_scene) if allow_hole_fills else []
   # Every input shape's (top, left, shadow): a hole fill has to be NEW, so an output shape with the exact same shadow at
   # the exact same spot as an input shape (e.g. a square nested in a frame's hole, left alone) just claims it instead
   input_spots:set[tuple] = {(input_shape['top'], input_shape['left'], _shadow_key(input_shape['_shadow']))
                             for input_shape in input_shapes} if input_hole_masks else set()
   # The input's pattern background (a pattern parse has at most one), so a new shape cut out of it is cheap. (None = no pattern)
   input_pattern_pixels:np.ndarray|None = next((input_shape['_pixels'] for input_shape in input_shapes
                                                if input_shape.get('_pattern_background')), None)

   ########################################
   # Each output shape: which input shape did it (most cheaply) come from, if any?
   ########################################

   # NOTE: claiming is greedy, so it depends on order! If two output shapes want the same
   # input shape, whichever comes first in the (shuffled) output list gets it. Within one output shape's pick,
   # a tie between equally cheap input shapes goes to the closest one (see the tie key below), never the shuffle.
   # Each output shape's points, which input shape it came from (None = de-novo), and its match keys (for the breakdown)
   output_points:list[float] = []
   output_came_from_indices:list[int|None] = []
   output_keys:list[tuple[tuple, set[tuple]]] = []
   for output_shape in output_shapes:

      # Precompute this output shape's match keys once: as-is, and under the 7 other non-identity D4 transforms
      output_grid:np.ndarray = _match_grid(output_shape)
      output_key:tuple = _shadow_key(output_grid)
      # (Looked up from the cache: the same few shadows come up over and over, across all configs)
      output_transformed_keys:set[tuple] = _cached_d4_keys(output_grid, output_key)

      #  Worst case: it came from nothing, made fully de-novo.
      # Pay only for the pixels actually drawn (area), not its whole bounding box, so a sparse shape
      # isn't charged for the empty cells around it. (Outlines and rays get their own cheap price below)
      # Shaped like one of the config's chunks? Then it's as cheap as a new dot: POINTS_SHAPE_ADDED (x, y, color)
      # +1 (which chunk). Matched by shadow, so any color & any D4 orientation, no extra points for either.
      if _shadow_key(output_shape['_shadow']) in chunk_shadow_keys:
         de_novo_price:float = POINTS_SHAPE_ADDED + 1
      else:
         de_novo_price:float = POINTS_SHAPE_ADDED + output_shape['area']
      # A one-color filled rectangle (lines included) is just corner + color + width + height: POINTS_SHAPE_ADDED + 2.
      # (Only one-color ones: a multicolor rectangle still has to say every pixel's color. A dot stays cheaper)
      if output_shape['monochrome'] and output_shape['_shadow'].all():
         de_novo_price = min(de_novo_price, POINTS_SHAPE_ADDED + 2)
      # A piece cut straight out of the input's pattern background (same colors, unturned) is just where it's cut from +
      # width + height, priced like a rectangle. (Only checked when that'd be cheaper, i.e. 3+ pixels, so it stays cheap)
      if (input_pattern_pixels is not None and de_novo_price > POINTS_SHAPE_ADDED + 2
          and _is_cut_from_pattern(output_shape, input_pattern_pixels)):
         de_novo_price = POINTS_SHAPE_ADDED + 2
      # A one-color shape that's exactly one or more whole input holes? That's just fill_holes(): cheapest of all.
      # Priced here (not after the claims, like outlines), so it never "comes from" a same-shaped input shape it'd recolor
      if (input_hole_masks and output_shape['monochrome']
          and (output_shape['top'], output_shape['left'], _shadow_key(output_shape['_shadow'])) not in input_spots
          and _fills_holes(output_shape, input_hole_masks)):
         de_novo_price = POINTS_SHAPE_FILLS_HOLE
      min_edit_distance:float = de_novo_price
      came_from_index:int|None = None
      # The claimed input shape's tie-break key: (Chebyshev distance between the top-left corners, its top, its left).
      # Equally cheap input shapes go to the closest one, then the top-left-most, so the claim never depends on the shuffle
      best_tie_key:tuple[int, int, int]|None = None
      output_top:int = output_shape['top']
      output_left:int = output_shape['left']

      # for input_index, input_shape in enumerate(input_shapes): # (the walk-every-input-shape loop, see the assert at the top)
      # Only the input shapes with a matching key (as-is or any flip/rotation). (A differently-shaped one costs >= de-novo, never wins)
      matching_input_indices:set[int] = {input_index for matching_key in {output_key} | output_transformed_keys
                                         for input_index in input_indices_by_key.get(matching_key, ())}
      for input_index in matching_input_indices:
         input_shape:dict = input_shapes[input_index]

         # Already claimed by an earlier output shape? Skip
         if input_index in claimed_input_indices:
            continue

         # Otherwise, how far is it from this output shape? Pricier than the best so far = no chance
         current_distance:float = _edit_distance(input_shape, input_keys[input_index], output_shape, output_key, output_transformed_keys, de_novo_price, move_points)
         if current_distance > min_edit_distance:
            continue
         # Cheaper than the best so far? It wins. Equally cheap as an input shape we already claim? The closer one wins.
         # (Equal to de-novo with nothing claimed yet doesn't count: then de-novo stays, like before)
         tie_key:tuple[int, int, int] = (max(abs(input_shape['top'] - output_top), abs(input_shape['left'] - output_left)),
                                         input_shape['top'], input_shape['left'])
         if current_distance < min_edit_distance or (best_tie_key is not None and tie_key < best_tie_key):
            min_edit_distance = current_distance
            came_from_index = input_index
            best_tie_key = tie_key

            # Efficiency: as cheap as a claim can possibly be, at the very same spot? Nothing can beat it, early stop
            # (If moves are a reward, a moved twin beats a same-spot one, so stop at the true minimum, not just 0)
            if current_distance <= cheapest_possible_distance and tie_key[0] == 0:
               break

      # Remember its cost, and claim its input shape (only if there IS one cheaper than de-novo)
      output_points.append(min_edit_distance)
      output_came_from_indices.append(came_from_index)
      output_keys.append((output_key, output_transformed_keys))
      if came_from_index is not None:
         claimed_input_indices.add(came_from_index)

   ########################################
   # Each de-novo output shape: is it just an outline of another shape, or a ray shot from one? Then it's cheap
   ########################################

   # Done AFTER all the claims: a de-novo shape never claims anything, so its price can't change any claim.
   # Which shapes can an outline be drawn around (or a ray shot from)? Every input shape, plus the output shapes that came from an
   # input shape and changed on the way (an unchanged one is the same pixels as its input shape, already a source).
   # New output shapes don't count as sources: otherwise two new shapes side by side would "explain" each other.
   outline_source_shapes:list[dict] = input_shapes + [output_shape for output_shape, came_from_index, output_price
                                                      in zip(output_shapes, output_came_from_indices, output_points)
                                                      if came_from_index is not None and output_price > 0]
   # A ray is something NEW: so every input shadow (as-is), to rule out new shapes that are just copies of input shapes
   input_shadow_keys:set[tuple] = {_shadow_key(input_shape['_shadow']) for input_shape in input_shapes}
   for output_index, output_shape in enumerate(output_shapes):
      # Only one-color new shapes can be outlines or rays: outline() and rays paint one color,
      # so a multicolor new shape (like a ring + its fill) always pays the normal price
      # Only new shapes that'd cost more than an outline (or ray)
      if (allow_outlines_and_rays and output_shape['monochrome']
          and output_came_from_indices[output_index] is None and output_points[output_index] > POINTS_SHAPE_OUTLINE_OR_RAY):
         if (_is_outline(output_shape, outline_source_shapes, output_scene['width'], output_scene['height'])
             or _is_ray(output_shape, outline_source_shapes, input_shadow_keys)):
            output_points[output_index] = POINTS_SHAPE_OUTLINE_OR_RAY

   # Add up every output shape's cost
   points += sum(output_points)

   # Debug switch on? Write down what each output shape was matched as, and its points
   if breakdown is not None:
      for output_shape, (output_key, output_transformed_keys), came_from_index, output_price in zip(
            output_shapes, output_keys, output_came_from_indices, output_points):
         _record_output_shape(breakdown, output_shape, output_key, output_transformed_keys,
                              input_shapes, input_keys, came_from_index, output_price)

   ########################################
   # Each unclaimed input shape: deleted! (Free if it was just cropped out by a resize)
   ########################################

   # Same size? Every deletion costs POINTS_SHAPE_DELETED.
   unclaimed_shapes:list[dict] = [input_shape for input_index, input_shape in enumerate(input_shapes) if input_index not in claimed_input_indices]
   if is_same_size:
      paid_deletions:list[dict] = unclaimed_shapes
   else:
      # Resized? Then a missing shape probably just got cropped out of frame, so it's free...
      # ...unless its bbox is bigger than the whole output grid (e.g. a giant bg piece): that can't just be a crop
      # TODO: maybe something cleverer, like convolving the output over the input to see what really got cropped out
      output_area:int = output_scene['width'] * output_scene['height']
      paid_deletions = [input_shape for input_shape in unclaimed_shapes
                        if input_shape['width'] * input_shape['height'] > output_area]
   deletion_points:float = len(paid_deletions) * POINTS_SHAPE_DELETED
   points += deletion_points
   if breakdown is not None:
      breakdown['deleted'] = (len(unclaimed_shapes), deletion_points)

   ########################################
   # Score the scene as a whole: did the size or background change?
   ########################################

   if not is_same_size:
      points += POINTS_SCENE_SIZE_CHANGED
   if input_scene['background'] != output_scene['background']:
      points += POINTS_SCENE_BG_CHANGED

   # Debug breakdown: the scene changes, plus this pair's real total (to check the parts add up)
   if breakdown is not None:
      breakdown['size_changed'] = 0.0 if is_same_size else POINTS_SCENE_SIZE_CHANGED
      breakdown['bg_changed'] = POINTS_SCENE_BG_CHANGED if input_scene['background'] != output_scene['background'] else 0.0
      breakdown['input_shape_count'] = len(input_shapes)
      breakdown['output_shape_count'] = len(output_shapes)
      breakdown['total'] = points

   return points


def _edit_distance(input_shape:dict, input_key:tuple, output_shape:dict, output_key:tuple, output_transformed_keys:set[tuple], de_novo_price:float,
                   move_points:float) -> float: # move_points: POINTS_SHAPE_MOVE, or POINTS_SHAPE_MOVE_WHEN_RESIZED if the scene changed size
   """
   How many edits turn this input shape into this output shape?
   (Takes the precomputed match keys, so we don't recompute D4 transforms for every pair of shapes)
   - 0 if it's the exact same shadow, color & position
   - +move_points if it moved (left and/or top changed: one charge, not one per axis)
   - +POINTS_SHAPE_RECOLOR if color changed
   - +POINTS_SHAPE_TRANSFORMED if it's a non-identity flip/rotate
   - + the output shape's de-novo price if the input's shadow isn't a D4 transform of the output's at all
     (see _score_change: POINTS_SHAPE_ADDED + area, or cheaper if it's chunk-shaped or a one-color rectangle)
   """
   distance:float = 0.0

   # Moved? One move = one charge, not one per axis (so a move costs the same as a deletion)
   # TODO: account for cropping! After a crop, EVERY shape's left/top shifts by the crop's offset
   if input_shape['left'] != output_shape['left'] or input_shape['top'] != output_shape['top']:
      distance += move_points

   # Recolored? (For multicolor shapes, this is their majority color)
   # TODO: multicolor shapes match by their exact pixels, so ANY recoloring inside one (even a
   # subtle, uniform one) counts as a whole new shape (the de-novo price). This may cause failures later
   # for subtle recolorings, but we'd rather it be a loud failure than a silent one.
   if input_shape['color'] != output_shape['color']:
      distance += POINTS_SHAPE_RECOLOR

   # Same shape as-is: no reshaping cost. (Checked FIRST, so symmetrical shapes that
   # also match under some flip don't wrongly count as "flipped")
   # TODO: scale too, eventually (a shape that got 2x bigger is cheaper than a brand new shape)
   if input_key == output_key:
      pass
   # Same shape after a flip/rotate: +1
   elif input_key in output_transformed_keys:
      distance += POINTS_SHAPE_TRANSFORMED
   # Not the same shape at all: a different silhouette isn't "this shape, edited", it's a brand-new shape.
   # So it pays the same de-novo price _score_change worked out (plus move/recolor above). That way a mismatched
   # claim can never beat just making the shape de-novo, so a new shape never "steals" an unrelated input shape
   # from the output shape that really came from it (which would make scores depend on the shape order)
   else:
      distance += de_novo_price

   return distance


def _is_outline(new_shape:dict, source_shapes:list[dict], scene_width:int, scene_height:int) -> bool:
   """Does this new shape hold (nearly) all of some source shape's 1-pixel ring? (= an outline of it)"""
   new_top:int = new_shape['top']
   new_left:int = new_shape['left']
   new_bottom:int = new_top + new_shape['height']
   new_right:int = new_left + new_shape['width']
   # By shadow, not pixels: so an outline can be any color, even different from its shape's
   new_shadow:np.ndarray = new_shape['_shadow'] != 0

   # Which sides of the new shape touch the scene's edge? There, the outline may have been cropped off by the camera
   touches_top:bool = new_top <= 0
   touches_left:bool = new_left <= 0
   touches_bottom:bool = new_bottom >= scene_height
   touches_right:bool = new_right >= scene_width

   for source_shape in source_shapes:
      # Cheap reject first: an outline SURROUNDS its shape, so the outline's box must contain the shape's box
      # grown by 1 on every side (except cropped sides). This stops a strip from "outlining" its same-width neighbor.
      source_top:int = source_shape['top']
      source_left:int = source_shape['left']
      source_bottom:int = source_top + source_shape['height']
      source_right:int = source_left + source_shape['width']
      if ((not touches_top and new_top > source_top - 1) or (not touches_left and new_left > source_left - 1)
          or (not touches_bottom and new_bottom < source_bottom + 1) or (not touches_right and new_right < source_right + 1)):
         continue
      # Skip the new shape itself (it can be in the sources if it's an output shape that changed)
      if source_shape is new_shape:
         continue

      # The ring outline() would paint around the source. Just the ring, NOT the source's own pixels:
      # else any shape that changed in place would count as its own "outline".
      # Its grid starts PARSE_OUTLINE_WIDTH up-left of the source.
      ring_mask:np.ndarray = _cached_outline_ring(source_shape['_shadow'])
      ring_top:int = source_top - PARSE_OUTLINE_WIDTH
      ring_left:int = source_left - PARSE_OUTLINE_WIDTH

      # Only the ring pixels inside the scene count: the camera cropped the rest, so nobody could've drawn them
      visible_top:int = max(ring_top, 0)
      visible_left:int = max(ring_left, 0)
      visible_bottom:int = min(ring_top + ring_mask.shape[0], scene_height)
      visible_right:int = min(ring_left + ring_mask.shape[1], scene_width)
      visible_ring:np.ndarray = ring_mask[visible_top - ring_top:visible_bottom - ring_top, visible_left - ring_left:visible_right - ring_left]
      visible_ring_count:int = int(np.count_nonzero(visible_ring))
      if visible_ring_count == 0:
         continue

      # Where do the new shape's box and the visible ring's box overlap? (In grid coordinates)
      overlap_top:int = max(new_top, visible_top)
      overlap_left:int = max(new_left, visible_left)
      overlap_bottom:int = min(new_bottom, visible_bottom)
      overlap_right:int = min(new_right, visible_right)
      if overlap_top >= overlap_bottom or overlap_left >= overlap_right:
         continue

      # How much of the visible ring does the new shape cover? Nearly all of it = an outline.
      # (The new shape may hold more than this one ring: several touching rings merge into one shape)
      new_slice:np.ndarray = new_shadow[overlap_top - new_top:overlap_bottom - new_top, overlap_left - new_left:overlap_right - new_left]
      ring_slice:np.ndarray = visible_ring[overlap_top - visible_top:overlap_bottom - visible_top, overlap_left - visible_left:overlap_right - visible_left]
      covered_count:int = int(np.count_nonzero(new_slice & ring_slice))
      if covered_count >= PARSE_OUTLINE_COVERED_FRACTION * visible_ring_count:
         return True

   return False

# The 8 ray directions (dy, dx)
_RAY_DIRECTIONS:list[tuple[int, int]] = [(direction_y, direction_x) for direction_y in (-1, 0, 1)
                                         for direction_x in (-1, 0, 1) if (direction_y, direction_x) != (0, 0)]

def _is_ray(new_shape:dict, caster_shapes:list[dict], input_shadow_keys:set[tuple]) -> bool:
   """Is this new shape nothing but rays shot out of a shape it touches, with at least one full PARSE_RAY_LENGTH-pixel ray?"""
   # Too small to hold a whole ray? Then it can't be one
   if new_shape['area'] < PARSE_RAY_LENGTH:
      return False
   # Filter 1: a ray is something new, so its shadow can't be anywhere in the input (at any spot, in any flip/rotation).
   # Else it's more likely a copy of that input shape
   new_key:tuple = _shadow_key(new_shape['_shadow'])
   if new_key in input_shadow_keys or not _cached_d4_keys(new_shape['_shadow'], new_key).isdisjoint(input_shadow_keys):
      return False

   new_top:int = new_shape['top']
   new_left:int = new_shape['left']
   new_height:int = new_shape['height']
   new_width:int = new_shape['width']
   # By shadow, not pixels: so a ray can be any color, even different from its caster's
   new_shadow:np.ndarray = new_shape['_shadow'] != 0

   for caster_shape in caster_shapes:
      # Skip the new shape itself (it can be in the casters if it's an output shape that changed)
      if caster_shape is new_shape:
         continue
      # Filter 2: only a shape touching the new one (diagonals too) can have shot it. Cheap bbox check here:
      # the caster's box grown by 1 must overlap the new shape's box. (Filter 3 then needs real touching anyway:
      # a ray's first pixel is right next to its caster)
      caster_top:int = caster_shape['top']
      caster_left:int = caster_shape['left']
      if (caster_top - 1 >= new_top + new_height or caster_top + caster_shape['height'] + 1 <= new_top
          or caster_left - 1 >= new_left + new_width or caster_left + caster_shape['width'] + 1 <= new_left):
         continue

      # The caster's pixels, padded by 1 so "the next pixel over" never falls off its grid
      caster_filled:np.ndarray = np.pad(caster_shape['_shadow'] != 0, 1, constant_values=False)
      caster_ys, caster_xs = np.nonzero(caster_filled)
      # Rays this long reach past the far side of the new shape's box, from anywhere on the caster
      ray_steps:np.ndarray = np.arange(1, max(caster_shape['height'] + new_height, caster_shape['width'] + new_width, PARSE_RAY_LENGTH) + 1)

      # Shoot a ray from every caster pixel on its edge, in all 8 directions (no blockers, so no collisions)
      on_rays:np.ndarray = np.zeros((new_height, new_width), dtype=bool)
      has_full_ray:bool = False
      for direction_y, direction_x in _RAY_DIRECTIONS:
         # Only from the caster's edge in this direction: pixels whose next pixel that way isn't the caster
         on_edge:np.ndarray = ~caster_filled[caster_ys + direction_y, caster_xs + direction_x]
         # Where each ray's pixels land, relative to the new shape's top-left (the -1 undoes the padding)
         ray_ys:np.ndarray = (caster_ys[on_edge] - 1 + caster_top - new_top)[:, None] + direction_y * ray_steps[None, :]
         ray_xs:np.ndarray = (caster_xs[on_edge] - 1 + caster_left - new_left)[:, None] + direction_x * ray_steps[None, :]
         inside_box:np.ndarray = (ray_ys >= 0) & (ray_ys < new_height) & (ray_xs >= 0) & (ray_xs < new_width)
         # Mark every ray pixel that lands in the new shape's box
         on_rays[ray_ys[inside_box], ray_xs[inside_box]] = True

         # Filter 3: the new shape must cover ALL of the first PARSE_RAY_LENGTH pixels of at least one ray.
         # (A ray cut short by the grid's edge doesn't count)
         if not has_full_ray:
            first_inside:np.ndarray = inside_box[:, :PARSE_RAY_LENGTH].all(axis=1)
            if first_inside.any():
               first_covered:np.ndarray = new_shadow[ray_ys[first_inside, :PARSE_RAY_LENGTH], ray_xs[first_inside, :PARSE_RAY_LENGTH]]
               has_full_ray = bool(first_covered.all(axis=1).any())

      # ...AND be nothing but this caster's rays: else a big blob next to a shape would swallow a ray and pass too
      if has_full_ray and not (new_shadow & ~on_rays).any():
         return True

   return False

def _scene_mask(local_mask:np.ndarray, top:int, left:int, scene_height:int, scene_width:int) -> np.ndarray:
   """A shape-local True/False mask, pasted into a whole-scene mask at (top, left). Clipped to the scene."""
   scene_mask:np.ndarray = np.zeros((scene_height, scene_width), dtype=bool)
   # Shapes can stick out past the scene's edges (e.g. completed rectangles), so only paste the part inside
   local_height, local_width = local_mask.shape
   scene_top, scene_left = max(top, 0), max(left, 0)
   scene_bottom, scene_right = min(top + local_height, scene_height), min(left + local_width, scene_width)
   if scene_top < scene_bottom and scene_left < scene_right:
      scene_mask[scene_top:scene_bottom, scene_left:scene_right] = local_mask[scene_top - top:scene_bottom - top, scene_left - left:scene_right - left]
   return scene_mask

def _hole_masks(scene:dict) -> list[np.ndarray]:
   """Every hole of this scene's shapes, each as a True/False mask over the whole scene."""
   # The hole grids cover their shape's box (1 = hole), so each gets pasted in at its shape's top-left corner
   return [_scene_mask(hole_grid == 1, shape['top'], shape['left'], scene['height'], scene['width'])
           for shape in scene['shapes'] for hole_grid in shape['_hole_grids']]

def _is_cut_from_pattern(shape:dict, pattern_pixels:np.ndarray) -> bool:
   """Do the shape's pixels show up, same colors, somewhere inside the pattern background's pixels?"""
   shape_pixels:np.ndarray = shape['_pixels']
   shape_height, shape_width = shape_pixels.shape
   # Bigger than the pattern? Then it can't have been cut out of it
   if shape_height > pattern_pixels.shape[0] or shape_width > pattern_pixels.shape[1]:
      return False
   # Every shape-sized window of the pattern, at once: does one agree with the shape on all of the shape's own pixels?
   # (The empty cells of the shape's box don't have to match anything)
   windows:np.ndarray = np.lib.stride_tricks.sliding_window_view(pattern_pixels, (shape_height, shape_width))
   empty_cells:np.ndarray = shape['_shadow'] == 0
   return bool(((windows == shape_pixels) | empty_cells).all(axis=(2, 3)).any())


def _fills_holes(shape:dict, hole_masks:list[np.ndarray]) -> bool:
   """Are this shape's pixels exactly the union of one or more WHOLE holes? (Holes from any shapes)"""
   # The shape's pixels, as a mask over the whole input scene (same as the holes)
   scene_height, scene_width = hole_masks[0].shape
   shape_pixels:np.ndarray = shape['_shadow'] != 0
   shape_mask:np.ndarray = _scene_mask(shape_pixels, shape['top'], shape['left'], scene_height, scene_width)
   # Any pixel off the input scene? (The output can be bigger) Then it can't be all holes
   if shape_mask.sum() != shape_pixels.sum():
      return False

   # Add up every hole that lies fully inside the shape. A hole it only partly covers doesn't count, so if the
   # shape covers part of one, those pixels stay uncovered below. (Holes can nest: a ring inside another ring's
   # hole, so a fill of just the inner hole only partly covers the outer one. That outer hole just doesn't count)
   covered_mask:np.ndarray = np.zeros_like(shape_mask)
   for hole_mask in hole_masks:
      if not (hole_mask & ~shape_mask).any():
         covered_mask |= hole_mask

   # Those whole holes make up exactly the shape? Then it's a fill (of one hole, or of several touching ones,
   # e.g. two holes that only touch diagonally, merged into one shape by an 8-connected parse)
   return bool(np.array_equal(covered_mask, shape_mask))

# Cache: a shadow's key -> its PARSE_OUTLINE_WIDTH-wide outline ring. The same few shapes come up
# in config after config, so each gets dilated once. A pure function of the shadow, so it can't change any score.
_outline_ring_cache:dict[tuple, np.ndarray] = {}

def _cached_outline_ring(shadow:np.ndarray) -> np.ndarray:
   """The ring outline() paints at line_width=PARSE_OUTLINE_WIDTH: empty cells within that many pixels (square corners)."""
   shadow_key:tuple = _shadow_key(shadow)
   ring_mask:np.ndarray|None = _outline_ring_cache.get(shadow_key)
   if ring_mask is None:
      # Pad first, so the ring has room to grow, then keep only the grown cells that aren't the shape itself
      # (holes included, same as outline() without ignore_holes)
      filled:np.ndarray = np.pad(shadow != 0, PARSE_OUTLINE_WIDTH, constant_values=False)
      ring_mask = _chebyshev_dilate(filled, PARSE_OUTLINE_WIDTH) & ~filled
      _outline_ring_cache[shadow_key] = ring_mask
   return ring_mask


############################################################
# Debug breakdown of a score (only runs when DEBUG_PARSE_SCORE_BREAKDOWN is on)
############################################################

def _record_output_shape(breakdown:dict, output_shape:dict, output_key:tuple, output_transformed_keys:set[tuple],
                         input_shapes:list[dict], input_keys:list[tuple], came_from_index:int|None, points:float) -> None:
   """Write down what one output shape got matched as (or that it was made de-novo), and its points."""

   # Made de-novo? Keep its area, colors & points, so we can list the biggest ones
   if came_from_index is None:
      shape_colors:list[int] = sorted(np.unique(output_shape['_pixels'][output_shape['_pixels'] >= 0]).tolist())
      breakdown.setdefault('de_novo', []).append({
         'points': points, 'area': output_shape['area'], 'colors': shape_colors,
         'left': output_shape['left'], 'top': output_shape['top'],
         # Got one of the cheap prices? Filling a hole is the cheapest, then outline/ray, then chunk-shaped or a one-color rectangle (see _score_change)
         'cheap_kind': ('fills a hole' if points <= POINTS_SHAPE_FILLS_HOLE
                        else 'outline or ray' if points <= POINTS_SHAPE_OUTLINE_OR_RAY
                        else 'chunk or rectangle' if points < POINTS_SHAPE_ADDED + output_shape['area'] else None),
      })
      return

   # Matched to an input shape: as what? (Same checks as _edit_distance, just spelled out as words)
   input_shape:dict = input_shapes[came_from_index]
   edits:list[str] = []
   if input_shape['left'] != output_shape['left'] or input_shape['top'] != output_shape['top']:
      edits.append('moved')
   if input_shape['color'] != output_shape['color']:
      edits.append('recolored')
   if input_keys[came_from_index] == output_key:
      pass
   elif input_keys[came_from_index] in output_transformed_keys:
      edits.append('flipped/rotated')
   else:
      edits.append('reshaped') # Pays the de-novo price on top, so it can't actually win a claim today, but just in case
   match_kind:str = '+'.join(edits) if edits else 'exact same'

   # Tally it: how many of this kind, and their points
   matched:dict = breakdown.setdefault('matched', {})
   count, kind_points = matched.get(match_kind, (0, 0.0))
   matched[match_kind] = (count + 1, kind_points + points)


# complexity_parts is "dict|None" at the call site (it's a dict whenever this runs, but the
# type checker can't tell, and the print below already treats an empty/None one as "(all defaults)")
def _print_score_breakdown(parse_config:dict, score:float, complexity_parts:dict|None, config_complexity_points:float,
                           pair_breakdowns:list[dict]) -> None:
   """Print one config's score, part by part, and check the parts add up to the real score."""
   print(f"===== score_parse breakdown: score={score:.2f}  config={parse_config}")

   # Config complexity first: which settings cost points?
   print(f"  config complexity: {config_complexity_points:.2f} pts x COEFF {COEFF_CONFIG_COMPLEXITY}  "
         f"{complexity_parts if complexity_parts else '(all defaults)'}")
   parts_sum:float = config_complexity_points * COEFF_CONFIG_COMPLEXITY

   # Then each demo pair's change cost
   for pair_index, breakdown in enumerate(pair_breakdowns):
      print(f"  demo {pair_index}: {breakdown['total']:.2f} pts  "
            f"({breakdown['input_shape_count']} input shapes -> {breakdown['output_shape_count']} output shapes)")

      # Output shapes matched to an input shape, by kind
      matched:dict = breakdown.get('matched', {})
      for match_kind, (count, kind_points) in sorted(matched.items()):
         print(f"    matched as {match_kind}: {count} shapes, {kind_points:.2f} pts")

      # De-novo shapes: the total, then the biggest few
      de_novo_shapes:list[dict] = breakdown.get('de_novo', [])
      de_novo_points:float = sum(shape['points'] for shape in de_novo_shapes)
      print(f"    de-novo: {len(de_novo_shapes)} shapes, total area {sum(shape['area'] for shape in de_novo_shapes)}, {de_novo_points:.2f} pts")
      for shape in sorted(de_novo_shapes, key=lambda shape: -shape['area'])[:5]:
         print(f"      area {shape['area']}, colors {shape['colors']}, at (left={shape['left']}, top={shape['top']}), "
               f"{shape['points']:.2f} pts{' (' + shape['cheap_kind'] + ')' if shape['cheap_kind'] else ''}")

      # Deleted input shapes, and the scene changes
      deleted_count, deleted_points = breakdown['deleted']
      print(f"    deleted: {deleted_count} input shapes, {deleted_points:.2f} pts")
      print(f"    scene: size changed {breakdown['size_changed']:.2f} pts, bg changed {breakdown['bg_changed']:.2f} pts")

      # Do this pair's parts add up to its real total?
      pair_parts_sum:float = (sum(kind_points for _, kind_points in matched.values()) + de_novo_points
                              + deleted_points + breakdown['size_changed'] + breakdown['bg_changed'])
      if abs(pair_parts_sum - breakdown['total']) > 1e-6:
         print(f"    ⚠️ MISMATCH: parts add up to {pair_parts_sum:.2f}, but the real total is {breakdown['total']:.2f}")
      parts_sum += pair_parts_sum

   # And does everything add up to the real score?
   print(f"  parts sum = {parts_sum:.2f}, real score = {score:.2f}" + ("" if abs(parts_sum - score) < 1e-6 else "  ⚠️ MISMATCH!"))


def _match_grid(shape:dict) -> np.ndarray:
   # Which grid do we compare shapes by?
   # Monochrome shapes: their shadow (silhouette).
   # Multicolor shapes should compare by their PIXELS instead (same as old scoring did),
   # because their pattern matters. Otherwise, 1 big multicolor blob can take whole scene & win
   return shape['_shadow'] if shape['monochrome'] else shape['_pixels']


def _shadow_key(shadow:np.ndarray) -> tuple:
   # A hashable key for a shadow grid: its dimensions + raw bytes
   # (dimensions too, so e.g. a 2x3 and a 3x2 grid with the same bytes don't collide)
   return (shadow.shape, shadow.tobytes())

# Cache: a match key -> the keys of its 7 non-identity D4 transforms. On big levels, the same few shadows
# show up in thousands of shapes (e.g. ~40 distinct shadows among ~1,000 shapes), so we flip/rotate each one just once.
# It's a pure function of the grid (its key = dimensions + bytes), so caching can't change any score.
# search_parse clears it at the start of each call, so it doesn't grow forever across levels.
_d4_keys_cache:dict[tuple, set[tuple]] = {}

def _cached_d4_keys(grid:np.ndarray, grid_key:tuple) -> set[tuple]:
   """The keys of a grid's 7 non-identity D4 transforms, computed once per distinct grid."""
   transformed_keys:set[tuple]|None = _d4_keys_cache.get(grid_key)
   if transformed_keys is None:
      # First time seeing this grid: flip/rotate it for real, and remember the result
      transformed_keys = {_shadow_key(transformed_grid) for transformed_grid in _non_identity_d4_transforms(grid)}
      _d4_keys_cache[grid_key] = transformed_keys
   return transformed_keys

def clear_score_parse_caches() -> None:
   """Forget every cached D4 key and outline ring (search_parse calls this at the start of each search)."""
   _d4_keys_cache.clear()
   _outline_ring_cache.clear()

def _non_identity_d4_transforms(grid:np.ndarray) -> list[np.ndarray]:
   # All 7 non-identity D4 transforms of a grid (3 rotations + 4 flips), using the
   # existing grid helpers. (Some may be identical to the original, e.g. for symmetrical shapes.)
   transformed_grids:list[np.ndarray] = [_rotate_grid(grid, quarters) for quarters in (1, 2, 3)]
   transformed_grids += [_flip_grid(grid, mode) for mode in ('horizontal', 'vertical', 'diagonal_1', 'diagonal_2')]
   return transformed_grids

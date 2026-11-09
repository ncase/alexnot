import random
from collections import Counter
from typing import Optional
import numpy as np
import scipy.ndimage
import scipy.signal # FFT correlation, for the partial chunk fits
from ..grid import _flood_fill_select_all_shapes
from .score_parse import _shadow_key, _non_identity_d4_transforms # D4 helpers, for grabbing repeated chunks in any orientation
from ..utils.random_ties import _random_most_common # most_common, with random tie-breaks
from .pattern_background import find_pattern_background, PatternBackground # a big tiled / mirrored picture behind everything

# CONFIG KNOBS
from ..config import MAJORITY_TIE_FRACTION
from ..config import GREEDY_MOVING_MIN_AREA # smallest input shape the moving grab tries

# Optional input_scenes: already-parsed input scenes (one per raw pair, same order), so search_parse can parse
# each input once and share it across the greedy configs (greedy flags only change the OUTPUT parse). None = parse them here
# Optional plain_output_scenes: the same pairs' outputs parsed with NO greedy grabs (one per raw pair, same order,
# None for tests). A greedy output parse that grabs nothing is exactly that plain parse, so it reuses it. None = don't
# Optional max_shapes: search_parse's shape cap, passed on to every DEMO grid's parse (see apply_parse_one_grid). Test
# inputs never get it, so they always get their full parse. None = no cap
def apply_parse(parse_config:dict, raw_pairs:list[dict], input_scenes:Optional[list[dict]]=None,
                plain_output_scenes:Optional[list[dict|None]]=None, max_shapes:Optional[int]=None) -> list[dict]: # TODO flip param names for consistency with one_grid?
    """
    Apply the parser config to ALL pairs
    """
    parsed_train_pairs:list[dict] = []

    for pair_index, raw_pair in enumerate(raw_pairs): # (index, to find this pair's shared input scene)

        # Get raws
        raw_input_grid:np.ndarray = raw_pair['input']
        raw_output_grid:np.ndarray|None = raw_pair['output'] # CAN BE NONE coz we're taking test pairs, too

        # Parse input first, then output (output may use input's bg via 'most_common_input_color')
        # Reuse the shared input scene if we got one. (Read-only from here on: the output parse & the
        # greedy grabs only read it. The ARC-3 controls below DO write to it, but always the same values for the same pair)
        # (The shape cap only goes to demo grids: a test input has no output, and must always get its full parse)
        demo_max_shapes:int|None = max_shapes if raw_output_grid is not None else None
        input_scene:dict = input_scenes[pair_index] if input_scenes is not None else apply_parse_one_grid(raw_input_grid, parse_config, max_shapes=demo_max_shapes)
        # The output parse also gets the already-parsed input scene (so the greedy grabs can reuse its shapes, no
        # re-flood-fill of the input), + this pair's plain (no-greedy) output scene, if we got one, for when the greedy grabs grab nothing
        plain_output_scene:dict|None = plain_output_scenes[pair_index] if plain_output_scenes is not None else None
        output_scene:dict|None = apply_parse_one_grid(raw_output_grid, parse_config, input_grid=raw_input_grid, input_scene=input_scene,
                                                      plain_output_scene=plain_output_scene, max_shapes=demo_max_shapes) if raw_output_grid is not None else None # again, can be None coz test

        # (FOR ARC-3: Transfer over CONTROLS & GAME_STATE data!)
        if 'controls' in raw_pair:

            action:int = raw_pair['controls']['action']

            # ACTION1 = Up, ACTION2 = Down, ACTION3 = Left, ACTION4 = Right
            input_scene['controls_x'] = 0
            input_scene['controls_y'] = 0
            input_scene['controls_direction'] = "none"
            input_scene['controls_vertical'] = (action==1 or action==2)
            input_scene['controls_horizontal'] = (action==3 or action==4)
            if action>=1 and action<=4:
                input_scene['controls_arrow'] = True
                if action==1:
                    input_scene['controls_direction'] = "up"
                    input_scene['controls_y'] = -1
                if action==2:
                    input_scene['controls_direction'] = "down"
                    input_scene['controls_y'] = +1
                if action==3:
                    input_scene['controls_direction'] = "left"
                    input_scene['controls_x'] = -1
                if action==4:
                    input_scene['controls_direction'] = "right"
                    input_scene['controls_x'] = +1
            else:
                input_scene['controls_arrow'] = False

            # ACTION5 = Spacebar/Interact
            input_scene['controls_space'] = (action==5)

            # ACTION6 = Click at (x, y), coords 0–63
            input_scene['controls_click_x'] = 0
            input_scene['controls_click_y'] = 0
            if action==6:
                input_scene['controls_click'] = True
                input_scene['controls_click_x'] = raw_pair['controls']['data'][0]
                input_scene['controls_click_y'] = raw_pair['controls']['data'][1]
            else:
                input_scene['controls_click'] = False

            # GAME STATE: 0 = NOT_PLAYED / 1 = NOT_FINISHED / 2 = WIN / 3 = GAME_OVER
            if output_scene:
                if raw_pair['state']==2:
                    output_scene['WIN'] = True
                if raw_pair['state']==3:
                    output_scene['LOSE'] = True


        # Save into results
        parsed_train_pairs.append({
            'input': input_scene,
            'output': output_scene,
        })

    return parsed_train_pairs

def _most_common_color(grid:np.ndarray) -> int:
    """Most common pixel value in a grid, or -1 (no bg) if the runner-up is within MAJORITY_TIE_FRACTION of it. (Not a 50% majority)"""
    counter:Counter = Counter(grid.ravel().tolist())

    # Grab the top two colors (a one-color grid has no runner-up, so it's a clear majority)
    # No random tie-break needed: an exact tie for the top always lands in the "too close" case below
    # (runner-up count == top count), so it returns -1 no matter which tied color most_common happened to list first
    top_colors:list[tuple[int, int]] = counter.most_common(2)
    top_color, top_count = top_colors[0]
    if len(top_colors) == 1:
        return top_color

    # Runner-up too close to the top color? Then there's no clear background: act like 'no_bg'
    runner_up_count:int = top_colors[1][1]
    if runner_up_count >= top_count * (1 - MAJORITY_TIE_FRACTION):
        return -1
    return top_color

# For bg_mode 'most_common_edge_color': the most common color along the grid's outer border
def _most_common_edge_color(grid:np.ndarray) -> int:
    """Most common pixel value along the grid's edges (its outer border). Ties: a random one of them."""

    # Grab every border pixel: top & bottom rows, then the left & right columns (minus their corners, already counted)
    edge_pixels:np.ndarray = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])

    # A 1-row grid's top & bottom rows are the same row, so it gets counted twice (same for 1-column grids).
    # Harmless: doubling every count doesn't change which color wins
    edge_color, _ = _random_most_common(Counter(edge_pixels.tolist()))
    return edge_color

# What bg color does this bg mode give this grid? (Its own function, so search_parse can spot bg modes that
# land on the same color everywhere, resolved the exact same way the parse does). input_grid = None means this IS an input grid
def resolve_bg_color(grid:np.ndarray, bg_mode:str, input_grid:Optional[np.ndarray]=None) -> int:
    """The bg color (-1 = no bg) that bg_mode picks for this grid."""
    if bg_mode == 'no_bg':
        return -1
    elif bg_mode == 'most_common_input_color' and input_grid is not None:
        # This is an output scene — use the input grid's most common color as bg
        return _most_common_color(input_grid)
    elif bg_mode == 'most_common_edge_color':
        # The INPUT grid's most common border color: an output uses its input's (like most_common_input_color),
        # so a cropped or redrawn output keeps the same bg as its input. (Input & test grids: their own border)
        return _most_common_edge_color(input_grid if input_grid is not None else grid)
    elif bg_mode.startswith('c'):
        # Hardcoded color like 'c0', 'c3', 'c15'
        return int(bg_mode[1:])
    else:
        # 'most_common_scene_color' or 'most_common_input_color' when we ARE the input — same thing
        return _most_common_color(grid)

# Optional input_scene: the already-parsed input scene, only passed when parsing an OUTPUT grid.
# Needed for the greedy input shape grabs (see below); input grids & test scripts just leave it None.
# Optional plain_output_scene: this same grid parsed with the same settings but NO greedy grabs (see below)
# Optional max_shapes: search_parse's shape cap. If the flood fill finds more pieces than this, stop right there and hand back
# {'background': ..., 'too_many_shapes': True} instead of a scene, which search_parse throws out (see below). None = no cap: everything else (main_solver, the test scripts) gets the full parse
def apply_parse_one_grid(grid:np.ndarray, parse_config:dict, input_grid:Optional[np.ndarray]=None, input_scene:Optional[dict]=None,
                         plain_output_scene:Optional[dict]=None, max_shapes:Optional[int]=None)->dict:

    if not isinstance(grid, np.ndarray):
        raise TypeError(f"apply_parse_one_grid expects a numpy array, got {type(grid).__name__}")

    #grid = [ row[:] for row in original_grid ] # copy just in case

    # Dimensions
    grid_arr:np.ndarray = grid
    height:int
    width:int
    height, width = grid_arr.shape
    center_x:float = width/2
    center_y:float = height/2

    # What's the background? Determined by parse_config['bg_mode']
    bg_mode:str = parse_config.get('bg_mode', 'most_common_scene_color') # majority of this scene by default
    bg_color:int = resolve_bg_color(grid_arr, bg_mode, input_grid)

    # Get all shapes!
        
    # Greedy input shapes BEFORE flood fill: if this is an output grid, and the config says so,
    # first carry over every monochrome input shape whose pixels are still a single non-bg color here
    # (same position, maybe a new color). Those pixels get blanked to bg in a COPY of the grid, so the
    # flood fill below leaves them alone and doesn't merge them with same-colored neighbors.
    shapes:list[dict] = []
    grid_for_flood_fill:np.ndarray = grid_arr

    # Repeated chunks go even BEFORE the greedy input shapes: wherever a chunk's shadow (any D4 orientation)
    # fits on non-bg pixels, grab those pixels as one shape, and blank them out for everything after
    if parse_config.get('chunks'):
        # Reused from a per-search cache when another config already grabbed these same chunks on this same grid
        shapes, grid_for_flood_fill = _cached_greedily_grab_chunks(grid_arr, bg_color, parse_config['chunks'], parse_config.get('multicolor', False))

    # Static grab first (same position only). Which input shapes got claimed already (by index into input_scene['shapes'])? So the moving grab skips them
    claimed_input_shape_indices:set[int] = set()
    if parse_config.get('greedy_input_shapes_static', False) and input_scene is not None:
        # Start from the chunk-blanked grid (pixels already grabbed as chunks are bg there, so no input shape grabs them twice)
        greedy_input_shapes, grid_for_flood_fill, claimed_input_shape_indices = _greedily_grab_input_shapes_static(grid_for_flood_fill, bg_color, input_scene)
        shapes += greedy_input_shapes

    # Then input shapes that MOVED: wherever an (unclaimed, big enough) input shape's exact pixels show up on
    # still-unclaimed output pixels, grab them. (Starts from the chunk- & static-blanked grid, so no pixel gets grabbed twice)
    if parse_config.get('greedy_input_shapes_moving', False) and input_scene is not None:
        # greedy_input_shapes_overlap: moved shapes may share pixels with each other (e.g. pieces that
        # overlap on their connector dots). Only means anything when the moving grab is on, so it's read in here
        moved_input_shapes, grid_for_flood_fill = _greedily_grab_input_shapes_moving(grid_for_flood_fill, bg_color, input_scene, claimed_input_shape_indices,
                                                                                    allow_overlap=parse_config.get('greedy_input_shapes_overlap', False))
        shapes += moved_input_shapes

    # The greedy grabs grabbed NOTHING (and no chunks either)? Then the flood fill below would redo the exact plain,
    # no-greedy parse, so just hand back that one if we have it. (Shared, not copied: like the shared input scenes, nothing
    # downstream in search_parse changes it. It skips this parse's shuffle, so seeded tie-breaks shift a bit)
    if not shapes and plain_output_scene is not None:
        return plain_output_scene

    # Pattern background on, and this grid has a big tiled or mirrored pattern? Then that pattern, completed under
    # whatever covers it, is ONE shape at the very back, and only the pixels that break it get flood-filled below.
    # (The flood fill skips the pattern's pixels by their own marker color, not the bg color, so a break pixel
    # that happens to have the bg color, like a marker dot on a grid line, still becomes a shape)
    flood_fill_bg_color:int = bg_color
    # The color the scene says is its background (the parse itself keeps using bg_color)
    scene_bg_color:int = bg_color
    if parse_config.get('pattern_background', False):
        pattern_background = find_pattern_background(grid_arr)
        if pattern_background is not None:
            pattern_shape:dict = _pattern_background_shape(pattern_background)
            shapes.append(pattern_shape)
            # Does the pattern cover the whole grid? (It can hang past the edges, e.g. a mirrored one.) Then no single
            # color is behind everything, so the scene gets "no background" (-1), same as a no_bg parse
            if pattern_shape['top'] <= 0 and pattern_shape['left'] <= 0 and pattern_shape['bottom'] >= height and pattern_shape['right'] >= width:
                scene_bg_color = -1
            grid_for_flood_fill = np.where(pattern_background.breaks_pattern, grid_for_flood_fill.astype(np.int32), _PATTERN_PIXEL_MARKER)
            flood_fill_bg_color = _PATTERN_PIXEL_MARKER

    # Get all shapes! First, get coords...
    # (flood fill returns list of (ys, xs) numpy arrays per component)
    shape_components:list[tuple[np.ndarray, np.ndarray]] = _flood_fill_select_all_shapes(grid_for_flood_fill, # greedily-grabbed pixels already blanked out
                                             bg_color=flood_fill_bg_color,
                                             multicolor_shapes=parse_config.get('multicolor',False),
                                             diagonal_pixels=parse_config.get('diagonals',True),
                                             column_pixels=parse_config.get('columns',True),
                                             row_pixels=parse_config.get('rows',True),
                                             radius=parse_config.get('radius',1)) # connection radius (1 = plain, the default)

    # Over the shape cap already? Then skip building the shape dicts (most of a parse's time) and hand back just a flag
    # scene, which search_parse throws out. It has NO 'shapes' on purpose, so nothing can read it as a real scene by mistake.
    # (It keeps the bg, since search_parse's bg-mode twin lookup reads every demo scene's bg before throwing anything out)
    # (Counted before de-noise, which can change the count: fine, since search_parse never passes a cap with de-noise on)
    if max_shapes is not None and len(shapes) + len(shape_components) > max_shapes:
        return {'background': "c" + str(scene_bg_color), 'too_many_shapes': True}

    # ...and convert to dict, annotated with useful properties (colors from the REAL grid)
    flood_fill_shapes:list[dict] = [shape_coords_to_shape_dict(grid_arr, ys, xs) for ys, xs in shape_components]
    # De-noise on? Then ragged shapes get cleaned up (noise on top), and every shape gets a stacking order 'z'.
    # Autocomplete on too? Then partly hidden rectangles & frames get completed as well. (Autocomplete always comes with de-noise)
    # One-color parses only: a cleaned-up shape gets redrawn in its one color, so in a multicolor blob every
    # other-colored pixel that isn't noise would just vanish from the scene (e.g. a highlight ring drawn around a dot)
    if parse_config.get('denoise', False) and not parse_config.get('multicolor', False):
        flood_fill_shapes = _autocomplete_rects(grid_arr, grid_for_flood_fill, bg_color, flood_fill_shapes,
                                                complete_rects=parse_config.get('autocomplete_rects', False))
    shapes += flood_fill_shapes # appended after any greedy shapes

    # Shuffle them, so nothing order-dependent downstream (greedy claiming in score_parse, one(), monkeys...)
    # quietly favors top-left shapes, like flood-fill order did. Uses the global `random`, which solve() seeds.
    random.shuffle(shapes)

    # GIMME! ('Annotate' will add more advanced data later)
    return {
        'background': "c" + str(scene_bg_color), # cast to string (so selector/monkeys know it's NOT a numerical value

        'left':0,
        'top':0,
        'right': width,
        'bottom': height,
        'width': width,
        'height': height,
        'center_x': center_x,
        'center_y': center_y,

        'shapes': shapes,

        # Save raw pixels (as Numpy) just in case
        '_raw': grid

    }

########################################################
## Repeated chunks: greedily grab every spot a chunk's shadow fits, before anything else
########################################################

def _greedily_grab_chunks(grid:np.ndarray, bg_color:int, chunks:list[np.ndarray], multicolor:bool) -> tuple[list[dict], np.ndarray]:
    """Grab every (non-overlapping) spot where a chunk's shadow, in any D4 orientation, sits on non-bg pixels. Returns (grabbed shapes, grid with those pixels blanked to bg)."""

    grabbed_shapes:list[dict] = []
    grid_height, grid_width = grid.shape

    # Which pixels are still up for grabs? (Not bg, and not grabbed by an earlier chunk)
    is_free:np.ndarray = grid != bg_color
    blanked_grid:np.ndarray = grid.copy()

    # Most-repeated chunks first (search_parse already sorted them that way)
    for chunk in chunks:

        # All of this chunk's distinct D4 orientations (a symmetric chunk has fewer than 8)
        orientations:dict[tuple, np.ndarray] = {_shadow_key(transformed): transformed.astype(bool) for transformed in [chunk] + _non_identity_d4_transforms(chunk)}

        # Every (top, left, orientation's pixel coords) where the shadow fits entirely on free pixels, in reading order (top-left first)
        candidate_spots:list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for shadow in orientations.values():
            shadow_height, shadow_width = shadow.shape
            if shadow_height > grid_height or shadow_width > grid_width:
                continue
            # Slide the shadow over the grid: a spot fits if every shadow pixel lands on a free pixel
            windows:np.ndarray = np.lib.stride_tricks.sliding_window_view(is_free, (shadow_height, shadow_width))
            fits:np.ndarray = np.all(windows | ~shadow, axis=(2, 3))
            # (Efficiency: the shadow's own pixel coords, once per orientation instead of once per spot)
            local_ys, local_xs = np.nonzero(shadow)
            candidate_spots += [(int(top), int(left), local_ys, local_xs) for top, left in zip(*np.nonzero(fits))]
        # Reading order (top-left first) is a deliberate scan order for the greedy grab, NOT a tie-break, so it stays.
        # The only TRUE ties are different orientations at the very same (top, left): shuffle first (sort is stable)
        # so those come out in random order, instead of D4-transform order
        random.shuffle(candidate_spots)
        candidate_spots.sort(key=lambda spot: (spot[0], spot[1]))

        # Greedily grab each spot that still fits (an earlier grab may have taken some of its pixels)
        for top, left, local_ys, local_xs in candidate_spots:
            ys:np.ndarray = local_ys + top
            xs:np.ndarray = local_xs + left
            if not is_free[ys, xs].all():
                continue

            # Monochrome parse? Then the chunk has to be one color, too (same rule as the flood fill)
            chunk_colors:np.ndarray = grid[ys, xs]
            if not multicolor and not (chunk_colors == chunk_colors[0]).all():
                continue

            # Grab it! (Colors from the real grid)
            grabbed_shapes.append(shape_coords_to_shape_dict(grid, ys, xs))
            is_free[ys, xs] = False
            blanked_grid[ys, xs] = bg_color

        # Now the PARTIAL fits: spots where part of the shadow hangs off the canvas (e.g. small inputs that are
        # crops of a bigger picture). Grab the best one (most in-frame pixels), blank it, scan again, until no partial
        # fit has enough pixels in frame.
        # (Only claims pixels, never frees them, so no NEW perfect fit can show up in here: no need to redo the scan above)
        # (How many pixels are in frame at each spot only depends on the grid's size, so count once per orientation,
        # looked up from a per-level cache: the same few shadows & grid sizes come up thousands of times)
        shadows_with_in_frame_counts:list[tuple[np.ndarray, np.ndarray]] = [(shadow, _cached_chunk_in_frame_counts(shadow_key, shadow, grid.shape))
                                                                           for shadow_key, shadow in orientations.items()]
        # Each shadow's "are its in-frame pixels one color?" map, filled in by the first call that has candidates
        # (the grid never changes inside this loop, so it never needs redoing: see _best_partial_chunk_fit)
        is_monochrome_spot_per_shadow:list[Optional[np.ndarray]] = [None] * len(shadows_with_in_frame_counts)
        while True:
            best_partial_fit = _best_partial_chunk_fit(grid, is_free, shadows_with_in_frame_counts, multicolor, is_monochrome_spot_per_shadow)
            if best_partial_fit is None:
                break
            ys, xs = best_partial_fit

            # Which of its pixels are actually on the canvas? Those are the ones it claims
            in_frame:np.ndarray = (ys >= 0) & (ys < grid_height) & (xs >= 0) & (xs < grid_width)
            in_frame_ys:np.ndarray = ys[in_frame]
            in_frame_xs:np.ndarray = xs[in_frame]

            # The off-canvas pixels get the in-frame pixels' majority color (= THE color, for a monochrome parse).
            # TODO: cleverer color-pattern detection, e.g. repeat the chunk's color layout from where it's fully seen
            off_canvas_color, _ = _random_most_common(Counter(grid[in_frame_ys, in_frame_xs].tolist()))

            # Grab it, with its FULL bounding box & shadow (so left/top can be negative, right/bottom past the grid's edge)
            grabbed_shapes.append(_shape_coords_to_shape_dict_off_canvas(grid, ys, xs, off_canvas_color))
            is_free[in_frame_ys, in_frame_xs] = False
            blanked_grid[in_frame_ys, in_frame_xs] = bg_color

    return grabbed_shapes, blanked_grid


# Cache: (grid, bg color, multicolor, chunk list) -> _greedily_grab_chunks' result. Chunk configs that only differ
# in things the grab doesn't look at (diagonals, or bg modes that land on the same color) would otherwise redo the exact same grab.
# Only filled during search_parse (which clears it at the start AND end), so main_solver's later re-parses grab fresh.
# (A cache hit skips the grab's random tie-breaks, so seeded results shift a bit. Nicky's fine with that)
_greedily_grab_chunks_cache:dict[tuple, tuple[list[dict], np.ndarray]] = {}

def _cached_greedily_grab_chunks(grid:np.ndarray, bg_color:int, chunks:list[np.ndarray], multicolor:bool) -> tuple[list[dict], np.ndarray]:
    """_greedily_grab_chunks, done once per distinct (grid, bg color, multicolor, chunk list) within one search_parse."""

    # The key: the grid's & chunks' actual pixels (not object ids), so equal grids/chunks always match
    chunks_key:tuple = tuple((chunk.shape, chunk.tobytes()) for chunk in chunks)
    cache_key:tuple = (grid.shape, grid.tobytes(), bg_color, multicolor, chunks_key)
    cached_result:Optional[tuple[list[dict], np.ndarray]] = _greedily_grab_chunks_cache.get(cache_key)
    if cached_result is None:
        cached_result = _greedily_grab_chunks(grid, bg_color, chunks, multicolor)
        _greedily_grab_chunks_cache[cache_key] = cached_result

    # Hand back a COPY of the shape list, since apply_parse_one_grid appends to it & shuffles it in place.
    # (The shape dicts & blanked grid themselves are shared: nothing inside search_parse changes them, only reads them)
    grabbed_shapes, blanked_grid = cached_result
    return list(grabbed_shapes), blanked_grid

def clear_greedily_grab_chunks_cache() -> None:
    """Forget every cached chunk grab (search_parse calls this at the start & end of each search)."""
    _greedily_grab_chunks_cache.clear()


# A partial fit needs at least this many shadow pixels on the canvas (fewer = too likely a coincidence)
_CHUNK_PARTIAL_FIT_MIN_IN_FRAME_PIXELS:int = 5

# For every spot the shadow can slide to (incl. hanging off the canvas), how many of its pixels are in frame?
def _chunk_in_frame_counts(shadow:np.ndarray, grid_shape:tuple[int, int]) -> np.ndarray:
    """In-frame shadow pixel count at every (padded) spot, from only its bottom-right pixel in frame at the top-left, to only its top-left pixel in frame at the bottom-right."""
    shadow_height, shadow_width = shadow.shape

    # Pad an all-ones "in frame" grid with zeros, so the shadow can slide off every edge, down to just its corner pixel in frame
    padding:tuple = ((shadow_height - 1, shadow_height - 1), (shadow_width - 1, shadow_width - 1))
    padded_in_frame:np.ndarray = np.pad(np.ones(grid_shape, dtype=np.int32), padding, constant_values=0)
    in_frame_windows:np.ndarray = np.lib.stride_tricks.sliding_window_view(padded_in_frame, (shadow_height, shadow_width))
    return np.einsum('ijkl,kl->ij', in_frame_windows, shadow.astype(np.int32))

# Cache: (shadow key, grid size) -> _chunk_in_frame_counts. It's a pure function of those two, and one level asks
# for the same few combos thousands of times. Only read (never written to) by _best_partial_chunk_fit, so sharing is safe.
# search_parse clears it at the start of each call, so it only ever holds one level's shadows.
_chunk_in_frame_counts_cache:dict[tuple, np.ndarray] = {}

def _cached_chunk_in_frame_counts(shadow_key:tuple, shadow:np.ndarray, grid_shape:tuple[int, int]) -> np.ndarray:
    """_chunk_in_frame_counts, computed once per distinct (shadow, grid size)."""
    cache_key:tuple = (shadow_key, grid_shape)
    in_frame_counts:Optional[np.ndarray] = _chunk_in_frame_counts_cache.get(cache_key)
    if in_frame_counts is None:
        # First time seeing this combo: count for real, and remember it
        in_frame_counts = _chunk_in_frame_counts(shadow, grid_shape)
        _chunk_in_frame_counts_cache[cache_key] = in_frame_counts
    return in_frame_counts

def clear_chunk_in_frame_counts_cache() -> None:
    """Forget every cached in-frame count (search_parse calls this at the start of each search)."""
    _chunk_in_frame_counts_cache.clear()

# For _greedily_grab_chunks: the best spot where a chunk's shadow PARTLY hangs off the canvas.
# (Each shadow comes with its pre-counted in-frame pixel counts, from _chunk_in_frame_counts,
# + each shadow's monochrome-spot map, shared by every call in the same while loop (None = not worked out yet; this fills it in))
def _best_partial_chunk_fit(grid:np.ndarray, is_free:np.ndarray, shadows_with_in_frame_counts:list[tuple[np.ndarray, np.ndarray]], multicolor:bool,
                            is_monochrome_spot_per_shadow:list[Optional[np.ndarray]]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Best partly-off-canvas spot for any of these shadows (most in-frame pixels, random ties), as the FULL shadow's global (ys, xs). None if no spot has enough pixels in frame."""

    grid_height, grid_width = grid.shape

    # Which in-frame pixels block a spot? Every pixel that's NOT free
    # (bg or claimed). Off-canvas never blocks. As 0/1 ints, for the correlation below
    is_blocked:np.ndarray = (~is_free).astype(np.int32)

    # Every partial fit, over all orientations: (how many pixels in frame, top, left, which shadow).
    # One numpy array per shadow for each of those, glued together after the loop (no Python tuple per spot)
    in_frame_counts_per_shadow:list[np.ndarray] = []
    tops_per_shadow:list[np.ndarray] = []
    lefts_per_shadow:list[np.ndarray] = []
    shadow_indices_per_shadow:list[np.ndarray] = []
    is_monochrome_per_shadow:list[np.ndarray] = [] # does each candidate pass the monochrome check?
    for shadow_index, (shadow, in_frame_counts) in enumerate(shadows_with_in_frame_counts):
        shadow_height, shadow_width = shadow.shape
        shadow_area:int = int(shadow.sum())

        # A spot fits if every IN-FRAME shadow pixel lands on a free pixel (not bg, not claimed). Off-canvas counts
        # as "free" (anything goes out there). TODO: a shadow pixel on a non-matching pixel could be an *obscured*
        # shape, but we don't have a way to manage z-indexes on shapes yet. So for now, any in-frame miss = not a fit
        # Fast way: a spot fits if ZERO shadow pixels land on a blocking pixel. How many land on one, at every spot at
        # once = the correlation of the blocked map with the shadow. ("full" mode = every spot down to just a corner
        # pixel in frame, the exact same spots as in_frame_counts. Done with an FFT, but scipy rounds integer inputs
        # back to exact ints, so there's no float fuzz in the == 0)
        blocked_pixels_hit:np.ndarray = scipy.signal.correlate(is_blocked, shadow.astype(np.int32), mode='full', method='fft')
        fits:np.ndarray = blocked_pixels_hit == 0

        # (Its score, how many shadow pixels are in frame, is in_frame_counts: same padded spots as fits)

        # Partial = not the whole shadow in frame (those are perfect fits, already grabbed), but enough of it.
        # (This mask is the "skip it if fewer than 5 in frame" check, for every spot at once)
        is_partial_fit:np.ndarray = fits & (in_frame_counts >= _CHUNK_PARTIAL_FIT_MIN_IN_FRAME_PIXELS) & (in_frame_counts < shadow_area)
        padded_tops, padded_lefts = np.nonzero(is_partial_fit)
        # (Back to real grid coords: padded position 0 = the shadow hanging (size - 1) pixels off the top/left)
        in_frame_counts_per_shadow.append(in_frame_counts[padded_tops, padded_lefts])
        tops_per_shadow.append(padded_tops - (shadow_height - 1))
        lefts_per_shadow.append(padded_lefts - (shadow_width - 1))
        shadow_indices_per_shadow.append(np.full(len(padded_tops), shadow_index))

        # Monochrome parse? Then the in-frame pixels have to be one color. Inside one while loop the grid never changes
        # and is_free only shrinks, so every later call's candidates are a subset of the first call's, and a spot's colors
        # never change. So check this shadow's candidates ONCE (the first time it has any), and just look them up after that
        if multicolor:
            is_monochrome_per_shadow.append(np.ones(len(padded_tops), dtype=bool))
            continue
        if is_monochrome_spot_per_shadow[shadow_index] is None and len(padded_tops) > 0:
            # All its candidates' shadow pixels' global coords at once (one row per candidate)
            local_ys, local_xs = np.nonzero(shadow)
            ys = (padded_tops - (shadow_height - 1))[:, None] + local_ys[None, :]
            xs = (padded_lefts - (shadow_width - 1))[:, None] + local_xs[None, :]
            in_frame = (ys >= 0) & (ys < grid_height) & (xs >= 0) & (xs < grid_width)
            # One color = the in-frame pixels' highest color is also their lowest (off-canvas pixels left out of both)
            colors:np.ndarray = grid[np.clip(ys, 0, grid_height - 1), np.clip(xs, 0, grid_width - 1)].astype(np.int32)
            highest_colors:np.ndarray = np.where(in_frame, colors, np.iinfo(np.int32).min).max(axis=1)
            lowest_colors:np.ndarray = np.where(in_frame, colors, np.iinfo(np.int32).max).min(axis=1)
            # Remember it as a map over the padded spots (spots that weren't candidates stay False: they never come back)
            is_monochrome_spot:np.ndarray = np.zeros(in_frame_counts.shape, dtype=bool)
            is_monochrome_spot[padded_tops, padded_lefts] = highest_colors == lowest_colors
            is_monochrome_spot_per_shadow[shadow_index] = is_monochrome_spot
        # (No candidates now = none ever again in this loop, so there's nothing to look up)
        if len(padded_tops) == 0:
            is_monochrome_per_shadow.append(np.zeros(0, dtype=bool))
        else:
            # (Never None here: the block above filled it in, the first time this shadow had candidates. Pyright can't see that)
            is_monochrome_per_shadow.append(is_monochrome_spot_per_shadow[shadow_index][padded_tops, padded_lefts])  # pyright: ignore[reportOptionalSubscript]
    candidate_in_frame_counts:np.ndarray = np.concatenate(in_frame_counts_per_shadow)
    candidate_tops:np.ndarray = np.concatenate(tops_per_shadow)
    candidate_lefts:np.ndarray = np.concatenate(lefts_per_shadow)
    candidate_shadow_indices:np.ndarray = np.concatenate(shadow_indices_per_shadow)
    candidate_is_monochrome:np.ndarray = np.concatenate(is_monochrome_per_shadow)
    candidate_count:int = len(candidate_in_frame_counts)

    # Most in-frame pixels first, ties broken randomly (by a shuffled order)
    candidate_order:list[int] = list(range(candidate_count))
    random.shuffle(candidate_order)

    # The pick: among the candidates that pass the monochrome check, the most in-frame pixels,
    # then whichever the shuffle put first
    passing_indices:np.ndarray = np.flatnonzero(candidate_is_monochrome)
    if len(passing_indices) == 0:
        return None
    passing_in_frame_counts:np.ndarray = candidate_in_frame_counts[passing_indices]
    passing_indices = passing_indices[passing_in_frame_counts == passing_in_frame_counts.max()]
    # (Where did the shuffle put each candidate?)
    shuffled_positions:np.ndarray = np.empty(candidate_count, dtype=np.int64)
    shuffled_positions[candidate_order] = np.arange(candidate_count)
    best_candidate_index:int = passing_indices[np.argmin(shuffled_positions[passing_indices])]

    # Its FULL shadow's global coords (incl. the off-canvas pixels)
    local_ys, local_xs = np.nonzero(shadows_with_in_frame_counts[candidate_shadow_indices[best_candidate_index]][0])
    ys:np.ndarray = local_ys + candidate_tops[best_candidate_index]
    xs:np.ndarray = local_xs + candidate_lefts[best_candidate_index]
    return ys, xs


########################################################
## Greedy input shapes: carry input shapes over into the output parse, before flood fill
########################################################

def _greedily_grab_input_shapes_static(output_grid:np.ndarray, bg_color:int, input_scene:dict) -> tuple[list[dict], np.ndarray, set[int]]:
    """For each monochrome input shape still monochrome & non-bg in the output: grab it as-is. Returns (grabbed shapes, output grid with those pixels blanked to bg, indices of the grabbed input shapes)."""

    grabbed_shapes:list[dict] = []
    claimed_input_shape_indices:set[int] = set() # index into input_scene['shapes'], so the moving grab skips these

    # Only when input & output grids are the same dimensions: if they differ, skip greedy entirely for this pair.
    # (TODO: we may want to make this more flexible in the future, e.g. match up the shapes anyway if the
    # output is just a crop/pad of the input: things can "move" by being cropped, so it's not trivial.)
    input_grid:np.ndarray = input_scene['_raw']
    if input_grid.shape != output_grid.shape:
        return grabbed_shapes, output_grid, claimed_input_shape_indices # nothing claimed

    # Blank out grabbed pixels in a COPY, so the flood fill (which runs after us) skips them.
    # (bg_color may be -1 for 'no_bg': flood fill skips that value too, so the blanking still works)
    blanked_grid:np.ndarray = output_grid.copy()

    for input_shape_index, input_shape in enumerate(input_scene['shapes']): # (index, to say which ones got claimed)

        # Only monochrome input shapes. (MAYBE: also allow a multicolor shape if its pixels are still the same pattern?)
        if not input_shape['monochrome']:
            continue

        # Where did this input shape's pixels sit? (Shadow is local to its bounding box, so shift to global coords)
        local_ys, local_xs = np.nonzero(input_shape['_shadow'])
        ys:np.ndarray = local_ys + input_shape['top']
        xs:np.ndarray = local_xs + input_shape['left']

        # Are the output pixels at those same positions still ONE color (not necessarily the input's color)...
        output_colors:np.ndarray = output_grid[ys, xs]
        if not np.all(output_colors == output_colors[0]):
            continue

        # ...and not the output's background? (Otherwise a deleted or moved shape would get grabbed as a blob of bg)
        if int(output_colors[0]) == bg_color:
            continue

        # Grab it! Same pixels, same position, colors read from the real output grid (so it might be recolored)
        grabbed_shapes.append(shape_coords_to_shape_dict(output_grid, ys, xs))
        blanked_grid[ys, xs] = bg_color
        claimed_input_shape_indices.add(input_shape_index)

    # (Shapes that MOVED get their own grab: see _greedily_grab_input_shapes_moving below)

    return grabbed_shapes, blanked_grid, claimed_input_shape_indices


########################################################
## Greedy MOVING input shapes: input shapes whose exact pixels show up somewhere (else) in the output
########################################################

# allow_overlap: if True, a grab does NOT take its pixels away from the NEXT moving grabs (shapes may share pixels).
# Every grabbed pixel still gets blanked in the returned grid, so the flood fill can't claim them again either way
def _greedily_grab_input_shapes_moving(output_grid:np.ndarray, bg_color:int, input_scene:dict, claimed_input_shape_indices:set[int], allow_overlap:bool=False) -> tuple[list[dict], np.ndarray]:
    """For each unclaimed input shape (area >= GREEDY_MOVING_MIN_AREA, biggest first): grab one random spot where its exact pixels sit on unclaimed output pixels. Returns (grabbed shapes, output grid with those pixels blanked to bg)."""

    grabbed_shapes:list[dict] = []
    grid_height, grid_width = output_grid.shape

    # Blank out grabbed pixels in a COPY, so later grabs & the flood fill skip them.
    # (Works for different-sized input & output grids too: we only slide the input shape over the OUTPUT grid)
    blanked_grid:np.ndarray = output_grid.copy()

    # Which input shapes are still up for grabs? (Not claimed by the static grab, and big enough to not be a coincidence)
    candidate_input_shapes:list[dict] = [input_shape for input_shape_index, input_shape in enumerate(input_scene['shapes'])
                                         if input_shape_index not in claimed_input_shape_indices and input_shape['area'] >= GREEDY_MOVING_MIN_AREA]

    # Biggest to smallest by area. (Same-area ties stay in the input scene's order, which apply_parse already shuffled)
    candidate_input_shapes.sort(key=lambda input_shape: -input_shape['area'])

    for input_shape in candidate_input_shapes:

        # The input shape's exact pixels (colors too, no rotations/flips), and which of them are actually in the shape
        shape_pixels:np.ndarray = input_shape['_pixels']
        shape_mask:np.ndarray = input_shape['_shadow'].astype(bool)
        shape_height, shape_width = shape_mask.shape
        if shape_height > grid_height or shape_width > grid_width:
            continue

        # Slide it over the (blanked) output: a spot matches if every shape pixel lands on the SAME color, and that
        # output pixel isn't bg. (Already-claimed pixels are bg in blanked_grid, so this also skips claimed pixels)
        # Overlap allowed? Then slide over the grid we were handed (chunk & static grabs already blanked there,
        # so those pixels stay off-limits), NOT the one this loop keeps blanking, so earlier moving grabs don't block later ones
        grid_to_match:np.ndarray = output_grid if allow_overlap else blanked_grid
        windows:np.ndarray = np.lib.stride_tricks.sliding_window_view(grid_to_match, (shape_height, shape_width))
        pixel_matches:np.ndarray = (windows == shape_pixels) & (windows != bg_color)
        matching_spots:np.ndarray = np.argwhere(np.all(pixel_matches | ~shape_mask, axis=(2, 3)))
        if len(matching_spots) == 0:
            continue

        # Pick one matching spot at random (global `random`, which solve() seeds)
        top, left = matching_spots[random.randrange(len(matching_spots))]
        local_ys, local_xs = np.nonzero(shape_mask)
        ys:np.ndarray = local_ys + int(top)
        xs:np.ndarray = local_xs + int(left)

        # Grab it! (Colors from the real output grid), and blank those pixels so nothing grabs them again
        grabbed_shapes.append(shape_coords_to_shape_dict(output_grid, ys, xs))
        # (With allow_overlap, this blanking only keeps the flood fill off them: later moving grabs don't look here)
        blanked_grid[ys, xs] = bg_color

    return grabbed_shapes, blanked_grid


########################################################
## Autocomplete rectangles: rectangles, lines & frames partly hidden behind other shapes
########################################################

# A completed shape has to be at least this big (smaller ones are too likely a coincidence)
_AUTOCOMPLETE_MIN_AREA:int = 5

# A filled rectangle may still have bg pixels inside, if they're only small noise holes: each one at most this big
# (4-connected). Then it's completed anyway, and each hole becomes a bg-colored shape on top of it.
# De-noise fills a shape's holes too, judging each one-color piece of a hole on its own against this cap
# (touching noise pieces get merged back into one shape afterwards, so a multicolor stamp can end up bigger)
_AUTOCOMPLETE_MAX_HOLE_AREA:int = 6

# ...and all its bg pixels together have to be less than this fraction of its box (more = real holes, not noise)
_AUTOCOMPLETE_MAX_HOLE_FRACTION:float = 0.10

# De-noise's version: a shape's filled holes & notches together have to be less than this fraction of the filled shape.
# Looser than the rectangles' cap, so a shape with several small stamps on it still counts as noisy, not holey
_DENOISE_MAX_HOLE_FRACTION:float = 0.20

# Shapes this big or smaller are too small for de-noise to clean (before and after cleaning): they ARE the noise.
# So a small stamp sitting on a bigger shape stays whole, instead of getting its own center pixel cut out as noise
_DENOISE_MAX_TOO_SMALL_AREA:int = 5

# Every pixel's left/right/top/bottom neighbours (not itself), for counting how many of them are in a shape
_PLUS_NEIGHBOURS:np.ndarray = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.int8)

# ...and all 8 of its neighbours, diagonals too
_RING_NEIGHBOURS:np.ndarray = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int8)

def denoise_scene_both_ways(scene:dict) -> tuple[dict, dict]:
    """An already-parsed scene (no chunks or greedy grabs), de-noised only, and de-noised + its partly hidden rectangles & frames completed. Every shape gets a 'z'."""
    # Reuses the scene's shapes instead of flood-filling all over again, so search_parse can try de-noise on
    # every base config cheaply. (No chunks or greedy grabs = every shape came from the flood fill of the raw grid)
    grid:np.ndarray = scene['_raw']
    bg_color:int = int(scene['background'][1:])

    # Both ways share one de-noising pass, since it's the same for both. (Same steps as _autocomplete_rects)
    if not (grid == bg_color).any():
        denoised_shapes:list[dict] = [{**shape, 'z': 0} for shape in scene['shapes']]
        completed_shapes:list[dict] = list(denoised_shapes)
    else:
        clean_shapes, noise_shapes, noise_mask = _denoise_shapes(grid, scene['shapes'], grid != bg_color)
        denoised_shapes = _stack_noise_on_top(grid, clean_shapes, noise_shapes)
        completed_shapes = _complete_rects(grid, grid, bg_color, clean_shapes, noise_shapes, noise_mask)

    # (Each shuffled, same as apply_parse_one_grid's shuffle)
    random.shuffle(denoised_shapes)
    random.shuffle(completed_shapes)
    return {**scene, 'shapes': denoised_shapes}, {**scene, 'shapes': completed_shapes}

def _autocomplete_rects(grid:np.ndarray, flood_fill_grid:np.ndarray, bg_color:int, flood_fill_shapes:list[dict], complete_rects:bool) -> list[dict]:
    """The flood fill's one-color shapes, de-noised (1px danglers, 1px notches & small holes become their own shapes on top), and (if complete_rects) the rectangles & frames hidden under other shapes (or under small bg-colored noise holes) completed. Every shape gets a 'z' (new dicts: the ones passed in aren't changed)."""

    # No bg pixel anywhere in the grid? Then "no bg inside the box" is true of every box, so hidden and plain missing
    # can't be told apart: complete nothing
    if not (flood_fill_grid == bg_color).any():
        return [{**shape, 'z': 0} for shape in flood_fill_shapes]

    # Step 1: de-noise every shape: 1px danglers peeled off, 1px notches & small holes filled in. Each noise pixel
    # becomes its own small shape on top, so the completion only ever sees clean shapes
    clean_shapes, noise_shapes, noise_mask = _denoise_shapes(grid, flood_fill_shapes, flood_fill_grid != bg_color)

    # De-noise only? Then that's it. Else step 2: complete the hidden rectangles & frames
    if not complete_rects:
        return _stack_noise_on_top(grid, clean_shapes, noise_shapes)
    return _complete_rects(grid, flood_fill_grid, bg_color, clean_shapes, noise_shapes, noise_mask)

def _stack_noise_on_top(grid:np.ndarray, clean_shapes:list[dict], noise_shapes:list[dict]) -> list[dict]:
    """De-noise only: the clean shapes at the back (z 0), the noise shapes on top of them (z 1)."""
    clean_shapes = [{**shape, 'z': 0} for shape in clean_shapes]
    return clean_shapes + [{**noise_shape, 'z': 1} for noise_shape in _merge_touching_noise(grid, noise_shapes, clean_shapes)]

def _merge_touching_noise(grid:np.ndarray, noise_shapes:list[dict], shapes_underneath:list[dict]) -> list[dict]:
    """Noise shapes that sit on the same shape and touch (corners too) become one noise shape (multicolor if their colors differ)."""
    if len(noise_shapes) < 2:
        return noise_shapes

    # Which shape is each pixel on? Drawn back to front, so where shapes overlap the top one wins
    underneath_grid:np.ndarray = np.full(grid.shape, -1, dtype=np.int32)
    for shape_index in sorted(range(len(shapes_underneath)), key=lambda index: shapes_underneath[index]['z']):
        shape:dict = shapes_underneath[shape_index]
        local_ys, local_xs = np.nonzero(shape['_shadow'])
        underneath_grid[local_ys + shape['top'], local_xs + shape['left']] = shape_index

    # Each noise shape sits on one shape (or it's not mergeable: off every shape, or across two of them)
    noise_index_grid:np.ndarray = np.full(grid.shape, -1, dtype=np.int32)
    merged_noise_shapes:list[dict] = []
    for noise_index, noise_shape in enumerate(noise_shapes):
        local_ys, local_xs = np.nonzero(noise_shape['_shadow'])
        ys:np.ndarray = local_ys + noise_shape['top']
        xs:np.ndarray = local_xs + noise_shape['left']
        shapes_below:np.ndarray = underneath_grid[ys, xs]
        if shapes_below[0] == -1 or (shapes_below != shapes_below[0]).any():
            merged_noise_shapes.append(noise_shape)
            continue
        noise_index_grid[ys, xs] = noise_index

    # Clumps of touching noise, one shape underneath at a time (so noise on two neighbouring shapes stays apart,
    # and can't chain through each other). 8-connected, whatever the parse's diagonals setting: noise is noise
    is_mergeable:np.ndarray = noise_index_grid >= 0
    for shape_index in np.unique(underneath_grid[is_mergeable]).tolist():
        clump_labels, clump_count = scipy.ndimage.label(is_mergeable & (underneath_grid == shape_index), structure=np.ones((3, 3))) # type: ignore
        for clump_ys, clump_xs in (np.nonzero(clump_labels == clump_label) for clump_label in range(1, clump_count + 1)):
            noise_indices:np.ndarray = np.unique(noise_index_grid[clump_ys, clump_xs])
            # A lone noise shape stays as it was. Touching ones get remade as one shape, colors from the real grid
            if len(noise_indices) == 1:
                merged_noise_shapes.append(noise_shapes[int(noise_indices[0])])
            else:
                merged_noise_shapes.append(shape_coords_to_shape_dict(grid, clump_ys, clump_xs))
    return merged_noise_shapes

def _complete_rects(grid:np.ndarray, flood_fill_grid:np.ndarray, bg_color:int, flood_fill_shapes:list[dict],
                    noise_shapes:list[dict], noise_mask:np.ndarray) -> list[dict]:
    """The de-noised shapes with the rectangles & frames hidden under other shapes (or under small bg-colored noise holes) completed, noise shapes on top. Every shape gets a 'z'."""
    # (A copy: the hole shapes below get added to it, and the caller may still use the de-noised list)
    noise_shapes = list(noise_shapes)
    component_count:int = len(flood_fill_shapes)

    # How many bg pixels are in any box? A running 2D sum, so each box is 4 lookups. (Already-grabbed pixels are
    # bg in the flood fill grid, so nothing can hide under those either. Noise pixels aren't bg: a noise shape covers them)
    is_bg:np.ndarray = (flood_fill_grid == bg_color) & ~noise_mask
    bg_sums:np.ndarray = np.pad(is_bg.astype(np.int32).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    def count_bg(top, left, bottom, right): # (works on ints & on numpy arrays of boxes alike)
        return bg_sums[bottom, right] - bg_sums[top, right] - bg_sums[bottom, left] + bg_sums[top, left]

    # Complete the hidden ones. Each shape's pixels, in grid coords (its shadow is local to its bounding box)
    shape_components:list[tuple[np.ndarray, np.ndarray]] = []
    for shape in flood_fill_shapes:
        local_ys, local_xs = np.nonzero(shape['_shadow'])
        shape_components.append((local_ys + shape['top'], local_xs + shape['left']))

    # Which shape does each visible pixel belong to? (-1 = bg, or grabbed by a chunk/greedy grab before the flood fill)
    owner_grid:np.ndarray = np.full(grid.shape, -1, dtype=np.int32)
    for component_index, (ys, xs) in enumerate(shape_components):
        owner_grid[ys, xs] = component_index
    # (Noise pixels are owned by nobody: their noise shapes go on top of everything later, so no facts about them)
    owner_grid[noise_mask] = -1

    # Every shape's bounding box (bottom & right exclusive) and color, as arrays
    tops:np.ndarray = np.array([shape['top'] for shape in flood_fill_shapes], dtype=np.int32)
    lefts:np.ndarray = np.array([shape['left'] for shape in flood_fill_shapes], dtype=np.int32)
    bottoms:np.ndarray = np.array([shape['bottom'] for shape in flood_fill_shapes], dtype=np.int32)
    rights:np.ndarray = np.array([shape['right'] for shape in flood_fill_shapes], dtype=np.int32)
    colors:np.ndarray = np.array([int(shape['color'][1:]) for shape in flood_fill_shapes], dtype=np.int32)

    # Every candidate completion: (its member components, top, left, local shadow of the completed shape, color)
    candidates:list[tuple[np.ndarray, int, int, np.ndarray, int]] = []
    for color in np.unique(colors).tolist():
        members:np.ndarray = np.flatnonzero(colors == color)

        # Only pairs that line up on at least one side (same left, right, top or bottom) can be pieces of one
        # rectangle. Most pairs don't, so this cuts the gluing down to the few that might be
        member_tops, member_lefts, member_bottoms, member_rights = tops[members], lefts[members], bottoms[members], rights[members]
        pair_shares_a_side:np.ndarray = (np.equal.outer(member_tops, member_tops) | np.equal.outer(member_lefts, member_lefts)
                                         | np.equal.outer(member_bottoms, member_bottoms) | np.equal.outer(member_rights, member_rights))
        np.fill_diagonal(pair_shares_a_side, False) # (A shape alone gets checked anyway, as its own group)

        # Every such PAIR of this color's components: could their combined box be one shape? Filled = no bg anywhere inside.
        # Frame = at least 3x3 with no bg on its 1px border (its inner box holds every bg pixel of the box)
        pair_tops:np.ndarray = np.minimum.outer(tops[members], tops[members])
        pair_lefts:np.ndarray = np.minimum.outer(lefts[members], lefts[members])
        pair_bottoms:np.ndarray = np.maximum.outer(bottoms[members], bottoms[members])
        pair_rights:np.ndarray = np.maximum.outer(rights[members], rights[members])
        pair_bg_counts:np.ndarray = count_bg(pair_tops, pair_lefts, pair_bottoms, pair_rights)
        is_frame_sized:np.ndarray = (pair_bottoms - pair_tops >= 3) & (pair_rights - pair_lefts >= 3)
        # (Boxes too small to be frames get an empty inner box, so the lookup stays in bounds. They're masked out anyway)
        inner_bottoms:np.ndarray = np.maximum(pair_bottoms - 1, pair_tops + 1)
        inner_rights:np.ndarray = np.maximum(pair_rights - 1, pair_lefts + 1)
        inner_bg_counts:np.ndarray = count_bg(pair_tops + 1, pair_lefts + 1, inner_bottoms, inner_rights)
        pair_could_merge:np.ndarray = pair_shares_a_side & ((pair_bg_counts == 0) | (is_frame_sized & (pair_bg_counts == inner_bg_counts)))

        # Glue together every chain of mergeable pairs. (Merging only ever happens when the combined box passes,
        # so two separate parallel lines, with bg between them, stay separate)
        for group_member_indices in _glue_groups(len(members), *np.nonzero(np.triu(pair_could_merge))):
            group_members:np.ndarray = members[group_member_indices]
            candidate = _completion_candidate(group_members, shape_components, tops, lefts, bottoms, rights, count_bg, is_bg)
            if candidate is not None:
                candidates.append((*candidate, color))
                continue
            # A chain of mergeable pairs isn't always mergeable as a WHOLE: then try each of its components on its own
            if len(group_members) > 1:
                for member in group_members.tolist():
                    candidate = _completion_candidate(np.array([member]), shape_components, tops, lefts, bottoms, rights, count_bg, is_bg)
                    if candidate is not None:
                        candidates.append((*candidate, color))

    # Add completions greedily, biggest first (ties in random order: shuffle, then a stable sort),
    # keeping each only if some stacking order still explains every overlap
    random.shuffle(candidates)
    candidates.sort(key=lambda candidate: -int(candidate[3].sum()))
    accepted_candidates:list[tuple[np.ndarray, int, int, np.ndarray, int]] = []
    stacking:_StackingFacts = _StackingFacts(component_count)
    for candidate in candidates:
        if stacking.try_add(candidate, owner_grid):
            accepted_candidates.append(candidate)

    # Every accepted completion fits, so now sort them all into layers, just once
    z_per_component:np.ndarray = np.zeros(component_count, dtype=np.int32)
    if accepted_candidates:
        z_per_component = _stacking_order(accepted_candidates, owner_grid, component_count) # type: ignore (never None: try_add only kept completions that fit together)

    # Now make the shapes! Completed ones first: drawn in their color over a copy of the grid, so their hidden pixels have a color
    shapes:list[dict] = []
    completed_components:set[int] = set()
    for members, top, left, completed_shadow, color in accepted_candidates:
        local_ys, local_xs = np.nonzero(completed_shadow)
        completed_ys:np.ndarray = local_ys + top
        completed_xs:np.ndarray = local_xs + left
        completed_grid:np.ndarray = grid.copy()
        completed_grid[completed_ys, completed_xs] = color
        completed_shape:dict = shape_coords_to_shape_dict(completed_grid, completed_ys, completed_xs)
        # (A group shares one z: _stacking_order gives every member its group's)
        completed_shape['z'] = int(z_per_component[members[0]])
        shapes.append(completed_shape)
        completed_components.update(members.tolist())

    # Completed rectangles with small noise holes: each hole becomes a bg-colored shape, so the redraw puts the bg back.
    # On top of everything, since a hole pixel can be inside more than one completion (each hole is made just once)
    hole_mask:np.ndarray = np.zeros(grid.shape, dtype=bool)
    for members, top, left, completed_shadow, color in accepted_candidates:
        height, width = completed_shadow.shape
        hole_mask[top:top + height, left:left + width] |= is_bg[top:top + height, left:left + width] & completed_shadow.astype(bool)
    # (The noise shapes from de-noising go up there too)
    if hole_mask.any():
        hole_labels, hole_count = scipy.ndimage.label(hole_mask) # type: ignore
        for hole_ys, hole_xs in (np.nonzero(hole_labels == hole_label) for hole_label in range(1, hole_count + 1)):
            noise_shapes.append(shape_coords_to_shape_dict(grid, hole_ys, hole_xs))

    # ...then every other shape, as-is (a copy, plus its z)
    for component_index, shape in enumerate(flood_fill_shapes):
        if component_index in completed_components:
            continue
        shapes.append({**shape, 'z': int(z_per_component[component_index])})

    # Last, the noise on top. (Merged once every shape's in place, since touching noise only merges on the same shape)
    top_z:int = int(z_per_component.max(initial=0)) + 1
    return shapes + [{**noise_shape, 'z': top_z} for noise_shape in _merge_touching_noise(grid, noise_shapes, shapes)]

def _glue_groups(item_count:int, pair_firsts:np.ndarray, pair_seconds:np.ndarray) -> list[np.ndarray]:
    """Split items 0..item_count-1 into groups, where each (first, second) pair is glued into the same group (a union-find)."""

    # Nothing glued? Then every item is its own group (the usual case), no gluing work needed
    if len(pair_firsts) == 0:
        return [np.array([item]) for item in range(item_count)]

    # Every item starts as its own group, pointing at itself. Gluing two groups = pointing one's root at the other's
    parent_of:list[int] = list(range(item_count))
    def root_of(item:int) -> int:
        # Follow the pointers up to the group's root, shortcutting the path as we go so later lookups are quick
        while parent_of[item] != item:
            parent_of[item] = parent_of[parent_of[item]]
            item = parent_of[item]
        return item
    for first, second in zip(pair_firsts.tolist(), pair_seconds.tolist()):
        parent_of[root_of(first)] = root_of(second)

    # Collect each root's items into one group
    groups_by_root:dict[int, list[int]] = {}
    for item in range(item_count):
        groups_by_root.setdefault(root_of(item), []).append(item)
    return [np.array(group) for group in groups_by_root.values()]

def _completion_candidate(members:np.ndarray, shape_components:list[tuple[np.ndarray, np.ndarray]],
                          tops:np.ndarray, lefts:np.ndarray, bottoms:np.ndarray, rights:np.ndarray,
                          count_bg, is_bg:np.ndarray) -> Optional[tuple[np.ndarray, int, int, np.ndarray]]:
    """If these same-color components' combined box is a filled rectangle (maybe with small noise holes) or a 1px frame with hidden pixels, return (members, top, left, completed local shadow). Else None."""

    # The combined box
    top:int = int(tops[members].min())
    left:int = int(lefts[members].min())
    bottom:int = int(bottoms[members].max())
    right:int = int(rights[members].max())

    # Is it a rectangle or frame at all?
    member_components:list[tuple[np.ndarray, np.ndarray]] = [shape_components[member] for member in members.tolist()]
    completed_shadow:Optional[np.ndarray] = _box_shadow(member_components, top, left, bottom, right, count_bg, is_bg)
    if completed_shadow is None:
        return None

    # Big enough, and actually hides something? (No hidden pixel = nothing to complete, nothing changes)
    completed_area:int = int(completed_shadow.sum())
    member_area:int = sum(len(ys) for ys, _ in member_components)
    if completed_area < _AUTOCOMPLETE_MIN_AREA or completed_area <= member_area:
        return None
    return members, top, left, completed_shadow

def _box_shadow(components:list[tuple[np.ndarray, np.ndarray]], top:int, left:int, bottom:int, right:int,
                count_bg, is_bg:np.ndarray) -> Optional[np.ndarray]:
    """If these same-color pixels' box (bottom & right exclusive) is a filled rectangle (maybe with small noise holes) or a 1px frame, return its completed local shadow. Else None."""
    height:int = bottom - top
    width:int = right - left

    # Filled rectangle (lines too): not a single bg pixel inside, so every missing pixel is hidden under something
    bg_count:int = count_bg(top, left, bottom, right)
    if bg_count == 0:
        return np.ones((height, width), dtype=np.int8)

    # Frame: at least 3x3, no bg on the 1px border (all of the box's bg pixels are in its inner box),
    # and every one of the pixels sits ON that border (else the frame would leave some of them out).
    # (Its inside has bg, so it's never all this color: the filled check above didn't pass)
    if height >= 3 and width >= 3 and bg_count == count_bg(top + 1, left + 1, bottom - 1, right - 1):
        all_ys:np.ndarray = np.concatenate([ys for ys, _ in components])
        all_xs:np.ndarray = np.concatenate([xs for _, xs in components])
        if ((all_ys == top) | (all_ys == bottom - 1) | (all_xs == left) | (all_xs == right - 1)).all():
            frame_shadow:np.ndarray = np.ones((height, width), dtype=np.int8)
            frame_shadow[1:-1, 1:-1] = 0
            return frame_shadow

    # Filled rectangle with noise: its bg pixels are all small holes, so they're bg-colored shapes on top of it, not
    # missing pixels. (Holes on the box's edge count too: noise lands on a rectangle's edge just as often)
    # (Too much bg overall = a shape with real holes. Checked first, since it's cheap)
    if bg_count >= _AUTOCOMPLETE_MAX_HOLE_FRACTION * height * width:
        return None
    hole_labels, _ = scipy.ndimage.label(is_bg[top:bottom, left:right]) # type: ignore
    if np.bincount(hole_labels.ravel())[1:].max() > _AUTOCOMPLETE_MAX_HOLE_AREA: # type: ignore (scipy types the labels loosely)
        return None
    return np.ones((height, width), dtype=np.int8)

def _denoise_shapes(grid:np.ndarray, shapes:list[dict], is_not_bg:np.ndarray) -> tuple[list[dict], list[dict], np.ndarray]:
    """These one-color shapes with their noise cleaned up (1px danglers peeled off, 1px notches & small holes filled in), plus one small shape per noise spot and a grid mask of every noise pixel."""
    dangler_mask:np.ndarray = np.zeros(grid.shape, dtype=bool)
    fill_mask:np.ndarray = np.zeros(grid.shape, dtype=bool)

    # First find every shape's noise, all on the ORIGINAL shapes (one pass, so cleaning one bit can't make another)
    cleaned_shadows:list[Optional[np.ndarray]] = []
    for shape in shapes:
        shadow:np.ndarray = shape['_shadow'].astype(bool)

        # Too small to be a clean shape, or a plain full rectangle (nothing can stick in or out of it)? Nothing to clean
        if shadow.sum() <= _DENOISE_MAX_TOO_SMALL_AREA or shadow.all():
            cleaned_shadows.append(None)
            continue

        # Danglers & notches, each tangled spot cleaned whichever way leaves the smoother shape
        # (A notch that's some other color might be another shape hiding this one there, unlike a bg notch)
        hidden_in_box:np.ndarray = is_not_bg[shape['top']:shape['bottom'], shape['left']:shape['right']] & ~shadow
        danglers, notches = _smoothest_danglers_and_notches(shadow, hidden_in_box)

        # Holes: pixels fully enclosed by the shape, split into one-color pieces (4-connected, colors from the real grid).
        # Each piece is small enough to be noise, or not, on its own
        enclosed:np.ndarray = scipy.ndimage.binary_fill_holes(shadow) & ~shadow
        small_holes:np.ndarray = np.zeros_like(enclosed)
        if enclosed.any():
            box_grid:np.ndarray = grid[shape['top']:shape['bottom'], shape['left']:shape['right']]
            for color in np.unique(box_grid[enclosed]).tolist():
                piece_labels, _ = scipy.ndimage.label(enclosed & (box_grid == color)) # type: ignore
                # (Label 0 = not this color's hole, so it's never kept)
                piece_is_small:np.ndarray = np.bincount(piece_labels.ravel()) <= _AUTOCOMPLETE_MAX_HOLE_AREA # type: ignore (scipy types the labels loosely)
                piece_is_small[0] = False
                small_holes |= piece_is_small[piece_labels]
        fills:np.ndarray = notches | small_holes
        filled:np.ndarray = shadow | fills
        if not fills.any() and not danglers.any():
            cleaned_shadows.append(None)
            continue

        # Too many holes & notches = a shape with real holes or dents, not noise: leave it alone.
        # And what's left has to be big enough to be a real shape too (a plus sign is just 4 "danglers" on a dot)
        cleaned_shadow:np.ndarray = filled & ~danglers
        if fills.sum() >= _DENOISE_MAX_HOLE_FRACTION * filled.sum() or cleaned_shadow.sum() <= _DENOISE_MAX_TOO_SMALL_AREA:
            cleaned_shadows.append(None)
            continue
        cleaned_shadows.append(cleaned_shadow)
        dangler_mask[shape['top']:shape['bottom'], shape['left']:shape['right']] |= danglers
        fill_mask[shape['top']:shape['bottom'], shape['left']:shape['right']] |= fills

    # Nothing noisy anywhere? Then every shape stays as it is
    noise_mask:np.ndarray = dangler_mask | fill_mask
    if not noise_mask.any():
        return shapes, [], noise_mask

    # Now rebuild the shapes. A pixel that's another shape's notch or hole gets carved out of its owner too:
    # it's a noise shape on top now, and the owner may be hidden under the shape that filled it
    cleaned_shapes:list[dict] = []
    # (None = de-noising didn't touch this shape. A new name: the loop above already declared cleaned_shadow as an array)
    for shape, maybe_cleaned_shadow in zip(shapes, cleaned_shadows):
        top, left, bottom, right = shape['top'], shape['left'], shape['bottom'], shape['right']
        shadow = shape['_shadow'].astype(bool)
        carved:np.ndarray = shadow & fill_mask[top:bottom, left:right]
        if maybe_cleaned_shadow is None and not carved.any():
            cleaned_shapes.append(shape)
            continue
        kept_shadow:np.ndarray = (shadow if maybe_cleaned_shadow is None else maybe_cleaned_shadow) & ~carved
        if not kept_shadow.any(): # (all of it was someone's noise: its noise shape replaces it)
            continue

        # Drawn in its color over a copy of the grid, so the filled-in pixels have its color
        local_ys, local_xs = np.nonzero(kept_shadow)
        ys:np.ndarray = local_ys + top
        xs:np.ndarray = local_xs + left
        painted_grid:np.ndarray = grid.copy()
        painted_grid[ys, xs] = int(shape['color'][1:])
        cleaned_shapes.append(shape_coords_to_shape_dict(painted_grid, ys, xs))

    # The noise shapes, colors from the real grid. Every dangler is its own 1px shape...
    noise_shapes:list[dict] = [shape_coords_to_shape_dict(grid, np.array([y]), np.array([x])) for y, x in zip(*np.nonzero(dangler_mask))]
    # ...and the filled notches & holes are one shape per same-color clump (4-connected, like holes)
    fill_only_mask:np.ndarray = fill_mask & ~dangler_mask
    for color in np.unique(grid[fill_only_mask]).tolist():
        clump_labels, clump_count = scipy.ndimage.label(fill_only_mask & (grid == color)) # type: ignore
        for clump_ys, clump_xs in (np.nonzero(clump_labels == clump_label) for clump_label in range(1, clump_count + 1)):
            noise_shapes.append(shape_coords_to_shape_dict(grid, clump_ys, clump_xs))
    return cleaned_shapes, noise_shapes, noise_mask

def _neighbour_counts(shadow:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How many of each pixel's 4 neighbours, and of its 8 neighbours, are in the shape. (Outside its box & off-grid = not in it)"""
    as_ints:np.ndarray = shadow.astype(np.int8)
    return (scipy.ndimage.convolve(as_ints, _PLUS_NEIGHBOURS, mode='constant', cval=0),
            scipy.ndimage.convolve(as_ints, _RING_NEIGHBOURS, mode='constant', cval=0))

def _find_danglers(shadow:np.ndarray) -> np.ndarray:
    """Shape pixels sticking out by 1px: 1 side neighbour, and 3 neighbours in all (on an edge) or 2 (off a corner)."""
    # (The 8-count tells them apart from a 1px line's tip, which has just 1)
    four_counts, eight_counts = _neighbour_counts(shadow)
    return shadow & (four_counts == 1) & ((eight_counts == 3) | (eight_counts == 2))

def _find_notches(shadow:np.ndarray) -> np.ndarray:
    """Non-shape pixels (any color) biting 1px into the shape: 3 side & 5 in all (in an edge), or 2 side & 3 in all (a missing corner)."""
    # (The 4-count tells them apart from a plain rectangle corner or a pixel along a flat edge, which have the same 8-counts)
    four_counts, eight_counts = _neighbour_counts(shadow)
    return ~shadow & (((four_counts == 3) & (eight_counts == 5)) | ((four_counts == 2) & (eight_counts == 3)))

def _outline_length(shadow:np.ndarray) -> int:
    """How many in/out side-by-side pixel boundaries the shape has (off its box = out). Smoother shape = shorter outline."""
    padded:np.ndarray = np.pad(shadow, 1).astype(np.int8)
    return int(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())

def _smoothest_danglers_and_notches(shadow:np.ndarray, hidden:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shape's 1px danglers to peel & notches to fill. Each tangled spot of them gets whichever order leaves the shorter outline, unless all its notches are hidden under other colors and it has no edge danglers (then they're filled)."""
    # Neither order works everywhere: peeled danglers can leave fake notches between them, and filled notches can
    # leave a fake dangler between them. So first find both on the original shape...
    first_danglers:np.ndarray = _find_danglers(shadow)
    first_notches:np.ndarray = _find_notches(shadow)
    danglers:np.ndarray = np.zeros_like(shadow)
    notches:np.ndarray = np.zeros_like(shadow)
    if not first_danglers.any() and not first_notches.any():
        return danglers, notches

    # Danglers sticking out of the middle of an edge (3 neighbours in all), not off a corner (2)
    _, eight_counts = _neighbour_counts(shadow)
    edge_danglers:np.ndarray = first_danglers & (eight_counts == 3)

    # ...then group the touching ones (8-connected) into spots, and clean each spot (plus a 1px margin) on its own
    spot_labels, spot_count = scipy.ndimage.label(first_danglers | first_notches, structure=np.ones((3, 3))) # type: ignore
    for spot_label in range(1, spot_count + 1):
        spot:np.ndarray = scipy.ndimage.binary_dilation(spot_labels == spot_label, structure=np.ones((3, 3)))

        # Order (a): peel the spot's danglers, then fill the notches that are left
        peeled_first:np.ndarray = first_danglers & spot
        filled_second:np.ndarray = _find_notches(shadow & ~peeled_first) & spot
        danglers_first_result:np.ndarray = (shadow & ~peeled_first) | filled_second

        # Order (b): fill the spot's notches, then peel the danglers that are left
        filled_first:np.ndarray = first_notches & spot
        peeled_second:np.ndarray = _find_danglers(shadow | filled_first) & spot & shadow
        notches_first_result:np.ndarray = (shadow | filled_first) & ~peeled_second

        # Every notch of the spot is under another (non-bg) color? Then it's likely just hidden, so fill it: e.g. a
        # rectangle's corner pixel whose neighbours are covered looks like a dangler, but it's really still the corner.
        # Only for corner danglers though: dots sticking out of the middle of an edge are noise even when the pixels
        # between them are covered (filling those would grow the rectangle by a whole row of mostly nothing)
        notches_all_hidden:bool = (bool(filled_first.any()) and not (filled_first & ~hidden).any()
                                   and not (edge_danglers & spot).any())
        # Else keep the smoother one, MDL-style. Ties go to notches first (cutting a corner pixel off doesn't change the outline)
        if notches_all_hidden or _outline_length(notches_first_result) <= _outline_length(danglers_first_result):
            danglers |= peeled_second
            notches |= filled_first
        else:
            danglers |= peeled_first
            notches |= filled_second
    return danglers, notches

class _StackingFacts:
    """The "A is above B" facts of the completions accepted so far, so each new one can be checked for a contradiction without re-sorting them all."""

    def __init__(self, component_count:int):
        # Each component's shape (a completion's components become one shape, named by its first member)
        self.shape_of_component:np.ndarray = np.arange(component_count)
        # Which shapes are directly above each shape (only shapes with any facts are in here)
        self.shapes_above:dict[int, set[int]] = {}

    def try_add(self, completion:tuple[np.ndarray, int, int, np.ndarray, int], owner_grid:np.ndarray) -> bool:
        """Add this completion's facts if they don't contradict the ones so far (True), else change nothing (False)."""
        members, top, left, completed_shadow, _ = completion
        this_shape:int = int(members[0])
        member_set:set[int] = set(members.tolist())
        def merged(shape:int) -> int: # (the completion's members all count as this one shape)
            return this_shape if shape in member_set else shape

        # The new facts: whoever owns a pixel this completion covers is on top of it
        # (Noise holes are bg, owned by nobody: they go on top of everything later, so no facts about them)
        local_ys, local_xs = np.nonzero(completed_shadow)
        covered_owners:np.ndarray = owner_grid[local_ys + top, local_xs + left]
        owner_shapes:set[int] = {merged(shape) for shape in np.unique(self.shape_of_component[covered_owners[covered_owners >= 0]]).tolist()}
        owner_shapes.discard(this_shape)

        # Everything directly above the merged shape: its new owners, plus whatever was above any of its members
        above_this_shape:set[int] = set(owner_shapes)
        for member in member_set:
            above_this_shape.update(merged(shape) for shape in self.shapes_above.get(member, ()))
        above_this_shape.discard(this_shape)

        # The facts so far had no loop, and only this shape changed, so a new loop has to pass through it:
        # climb up from it, and if we ever get back to it, it contradicts something (A above B, B above A)
        to_visit:list[int] = list(above_this_shape)
        visited:set[int] = set()
        while to_visit:
            shape:int = to_visit.pop()
            if shape == this_shape:
                return False
            if shape in visited:
                continue
            visited.add(shape)
            to_visit.extend(merged(above) for above in self.shapes_above.get(shape, ()))

        # It fits! Merge the members into one shape, everywhere the facts mention them, and add the new facts
        self.shape_of_component[members] = this_shape
        for shape, above_shapes in self.shapes_above.items():
            if above_shapes & member_set:
                self.shapes_above[shape] = {merged(above) for above in above_shapes}
        for member in member_set:
            self.shapes_above.pop(member, None)
        self.shapes_above[this_shape] = above_this_shape
        return True

def _stacking_order(completions:list[tuple[np.ndarray, int, int, np.ndarray, int]], owner_grid:np.ndarray, component_count:int) -> Optional[np.ndarray]:
    """Every component's z (its group's, for completed ones), if one stacking order explains all these completions' overlaps. None if they contradict."""

    # Each completion's components become one shape: name it by its first member
    shape_of_component:np.ndarray = np.arange(component_count)
    for members, _, _, _, _ in completions:
        shape_of_component[members] = members[0]

    # "A is above B" facts: wherever a completion covers a pixel it doesn't own, the pixel's owner is on top of it
    # (Two completions covering the same pixel of a third shape: that one's above both, no fact between the two)
    shapes_below:dict[int, set[int]] = {}
    shapes_above:dict[int, set[int]] = {}
    for members, top, left, completed_shadow, _ in completions:
        # (Noise holes are owned by nobody, so they're left out)
        local_ys, local_xs = np.nonzero(completed_shadow)
        covered_owners:np.ndarray = owner_grid[local_ys + top, local_xs + left]
        owner_shapes:np.ndarray = shape_of_component[covered_owners[covered_owners >= 0]]
        this_shape:int = int(members[0])
        for owner_shape in np.unique(owner_shapes[owner_shapes != this_shape]).tolist():
            shapes_below.setdefault(owner_shape, set()).add(this_shape)
            shapes_above.setdefault(this_shape, set()).add(owner_shape)

    # Sort them bottom-up (a topological sort): z = the length of the longest chain of shapes below.
    # Start from the shapes with nothing below them, and only move up to a shape once everything below it is done
    z_per_shape:np.ndarray = np.zeros(component_count, dtype=np.int32)
    unfinished_below_counts:dict[int, int] = {shape: len(below) for shape, below in shapes_below.items()}
    ready_shapes:list[int] = [shape for shape in shapes_above if shape not in shapes_below]
    finished_count:int = 0
    while ready_shapes:
        shape:int = ready_shapes.pop()
        finished_count += 1
        for above_shape in shapes_above.get(shape, ()):
            z_per_shape[above_shape] = max(z_per_shape[above_shape], z_per_shape[shape] + 1)
            unfinished_below_counts[above_shape] -= 1
            if unfinished_below_counts[above_shape] == 0:
                ready_shapes.append(above_shape)

    # Some shape never got finished? Then it's in a loop (A above B somewhere, B above A elsewhere): no stacking explains that
    stacked_shapes:set[int] = set(shapes_below) | set(shapes_above)
    if finished_count < len(stacked_shapes):
        return None

    # Every component gets its shape's z
    return z_per_shape[shape_of_component]


########################################################
## Pattern backgrounds: one big shape at the very back (the pattern itself is found in pattern_background.py)
########################################################

# Stands in for the pattern's own pixels in the flood fill grid, so the flood fill skips them (no real color is negative)
_PATTERN_PIXEL_MARKER:int = -2

def _pattern_background_shape(pattern_background:PatternBackground) -> dict:
    """The completed pattern as one multicolor shape, behind every other shape (z = -1), its full box even where it hangs off the grid."""
    # Make the shape on the pattern's own canvas (every pixel exists there), then shift it back to grid coords
    pattern_ys, pattern_xs = np.nonzero(np.ones(pattern_background.pixels.shape, dtype=bool))
    shape:dict = shape_coords_to_shape_dict(pattern_background.pixels, pattern_ys, pattern_xs)
    for key in ('top', 'bottom', 'center_y'):
        shape[key] += pattern_background.top
    for key in ('left', 'right', 'center_x'):
        shape[key] += pattern_background.left

    # Every other shape sits at z 0 (no z = 0), so this one's behind them all.
    # Marked with a private flag (not a 'type': types like 'rect' mean something to the rest of the code)
    shape['z'] = -1
    shape['_pattern_background'] = True
    return shape


########################################################
## A helper method to get all shapes, and add useful data
########################################################

# 4-connectivity structure for hole detection (no diagonals)
_STRUCT_4:np.ndarray = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.int8)

# Shapes with at least this many pixels get their pixel grid built with numpy, smaller ones with a plain loop
# (numpy's per-call overhead loses on tiny shapes; 12 was the fastest of 8/10/12/16 on real shapes, measured)
_VECTORIZE_MIN_PIXELS:int = 12

def shape_coords_to_shape_dict(parent_grid:np.ndarray, ys:np.ndarray, xs:np.ndarray)->dict:

    # (Instead of passing a list of (x,y) tuples, pass in two same-length lists: Y's and X's)
    # (This is helpful for Numpy/Scipy)

    #########################################
    # Bounding box, dimensions, center
    #########################################

    left:int = int(xs.min())
    right:int = int(xs.max()) + 1
    top:int = int(ys.min())
    bottom:int = int(ys.max()) + 1
    width:int = right-left
    height:int = bottom-top
    center_x:float = left + width/2
    center_y:float = top + height/2

    #########################################
    # Normalized pixel grid => pixels, shadow
    #########################################

    # Pixel grid: -1 for no color, [0,9] for ARC color
    pixels_grid:np.ndarray = np.full((height, width), -1, dtype=np.int8)
    # Shadow grid: 0 for not in mask, 1 for in mask
    shadow_grid:np.ndarray = np.zeros((height, width), dtype=np.int8)
    shape_colors:list[int] = [] # For later use: Majority color
    count_edge_pixels:int = 0 # For later use: Is this a rectangle?
    # Big shapes: do the per-pixel stuff with numpy, all at once. Small shapes: keep the plain loop below,
    # because numpy's per-call overhead makes it SLOWER there (measured on real shapes: numpy only
    # wins from ~10 pixels up, e.g. 1 pixel: loop 3.6us vs numpy 6.5us; 30+ pixels: loop 43us vs numpy 12us).
    # Both give the exact same grids, colors (in the same pixel order, so the Counter's tie-breaks don't change) & edge count.
    if len(ys) >= _VECTORIZE_MIN_PIXELS:
        # Every pixel's color (from its global position), and where it sits in the shape's own little grid
        pixel_colors:np.ndarray = parent_grid[ys, xs]
        local_ys:np.ndarray = ys - top
        local_xs:np.ndarray = xs - left
        pixels_grid[local_ys, local_xs] = pixel_colors # local pixels!
        shadow_grid[local_ys, local_xs] = 1 # local shadow!

        # Data for later use... (same order as the loop's, so the majority color comes out the same)
        shape_colors = pixel_colors.tolist()
        # On-edge pixels = all pixels, minus the ones strictly inside the bounding box
        count_edge_pixels = len(ys) - int(np.count_nonzero(shadow_grid[1:-1, 1:-1]))
    else:
        for x,y in zip(xs.tolist(), ys.tolist()):
            color:int = int( parent_grid[y, x] ) # from global position
            local_x:int = x-left
            local_y:int = y-top
            pixels_grid[local_y, local_x] = color # local pixels!
            shadow_grid[local_y, local_x] = 1 # local shadow!

            # Data for later use...
            shape_colors.append(color)
            if local_x==0 or local_y==0 or local_x==width-1 or local_y==height-1: # on edge?
                count_edge_pixels += 1

    #########################################
    # Color: assume majority color is THE color (relevant for multicolor shapes)
    #########################################

    counter:Counter = Counter(shape_colors) # tally up every pixel's color
    most_common_color:int
    count:int
    # (A tie, e.g. a 50/50 two-color shape, picks a random winner, not the first color in raster order)
    most_common_color, count = _random_most_common(counter) # grab the winner
    color = most_common_color
    monochrome:bool = len(counter) == 1 # is it a single color?

    #########################################
    # What type of shape am I?
    #########################################

    area:int = len(ys) # note that Y's & X's have same length, so it doesn't matter which one

    type:str = 'blob' # default: lol i dunno
    # Lines also get a subtype, which way they point: "horizontal", "vertical", "diag_up" or "diag_down" (everything else gets none)
    subtype:str|None = None
    if width>1 and height>1:
        # If 90% of the bounding edge is filled, I'm probably a rectangle
        total_edge_pixels:int = 2*width + 2*height - 4 # 4 lines & minus corners
        filled_edge_ratio:float = count_edge_pixels/total_edge_pixels
        if filled_edge_ratio>=0.9:
            type = 'rect'
        # A diagonal line has one pixel per row & column (area = width = height)...
        # ...and ALL of them sit on one diagonal of the bounding box (not just any n pixels in an n x n box).
        # (The pixel check can't be skipped: a connection-radius parse can join n pixels that aren't diagonal neighbours)
        elif area==width==height:
            if np.trace(shadow_grid)==area: # top-left to bottom-right
                type = 'line'
                subtype = 'diag_down'
            elif np.trace(np.fliplr(shadow_grid))==area: # bottom-left to top-right
                type = 'line'
                subtype = 'diag_up'
    elif min(width,height)==1 and max(width,height)>1: # 1 in one direction, >1 in another
        type = 'line'
        subtype = 'horizontal' if height==1 else 'vertical'
    elif width==1 and height==1:
        type = 'dot'

    #########################################
    # HOLES & GAPS: label the empty pockets (4-connected zero-regions) in the shadow grid.
    # A pocket that doesn't touch the bbox edge is a hole. One that touches the edge but none of the 4 bbox
    # corner pixels is a gap (a dent in a side, like the opening of a "C"). One that touches a corner is just
    # a cut-off corner, so it counts as neither.
    # Fast-path: a shape that fills its whole bbox has no empty pockets at all.
    #########################################

    hole_count:int = 0
    gap_count:int = 0
    hole_grids:list[np.ndarray] = []
    if area < width * height:
        pocket_labels:np.ndarray
        pocket_count:int
        pocket_labels, pocket_count = scipy.ndimage.label(shadow_grid == 0, structure=_STRUCT_4) # type: ignore

        # Which pockets touch the bbox edge, and which touch a corner? (Label 0 is the shape itself, so drop it)
        edge_labels:set[int] = set(np.concatenate((pocket_labels[0, :], pocket_labels[-1, :],
                                                   pocket_labels[:, 0], pocket_labels[:, -1])).tolist()) - {0}
        corner_labels:set[int] = {int(pocket_labels[0, 0]), int(pocket_labels[0, -1]),
                                  int(pocket_labels[-1, 0]), int(pocket_labels[-1, -1])} - {0}
        gap_count = len(edge_labels - corner_labels)

        # Every pocket off the edge is a hole: keep its grid too (fill_holes, contains_count... need where it is)
        for pocket_label in range(1, pocket_count + 1):
            if pocket_label not in edge_labels:
                hole_count += 1
                hole_grid:np.ndarray = np.full((height, width), -1, dtype=np.int8) # -1 for no, 1 for yes
                hole_grid[pocket_labels == pocket_label] = 1
                hole_grids.append(hole_grid)

    #########################################
    # Finally, APPEND THIS SHAPE'S DATA TO RESULTS
    # ("Annotate") will come up with more advanced info later
    #########################################

    shape_dict:dict = {

        'type': type,
        # (No 'section' prop for now: sections are off in annotate, so selectors can't use it)

        # Color (cast to string, so that Selectors/Causality know it's NOT really a numerical property)
        'color': 'c'+str(color),
        'start_color': 'c'+str(color),
        'monochrome': monochrome,

        # Basic x,y,w,h info
        'top': top,
        'left': left,
        'right': right,
        'bottom': bottom,
        'center_x': center_x,
        'center_y': center_y,
        'width': width,
        'height': height,

        # Extra info
        'holes': hole_count,
        'gaps': gap_count,
        'area': area,

        # Grid data (private)
        '_pixels': pixels_grid,
        '_shadow': shadow_grid, # todo: public for DSL so we can match on this?
        '_hole_grids': hole_grids,
        # The shape's pixels as originally parsed, so paint() recolors ONLY those (not an outline ring, filled holes...).
        # Same size as _shadow, always: ops that grow the grid (outline, fit's glue) pad it, flip/rotate turn it along.
        # (Every parse path, chunks/greedy/flood fill, makes its shapes right here, so this covers them all)
        '_original_shadow': shadow_grid,

        # Has an action actually changed this shape yet? (Public, so programs can select e.g. all(edited=True))
        'edited': False,

        # Track edits! (private)
        '_edits_cost': 0, # to regularize against unnecessary edits
        '_moves': 0, # so most-recently-moved is drawn on top

    }

    # Only lines get a 'subtype' key at all, so non-lines don't add a useless prop for the selectors to try
    if subtype is not None:
        shape_dict['subtype'] = subtype
    return shape_dict


# shape_coords_to_shape_dict, but some of the pixels may be OFF the canvas (a partly-out-of-frame chunk).
# Same output as the original for in-frame pixels: it just pads the grid with the off-canvas color, then shifts back
def _shape_coords_to_shape_dict_off_canvas(parent_grid:np.ndarray, ys:np.ndarray, xs:np.ndarray, off_canvas_color:int) -> dict:
    """shape_coords_to_shape_dict for pixels that may hang off the grid: off-canvas pixels get off_canvas_color."""

    # How far past each edge do the pixels go? (0 if they don't)
    grid_height, grid_width = parent_grid.shape
    padding_top:int = max(0, -int(ys.min()))
    padding_left:int = max(0, -int(xs.min()))
    padding_bottom:int = max(0, int(ys.max()) + 1 - grid_height)
    padding_right:int = max(0, int(xs.max()) + 1 - grid_width)

    # A bigger grid with the off-canvas area filled in with that color, and the real grid in the middle
    padded_grid:np.ndarray = np.pad(parent_grid, ((padding_top, padding_bottom), (padding_left, padding_right)), constant_values=off_canvas_color)

    # Make the shape on the padded grid (so every pixel exists there), then shift its position back to real grid coords
    shape:dict = shape_coords_to_shape_dict(padded_grid, ys + padding_top, xs + padding_left)
    shape['top'] -= padding_top
    shape['bottom'] -= padding_top
    shape['center_y'] -= padding_top
    shape['left'] -= padding_left
    shape['right'] -= padding_left
    shape['center_x'] -= padding_left
    return shape

import random # to break ties randomly
from itertools import product
from functools import cached_property # to build each de-dupe key only once a comparison needs it
from collections import Counter
import numpy as np
from .apply_parse import apply_parse, apply_parse_one_grid # one_grid: to parse each input grid once & share it
from .apply_parse import resolve_bg_color # to spot bg modes that land on the same bg colors (= the same parse)
from .apply_parse import denoise_scene_both_ways # to try denoise (+ autocomplete_rects) on an already-parsed base scene
from .apply_parse import clear_chunk_in_frame_counts_cache # to empty apply_parse's in-frame count cache at the start of each search
from .apply_parse import clear_greedily_grab_chunks_cache # to empty apply_parse's chunk grab cache at the start & end of each search
from .pattern_background import find_pattern_background, clear_pattern_background_cache # big tiled / mirrored backgrounds (+ their per-grid cache)
from .score_parse import score_parse, _shadow_key, _non_identity_d4_transforms # + D4 helpers, to spot repeated chunks
from .score_parse import clear_score_parse_caches # to empty score_parse's D4 key + outline caches at the start of each search
from .score_parse import _find_free_color # to find the puzzle's free color once per search, not once per config

# CONFIG KNOBS
from ..config import PARSE_CHUNK_MIN_AREA
from ..config import PARSE_CONNECTION_RADIUS_ENABLED, PARSE_MAX_CONNECTION_RADIUS # connection radius configs (off by default)
from ..config import PARSE_MAX_SHAPES_PER_GRID # parses with more shapes than this on any demo grid get thrown out

# A hardcoded bg color (cN) must cover at least this fraction of the pixels of every demo input grid
HARDCODED_BG_MIN_INPUT_FRACTION:float = 0.10
# ...and at least this fraction of at least one demo input grid (somewhere it really looks like the background)
HARDCODED_BG_MIN_MAX_INPUT_FRACTION:float = 0.33

def search_parse(raw_pairs:list[dict], trace=None) -> list[dict]:
    """
    Given raw grid pairs (train & test *with no output*. hence, 'after' grids must be optional),
    try all Parse configs, then return them from best-to-worst.
    "Best" as in most compressed info + least change before/after + other human priors

    The plan:
    1. Generate ALL combinations of possible parse configs
    2. Apply each one to the raw train pairs + test inputs
    3. Score each parsed result via score_parse (approximate MDL / Kolmogorov complexity)
    4. Sort from best (lowest complexity) to worst
    5. De-dupe: configs with the same bg & shapes on every demo grid are one parse (the best one's kept), return the rest

    TODO: instead of a fixed top-N cutoff, try a score threshold
    (e.g. "configs within 2x of the best score") so we adapt to levels
    where there's 1 obvious winner vs 6 that are all roughly tied.
    """

    # Fresh D4 key + outline caches for this level (so they only ever hold one level's shadows, and never grow forever)
    clear_score_parse_caches()
    # Same deal for the chunks' in-frame pixel counts
    clear_chunk_in_frame_counts_cache()
    # ...and the chunk grabs themselves (shared between chunk configs that'd grab the exact same chunks)
    clear_greedily_grab_chunks_cache()
    # ...and the pattern backgrounds found per grid
    clear_pattern_background_cache()

    # first, is this ARC-3?
    first_grid:np.ndarray = raw_pairs[0]['input']
    is_arc_3:bool = first_grid.shape[1] == 64 and first_grid.shape[0] == 64
    # TODO: check docs that arc_3 will always be 64x64?

    # Step 1: generate all possible (base) parse configs
    all_base_configs:list[dict] = _generate_all_parse_configs(is_arc_3)

    # If NO demo pair keeps its grid size, the static greedy grab bails out on every pair (it only works on
    # same-sized grids), so every static=True config is an exact copy of its static=False twin (plus its config points):
    # it can never win. Don't parse or score those at all. (If even ONE pair keeps its size, keep them all.)
    # (Test pairs have no output, so they can't tell us anything here)
    any_pair_keeps_size:bool = any(raw_pair['output'] is not None and raw_pair['input'].shape == raw_pair['output'].shape
                                   for raw_pair in raw_pairs)
    if not any_pair_keeps_size:
        all_base_configs = [base_config for base_config in all_base_configs if not base_config['greedy_input_shapes_static']]

    # No bg + multicolor + rows AND columns connected = one whole-grid blob per grid (4- vs 8-connected doesn't matter,
    # since every pixel already touches its neighbors). Some tiny levels really are "the whole grid is the shape",
    # but on bigger grids that parse almost never is the answer, and it can still score well. So if any demo input
    # is bigger than 5 in width or height, don't parse or score it at all.
    # (Demo = train pairs, the ones with an output. Also bans the connection-radius variants: same blob)
    any_demo_input_bigger_than_5:bool = any(raw_pair['output'] is not None and max(raw_pair['input'].shape) > 5
                                            for raw_pair in raw_pairs)
    if any_demo_input_bigger_than_5:
        all_base_configs = [base_config for base_config in all_base_configs
                            if not (base_config['bg_mode'] == 'no_bg' and base_config['multicolor']
                                    and base_config['rows'] and base_config['columns'])]

    # A hardcoded bg color (cN) has to pass two checks, on the demo INPUT grids only (outputs can look very different):
    # 1. It covers a decent chunk (HARDCODED_BG_MIN_INPUT_FRACTION) of EVERY demo input. Else a color that's missing or
    #    rare somewhere would play bg there, and every real shape of that color would turn into "background"
    # 2. It covers a big chunk (HARDCODED_BG_MIN_MAX_INPUT_FRACTION) of at least ONE demo input, so it really looks like
    #    a background somewhere. Else a common shape color plays bg and its shapes vanish. (It also means most cN modes
    #    never get parsed at all, which keeps the parse search fast)
    demo_inputs:list[np.ndarray] = [raw_pair['input'] for raw_pair in raw_pairs if raw_pair['output'] is not None] # skip no-output Tests
    # Checked once per color, not once per config: each color shows up in many configs
    hardcoded_bg_modes:set[str] = {base_config['bg_mode'] for base_config in all_base_configs if base_config['bg_mode'][1:].isdigit()} # ('cN' = hardcoded)
    allowed_hardcoded_bg_modes:set[str] = set()
    for bg_mode in hardcoded_bg_modes:
        # What fraction of each demo input is this color?
        fractions_per_demo_input:list[float] = [float(np.mean(demo_input == int(bg_mode[1:]))) for demo_input in demo_inputs]
        if min(fractions_per_demo_input) >= HARDCODED_BG_MIN_INPUT_FRACTION and max(fractions_per_demo_input) >= HARDCODED_BG_MIN_MAX_INPUT_FRACTION:
            allowed_hardcoded_bg_modes.add(bg_mode)
    all_base_configs = [base_config for base_config in all_base_configs
                        if base_config['bg_mode'] not in hardcoded_bg_modes or base_config['bg_mode'] in allowed_hardcoded_bg_modes]

    # The puzzle's free color only depends on the raw grids, not the config, so find it once here and hand it to
    # every score_parse below. (It breaks 50/50 ties randomly, so finding it per config could make two configs disagree)
    free_color:str|None = _find_free_color(raw_pairs)

    # Step 2: apply & score each (base) config. iterate with chunks-saved configs
    scored_configs:list[dict] = []
    # Each kept base config's parsed pairs, in the same order as scored_configs, so we can look for repeated chunks below
    cached_parsed_pairs:list[list[dict]] = []
    # ...and the kept base configs themselves, same order. (A base config with too many shapes doesn't get kept)
    kept_base_configs:list[dict] = []
    # Which raw pairs are demos (they have an output), to check the shared input scenes for too many shapes
    demo_pair_indices:list[int] = [pair_index for pair_index, raw_pair in enumerate(raw_pairs) if raw_pair['output'] is not None]
    # How many base configs got thrown out for too many shapes (just for the trace)
    number_of_base_configs_with_too_many_shapes:int = 0
    # Each input grid's parse, keyed by the only settings an INPUT parse reads (bg_mode, connectivity, radius, multicolor).
    # The greedy flags only change the OUTPUT parse, so all 6 greedy combos share one input parse instead of redoing it.
    # (Safe to share: apply_parse's output parse + greedy grabs and score_parse only READ input scenes, never change them.
    # Chunks change the input parse too, but the chunk re-parses below don't use this cache)
    shared_input_scenes:dict[tuple, list[dict]] = {}
    # Same key, but each pair's OUTPUT parsed by the plain (no-greedy) config. A greedy config whose grabs grab
    # nothing on some output gives exactly this parse, so apply_parse reuses it instead of re-flood-filling.
    # (The plain config comes first among its greedy combos in config order, so it's always here before they need it)
    plain_output_scenes:dict[tuple, list[dict|None]] = {}

    # Bg-mode twins: two configs that differ only in bg_mode, whose bg modes land on the same bg color on every demo
    # grid, parse every demo grid the exact same way (the parse only reads bg_mode through that color). So each bg
    # mode's bg colors get resolved once up front (the same way the parse does), and a twin reuses the first one's
    # parsed pairs instead of parsing again. (Still scored on its own, so score_parse can price bg modes differently)
    # Test inputs are kept apart: they don't decide the parse, but the chunk & autocomplete steps below read them
    demo_bg_colors_per_bg_mode:dict[str, tuple[str, ...]] = {}
    test_bg_colors_per_bg_mode:dict[str, tuple[str, ...]] = {}
    for bg_mode in dict.fromkeys(base_config['bg_mode'] for base_config in all_base_configs): # unique, in config order
        # Every demo's input, then output (outputs pass their input grid, for most_common_input_color).
        # Same 'cN' strings as a scene's 'background', so they compare straight against parsed scenes
        demo_bg_colors_per_bg_mode[bg_mode] = tuple(f"c{bg_color}" for raw_pair in raw_pairs if raw_pair['output'] is not None
                                                    for bg_color in (resolve_bg_color(raw_pair['input'], bg_mode),
                                                                     resolve_bg_color(raw_pair['output'], bg_mode, input_grid=raw_pair['input'])))
        test_bg_colors_per_bg_mode[bg_mode] = tuple(f"c{resolve_bg_color(raw_pair['input'], bg_mode)}" for raw_pair in raw_pairs if raw_pair['output'] is None)
    # Parsed pairs per (actual demo bg colors, every other setting). Keyed by the bg colors the parse really landed on,
    # not the resolved ones above: most_common_edge_color breaks border ties randomly, so the two can disagree
    parsed_pairs_per_twin_key:dict[tuple, list[dict]] = {}
    # The test raw pairs alone, for a twin whose test inputs get different bg colors
    test_raw_pairs:list[dict] = [raw_pair for raw_pair in raw_pairs if raw_pair['output'] is None]

    # Bg-mode twins share the very same parsed pairs, so score_parse works their change cost out only once (see score_parse)
    change_cost_cache:dict[tuple, float] = {}

    for base_config in all_base_configs:

        # Is there a bg-mode twin parsed already? Then reuse its demo pairs (and its test pairs too, if this bg mode
        # gives the test inputs the same bg colors; else just the test inputs get parsed again, with this config)
        other_settings:tuple = tuple((key, value) for key, value in base_config.items() if key != 'bg_mode')
        twin_parsed_pairs:list[dict]|None = parsed_pairs_per_twin_key.get((demo_bg_colors_per_bg_mode[base_config['bg_mode']], other_settings))
        parsed_pairs:list[dict]
        if twin_parsed_pairs is not None:
            twin_test_bg_colors:tuple[str, ...] = tuple(pair['input']['background'] for pair in twin_parsed_pairs if pair['output'] is None)
            if twin_test_bg_colors == test_bg_colors_per_bg_mode[base_config['bg_mode']]:
                parsed_pairs = twin_parsed_pairs
            else:
                test_parsed_pairs = iter(apply_parse(base_config, test_raw_pairs))
                parsed_pairs = [pair if pair['output'] is not None else next(test_parsed_pairs) for pair in twin_parsed_pairs]

        # No twin yet? Then parse it for real
        else:
            # Parse the inputs for this combo of input settings, if no earlier config did yet.
            # (The connection radius changes the input's flood fill too: 1 = plain, when the key's missing)
            input_settings_key:tuple = (base_config['bg_mode'], base_config['diagonals'], base_config['columns'], base_config['rows'], base_config['multicolor'],
                                        base_config.get('radius', 1))
            if input_settings_key not in shared_input_scenes:
                # (Demo inputs stop early past the shape cap, since they'd get thrown out right below. Test inputs never do)
                shared_input_scenes[input_settings_key] = [apply_parse_one_grid(raw_pair['input'], base_config,
                                                                                max_shapes=PARSE_MAX_SHAPES_PER_GRID if raw_pair['output'] is not None else None)
                                                           for raw_pair in raw_pairs]

            # Too many shapes on a demo input already? Then don't even parse the outputs. (The greedy flags only change
            # the output parse, so all of this input parse's greedy combos get thrown out right here too)
            if any(shared_input_scenes[input_settings_key][pair_index].get('too_many_shapes', False) for pair_index in demo_pair_indices):
                number_of_base_configs_with_too_many_shapes += 1
                continue

            # Apply, save baseline version. Reuses the shared inputs, + the plain config's outputs
            # (None while parsing the plain config itself, or if it got skipped)
            parsed_pairs = apply_parse(base_config, raw_pairs, input_scenes=shared_input_scenes[input_settings_key],
                                       plain_output_scenes=plain_output_scenes.get(input_settings_key), max_shapes=PARSE_MAX_SHAPES_PER_GRID)
            # Is this the plain, no-greedy config? Then save its outputs for its greedy twins
            if not (base_config['greedy_input_shapes_static'] or base_config['greedy_input_shapes_moving']):
                plain_output_scenes[input_settings_key] = [parsed_pair['output'] for parsed_pair in parsed_pairs]
            # ...and save the whole parse for its bg-mode twins
            actual_demo_bg_colors:tuple[str, ...] = tuple(scene['background'] for scene in _demo_scenes(parsed_pairs))
            parsed_pairs_per_twin_key.setdefault((actual_demo_bg_colors, other_settings), parsed_pairs)

        # Too many shapes on some demo grid? Then throw it out before scoring. (Checked after the saves above, so its
        # twins & greedy combos can still reuse its scenes, and a reused twin parse gets checked here too)
        if _too_many_shapes(parsed_pairs):
            number_of_base_configs_with_too_many_shapes += 1
            continue

        # Score it (every config on its own, twin or not)
        # ('_parsed_pairs' is only kept until the de-dupe in Step 5, which compares the parsed scenes)
        scored_configs.append({
            'config': base_config,
            'score': score_parse(base_config, parsed_pairs, free_color, change_cost_cache),
            '_parsed_pairs': parsed_pairs
        })
        cached_parsed_pairs.append(parsed_pairs)
        kept_base_configs.append(base_config)

    # Step 3: repeated chunks. For each base config, find shadows that show up again and again (ignoring color,
    # any D4 flip/rotation), and try that config again with them as 'chunks', which apply_parse greedily grabs
    # BEFORE the flood fill. Scored just like the base configs, so a chunk parse only wins if it's actually simpler.
    # EVERY eligible config gets a try (no top-N cap): a config whose chunks would win big often ranks badly
    # BEFORE chunking, since its touching chunks merge into big unique blobs.
    # (Walking best-first doesn't pick anything, so no random tie-break needed: the chunk configs get shuffled
    # with everyone else in Step 4)
    best_base_indices:list[int] = sorted(range(len(scored_configs)), key=lambda index: scored_configs[index]['score'])
    for base_index in best_base_indices:

        # Only configs with both rows AND columns connectivity (4- or 8-connected). Rows-only / columns-only
        # parses are all lines, so their "repeated chunks" are too trivial to be worth trying
        base_config:dict = scored_configs[base_index]['config']
        if not (base_config['rows'] and base_config['columns']):
            continue

        # No chunk re-parse when EITHER greedy grab is on (Nicky's call): chunking is ~70% of search_parse's
        # time, and the 4 greedy combos would quadruple it. So only the plain, no-greedy configs get chunked
        if base_config['greedy_input_shapes_static'] or base_config['greedy_input_shapes_moving']:
            continue

        # Any repeated chunks in this config's parse? (No? Then there's nothing new to try)
        chunks:list[np.ndarray] = _find_repeated_chunks(cached_parsed_pairs[base_index])
        if not chunks:
            continue

        # Same config, plus chunks. Apply, score, save chunks-saved version
        chunk_config:dict = {**scored_configs[base_index]['config'], 'chunks': chunks}
        chunk_parsed_pairs:list[dict] = apply_parse(chunk_config, raw_pairs, max_shapes=PARSE_MAX_SHAPES_PER_GRID)

        # Grabbing chunks can split the leftover pixels into more shapes, so check again. (Before reading any shapes:
        # a demo grid over the cap is just a flag, with no shapes at all)
        if _too_many_shapes(chunk_parsed_pairs):
            continue

        # Each chunk costs its pixel area, so drop the chunks no shape actually got parsed as.
        # (E.g. "two squares touching" can repeat too, as a 50-pixel blob, but the single square gets
        # grabbed first, so the blob chunk never grabs anything, and would cost 50 points for nothing.)
        # An unused chunk grabbed nothing, so dropping it doesn't change the parse: no re-parse needed.
        parsed_shadow_keys:set[tuple] = {_shadow_key(shape['_shadow'])
                                         for pair in chunk_parsed_pairs for scene in (pair['input'], pair['output']) if scene is not None
                                         for shape in scene['shapes']}
        used_chunks:list[np.ndarray] = [chunk for chunk in chunks
                                        if any(_shadow_key(transformed) in parsed_shadow_keys for transformed in [chunk] + _non_identity_d4_transforms(chunk))]
        # No chunk used at all? Then it's just the base config again, nothing new to score
        if not used_chunks:
            continue
        chunk_config = {**chunk_config, 'chunks': used_chunks}

        scored_configs.append({
            'config': chunk_config,
            'score': score_parse(chunk_config, chunk_parsed_pairs, free_color),
            '_parsed_pairs': chunk_parsed_pairs
        })

    # Step 3b: de-noise, and autocomplete rectangles. Every one-color, no-greedy base config again, twice:
    # de-noised only (ragged shapes cleaned up, noise shapes on top), and de-noised + rectangles, lines & frames that are
    # partly hidden behind other shapes completed into whole shapes, stacked by z. Scored like everyone else.
    # (No greedy combos, like the chunks: a greedy grab carries an input shape over before any stacking gets worked out)
    # (No multicolor: a cleaned-up shape gets redrawn in its one color, so a multicolor blob's other-colored pixels would
    # vanish, and a scene missing pixels can look much cheaper than the real one)
    # Built from each base config's own parse (no flood-filling again), which apply_parse_one_grid's de-noise matches exactly.
    # (cached_parsed_pairs is in kept_base_configs' order: Step 3's chunk configs never go in there)
    # Bg-mode twins share the very same base parse, so they share its de-noised parses too.
    # (Keyed by id: cached_parsed_pairs keeps every base parse alive, so ids can't get reused)
    denoised_parsed_pairs_per_base_parse:dict[int, list[tuple[dict, list[dict]]]] = {}
    for base_config, base_parsed_pairs in zip(kept_base_configs, cached_parsed_pairs):
        if base_config['multicolor'] or base_config['greedy_input_shapes_static'] or base_config['greedy_input_shapes_moving']:
            continue
        if id(base_parsed_pairs) not in denoised_parsed_pairs_per_base_parse:
            # (De-noising peels noise off into shapes of its own, so drop any de-noised parse with too many shapes.
            # Counted here, not flagged: de-noising works on full base parses, which never went past the cap in apply_parse)
            denoised_parsed_pairs_per_base_parse[id(base_parsed_pairs)] = [(denoise_settings, denoised_parsed_pairs)
                                                                           for denoise_settings, denoised_parsed_pairs in _denoised_parses(base_parsed_pairs)
                                                                           if all(len(scene['shapes']) <= PARSE_MAX_SHAPES_PER_GRID
                                                                                  for scene in _demo_scenes(denoised_parsed_pairs))]

        # Each de-noised parse of this base parse (0, 1 or 2 of them), with this base config's settings
        for denoise_settings, denoised_parsed_pairs in denoised_parsed_pairs_per_base_parse[id(base_parsed_pairs)]:
            denoised_config:dict = {**base_config, **denoise_settings}
            scored_configs.append({
                'config': denoised_config,
                'score': score_parse(denoised_config, denoised_parsed_pairs, free_color, change_cost_cache),
                '_parsed_pairs': denoised_parsed_pairs
            })

    # Step 3c: pattern backgrounds. Does any DEMO grid have a big tiled or mirrored pattern behind its shapes? (Checked once
    # per grid, cached, and most grids bail out after a color count.) Then every no-greedy base config again, with that pattern
    # as one shape at the very back and only what breaks it as shapes in front. Scored like everyone else.
    # (Demos only: a test input's pattern must never switch a parse option on. No greedy combos, like the chunks &
    # autocomplete: keeps the extra configs down to the plain ones)
    any_pattern_background:bool = any(find_pattern_background(grid) is not None
                                      for raw_pair in raw_pairs if raw_pair['output'] is not None
                                      for grid in (raw_pair['input'], raw_pair['output']))
    if any_pattern_background:
        for base_config in all_base_configs:
            if base_config['greedy_input_shapes_static'] or base_config['greedy_input_shapes_moving']:
                continue
            pattern_config:dict = {**base_config, 'pattern_background': True}
            pattern_parsed_pairs:list[dict] = apply_parse(pattern_config, raw_pairs, max_shapes=PARSE_MAX_SHAPES_PER_GRID)
            if _too_many_shapes(pattern_parsed_pairs):
                continue
            scored_configs.append({
                'config': pattern_config,
                'score': score_parse(pattern_config, pattern_parsed_pairs, free_color),
                '_parsed_pairs': pattern_parsed_pairs
            })

    # Step 4: sort all by score (lower = simpler = better)
    # Shuffle first, so configs with the SAME score come out in random order (sort is stable),
    # instead of config-list order. (A level that only parses right with one lucky tie-break should fail loudly)
    random.shuffle(scored_configs)
    scored_configs.sort(key=lambda x: x['score'])

    # Step 5: different configs often land on the exact same parse (e.g. most_common_scene_color and c0 when black is
    # every grid's bg, or a greedy grab that grabs nothing). Keep only the best of each, so main_solver's attempts
    # each get a genuinely different parse. (The dropped twins ride along in their winner's 'duplicates')
    number_of_configs_scored:int = len(scored_configs)
    scored_configs = _deduplicate_parses(scored_configs)

    # Log what we found
    if trace:
        top_3 = scored_configs[:3]
        trace.log("search_parse", f"Tested {number_of_configs_scored} parse configs, {len(scored_configs)} different parses. "
                  f"Top 3 scores: {[round(c['score'], 2) for c in top_3]}. "
                  f"Threw out {number_of_base_configs_with_too_many_shapes} base configs with over {PARSE_MAX_SHAPES_PER_GRID} shapes on a demo grid")

    # Empty the chunk grab cache again on the way out, so main_solver's later apply_parse calls (whose scenes
    # annotate goes on to change) grab their chunks fresh, instead of sharing shape dicts with the cache
    clear_greedily_grab_chunks_cache()
    clear_pattern_background_cache() # (just to keep memory down: the pattern cache only ever holds pure results)

    # Return the sorted, de-duped results
    return scored_configs


def _too_many_shapes(parsed_pairs:list[dict]) -> bool:
    """Did apply_parse flag any demo grid (input or output) for going past the PARSE_MAX_SHAPES_PER_GRID shape cap?"""
    # Demos only: a test input must never throw a parse out. (Test inputs never get the cap anyway, so never get flagged)
    # (Some config always passes on grids up to 30x30: e.g. no bg + multicolor + rows only = at most one shape per row)
    return any(scene.get('too_many_shapes', False) for scene in _demo_scenes(parsed_pairs))


########################################################
## De-dupe: configs that give the exact same parse on every demo
########################################################

def _deduplicate_parses(sorted_scored_configs:list[dict]) -> list[dict]:
    """Keep only the best-scored config of each group that parses every demo grid into the same bg & shapes (twins go in its 'duplicates')."""
    # Demos only (train inputs + outputs): test inputs shouldn't decide anything about the parse.
    # Walks best-first, so the first config of each group is its winner. (Ties were already shuffled, so that's random)
    kept_results:list[dict] = []

    # The cheapest check first: configs with different bg colors on any demo grid can't be the same parse, so each
    # config only gets compared against the winners that have all the same demo bg colors
    winners_per_bg_key:dict[tuple, list[tuple[dict, _DemoScenesKeys]]] = {}

    for scored_config in sorted_scored_configs:
        parsed_pairs:list[dict] = scored_config.pop('_parsed_pairs') # (not needed after this, so don't keep it around)
        scored_config['duplicates'] = []
        demo_scenes_keys:_DemoScenesKeys = _DemoScenesKeys(_demo_scenes(parsed_pairs))
        same_bg_winners:list[tuple[dict, _DemoScenesKeys]] = winners_per_bg_key.setdefault(demo_scenes_keys.bg_key, [])

        # Same shapes as an earlier (better or tied) winner? Then it's a twin: just remember its config
        twin_winner:dict|None = next((winner for winner, winner_keys in same_bg_winners if demo_scenes_keys.same_shapes_as(winner_keys)), None)
        if twin_winner is not None:
            twin_winner['duplicates'].append(scored_config['config'])
            continue
        same_bg_winners.append((scored_config, demo_scenes_keys))
        kept_results.append(scored_config)

    return kept_results

def _demo_scenes(parsed_pairs:list[dict]) -> list[dict]:
    """Every demo grid's scene (input, then output, per demo pair). Test pairs have no output, so they're skipped."""
    return [scene for pair in parsed_pairs if pair['output'] is not None for scene in (pair['input'], pair['output'])]

class _DemoScenesKeys:
    """One config's demo scenes + keys to compare them by, cheapest first. Each key only gets built once a comparison gets that far."""

    def __init__(self, demo_scenes:list[dict]):
        self.demo_scenes:list[dict] = demo_scenes
        # The bg colors are the cheapest key of all, so they're always built (the de-dupe groups by them)
        self.bg_key:tuple = tuple(scene['background'] for scene in demo_scenes)

    @cached_property
    def shape_counts(self) -> tuple[int, ...]:
        """How many shapes each demo scene has."""
        return tuple(len(scene['shapes']) for scene in self.demo_scenes)

    @cached_property
    def boxes_key(self) -> tuple:
        """Every demo scene's shapes by where they sit, how big they are & their z place (no pixels yet). Sorted, so shape order doesn't matter."""
        return tuple(tuple(sorted((shape['top'], shape['left'], shape['_pixels'].shape, z_rank)
                                  for shape, z_rank in zip(scene['shapes'], _z_ranks(scene))))
                     for scene in self.demo_scenes)

    @cached_property
    def pixels_key(self) -> tuple:
        """Like boxes_key, plus each shape's pixels (they already include its shadow, and tell multicolor shapes apart)."""
        return tuple(tuple(sorted((shape['top'], shape['left'], shape['_pixels'].shape, shape['_pixels'].tobytes(), z_rank)
                                  for shape, z_rank in zip(scene['shapes'], _z_ranks(scene))))
                     for scene in self.demo_scenes)

    def same_shapes_as(self, other:'_DemoScenesKeys') -> bool:
        """Same shapes on every demo scene? (The caller already checked the bg colors match.) Stops at the first key that differs."""
        # Literally the same scene objects (a bg-mode twin's reused parse, a greedy grab that grabbed nothing)? Then
        # they're the same without looking inside
        if all(my_scene is other_scene for my_scene, other_scene in zip(self.demo_scenes, other.demo_scenes)):
            return True
        # Else cheapest first: shape counts, then positions & sizes, then finally the pixels
        return (self.shape_counts == other.shape_counts
                and self.boxes_key == other.boxes_key
                and self.pixels_key == other.pixels_key)

def _z_ranks(scene:dict) -> list[int]:
    """Each shape's place in the scene's stacking order (tied z = same place), not its raw z value. Only RELATIVE z matters."""
    # (Parses without autocomplete have no z at all: everything's on one layer)
    sorted_z_values:list[int] = sorted({shape.get('z', 0) for shape in scene['shapes']})
    z_rank_per_z_value:dict[int, int] = {z_value: rank for rank, z_value in enumerate(sorted_z_values)}
    return [z_rank_per_z_value[shape.get('z', 0)] for shape in scene['shapes']]


########################################################
## De-noise: one base parse, de-noised only and de-noised + its partly hidden rectangles completed
########################################################

def _denoised_parses(base_parsed_pairs:list[dict]) -> list[tuple[dict, list[dict]]]:
    """The base parse de-noised, and de-noised + autocompleted, as (config settings, parsed pairs). Skips any that's the same parse as the base or the other."""
    # Both ways at once, per grid (they share one de-noising pass). Even when de-noising changes nothing, a clean
    # rectangle can still be hidden under something, so each one gets checked on its own
    denoised_parsed_pairs:list[dict] = []
    completed_parsed_pairs:list[dict] = []
    for pair in base_parsed_pairs:
        denoised_input, completed_input = denoise_scene_both_ways(pair['input'])
        denoised_output, completed_output = denoise_scene_both_ways(pair['output']) if pair['output'] is not None else (None, None)
        denoised_parsed_pairs.append({'input': denoised_input, 'output': denoised_output})
        completed_parsed_pairs.append({'input': completed_input, 'output': completed_output})

    # Nothing changed on any grid? Then it's just the base parse again, nothing new to score
    denoised_parses:list[tuple[dict, list[dict]]] = []
    denoise_changed_anything:bool = _anything_completed_or_peeled(denoised_parsed_pairs, base_parsed_pairs)
    if denoise_changed_anything:
        denoised_parses.append(({'denoise': True}, denoised_parsed_pairs))

    # Completion that completed nothing gives the very same demo scenes as de-noise only: then that one's enough.
    # (Demos only, like the de-dupe in Step 5)
    if _anything_completed_or_peeled(completed_parsed_pairs, base_parsed_pairs) and not (
            denoise_changed_anything
            and _DemoScenesKeys(_demo_scenes(completed_parsed_pairs)).same_shapes_as(_DemoScenesKeys(_demo_scenes(denoised_parsed_pairs)))):
        denoised_parses.append(({'denoise': True, 'autocomplete_rects': True}, completed_parsed_pairs))
    return denoised_parses

def _anything_completed_or_peeled(autocomplete_parsed_pairs:list[dict], base_parsed_pairs:list[dict]) -> bool:
    """Did de-noising (+ completion) change anything on any grid of the base parse?"""
    # Anything completed? (Every completion hides something under another shape, and that shape's z is at least 1.
    # De-noised noise shapes go on top, so they have z ≥ 1 too)
    anything_completed:bool = any(shape['z'] > 0 for pair in autocomplete_parsed_pairs for scene in (pair['input'], pair['output']) if scene is not None
                                  for shape in scene['shapes'])
    # ...or peeled? (Then the shape count changes)
    anything_peeled:bool = any(len(autocomplete_scene_of_grid['shapes']) != len(base_scene['shapes'])
                               for autocomplete_pair, base_pair in zip(autocomplete_parsed_pairs, base_parsed_pairs)
                               for autocomplete_scene_of_grid, base_scene in ((autocomplete_pair['input'], base_pair['input']), (autocomplete_pair['output'], base_pair['output']))
                               if base_scene is not None)
    return anything_completed or anything_peeled

########################################################
## Repeated chunks: shadows that show up again and again
########################################################

def _find_repeated_chunks(parsed_pairs:list[dict]) -> list[np.ndarray]:
    """Every "complicated" shadow that shows up in 2+ different pairs (ignoring color, up to D4), most-repeated first."""
    # Each demo pair (input AND output together) is one "unit", each test input is its own unit.
    # A chunk has to show up in 2+ units, so a shape that merely MOVES inside one pair is NOT a chunk

    # How many times does each shadow show up? (Keyed by its D4-canonical form, so a rotated copy counts as the same chunk)
    shadow_counts:Counter = Counter()
    canonical_shadows:dict[tuple, np.ndarray] = {}
    # Which units (= indices into parsed_pairs) each shadow showed up in. A test pair has no output,
    # so "one pair = one unit" already makes each test input its own unit
    shadow_units:dict[tuple, set[int]] = {}

    # (The pair's index = which unit each shadow came from)
    for unit_index, pair in enumerate(parsed_pairs):
        for scene in (pair['input'], pair['output']):
            if scene is None: # skip no-output Tests
                continue
            for shape in scene['shapes']:

                # Only "complicated" shadows: no dots or lines, and not too tiny
                if shape['width'] < 2 or shape['height'] < 2 or shape['area'] < PARSE_CHUNK_MIN_AREA:
                    continue

                # TODO: chunks only match up to D4 (+ any color) for now. Later we'll need to match up to
                # *scaling* too (a 2x-bigger copy of a chunk is still that chunk), here AND in score_parse/apply_parse.
                # Canonical form: whichever of its 8 D4 transforms has the smallest key (just needs to be consistent)
                shadow:np.ndarray = shape['_shadow']
                canonical_key, canonical_shadow = min(((_shadow_key(transformed), transformed) for transformed in [shadow] + _non_identity_d4_transforms(shadow)),
                                                      key=lambda key_and_shadow: key_and_shadow[0])
                shadow_counts[canonical_key] += 1
                canonical_shadows[canonical_key] = canonical_shadow
                shadow_units.setdefault(canonical_key, set()).add(unit_index)

    # Repeated = showed up in at least 2 different units (repeats inside one grid / one pair don't count, for now).
    # The sort below still uses the TOTAL count, for "most-repeated first"
    chunks:list[np.ndarray] = [canonical_shadows[key] for key, units in shadow_units.items() if len(units) >= 2]

    # Most-repeated chunks first (ties: biggest first), since apply_parse grabs them in this order.
    # (Not just biggest first: two touching chunks that happen to touch the same way twice ALSO repeat, as one
    # big blob, and would get grabbed before the real chunk. The real chunk repeats more often than that.)
    # Shuffle first, so chunks tied on BOTH count and size come out in random order (sort is stable), not order found
    random.shuffle(chunks)
    chunks.sort(key=lambda chunk: (-shadow_counts[_shadow_key(chunk)], -int(chunk.sum())))
    return chunks


def _generate_all_parse_configs(is_arc_3:bool) -> list[dict]:
    """
    Generate all combinations of parse hyperparameters.
    Cartesian product of bg_mode * connectivity * multicolor * 6 greedy combos (static / moving / overlap).
    """

    # each entry: (key_name_or_dict, list_of_options)
    # dicts get unpacked into the config, scalars get assigned to the key
    options = {

        # bg_mode: smart defaults first, then every specific color
        # ('most_common_edge_color': each input's most common border color, also used for its output)
        'bg_mode': ['most_common_scene_color', 'most_common_input_color', 'no_bg', 'most_common_edge_color']
              + [f'c{i}' for i in range(16 if is_arc_3 else 10)],

        # Pixel connectivity: which neighbors count as "same shape"?
        'connectivity': [
            {'diagonals': True,  'columns': True,  'rows': True},   # all 8 directions
            {'diagonals': False, 'columns': True,  'rows': False},  # columns only
            {'diagonals': False, 'columns': False, 'rows': True},   # rows only
            {'diagonals': False, 'columns': True,  'rows': True},   # 4-connectivity (no diags)
        ]
        # + (only with PARSE_CONNECTION_RADIUS_ENABLED) all 8 directions, but pixels up to `radius` apart connect too
        # (radius 2 = across a 1-pixel gap). Only with 8-connectivity: Chebyshev distance is an all-8-directions idea.
        # Appended at the end, so with the flag off the config list (and its order) is the plain one
        + ([{'diagonals': True, 'columns': True, 'rows': True, 'radius': radius} for radius in range(2, PARSE_MAX_CONNECTION_RADIUS + 1)]
           if PARSE_CONNECTION_RADIUS_ENABLED else []),

        # Are shapes allowed to have mixed colors?
        'multicolor': [False, True],
        #'multicolor': [False],

        # Greedy grabs, before flood-filling the output:
        # - static: grab each monochrome input shape's pixels as the same shape (same position, maybe recolored),
        #   if they're still monochrome & non-bg in the output
        # - moving: grab input shapes (area >= GREEDY_MOVING_MIN_AREA) whose exact pixels show up SOMEWHERE in
        #   the output (moved, or cropped into a different-sized grid)
        # - overlap: moved shapes may share pixels. Overlap without moving means nothing, so only these 6 of the 8 combos
        # (Whole dicts, unpacked like connectivity)
        'greedy': [
            {'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': False, 'greedy_input_shapes_overlap': False},
            {'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': True,  'greedy_input_shapes_overlap': False},
            {'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': True,  'greedy_input_shapes_overlap': True},
            {'greedy_input_shapes_static': True,  'greedy_input_shapes_moving': False, 'greedy_input_shapes_overlap': False},
            {'greedy_input_shapes_static': True,  'greedy_input_shapes_moving': True,  'greedy_input_shapes_overlap': False},
            {'greedy_input_shapes_static': True,  'greedy_input_shapes_moving': True,  'greedy_input_shapes_overlap': True},
        ],
    }

    # Cartesian product of all option lists
    keys:list[str] = list(options.keys())
    configs:list[dict] = []
    for combo in product(*options.values()):
        config:dict = {}
        for key, value in zip(keys, combo):
            if isinstance(value, dict): # unpack whole dicts, like connectivity
                config.update(value)
            else:
                config[key] = value
        configs.append(config)

    return configs

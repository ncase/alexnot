"""
The DEFAULT config: every tuning knob for the whole solver, all in one place.
Other modes (e.g. double_time.py) start with `from .default import *` and override only what changes.
Pick one with the env var ALEXNOT_CONFIG (or BLAST_EM's --config flag). See config/__init__.py.
"""

############################################################
# Main Solver (main_solver.py)
############################################################

# How many parse configs to try (one search_transform attempt each), and the wall-clock cap per attempt.
# (19 monkeys x 0.2s x 10 rounds is way more than 10s, so 10s attempts kept hitting the cap)
# 1 attempt: on our smoke-test levels, 3 attempts only won one extra level (a known coin flip) after the crash fixes, so the
# extra parses aren't worth it locally. (Kaggle has far more time: see kaggle.py)
TOTAL_ATTEMPTS:int = 1
TIME_LIMIT_PER_ATTEMPT:float = 30.0 # The slowest correct level in a daily blast took ~29s, so 20s was cutting some off
# HARD cap on scoring ONE program (all its train pairs), see utils/deadline.py. Heavy loops (apply_fit's
# search) check it as they go. A crop-then-fit program needs ~4s on its biggest demos, so don't go much lower
TIME_LIMIT_PER_PROGRAM:float = 5.0

############################################################
# Parser (apply_parse.py)
############################################################

# A "majority" bg mode only picks a background if the top color clearly wins:
# if the runner-up color's pixel count is within this fraction of the top color's,
# fallback to "no bg" (color = -1)
MAJORITY_TIE_FRACTION:float = 0.10

# Repeated chunks (search_parse.py): every rows+columns config with repeated shadows gets a chunked re-parse,
# with those shadows as greedily-grabbed "chunks". A shadow only counts as a chunk if it's "complicated" enough:
# not a dot or a line, and at least this many pixels. (Chunking dots or 2x2 squares would chop every big shape into tiny pieces)
PARSE_CHUNK_MIN_AREA:int = 5
# greedy_input_shapes_moving only tries input shapes at least this big (exact pixels, colors too).
# Small shapes (dots, short lines) match all over the place by coincidence, so they're left for the flood fill.
# (3 lets small 4-pixel pieces through, with a bit of margin)
GREEDY_MOVING_MIN_AREA:int = 3
# Connection radius: also try 8-connected parses where pixels up to N apart (Chebyshev distance, so radius 2 =
# a gap of 1 pixel) count as one shape, for N = 2..PARSE_MAX_CONNECTION_RADIUS. OFF by default.
# (Each radius > 1 costs +1 parse point, see score_parse.py)
PARSE_CONNECTION_RADIUS_ENABLED:bool = False
PARSE_MAX_CONNECTION_RADIUS:int = 4
# Too many shapes: search_parse throws out any parse config with more than this many shapes on some demo grid (input or
# output), before scoring it. (E.g. a big patterned background parsed as hundreds of single dots: slow, and never the answer.)
# Measured on the fully-solved levels of a full daily blast: the most shapes any winning parse had on one demo grid
# was 83 (one outlier: the runner-up had 42), so 100 leaves some headroom (Nicky's call).
# The parser also stops early once a demo grid's flood fill finds more pieces than this (see apply_parse_one_grid's max_shapes)
PARSE_MAX_SHAPES_PER_GRID:int = 100

############################################################
# Parser Config Scoring (score_parse.py)
############################################################

# Tunable coefficients (magic numbers, all in one place)

COEFF_CONFIG_COMPLEXITY:float = 1.0 # How much to penalize "complex" parse configs
# Config-complexity cost of greedy_input_shapes_static=True (grabbing input shapes, same position, from the output before flood fill)
POINTS_GREEDY_INPUT_SHAPES_STATIC:float = 0.5
# Config-complexity cost of greedy_input_shapes_moving=True (grabbing input shapes that MOVED in the output, before flood fill)
POINTS_GREEDY_INPUT_SHAPES_MOVING:float = 0.5
# Config-complexity cost of denoise=True (ragged shapes cleaned up, noise on top). Not free: it's a real claim about the scene
POINTS_DENOISE:float = 1.0
# ...and of autocomplete_rects=True on top (completing rectangles & frames hidden under other shapes, always with denoise).
# Not free either, so de-noise + rectangles (2) has to explain the scene better than de-noise alone (1) to win
POINTS_AUTOCOMPLETE_RECTS:float = 1.0
# Config-complexity cost of pattern_background=True (a big tiled / mirrored picture as one shape at the very back).
# Not free: else on demos with no pattern at all it parses exactly like its base config, ties it, and wins coin flips
POINTS_PATTERN_BACKGROUND:float = 5.0
# ...but a pattern behind EVERY demo pair (on its input, its output, or both) is the whole puzzle's background, not a
# coincidence: that earns this reward on top, so the parse comes out cheaper than without the toggle (net 5 - 10 = -5).
# (The pattern always covers 100%+ of its grid, mirrors can hang past the edges, so "every pair" is what decides it.
# One side is enough: crop / clean-up puzzles have the pattern on the input only. A pattern in only SOME pairs: no reward)
POINTS_PATTERN_BACKGROUND_EVERY_PAIR_REWARD:float = -10.0
# Each chunk in parse_config['chunks'] costs this much PER PIXEL in its shadow (25 pixels -> 12.5 points at 0.5),
# so a chunk parse has to actually save points to win. (A full point per pixel made a real 5x5 chunk lose to a
# greedy-moving parse, and 0.1 was so cheap that whole-grid "chunks" started winning)
POINTS_PARSE_CHUNK_PER_PIXEL:float = 0.5

# Penalty for shape changes
POINTS_SHAPE_MOVE:float = 1.0 # One charge per move (NOT per axis). NOTE: TIES with a deletion (1)
# The move price when the scene changes size (e.g. a crop shifts EVERY shape). Same as a normal move: a reward
# (or free moves) just rewarded shattering shapes into row fragments, since more fragments = more reward
POINTS_SHAPE_MOVE_WHEN_RESIZED:float = 1.0
POINTS_SHAPE_RECOLOR:float = 1.0
POINTS_SHAPE_TRANSFORMED:float = 1.0
POINTS_SHAPE_DELETED:float = 1.0 # 1 for now, see what happens TODO
POINTS_SHAPE_ADDED:float = 3.0 # A de-novo shape costs this + its pixel area (pixels actually drawn)
# ...unless it's one color and looks like an outline of another shape: then it costs just POINTS_SHAPE_OUTLINE_OR_RAY (outline width + color).
# Cheaper than a new dot, since an outline or ray is "explained" by the shape it hugs or shoots out of.
POINTS_SHAPE_OUTLINE_OR_RAY:float = 1.0
# "Looks like an outline" = it covers this fraction of the other shape's PARSE_OUTLINE_WIDTH-pixel ring
# (the ring outline() would paint). It may hold more than that one ring (touching rings merge into one shape).
# Only 1-pixel outlines for now; under 100% as a buffer.
PARSE_OUTLINE_WIDTH:int = 1
PARSE_OUTLINE_COVERED_FRACTION:float = 0.9
# ...or it looks like a ray shot from a shape it touches: then it also costs just POINTS_SHAPE_OUTLINE_OR_RAY (direction + color).
# "Looks like a ray" = its shadow fully covers a ray this many pixels long, shot straight out of the touching shape's edge
# (any of the 8 directions), every one of its pixels is on that shape's rays, and nothing in the input has its shadow.
# Long enough that 2-3 pixel blobs don't count.
PARSE_RAY_LENGTH:int = 5
# ...or it's one color and exactly fills one or more whole holes of the input's shapes: then it costs just this.
# 1 point (its color), so a dot that just happens to land on some 1-pixel hole doesn't get explained for free
POINTS_SHAPE_FILLS_HOLE:float = 1.0

# Penalty for scene props changed
POINTS_SCENE_SIZE_CHANGED:float = 1.0
POINTS_SCENE_BG_CHANGED:float = 1.0

# Debug switch: when True, score_parse PRINTS a per-demo breakdown of every config it scores (config
# complexity + matched/de-novo/deleted shapes + scene changes), to see WHY a parse wins or loses. Keep it OFF:
# it's slow & loud. (test_search_parse flips it on by itself, just for the configs it wants to show)
DEBUG_PARSE_SCORE_BREAKDOWN:bool = False

############################################################
# DSL (transform/dsl/)
############################################################

# The widest ring shape.outline() will ever draw. A monkey can set line_width from a shape
# property (e.g. line_width=other.area = 4160), and outline() then pads the shape's pixel grid by that much
# on every side: an 8000x8000 grid that ate a whole attempt. Anything wider than this is clamped down to it.
DSL_MAX_OUTLINE_LINE_WIDTH:int = 5

# apply_fit searches over placement orders instead of going greedy: at every step it tries each
# (moving shape, spot) scoring within this many points of that step's best. 0 = only ties. Bigger = slower (factorial-ish!)
# (1 was enough for the staircase puzzles we've seen, with ~10x fewer search nodes than 2)
FIT_SEARCH_MARGIN:int = 1
# ...and at most this many of those (moving shape, spot) options per step: best score first, ties broken
# randomly. Keeps the search at most WIDTH^depth (e.g. 4 small pieces: 3^4 = 81 orders, instead of ~10*8*6*5)
FIT_SEARCH_MAX_WIDTH:int = 3
# ...but only for the first FIT_SEARCH_WIDE_DEPTH placements (the big, early choices), then narrower.
# On a 15-piece test: 2.4-7.1s at width 3 everywhere -> 0.2-0.8s, still right on 5/5 seeds (width 2 everywhere
# was ~2x faster again, but keeps less room for early mistakes)
FIT_SEARCH_WIDE_DEPTH:int = 5
FIT_SEARCH_NARROW_WIDTH:int = 2
# The scene edge pinches too, but only as a tie-breaker: each corner pinched by the edge (edge+edge, or
# edge+target) is worth this much, vs 1 for a corner pinched by targets alone. (At full weight, 1, it broke a staircase puzzle)
FIT_EDGE_PINCH_WEIGHT:float = 0.1
# Alignment bonus (always on): +this for every touching target whose top & bottom match the moving shape's,
# and again if its left & right do. (A stacked staircase needs > 2)
FIT_ALIGNMENT_BONUS:int = 3
# ...but only for moving shapes with at least this much area, so tiny pieces (1x1s & 1x2s) don't
# get rewarded for stacking into trivial lines against each other
FIT_ALIGNMENT_MIN_AREA:int = 5
# apply_attract shares apply_fit's search (same width caps above), with its own margin: it tries each
# (moving shape, spot) within this many color-matched overlap pixels of that step's best. Same value as fitting's, for now
ATTRACT_SEARCH_MARGIN:int = 1

############################################################
# Approximate Bayesian posterior (bayes_posterior.py)
############################################################

# Webapp only (solve(..., bayes=True)), display only: never changes attempts 1 and 2.
# P(program) ∝ exp(-total_score / BAYES_TEMPERATURE), over the top BAYES_TOP_PROGRAMS programs plus one dummy
# "out-of-model" program with the constant score BAYES_DUMMY_SCORE. T and D were grid-searched for the lowest
# multi-class Brier score on half of ARC-1 Eval, and checked on the other half (paper/experiments/bayes_tuning/)
BAYES_TOP_PROGRAMS: int = 100
BAYES_TEMPERATURE: float = 15.5
BAYES_DUMMY_SCORE: float = 18.0

############################################################
# Program Scoring (score_transform.py)
############################################################

# Coefficients for scoring program results: accuracy > magic parameters > length
SCORE_COEFF_WRONG_PIXELS = 5**2
SCORE_COEFF_TOKEN_COST = 1 # Keep at 1! A plain word costs exactly 1, so the accuracy-vs-simplicity balance (wrong pixel = 25) stays put
SCORE_COEFF_UNIQUE_TOKEN_COST = 1.5  # Regularize towards fewer unique concepts. Same weight as a unique line below. (Not too high, or it swamps the TOKEN_BIASES: at 5, "rank" would really cost -0.2+5)
# Regularize towards programs that repeat the same line with different values: every literal (number, string,
# True/False) becomes X, then count the unique lines. So "color by holes" all the way through (2 unique lines) beats
# switching to "by color" for the last block (3). Slightly above a plain word (1.5), so a new concept or a new line
# outweighs a couple of extra tokens on a line the program already has. Much higher would block programs that
# really need more distinct steps
SCORE_COEFF_UNIQUE_LINE_COST = 1.5
SCORE_COEFF_EDIT_COST = 2  # Regularize against number of edits (à la Levin or "Resource-Bounded")
# Each one() call whose selector matches 2+ shapes picks one at RANDOM, so a program can win on a lucky roll.
# Every such call adds this much to the scene's _edits_cost (then times SCORE_COEFF_EDIT_COST, like any edit)
ONE_AMBIGUOUS_EDITS_COST = 100
SCORE_COEFF_GLOBAL_WRONG = 1
SCORE_COEFF_OVERALL_IN_PAIR = 0.2
# The shape-diff score (only used to pick parents, never the winner): one unit of shape difference (an extra shape,
# a wrong color, a flip...) is worth this many tokens. A missing output shape is several units, see below.
# It has no wrong pixels in it, so it can tell "right shapes, wrong spots" apart from "deleted everything"
# (which can get lots of pixels right on a mostly-background grid)
# Big on purpose: positions are a fraction of the grid size, so a small move only fixes ~0.3 shape diff, and it has
# to beat its own extra tokens for a "right idea, half done" program to rank above "do nothing" (with a clear margin)
SCORE_COEFF_SHAPE_DIFF = 50
# In the shape diff, an output shape that no predicted shape matches (it "appears out of nowhere") costs this much,
# vs +1 for a predicted shape the output doesn't have. Pricier, so deleting a near-miss shape isn't a cheap way out
SHAPE_DIFF_UNMATCHED_OUTPUT_SHAPE_COST = 5

# Some tokens are more human-priors than others. Just bias towards 'em
# Tokens split on underscore too ("rank_area" = "rank" + "area"), so these are keyed by
# the pieces. Every WORD token costs SCORE_COEFF_TOKEN_COST * TOKEN_BIASES.get(word, 1.0).
# Biggest/smallest/most/least are more natural for people to look for than "most left", so they're cheap.
TOKEN_BIASES = {

    # Modifiers to properties: should be cheap.
    # "rank" is even a small discount: a relative position ("the top-most one") is a likelier rule than an absolute
    # one ("top <= 1"), so ranking a property makes a selector cheaper. (It never goes below 0 overall: a rank
    # always comes with the property it ranks)
    'rank': -0.2,
    'reversed': 0.2,
    'start': 0.2,
    'most': 0.2,
    'least': 0.2,
    'touching': 0.2,
    # Same cost as 'rank': "transformed_pixels_twins" = twins up to rotation/flip, a modifier on plain (exact) twins
    'transformed': 0.2,
    # rank_shape_color_commonness = color, ranked by how common it is: 'commonness' only says HOW it's ranked, so it's free
    # (and not a new concept either, see _NOT_CONCEPT_WORDS in score_transform.py)
    'commonness': 0,
    # "Greater/less than or equal" are comparisons, not concepts: nearly free, and they don't count
    # towards the unique-concept cost either (see _NOT_CONCEPT_WORDS in score_transform.py)
    'GTE': 0.1,
    'LTE': 0.1,
    'NOT': 0.1, # color_NOT="c4" = "not yellow": a comparison too, priced like GTE/LTE

    # More human-familiar ideas punish less
    'area': 0.5,
    'color': 0.5,
    # "What kind of shape is it?" (a rect, a line...) is one of the first things a person sees, so it's cheaper still.
    # (Its value, e.g. "rect", costs extra on top.)
    'type': 0.2,
    # 'section': 0.5, # Off: sections are disabled, so no program can contain a 'section' token

    # These are not invariant to augmentations (D4) -- punish them a bit
    'left': 2,
    'right': 2,
    'top': 2,
    'bottom': 2,
    'vhalf': 2, # "In the top/bottom half" is just as direction-specific as top/bottom
    'hhalf': 2, # ibid, left/right half
    'width': 1.5, # survives half of D4 transforms, so less punished
    'height': 1.5, # ibid

    # Every other word seen in a best program, all at the default 1 (listed so they're easy to tune)

    # Program scaffolding: "for shape in all(...)"
    'shape': 1,
    'all': 1,
    'one': 1,
    'other': 1,
    'for': 1,
    'in': 1,
    'scene': 1,
    'apply': 1,

    # Actions
    'paint': 1,
    'hide': 1,
    'replace': 1,
    'move': 1,
    'towards': 1,
    'place': 1,
    'fit': 1,
    'fall': 1,
    'gravity': 1,
    'attract': 1,
    'rotate': 1,
    'rot': 1,
    'flip': 1,
    'fill': 1,
    'outline': 1,
    'shadow': 1,
    'focus': 1,
    'camera': 1,
    'resize': 1,
    'change': 1,
    'background': 1,
    'steps': 1,

    # Properties & settings
    'pixels': 1,
    'twins': 1,
    'holes': 1,
    'gaps': 1,
    'border': 1,
    'padding': 1,
    'quarters': 1,
    #'section': 1,
    'symm': 1,
    'monochrome': 1,
    'line': 1,
    'center': 1,
    'count': 1,
    'mode': 1,
    'overlap': 1,
    'ignore': 1,
    'dim': 1,
    'match': 1,
    'matches': 1,
    'non': 1,
    'any': 1,
    'max': 1,
    'min': 1,

    # Coordinates & single letters
    'x': 1,
    'y': 1,
    'd': 1,
    'h': 1,
    'v': 1,

    # Color names used as words (not quoted values)
    'c1': 1,
    'c2': 1,
    'c3': 1,
    'c5': 1,
    'c6': 1,
    'c7': 1,
    'c8': 1,
    'c9': 1,

}

# VALUE tokens (numbers, quoted strings, True/False) get their own lookup, keyed by the
# UNQUOTED value so "c8" and 'c8' hit the same entry. Anything not listed is a "magic number" and
# costs VALUE_DEFAULT_BIAS.
VALUE_DEFAULT_BIAS = 5
VALUE_BIASES = {
    # Zero, one, minus one, booleans: totally free (the word they fill in, like "padding", already paid)
    '0': 0, '1': 0, '-1': 0, 'True': 0, 'False': 0,
    # "No color" (nothing touching, nothing under me...): as free as -1
    'c-1': 0,
    # Subitizing: people see 2, 3, 4 at a glance, so they're nearly free
    '2': 1, '3': 1, '4': 1, '-2': 1, '-3': 1, '-4': 1,
    # Shape types
    'rect': 1, 'line': 1, 'dot': 1, 'blob': 1,
    # Colors (ten plain entries, readability over cleverness)
    'c0': 1, 'c1': 1, 'c2': 1, 'c3': 1, 'c4': 1, 'c5': 1, 'c6': 1, 'c7': 1, 'c8': 1, 'c9': 1,
    # Directions (rays & connect_with_lines share them): one line = like a color, two lines = 2, everything = 4
    'lr': 1, 'ud': 1, 'diag_up': 1, 'diag_down': 1, '+': 2, 'x': 2, 'all': 4,
    # Directions for move / fall: one word names the whole step, so it's priced like "lr" (and "dl" can beat x=-1,y=1)
    'u': 1, 'd': 1, 'l': 1, 'r': 1, 'ul': 1, 'ur': 1, 'dl': 1, 'dr': 1, 'in': 1, 'out': 1,
}

############################################################
# Evolutionary Search (search_transform.py)
############################################################

# Configuration magic numbers

EVO_MAX_GENERATIONS = 10            # Maximum rounds of evolution
EVO_EARLY_STOP_COUNTDOWN = 2        # Stop early, when the score's been stable for this many rounds
# Each round's parents: the top K by total_score (pixels + simplicity), plus the top K by the shape-diff score,
# plus the best on each demo pair. No random draw (a random draw let near-tied general programs drop out early)
# K = 1 each (just the overall best and the best by shape diff, no runner-ups): runner-ups scored no better
# on our smoke-test levels, concept-arc and mini-arc, and cost search transform time
PARENTS_TOP_K_BY_SCORE = 1          # How many of the best programs by total_score become parents
PARENTS_TOP_K_BY_SHAPE_DIFF = 1     # How many of the best programs by the shape-diff score (see SCORE_COEFF_SHAPE_DIFF) become parents

# For the paper's ablations only (the config/compression_*.py and config/search_*.py files): which part of the solver
# to switch off, for the few that need code, not just config values. "" = the normal solver. The modes:
# "NO DEMO ISLANDS" = no best-on-each-demo-pair parents
# "RANDOM SCORES" = every program's scores are random numbers (the winner, parents & attempt 2 = random picks)
ABLATION_STUDY_MODE:str = ""

MONKEY_CAP_START = 1000             # At the start, the maximum cap of how many programs each code monkey suggests
MONKEY_CAP_DISCOUNT = 1             # Exponentially bringing down the cap per round
MONKEY_CAP_MIN = 500                # The minimum cap

CLOCK_BUDGET_PER_MONKEY = 0.3       # Wall-clock seconds to spend scoring each monkey's suggestions
CLOCK_BUDGET_PER_BATCHER_MONKEY = 0.4 # ...but a bit more for the ShapeActionBatcher monkeys, which propose a lot (more still if BATCHER_PER_DEMO_PAIR is on)

############################################################
# Monkeys & Selectors (monkey_helpers.py, selector_helpers.py)
############################################################

# Only generate code proposals for the top N most-pixels-affecting actions
BATCHER_TOP_N_PIXELS = 5
# For each of those actions (with no target shape), propose this many actor selectors (e.g. paint, hide)
BATCHER_TOP_N_ACTIONS = 10
# For actions WITH a target shape (e.g. move_towards), propose this many actor selectors AND this many
# target selectors, then try every combo (5 x 5 = 25 proposals per action). (Fewer actors and more targets,
# like 6 x 16, lets rarer targets make the cut, but loses some plain all(color=...) actors)
BATCHER_TOP_N_ACTORS_WITH_TARGET = 5
BATCHER_TOP_N_TARGETS = 5
# Also batch each demo pair on its own (selectors picked against just that demo's shapes)? Off for now:
# its extra wins on arc-1-train didn't hold up on a closer look, and it costs ~20% more search transform time
BATCHER_PER_DEMO_PAIR = False
# Why so few? Only a small slice of what the batcher proposes ever gets scored (the monkey cap and the
# clock budget cut the rest at random), so these counts were halved: same scores, less search time.
# Halving again started to lose levels.

# Tunable coefficients (magic numbers, all in one place)
SELECTOR_KEY_COST:float = 0.2
# At most this many prop_NOT=value selectors (e.g. color_NOT="c4") proposed per _propose_selectors call
SELECTOR_MAX_NOT_PROPOSALS:int = 20

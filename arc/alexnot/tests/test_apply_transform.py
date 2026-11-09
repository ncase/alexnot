import json
import sys
import numpy as np
from pathlib import Path

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.parse.apply_parse import apply_parse_one_grid
from alexnot.parse.reverse_parse import reverse_parse_one
from alexnot.transform.apply_transform import apply_transform_one
from alexnot.annotate.apply_annotate import apply_annotate_one_scene
from alexnot.utils.deadline import start_deadline, clear_deadline
from alexnot.config import TIME_LIMIT_PER_PROGRAM
import time

ROOT = Path(__file__).resolve().parent.parent.parent
LEVELS_DIR = ROOT / "levels"

from alexnot.tests.print_grid import print_grid, print_grid_emoji, print_grid_comparison

def run_program_with_deadline(program:str, scene:dict, label:str) -> dict:
    """Run a program with the same hard time cap the solver uses (TIME_LIMIT_PER_PROGRAM), so a slow program can't hang
    the test. apply_transform_one swallows the timeout (only a debug log) and returns an empty scene, so say it LOUDLY here."""
    start_time:float = time.time()
    start_deadline(TIME_LIMIT_PER_PROGRAM)
    try:
        transformed_scene:dict = apply_transform_one(program, scene)
    finally:
        clear_deadline()
    if time.time() - start_time >= TIME_LIMIT_PER_PROGRAM:
        print(f"\n⏰ {label}: TIMED OUT after {TIME_LIMIT_PER_PROGRAM}s (the grid below is the solver's empty 'nothing' scene)")
    return transformed_scene

####################################################################################
####################################################################################

if __name__ == "__main__":

    #######################################################################
    ## CHOOSE THE LEVEL & PROGRAM
    #######################################################################

    #path = LEVELS_DIR / "arc-2-train/5034a0b5.json"  # move 1 to own color
    #program = """
#for shape in all(touching_scene_edge=False):
    #shape.move_towards(touching_scene_edge=True, color=shape.color, steps=1)
    #"""

    #path = LEVELS_DIR / "arc-1-eval/d56f2372.json"  # camera spotlight the symmetrical
    #program = """
#focus(symm_any=True)
    #"""    

    #path = LEVELS_DIR / "_old/custom/scene_prop.json"  # shapes move to bottom via scene.height
    #program = """
#for shape in all():
#\tshape.place(bottom=scene.height)
    #"""

    #path = LEVELS_DIR / "arc-1-eval/16b78196.json"  # fit pieces into the band's gaps, jigsaw-style
    #program = """
#for shape in all(rank_area_GTE=1):
#\tshape.fit()
#apply_fit()
    #"""
    # In-line the big region that's touching the pink corner pixel (expected: 1 wrong pixel near the
    # bottom-left in every train pair, plus an orange ring around the center blue shape in train 2).
    # touching_color_LEAST, not _most: the region around the pink pixel touches way more blue than pink,
    # so its touching_color_most is c1 (touching_color_most="c6" matches nothing, and the program does nothing)
    # path = LEVELS_DIR / "arc-2-eval/8f3a5a89.json"  # in-line the region the pink pixel sits in
    # program = """
# for shape in all(touching_color_least="c6"):
# \tshape.inline(color="c7", line_width=1)
    # """
    # Same, but ignore_holes=True: the blue shapes are holes in that region, so their edges shouldn't get
    # traced (expected: just the 1 wrong pixel near the bottom-left, plus train 0's 4 blue shapes that need deleting)
    # path = LEVELS_DIR / "arc-2-eval/8f3a5a89.json"  # in-line the region the pink pixel sits in, skipping its holes
    # program = """
# for shape in all(touching_color_least="c6"):
# \tshape.inline(color="c7", line_width=1, ignore_holes=True)
    # """
    # Recolor the yellow square pink, then in-line it back in yellow (1 px), so only the inside stays pink.
    # (Paint first: paint() only recolors the "original" pixels, and inline marks its ring as not-original anyway)
    # path = LEVELS_DIR / "mini-arc/mini_aeugn2pfna6pvwdt.json"  # yellow square -> pink square with a yellow rim
    # program = """
# for shape in all(color="c4"):
# \tshape.paint(color="c6")
# \tshape.inline(color="c4", line_width=1)
    # """
    # The ignore_holes inline, then patch the one wrong pixel next to the pink corner (-1 in the new grid = leave
    # that pixel alone). The wrong one is DIAGONALLY up-right of pink (orange, should be cyan). The pattern is 2x3,
    # not 2x2: the test's 2x2 corner looks exactly like the trains' ([[7,7],[6,7]]) but needs no fix, and the extra
    # cyan pixel at the top-right only shows up in the train pairs. (Train 0 still has its 4 deleted blue shapes wrong, 28 px.)
    # path = LEVELS_DIR / "arc-2-eval/8f3a5a89.json"  # in-line the pink pixel's region, then fix the corner pixel
    # program = """
# for shape in all(touching_color_least="c6"):
# \tshape.inline(color="c7", line_width=1, ignore_holes=True)
# replace_pixels([[7,7,8],[6,7,7]], [[-1,8,-1],[-1,-1,-1]])
    # """
    # Odd One Out: crop to the leftmost column of boxes (padding 1 keeps the grid lines), then slide each row's
    # odd box (the one with no exact twin) into that column. Use with the multicolor edge-bg parse config below.
    # path = LEVELS_DIR / "arc-2-eval/38007db0.json"  # odd one out per row
    # program = """
# focus(rank_left=0, padding=1)
# for shape in all(pixels_twins=0):
# \tshape.place(left=1)
    # """

    # Noisy bands: delete the small blue noise dots, orange-outline the small black ones.
    # Use with the blue-bg autocomplete parse config below
    path = LEVELS_DIR / "arc-2-eval/71e489b6.json"  # delete blue noise, outline black noise in orange
    program = """
for shape in all(color="c1", area_LTE=3):
\tshape.hide()
for shape in all(color="c0", area_LTE=3):
\tshape.outline(color="c7")
    """
    # To SEE the parse instead: paint every rectangle red (completed ones too), so whatever stays black or blue
    # wasn't parsed as a rectangle
    # program = """
# for shape in all(type="rect"):
# \tshape.paint(color="c2")
    # """
    # (focus() frames ALL matches by default. Plain pixels_twins means EXACT twins, the D4 one is transformed_pixels_twins)
    # Outline every blue shape in red, without painting into the blue shapes' holes (only the red pixels
    # should come out right; use with "bg_mode": "c4" in the parse config below)
    # path = LEVELS_DIR / "arc-1-eval/aa4ec2a5.json"  # outline the blue shapes in red, skipping their holes
    # program = """
# for shape in all(color="c1"):
# \tshape.outline(color="c2", line_width=1, ignore_holes=True)
    # """

    # Crop to the biggest shape(s) (frames), then jigsaw everything else into their gaps.
    # (focus() frames ALL matches by default, and train 1 has two tied frames)
    #path = LEVELS_DIR / "arc-2-eval/e8686506.json"  # crop to the frame(s), fit pieces into the gaps
    #program = """
#focus(rank_area=0)
#for shape in all(rank_area_GTE=1):
#\tshape.fit()
#apply_fit()
    #"""

    # Parse shuffles the shapes, and apply_fit breaks ties RANDOMLY, so pick a fresh seed every run, to see
    # the alternate outcomes. Set SEED to a number to replay a run
    import random
    SEED = None
    seed:int = SEED if SEED is not None else random.randrange(1_000_000)
    print(f"Random seed: {seed}")
    random.seed(seed)

    #######################################################################
    ## CHOOSE THE PARSE PROGRAM
    #######################################################################

    # Parse config
    # 38007db0 Odd One Out: test_search_parse's #1 config (grid lines = background, each box is ONE multicolor shape)
    # parse_config = {
    #     'bg_mode': 'most_common_edge_color', 'diagonals': True, 'columns': True, 'rows': True, 'multicolor': True,
    #     'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': False,
    # }
    # 71e489b6 noisy bands: search_parse's #1 config (blue bg, one-color, rectangles autocompleted).
    # (#2/#3 are the same with 'bg_mode': 'most_common_edge_color' / 'most_common_scene_color')
    parse_config = {
        'bg_mode': 'c1', 'diagonals': True, 'columns': True, 'rows': True, 'multicolor': False,
        'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': False, 'denoise': True, 'autocomplete_rects': True,
    }
    # Another config to pick from (a string, so it's off)
    """
    parse_config = {
        "no_bg": False,
        "diagonals": True,
        "columns": True,
        "rows": True,
        "multicolor": False,
        "bg_mode": "no_bg",
        # "bg_mode": "c0", 
        # "bg_mode": "c4",
    }
    """

    #######################################################################
    ## NOW parse, annotate, apply transform on ALL train pairs.
    #######################################################################

    # Load level
    with open(path, "r") as file:
        level = json.load(file)

    total_wrong = 0
    num_pairs = len(level['train'])

    for i, pair in enumerate(level['train']):
        # Parse & annotate input
        input_grid = np.array(pair['input'], dtype=np.int8)
        parsed_scene = apply_parse_one_grid(input_grid, parse_config)
        annotated_scene = apply_annotate_one_scene(parsed_scene)

        # Apply transform & reverse-parse to get predicted grid
        transformed_scene = run_program_with_deadline(program, annotated_scene, f"Train pair {i}") # Same time cap as the solver
        predicted_grid = reverse_parse_one(transformed_scene)

        # Compare against expected output
        expected_grid = np.array(pair['output'], dtype=np.int8)
        total_wrong += print_grid_comparison(predicted_grid, expected_grid, label=f"Train pair {i}")

    # Also run on test input(s) — no ground truth, just show the prediction
    for i, pair in enumerate(level.get('test', [])):
        input_grid = np.array(pair['input'], dtype=np.int8)
        parsed_scene = apply_parse_one_grid(input_grid, parse_config)
        annotated_scene = apply_annotate_one_scene(parsed_scene)
        transformed_scene = run_program_with_deadline(program, annotated_scene, f"Test pair {i}") # Same time cap as the solver
        predicted_grid = reverse_parse_one(transformed_scene)
        # Most level JSONs DO have the test output, so compare against it when it's there
        if 'output' in pair:
            print_grid_comparison(predicted_grid, np.array(pair['output'], dtype=np.int8), label=f"Test pair {i}")
        else:
            print(f"\n--- Test pair {i} (predicted, no ground truth) ---")
            print_grid_emoji(predicted_grid)

    # Final summary
    print(f"\n{'='*40}")
    if total_wrong == 0:
        print(f"ALL {num_pairs} TRAIN PAIRS PASS!")
    else:
        print(f"TOTAL WRONG PIXELS across {num_pairs} train pair(s): {total_wrong}")
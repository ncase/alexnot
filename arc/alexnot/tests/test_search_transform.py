# Given a HAND-PICKED parse config, can search_transform find the program? (main_solver's one_attempt(), minus search_parse)
import json
import sys
import time
import random
import numpy as np
from pathlib import Path

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Same imports as main_solver (package-level functions, NOT `import alexnot.parse.search_parse as x`, which grabs the function)
from alexnot import parse
from alexnot import annotate
from alexnot import transform
from alexnot.config import TIME_LIMIT_PER_ATTEMPT
from func_timeout import func_timeout, FunctionTimedOut

from alexnot.tests.print_grid import print_grid_emoji, print_grid_comparison
# Reuse test_apply_transform's helper: runs one program with the solver's TIME_LIMIT_PER_PROGRAM cap, says so loudly if it times out
from alexnot.tests.test_apply_transform import run_program_with_deadline

ROOT = Path(__file__).resolve().parent.parent.parent
LEVELS_DIR = ROOT / "levels"

# How many of the top programs to print
TOP_PROGRAMS_TO_SHOW:int = 5

####################################################################################
####################################################################################

if __name__ == "__main__":

    #######################################################################
    ## CHOOSE THE LEVEL & PARSE CONFIG
    #######################################################################

    # Sanity check: the full solver solves this one, with test_apply_transform's CHOOSE parse
    #path = LEVELS_DIR / "arc-1-eval/16b78196.json"  # fit pieces into the band's gaps, jigsaw-style
    #parse_config = {
        #"no_bg": False,
        #"diagonals": True,
        #"columns": True,
        #"rows": True,
        #"multicolor": False,
    #}

    # Want: focus(rank_area_LTE=1) / for shape in all(rank_area_GTE=2): shape.fit() / apply_fit()
    # path = LEVELS_DIR / "arc-2-eval/5dbc8537.json"  # crop to the frame, fit the other pieces into its holes
    # parse_config = {
    #     "bg_mode": "most_common_input_color",
    #     "multicolor": True,
    #     "diagonals": True,
    #     "columns": True,
    #     "rows": True,
    #     "greedy_input_shapes_moving": True, # Grab the input pieces that moved into the output's holes
    # }

    # Want: focus(rank_left=0, padding=1) / for shape in all(pixels_twins=0): shape.place(left=1)
    # (plain twins = exact; the D4 version is transformed_pixels_twins. rank_left=0 = the left-most column)
    # (test_search_parse's #1 config: grid lines = background, each box is ONE multicolor shape)
    path = LEVELS_DIR / "arc-2-eval/38007db0.json"  # odd one out per row
    parse_config = {
        'bg_mode': 'most_common_edge_color', 'diagonals': True, 'columns': True, 'rows': True, 'multicolor': True,
        'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': False,
    }

    # Fresh seed every run (search ties + apply_fit ties are random), printed so you can replay it
    SEED = None
    seed:int = SEED if SEED is not None else random.randrange(1_000_000)
    print(f"Random seed: {seed}")
    random.seed(seed)

    #######################################################################
    ## NOW parse the train pairs & search for the transform (like main_solver's one_attempt)
    #######################################################################

    # Load level, numpy-ify the grids (same as main_solver's preamble)
    with open(path, "r") as file:
        level = json.load(file)
    raw_train_pairs:list[dict] = [
        {'input': np.array(pair['input'], dtype=np.int8), 'output': np.array(pair['output'], dtype=np.int8)}
        for pair in level['train']
    ]
    # Test pairs WITHOUT their answers, like the solver (we keep the answers aside, just to compare at the end)
    raw_test_pairs:list[dict] = [{'input': np.array(pair['input'], dtype=np.int8), 'output': None} for pair in level['test']]

    # Search under the solver's per-attempt deadline. (The per-program cap is already inside score_transform.)
    # best_so_far gets refilled every round, so a timed-out search still has its best programs, just like the solver.
    best_so_far_program_results:list[dict] = []
    def one_attempt() -> list[dict]:
        parsed_train_pairs:list[dict] = parse.apply_parse(parse_config, raw_train_pairs)
        return transform.search_transform(parsed_train_pairs, best_so_far=best_so_far_program_results)

    search_start_time:float = time.time()
    try:
        # (func_timeout is untyped, so pyright guesses its result might be None. It's one_attempt's list)
        program_results:list[dict] = func_timeout(TIME_LIMIT_PER_ATTEMPT, one_attempt)  # pyright: ignore[reportAssignmentType]
    except FunctionTimedOut:
        print(f"\n⏰ Search hit TIME_LIMIT_PER_ATTEMPT ({TIME_LIMIT_PER_ATTEMPT}s), using its best programs so far")
        program_results = list(best_so_far_program_results)
    print(f"Search took {time.time() - search_start_time:.2f}s, {len(program_results)} programs ranked")

    # Nothing ranked at all? (Timed out before even round 0 finished.) Then there's no program to test
    if not program_results:
        print("\n❌ No programs found at all!")
        sys.exit(1)

    # Show the top few programs
    for rank, program_result in enumerate(program_results[:TOP_PROGRAMS_TO_SHOW]):
        print(f"\n#{rank+1}  score={program_result['total_score']}  train wrong px={program_result['pair_wrong_pixels']}  (monkey: {program_result.get('monkey')})")
        print(program_result['program'].strip())

    #######################################################################
    ## Run the BEST program on every train AND test pair (parsed the same way)
    #######################################################################

    best_program:str = program_results[0]['program']
    total_wrong:int = 0

    # Parse + annotate fresh, then apply the program, then reverse-parse back to a grid (same steps as main_solver's step 5a)
    parsed_train_pairs:list[dict] = annotate.apply_annotate(parse.apply_parse(parse_config, raw_train_pairs))
    for i, parsed_pair in enumerate(parsed_train_pairs):
        transformed_scene:dict = run_program_with_deadline(best_program, parsed_pair['input'], f"Train pair {i}")
        predicted_grid = parse.reverse_parse_one(transformed_scene)
        total_wrong += print_grid_comparison(predicted_grid, raw_train_pairs[i]['output'], label=f"Train pair {i}")

    parsed_test_pairs:list[dict] = annotate.apply_annotate(parse.apply_parse(parse_config, raw_test_pairs))
    missing_test_answer:bool = False
    for i, parsed_pair in enumerate(parsed_test_pairs):
        transformed_scene = run_program_with_deadline(best_program, parsed_pair['input'], f"Test pair {i}")
        predicted_grid = parse.reverse_parse_one(transformed_scene)
        # Most level JSONs have the test answer. If not, just show the prediction
        if 'output' in level['test'][i]:
            total_wrong += print_grid_comparison(predicted_grid, np.array(level['test'][i]['output'], dtype=np.int8), label=f"Test pair {i}")
        else:
            missing_test_answer = True
            print(f"\n--- Test pair {i} (predicted, no ground truth) ---")
            print_grid_emoji(predicted_grid)

    # Final summary. Exit 0 only if the best program gets EVERY train & test pair right
    print(f"\n{'='*40}")
    print(f"Best program:\n{best_program.strip()}\n")
    # Judgment call: a test pair with no answer can't be checked, so it counts as NOT passing (exit 1)
    if total_wrong == 0 and not missing_test_answer:
        print(f"ALL {len(raw_train_pairs)} TRAIN + {len(raw_test_pairs)} TEST PAIRS PASS!")
        sys.exit(0)
    else:
        print(f"TOTAL WRONG PIXELS across all train + test pairs: {total_wrong}" + (" (and some test answers were missing)" if missing_test_answer else ""))
        sys.exit(1)

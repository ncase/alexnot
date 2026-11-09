import json
import sys
import numpy as np
from pathlib import Path

from typing import Dict, List, Optional, Tuple

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.parse.apply_parse import apply_parse_one_grid
from alexnot.parse.reverse_parse import reverse_parse_one
from alexnot.transform.apply_transform import apply_transform_one
from alexnot.annotate.apply_annotate import apply_annotate, apply_annotate_one_scene
from alexnot.parse.apply_parse import apply_parse
from alexnot.transform.score_transform import get_transform_program_result
from alexnot.transform.monkeys import causality, shape_change, shape_change_fancy, scene_change, camera_change, compression, synonym

ROOT = Path(__file__).resolve().parent.parent.parent
LEVELS_DIR = ROOT / "levels"

from alexnot.tests.print_grid import print_grid_comparison

####################################################################################
####################################################################################

if __name__ == "__main__":

    #######################################################################
    ## CHOOSE THE LEVEL
    #######################################################################

    #path = LEVELS_DIR / "arc-2-train/5034a0b5.json"  # move 1 to own color
    path = LEVELS_DIR / "arc-1-eval/d56f2372.json"  # camera spotlight the symmetrical
    #path = LEVELS_DIR / "arc-1-train/00d62c1b.json"  # fill holes puzzle

    #######################################################################
    ## CHOOSE THE PARSE PROGRAM
    #######################################################################

    # Parse config
    parse_config = {
        "bg_mode": "most_common_input_color", # d56f2372's winning parse
        "diagonals": True,
        "columns": True,
        "rows": True,
        "multicolor": False,
        "greedy_input_shapes_static": False,
        "greedy_input_shapes_moving": False,
        "greedy_input_shapes_overlap": False,
    }

    #######################################################################
    ## CHOOSE THE MONKEY TO TEST
    #######################################################################

    #monkey = shape_change.shapes_move
    #monkey = shape_change_fancy.shapes_fill_holes
    monkey = camera_change.focus

    #######################################################################
    ## CHOOSE THE PARENT PROGRAM
    #######################################################################

    # The program the monkey evolves from ('' = the blank root program, like round 0 of the search)
    PARENT_PROGRAM:str = ''

    #######################################################################
    ## NOW parse, annotate, apply Code Monkey
    #######################################################################

    # Load level, parse & annotate
    with open(path, "r") as file:
        level_data = json.load(file)

    # Convert all grids to numpy arrays right away (int8 fits ARC colors 0-9 and empty -1)
    for pair in level_data.get('train', []):
        pair['input'] = np.array(pair['input'], dtype=np.int8)
        pair['output'] = np.array(pair['output'], dtype=np.int8)
    for pair in level_data.get('test', []):
        pair['input'] = np.array(pair['input'], dtype=np.int8)
        if 'output' in pair:
            pair['output'] = np.array(pair['output'], dtype=np.int8)
    raw_train_pairs:List[Dict] = level_data['train']
    
    # Parse all train pairs
    parsed_train_pairs:List[Dict] = apply_parse(parse_config, raw_train_pairs)

    # Annotate all (operates on already-parsed pairs)
    train_pairs:List[Dict] = apply_annotate(parsed_train_pairs)

    # Program results: all (already tested) and parents for monkeys to evolve on
    all_program_results:Dict = {}
    parent_program_results:List[Dict] = []

    # Start with the parent program (the blank one = identity transform, by default)
    root_program_result:Optional[Dict] = get_transform_program_result(PARENT_PROGRAM, train_pairs)
    assert root_program_result is not None, "The parent program crashed on some demo"
    root_program_result['monkey'] = '(parent)'
    all_program_results[PARENT_PROGRAM] = root_program_result
    parent_program_results.append(root_program_result)

    # 👉👉👉 DEBUG THIS!!! Run ONE monkey on it 👈👈👈
    suggested_programs:set = monkey(train_pairs, parent_program_results)
    for program in suggested_programs:
        new_program_result:Optional[Dict] = get_transform_program_result(program, train_pairs)
        # Crashed on some pair? The search drops those, so skip it here too
        if new_program_result is None:
            print(f"  (crashed, dropped) {program.strip()}")
            continue
        all_program_results[program] = new_program_result

    # Report top 10 programs, readable
    all_program_results_sorted = sorted(all_program_results.values(), key=lambda item:item['total_score'])
    top_ten = all_program_results_sorted[:10]
    print(f"\n{monkey.__name__} suggested {len(suggested_programs)} program(s). Top 10:\n")
    for rank, pr in enumerate(top_ten):
        print(f"  #{rank} | score={pr['total_score']:.1f} | wrong_px={pr['total_wrong_pixels']} | tokens={pr['token_cost']:.1f}")
        # indent the program for readability
        for line in pr['program'].strip().splitlines():
            print(f"       {line}")
        print()

    # Show the best program's predictions vs ground truth
    best = top_ten[0]
    if best['total_wrong_pixels'] == 0:
        print(f"Best program solves ALL train pairs perfectly!")
    else:
        print(f"Best program's predictions vs expected:")
        for i, pair in enumerate(train_pairs):
            expected_grid = pair['output']['_raw']
            predicted_grid = best['predicted_pixels'][i]
            print_grid_comparison(predicted_grid, expected_grid, label=f"Train pair {i}")

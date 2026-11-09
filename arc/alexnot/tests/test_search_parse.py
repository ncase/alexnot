import json
import sys
import numpy as np
from pathlib import Path

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.parse.search_parse import search_parse
from alexnot.parse.apply_parse import apply_parse # To re-parse the configs we want a score breakdown of

ROOT = Path(__file__).resolve().parent.parent.parent
LEVELS_DIR = ROOT / "levels" # in ARC2 coz they're before-after format

####################################################################################
####################################################################################

if __name__ == "__main__":

    #######################################################################
    ## CHOOSE THE LEVEL
    #######################################################################

    #path = LEVELS_DIR / "arc-2-train/825aa9e9.json"  # ✅ (1st) finds orange BG 
    path = LEVELS_DIR / "arc-1-train/1f85a75f.json"  # ✅ (1st) finds black BG
    #path = LEVELS_DIR / "arc-1-train/0d3d703e.json"  # ❌ finds no BG // todo: actually, hack: if there IS no majority
    #path = LEVELS_DIR / "arc-1-eval/84db8fc4.json"   # ✅ (in 2nd) finds no diagonal & no bg
    #path = LEVELS_DIR / "arc-1-eval/12eac192.json"   # ✅ (in 4th) finds no diagonal... (bg should be 0 but may still work...)
    #path = LEVELS_DIR / "arc-1-eval/32e9702f.json"   # ✅ (1st) finds BG majority
    #path = LEVELS_DIR / "arc-1-eval/d56f2372.json"   # ✅(1st) crop to symmetrical //  bg=0 or bg-majority-input
    #path = LEVELS_DIR / "arc-1-train/0b148d64.json"  # ✅(1st) crop to odd color out //  bg=0 or bg-majority-input
    #path = LEVELS_DIR / "arc-1-eval/f5aa3634.json"   # ✅(1st) crop to 1 MULTICOLOR twin // with bg=0 or bg-majority-input
    #path = LEVELS_DIR / "arc-1-eval/aee291af.json"   # ✅ crop to odd one out with padding // //  bg=2 or bg-majority-input // oh wait c8 iis better!!!

    #path = LEVELS_DIR / "arc-1-eval/16b78196.json" # fit pieces into the band's gaps (picks c4/c5 bg?)
    #path = LEVELS_DIR / "mini-arc/mini_aeugn2pfna6pvwdt.json" # recolor pink, then inline yellow (intended parse: bg c0, monochrome)

    # TODO: prefer bg-majority-input > bg=0 etc, least restrictive wins. (what's this prior called?)

    #######################################################################
    ## CHOOSE which configs get a score breakdown (WHY did they score what they did?)
    #######################################################################

    # Print score_parse's per-demo breakdown for the top N ranked configs (0 = none)
    # BREAKDOWN_TOP_N:int = 0
    BREAKDOWN_TOP_N:int = 1 # The winner, to compare against the intended parse below

    # ...and for these hand-picked configs too (any parse config dict; [] = none)
    BREAKDOWN_CONFIGS:list[dict] = [
        # mini_aeugn2pfna6pvwdt: the intended parse (bg c0, monochrome, so each yellow square is one shape)
        # {'bg_mode': 'c0', 'diagonals': True, 'columns': True, 'rows': True, 'multicolor': False, 'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': False},
        # 1f85a75f: the intended parse (bg = most common input color, monochrome, so the biggest shape is one piece)
        {'bg_mode': 'most_common_input_color', 'diagonals': True, 'columns': True, 'rows': True, 'multicolor': False, 'greedy_input_shapes_static': False, 'greedy_input_shapes_moving': False},
    ]

    #######################################################################
    ## NOW search for best parse on that level
    #######################################################################

    # Load level
    with open(path, "r") as file:
        level = json.load(file)

    # Convert all grids to numpy arrays right away (int8 fits ARC colors 0-9 and empty -1)
    # CONVERT TO NUMPY FIRST
    for pair in level['train']:
        pair['input'] = np.array(pair['input'], dtype=np.int8)
        pair['output'] = np.array(pair['output'], dtype=np.int8)
    for pair in level['test']:
        pair['input'] = np.array(pair['input'], dtype=np.int8)
        if 'output' in pair:
            pair['output'] = None # DESTROY THIS INFO

    # All pairs
    all_pairs:list[dict] = level['train'] + level['test']
    search_parse_results = search_parse(all_pairs)

    # Print all configs ranked by score
    for rank in range(10):
        result = search_parse_results[rank]
        print(f"#{rank+1}  score={result['score']:.2f}  config={result['config']}")

    # Score breakdowns: flip the debug switch on (just here, AFTER the search, so the search itself stays
    # quiet), then re-parse & re-score each chosen config, and score_parse prints its breakdown.
    # (Flipped on the score_parse MODULE, since that's where the knob got imported to. NOTE: the re-parse reshuffles
    # shapes, so a greedy-claiming tie could land differently than in the ranking above)
    breakdown_configs:list[dict] = [result['config'] for result in search_parse_results[:BREAKDOWN_TOP_N]] + BREAKDOWN_CONFIGS
    # Where did the hand-picked configs rank in the search above? (1-based; None = not among the searched configs)
    for breakdown_config in BREAKDOWN_CONFIGS:
        rank_found:int|None = next((rank + 1 for rank, result in enumerate(search_parse_results) if result['config'] == breakdown_config), None)
        score_found = search_parse_results[rank_found - 1]['score'] if rank_found else None
        print(f"Rank #{rank_found} of {len(search_parse_results)} (score={score_found}): {breakdown_config}")
    if breakdown_configs:
        # A plain import, so the type checker knows the module's attributes.
        # (alexnot.parse doesn't re-export a same-named function, so this really is the module)
        import alexnot.parse.score_parse as score_parse_module
        score_parse_module.DEBUG_PARSE_SCORE_BREAKDOWN = True
        for breakdown_config in breakdown_configs:
            print()
            # score_parse also takes the puzzle's free color (search_parse finds it once per search)
            score_parse_module.score_parse(breakdown_config, apply_parse(breakdown_config, all_pairs), score_parse_module._find_free_color(all_pairs))
        score_parse_module.DEBUG_PARSE_SCORE_BREAKDOWN = False

    # Test parse against ALL train (input,output) and test (input) scenes
    # TODO...


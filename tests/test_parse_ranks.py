"""
Where does each level's intended parse rank? Runs the real search_parse on every level in
arc/levels/parse_scoring.txt (one process per level) and prints one line per level.

    .venv/bin/python -m tests.test_parse_ranks
    .venv/bin/python -m tests.test_parse_ranks --levels 1f85a75f Count1

Ranks are in search_parse's own de-duplicated list (configs that give the same bg & shapes on every
demo are one parse), the same list main_solver's attempts walk through in order.
✅ = intended is #1, ⚠️ = in the top 3, ❌ = worse.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# alexnot lives in arc/
_root = str(Path(__file__).resolve().parent.parent / "arc")
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.parse import search_parse
from alexnot.parse.apply_parse import resolve_bg_color, apply_parse_one_grid
from tests.BLAST_EM import read_level_list, find_level_file

LIST_NAME: str = "parse_scoring"

#####################################################################
## What a parse does: its actual bg colors + connectivity
#####################################################################

def resolved_bg_colors(raw_pairs: List[Dict], bg_mode: str) -> tuple:
    """This bg mode's actual bg color on every grid (train in/out + test in)."""
    bg_colors: List[int] = []
    for raw_pair in raw_pairs:
        bg_colors.append(resolve_bg_color(raw_pair['input'], bg_mode))
        if raw_pair['output'] is not None: # test pairs have no output
            bg_colors.append(resolve_bg_color(raw_pair['output'], bg_mode, input_grid=raw_pair['input']))
    return tuple(bg_colors)

# Connectivity shorthands, so the predicates below read like the analysis doc
def is_8_connected(config: Dict) -> bool:
    return config['diagonals'] and config['columns'] and config['rows'] and config.get('radius', 1) == 1
def is_4_connected(config: Dict) -> bool:
    return not config['diagonals'] and config['columns'] and config['rows']
def is_rows_only(config: Dict) -> bool:
    return not config['diagonals'] and not config['columns'] and config['rows']
def is_columns_only(config: Dict) -> bool:
    return not config['diagonals'] and config['columns'] and not config['rows']

# Bg shorthands: compare actual bg colors, so e.g. c0 and most_common_scene_color both count when they're black everywhere
def bg_everywhere(color: int) -> Callable[[Dict, List[Dict]], bool]:
    return lambda config, raw_pairs: all(bg == color for bg in resolved_bg_colors(raw_pairs, config['bg_mode']))
def bg_per_scene_majority(config: Dict, raw_pairs: List[Dict]) -> bool:
    return resolved_bg_colors(raw_pairs, config['bg_mode']) == resolved_bg_colors(raw_pairs, 'most_common_scene_color')

black_bg = bg_everywhere(0)
red_bg = bg_everywhere(2)
yellow_bg = bg_everywhere(4)
blue_bg = bg_everywhere(1)
no_bg = bg_everywhere(-1)

# Completed rectangles: how many big ones did an autocomplete parse complete on each demo input?
# (Big = at least this many pixels, so completed noise dots & short lines don't count)
BIG_RECTANGLE_MIN_AREA: int = 20

def completed_big_rectangles_per_demo(config: Dict, raw_pairs: List[Dict]) -> List[int]:
    """Per demo input: how many big, solid rectangles this config's parse has that aren't fully visible in the grid (= got completed)."""
    counts: List[int] = []
    for raw_pair in raw_pairs:
        if raw_pair['output'] is None: # demos only
            continue
        grid: np.ndarray = raw_pair['input']
        scene: Dict = apply_parse_one_grid(grid, config)
        count: int = 0
        for shape in scene['shapes']:
            # A solid rectangle, big enough, whose box in the raw grid isn't all its own color
            is_solid_rectangle: bool = bool(shape['_shadow'].all()) and shape['_shadow'].size >= BIG_RECTANGLE_MIN_AREA
            box_pixels: np.ndarray = grid[shape['top']:shape['bottom'], shape['left']:shape['right']]
            if is_solid_rectangle and (box_pixels != int(shape['color'][1:])).any():
                count += 1
        counts.append(count)
    return counts

def completes_rectangles(bg_check: Callable[[Dict, List[Dict]], bool], minimum_per_demo: List[int]) -> Callable[[Dict, List[Dict]], bool]:
    """A one-color autocomplete parse with this bg that completes at least this many big rectangles on each demo input."""
    def is_intended(config: Dict, raw_pairs: List[Dict]) -> bool:
        if not config.get('autocomplete_rects', False) or config['multicolor'] or not bg_check(config, raw_pairs):
            return False
        counts: List[int] = completed_big_rectangles_per_demo(config, raw_pairs)
        return all(count >= minimum for count, minimum in zip(counts, minimum_per_demo))
    return is_intended

# The levels checked with completes_rectangles, and their bg: their report line also shows what got completed per demo
COMPLETION_LEVEL_BGS: Dict[str, Callable[[Dict, List[Dict]], bool]] = {
    "de809cff": black_bg,
    "71e489b6": blue_bg,
}

#####################################################################
## The intended parse, per level
#####################################################################

INTENDED_PARSES: Dict[str, Callable[[Dict, List[Dict]], bool]] = {
    # Crop to the biggest monochrome shape
    "1f85a75f": lambda config, pairs: black_bg(config, pairs) and not config['multicolor'],
    # Outline + hole fills on a yellow bg (no de-noise: the blue shapes' holes are real, not noise)
    "aa4ec2a5": lambda config, pairs: (yellow_bg(config, pairs) and not config['multicolor']
                                       and not config.get('denoise', False)),
    # Fill the closed loops' holes: 8-connected loops, so diagonal-touching holes fill as one shape
    "00d62c1b": lambda config, pairs: (black_bg(config, pairs) and is_8_connected(config) and not config['multicolor']
                                       and not config.get('denoise', False)),
    # Rays / bars extending to a wall
    "ExtendToBoundary2": lambda config, pairs: black_bg(config, pairs) and is_8_connected(config),
    "13713586": lambda config, pairs: black_bg(config, pairs) and is_8_connected(config),
    # The good levels: today's winners
    # (4-connected is fine too: it just splits each diagonal-stepping ray into straight pieces)
    "ExtendToBoundary8": lambda config, pairs: black_bg(config, pairs) and (is_8_connected(config) or is_4_connected(config)),
    "32e9702f": lambda config, pairs: bg_per_scene_majority(config, pairs) and is_columns_only(config),
    "12eac192": lambda config, pairs: bg_per_scene_majority(config, pairs) and is_4_connected(config),
    # (A red bg on every grid works too, with any connectivity: the recolor doesn't care)
    "84db8fc4": lambda config, pairs: (no_bg(config, pairs) and is_4_connected(config)) or red_bg(config, pairs),
    # Count into a rectangle: black bg on every grid (most common input color or c0), so the output bar isn't the bg
    "Count1": lambda config, pairs: black_bg(config, pairs),
    "Count3": lambda config, pairs: black_bg(config, pairs),
    # Partly hidden rectangles, lines & frames: any parse that autocompletes them (and stacks them by z)
    **{level_id: lambda config, pairs: config.get('autocomplete_rects', False)
       for level_id in ("TopBottom3D1", "TopBottom3D2", "TopBottom3D3", "TopBottom3D4", "TopBottom3D6", "TopBottom3D7", "TopBottom3D8",
                        "CleanUp2", "CleanUp3", "CleanUp4")},
    # Big rectangles with small bg-colored noise holes: completed anyway (the holes become bg shapes on top)
    "de809cff": completes_rectangles(black_bg, [3, 2]),
    # Bands with small noise dots, on a blue bg (demo 0's two black bands get completed; the other demos' black areas aren't rectangles)
    "71e489b6": completes_rectangles(blue_bg, [2, 0, 0]),
}

#####################################################################
## One level: search_parse, then find the intended parse's rank
#####################################################################

def describe_config(config: Dict) -> str:
    """A short config string (chunks are arrays, so just count them)."""
    shown: Dict = {key: value for key, value in config.items() if key != 'chunks'}
    if config.get('chunks'):
        shown['chunks'] = len(config['chunks'])
    return json.dumps(shown)

def rank_one_level(level_id: str) -> str:
    """Run search_parse on one level and return its report line."""
    # Load it like main_solver does: numpy grids, no test answers
    level_data: Dict = json.loads(find_level_file(LIST_NAME, level_id).read_text())
    raw_pairs: List[Dict] = ([{'input': np.array(pair['input'], dtype=np.int8), 'output': np.array(pair['output'], dtype=np.int8)} for pair in level_data['train']]
                             + [{'input': np.array(pair['input'], dtype=np.int8), 'output': None} for pair in level_data['test']])
    parse_results: List[Dict] = search_parse(raw_pairs)

    # The first (best) parse that does what the intended parse does: its own config, or one of the twins it beat.
    # (Twins give the same demo parse, but can differ on the test inputs, which the bg checks below also look at)
    is_intended: Callable[[Dict, List[Dict]], bool] = INTENDED_PARSES[level_id]
    intended_index: Optional[int] = next((index for index, result in enumerate(parse_results)
                                          if any(is_intended(config, raw_pairs) for config in [result['config']] + result['duplicates'])), None)
    winner: Dict = parse_results[0]
    winner_text: str = f"winner {winner['score']:.1f} {describe_config(winner['config'])}"
    if intended_index is None:
        return f"❌ {level_id}: intended parse not found at all | {winner_text}{completion_text(level_id, parse_results, raw_pairs)}"

    # Its rank: search_parse already de-duped the list, so every parse ahead of it is genuinely different
    rank: int = intended_index + 1

    status: str = "✅" if rank == 1 else "⚠️" if rank <= 3 else "❌"
    return f"{status} {level_id}: intended #{rank} ({parse_results[intended_index]['score']:.1f}) | {winner_text}{completion_text(level_id, parse_results, raw_pairs)}"

def completion_text(level_id: str, parse_results: List[Dict], raw_pairs: List[Dict]) -> str:
    """For the completion levels: big rectangles completed per demo input, by the best one-color autocomplete parse with the right bg."""
    if level_id not in COMPLETION_LEVEL_BGS:
        return ""
    bg_check: Callable[[Dict, List[Dict]], bool] = COMPLETION_LEVEL_BGS[level_id]
    best_config: Optional[Dict] = next((result['config'] for result in parse_results if result['config'].get('autocomplete_rects', False)
                                        and not result['config']['multicolor'] and bg_check(result['config'], raw_pairs)), None)
    if best_config is None:
        return " | no autocomplete parse with that bg"
    return f" | completed per demo: {completed_big_rectangles_per_demo(best_config, raw_pairs)} ({describe_config(best_config)})"

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--levels", nargs="+", help=f"a subset of {LIST_NAME}.txt (default: all)")
    arguments = argument_parser.parse_args()
    level_ids: List[str] = arguments.levels or read_level_list(LIST_NAME)

    # One process per level, so the whole list takes about as long as the slowest level
    start_time: float = time.time()
    with Pool(len(level_ids)) as pool:
        report_lines: List[str] = pool.map(rank_one_level, level_ids)
    print("\n".join(report_lines))
    print(f"\nWall time: {time.time() - start_time:.1f}s")

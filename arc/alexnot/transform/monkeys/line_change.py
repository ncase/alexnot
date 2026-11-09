# The connect-with-lines monkey: for every pair of shapes that line up exactly (straight or at 45 degrees),
# check whether the output filled the empty gap between them with one color. If so, propose connect_with_lines,
# with selectors for the connected shapes and the line directions & line color that fit.
from itertools import combinations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np

from ..dsl.lines import _doubled_center, _gap_pixels, line_kind, LINE_DIRECTION_KINDS
from ...utils import _propose_selectors, _dict_to_argstring

# CONFIG KNOBS
from ...config import BATCHER_TOP_N_ACTIONS


@dataclass
class _ConnectedGroup:
    """The connected pairs that share one line_color argument: who they are, what kinds of lines, all one-colored pairs?"""
    shapes:List[dict] = field(default_factory=list) # Both shapes of every pair (the selectors' ingroup)
    line_kinds:Set[str] = field(default_factory=set) # Any of 'lr', 'ud', 'diag_up', 'diag_down'
    every_pair_same_color:bool = True # So same_color can stay True


def shapes_connect_with_lines(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice lines drawn in the gap between two lined-up shapes, propose connect_with_lines
    """
    candidates:set = set()
    for program_result in parent_program_results:

        # Everything seen across the demos: every visible shape (the selectors' outgroup pool), and the connected
        # pairs, grouped by their line_color argument: '' = the shapes' own color (the default), 'line_color="cN"' = a
        # fixed color. (Grouped, so a stray pair, like a gap that some OTHER line happens to cross, can't spoil the rest)
        all_shapes:List[dict] = []
        groups:Dict[str, _ConnectedGroup] = {}

        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            predicted_grid:np.ndarray = predicted_scene['_raw']
            output_grid:np.ndarray = pair['output']['_raw']

            # Lines can't resize the grid, so skip pairs where the sizes differ
            if predicted_grid.shape != output_grid.shape:
                continue

            # Hidden shapes aren't drawn, so connect_with_lines skips them too
            visible_shapes:List[dict] = [shape for shape in predicted_scene['shapes'] if not shape.get('hidden')]
            all_shapes += visible_shapes

            # A line only shows on empty (background) pixels: the ones under other shapes stay hidden, since
            # new lines go behind. No empty pixel got painted in this demo? Then there's no line to find
            background_digit:int = int(predicted_scene['background'][1:])
            empty_pixels:np.ndarray = predicted_grid == background_digit
            grid_height, grid_width = predicted_grid.shape
            if not (empty_pixels & (output_grid != background_digit)).any():
                continue

            # Every pair of shapes, once. (The center maths is the DSL's own, so the proposals match what it draws)
            centers:List[Tuple[int, int]] = [_doubled_center(shape) for shape in visible_shapes]
            for index_a, index_b in combinations(range(len(visible_shapes)), 2):
                shape_a:dict = visible_shapes[index_a]
                shape_b:dict = visible_shapes[index_b]
                gap_xs, gap_ys = _gap_pixels(shape_a, shape_b, centers[index_a], centers[index_b], LINE_DIRECTION_KINDS['all'])
                if not gap_xs:
                    continue

                # Only the gap's on-grid pixels can be checked (a shape can sit partly off the canvas, e.g. after a crop)
                gap_xs_array:np.ndarray = np.array(gap_xs)
                gap_ys_array:np.ndarray = np.array(gap_ys)
                on_grid:np.ndarray = (gap_xs_array >= 0) & (gap_xs_array < grid_width) & (gap_ys_array >= 0) & (gap_ys_array < grid_height)
                gap_xs_array, gap_ys_array = gap_xs_array[on_grid], gap_ys_array[on_grid]

                # The gap's empty pixels have to ALL be one (non-background) color in the output
                gap_is_empty:np.ndarray = empty_pixels[gap_ys_array, gap_xs_array]
                if not gap_is_empty.any():
                    continue
                gap_output_colors:np.ndarray = output_grid[gap_ys_array, gap_xs_array][gap_is_empty]
                line_color_digit:int = int(gap_output_colors[0])
                if line_color_digit == background_digit or not (gap_output_colors == line_color_digit).all():
                    continue

                # Connected! Note what kind of line, what color, and who
                # (The kind comes from the centers, not the gap: a 1-pixel gap looks the same either way)
                line_color:str = f'c{line_color_digit}'
                matches_shape_color:bool = shape_a['color'] == line_color and shape_b['color'] == line_color
                group:_ConnectedGroup = groups.setdefault('' if matches_shape_color else f'line_color="{line_color}"', _ConnectedGroup())
                group.line_kinds.add(line_kind(centers[index_a], centers[index_b]))
                group.every_pair_same_color &= shape_a['color'] == shape_b['color']
                group.shapes += [shape_a, shape_b]

        # One set of proposals per line color group
        for line_color_argument, group in groups.items():

            # Directions are always spelled out: "all", plus the narrowest setting that covers every line seen
            # (fewest kinds first, so e.g. only horizontal lines = "lr", horizontal + vertical = "+")
            narrowest_directions:str = min((directions for directions, kinds in LINE_DIRECTION_KINDS.items()
                                            if group.line_kinds <= kinds), key=lambda directions: len(LINE_DIRECTION_KINDS[directions]))
            direction_arguments:List[str] = sorted({'directions="all"', f'directions="{narrowest_directions}"'})

            # same_color=True is the default, so only spell it out when a pair had two colors
            same_color_argument:str = '' if group.every_pair_same_color else 'same_color=False'

            # Who gets connected? The usual selector proposals, ingroup = every connected shape
            selectors:List[dict] = _propose_selectors(group.shapes, all_shapes, N=BATCHER_TOP_N_ACTIONS)
            for direction_argument in direction_arguments:
                for selector in selectors:
                    arguments:List[str] = [argument for argument in (line_color_argument, direction_argument,
                                           same_color_argument, _dict_to_argstring(selector)) if argument]
                    candidates.add(program_result['program'] + '\n' + f'connect_with_lines({", ".join(arguments)})')

    # All candidates!
    return candidates

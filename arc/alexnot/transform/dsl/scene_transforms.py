import random
from itertools import combinations
from typing import List, Optional, Tuple

from .top_level_methods import DSL_CONFIG
from .shape import Shape
from ...utils import _select_shape_datas

def change_background(**kwargs):
    scene = DSL_CONFIG['scene']
    if 'color' in kwargs:
        scene['background'] = kwargs['color']

def camera_resize(**kwargs):
    scene = DSL_CONFIG['scene']
    # int() because sizes like scene.center_x are floats (width/2), and a float scene size crashes numpy later
    # (e.g. np.zeros in collision.py). Yeah, this'll probably mess up: an odd
    # width's center_x = 4.5 gets silently floored to 4. A tension: failing gracefully is good for the evo search
    # (a weird program just scores badly), but bad for development & debugging (the weirdness gets hidden).
    # Then clamp to 1..30, the only sizes an ARC grid can be. A candidate program once asked for a NEGATIVE width,
    # and the np.full in reverse_parse_one crashed (outside the per-program crash guard), killing the whole attempt.
    # TODO: watch out! This "graceful" failure could hide real bugs (a broken size computation silently becomes 1 or 30
    # and just scores badly). Also: ARC-3 grids are 64x64, so this cap would need raising if ARC-3 ever comes back.
    if 'width' in kwargs:
        scene['width'] = min(max(int(kwargs['width']), 1), 30)
    if 'height' in kwargs:
        scene['height'] = min(max(int(kwargs['height']), 1), 30)
    # Keep right/bottom/center in step with the new size (focus() and hug both land here too)
    update_scene_bounds(scene)

def update_scene_bounds(scene) -> None:
    """Re-derive a scene's right/bottom/center_x/center_y from its left/top/width/height, exactly like the parse does."""
    # Right/bottom are exclusive edges (one past the last pixel), same as shapes
    scene['right'] = scene['left'] + scene['width']
    scene['bottom'] = scene['top'] + scene['height']
    # Centers stay floats (an odd width's center lands on a half pixel)
    scene['center_x'] = scene['left'] + scene['width'] / 2
    scene['center_y'] = scene['top'] + scene['height'] / 2


# Special (non-selector) params of focus(). Every OTHER kwarg is treated as a shape selector.
_FOCUS_SPECIAL_PARAMS = ('padding', 'hide_non_matches', 'one_match', 'hug', 'margin')
def focus(**kwargs):
    """
    Crop the camera to focus on the shapes matching a selector, all in one call, e.g.
        focus(rank_area=0, padding=1)
    
    Every kwarg is a shape selector (same as all(...)), EXCEPT these special params:
      padding:int            — adds (or if negative, removes) border pixels. Default 0.
      hide_non_matches:bool  — if True, hide every shape that's NOT in the focused set. Default False.
      one_match:bool         — if True, focus on just ONE of the matches, picked randomly
                               (seeded by solve(), so reproducible, same as one()). If False (default), frame ALL matches.
      hug:bool|str           — if True, squish the focused shapes side by side (no gaps, no overlaps) and
                               crop to just that group. Also hides every non-focused shape. Default False.
                               hug="2d" lays them out in a grid of cells instead (see _hug_shapes_2d).
      margin:int             — only for hug="2d": empty pixels BETWEEN cells (not at the camera edge). Default 0.
    """

    scene = DSL_CONFIG['scene']
    padding:int = int(kwargs.get('padding', 0))
    hide_non_matches:bool = kwargs.get('hide_non_matches', False)
    # hug=True squishes the focused shapes together in a line, hug="2d" in a grid (see _hug_shapes below)
    hug = kwargs.get('hug', False)
    margin:int = int(kwargs.get('margin', 0))
    # Default False (frame ALL matches): every real use so far wanted all of them,
    # and spelling out one_match=False would cost ~7 simplicity points. one_match=True still works when written.
    one_match:bool = kwargs.get('one_match', False)

    # Everything that's not a special param is the selector
    selector:dict = {key: value for key, value in kwargs.items() if key not in _FOCUS_SPECIAL_PARAMS}

    # Find all matching shapes (skip hidden ones)
    matches = [s for s in _select_shape_datas(scene['shapes'], selector) if not s.get('hidden')]
    if not matches:
        return

    # IF one_match, pick ONE random match to focus on
    if one_match:
        focused = [random.choice(matches)]
    else:
        focused = matches

    # IF hide_non_matches: hide every shape that isn't in the focused set
    # Note: with one_match=True, this also hides the OTHER (un-chosen) matches
    # hug=True always hides the non-focused shapes too (Nicky OK'd it): after the squish they'd otherwise
    # land in random spots on top of the group
    if hide_non_matches or hug:
        focused_ids = set(id(s) for s in focused)
        for shape_data in scene['shapes']:
            if id(shape_data) not in focused_ids:
                Shape(shape_data, scene).hide()

    # hug=True: squish the focused shapes together, and the camera becomes exactly that group (+ padding)
    if hug == "2d":
        _hug_shapes_2d(focused, scene, padding, margin)
        return
    if hug:
        _hug_shapes(focused, scene, padding)
        return

    # Get the combined bounding box of the focused shapes
    min_left   = min(s['left']   for s in focused)
    min_top    = min(s['top']    for s in focused)
    max_right  = max(s['right']  for s in focused)
    max_bottom = max(s['bottom'] for s in focused)

    # The new camera origin (top-left corner), accounting for padding
    origin_x:int = min_left - padding
    origin_y:int = min_top  - padding

    # Shift ALL shapes so the spotlight region lands at (padding, padding)
    for shape_data in scene['shapes']:
        Shape(shape_data, scene)._set_position(
            left = shape_data['left'] - origin_x,
            top  = shape_data['top']  - origin_y
        )

    # Resize the camera to fit the focused bounds + padding on both sides
    new_width  = (max_right  - min_left) + 2 * padding
    new_height = (max_bottom - min_top)  + 2 * padding
    camera_resize(width=new_width, height=new_height)

# The hug=True half of focus(): line the shapes up side by side, keeping their order
def _hug_shapes(focused:list, scene:dict, padding:int) -> None:
    """Squish the focused shapes side by side (no gaps, no overlaps), then crop the camera to the group (+ padding)."""

    # Which way do we line them up? Along the LONGER side of their union bounding box (ties go to x)
    union_width:int  = max(s['right']  for s in focused) - min(s['left'] for s in focused)
    union_height:int = max(s['bottom'] for s in focused) - min(s['top']  for s in focused)
    line_up_along_x:bool = union_width >= union_height

    # Keep their left-to-right (or top-to-bottom) order
    # (Ties keep the scene's shape order, which parse shuffles, so a tie is effectively random)
    sort_key:str = 'left' if line_up_along_x else 'top'
    ordered_shapes:list = sorted(focused, key=lambda shape_data: shape_data[sort_key])

    # The group's thickness on the OTHER axis = the biggest shape's, so the smaller ones get centered in it
    if line_up_along_x:
        group_thickness:int = max(s['height'] for s in ordered_shapes)
    else:
        group_thickness:int = max(s['width'] for s in ordered_shapes)

    # Walk along the layout axis, butting each shape right up against the previous one
    next_position:int = padding
    for shape_data in ordered_shapes:
        if line_up_along_x:
            # Center on y; on a half pixel, // rounds down (toward the top)
            centered_top:int = padding + (group_thickness - shape_data['height']) // 2
            Shape(shape_data, scene)._set_position(left=next_position, top=centered_top)
            next_position += shape_data['width']
        else:
            # Center on x; on a half pixel, // rounds down (toward the left)
            centered_left:int = padding + (group_thickness - shape_data['width']) // 2
            Shape(shape_data, scene)._set_position(left=centered_left, top=next_position)
            next_position += shape_data['height']

    # The camera is exactly the squished group, + padding on both sides
    group_length:int = next_position - padding
    if line_up_along_x:
        camera_resize(width=group_length + 2 * padding, height=group_thickness + 2 * padding)
    else:
        camera_resize(width=group_thickness + 2 * padding, height=group_length + 2 * padding)

# Where could a dividing line go along one axis, without cutting through any shape?
def _dividing_line_candidates(shape_spans:List[Tuple[int, int]]) -> List[int]:
    """One line position per gap between the shapes' (start, end) spans, end exclusive. Lines in the same gap split the same way."""
    # Walk the spans in order, merging the overlapping ones: every break between merged runs is a gap
    sorted_spans:list = sorted(shape_spans)
    candidates:List[int] = []
    run_end:int = sorted_spans[0][1]
    for start, end in sorted_spans[1:]:
        if start >= run_end:
            # A gap! A line right at this shape's start doesn't cut anything
            candidates.append(start)
        run_end = max(run_end, end)
    return candidates

# Which cell index does a span fall in, given the chosen line positions? (= how many lines sit before it)
def _cell_index(span_start:int, line_positions:tuple) -> int:
    return sum(1 for line_position in line_positions if line_position <= span_start)

# Max lines per axis, so the layout search stays a small brute force
_HUG_2D_MAX_LINES:int = 4

# The hug="2d" half of focus(): lay the shapes out in a grid of cells, then squish the cells together
def _hug_shapes_2d(focused:list, scene:dict, padding:int, margin:int) -> None:
    """Split the shapes with dividing lines (one shape per cell at most), pack the cells, then crop the camera to them (+ padding)."""

    # Where can vertical lines (split x) and horizontal lines (split y) go?
    x_candidates:List[int] = _dividing_line_candidates([(s['left'], s['right'])  for s in focused])
    y_candidates:List[int] = _dividing_line_candidates([(s['top'],  s['bottom']) for s in focused])

    # Brute force every combo of 1-4 lines each way. Best = the least cells (columns x rows), then the fewest lines,
    # then random among the ties
    best_key:Optional[Tuple[int, int]] = None
    best_layouts:list = []
    for number_of_x_lines in range(1, min(_HUG_2D_MAX_LINES, len(x_candidates)) + 1):
        for number_of_y_lines in range(1, min(_HUG_2D_MAX_LINES, len(y_candidates)) + 1):
            layout_key:Tuple[int, int] = ((number_of_x_lines + 1) * (number_of_y_lines + 1), number_of_x_lines + number_of_y_lines)
            # Cheapest check first: worse than the best so far, or too few cells to fit every shape?
            if best_key is not None and layout_key > best_key:
                continue
            if layout_key[0] < len(focused):
                continue
            for x_lines in combinations(x_candidates, number_of_x_lines):
                for y_lines in combinations(y_candidates, number_of_y_lines):
                    # Each shape's (column, row): valid only if no two shapes share a cell
                    cells:list = [(_cell_index(s['left'], x_lines), _cell_index(s['top'], y_lines)) for s in focused]
                    if len(set(cells)) < len(cells):
                        continue
                    if best_key is None or layout_key < best_key:
                        best_key, best_layouts = layout_key, []
                    best_layouts.append((number_of_x_lines + 1, number_of_y_lines + 1, cells))

    # No layout splits them all (e.g. two shapes overlap on both axes): leave the scene as it is
    # (the non-focused shapes are already hidden, so the program just scores badly)
    if not best_layouts:
        return
    number_of_columns, number_of_rows, cells = random.choice(best_layouts)

    # Each column is as wide as its widest shape, each row as tall as its tallest
    # (Every column & row holds at least one shape, since the lines only sit in gaps BETWEEN shapes)
    column_widths:List[int] = [0] * number_of_columns
    row_heights:List[int]   = [0] * number_of_rows
    for shape_data, (column, row) in zip(focused, cells):
        column_widths[column] = max(column_widths[column], shape_data['width'])
        row_heights[row]      = max(row_heights[row],      shape_data['height'])

    # Where each cell starts: padding at the camera edge, then margin only BETWEEN cells
    column_starts:List[int] = [padding + sum(column_widths[:column]) + margin * column for column in range(number_of_columns)]
    row_starts:List[int]    = [padding + sum(row_heights[:row])      + margin * row    for row    in range(number_of_rows)]

    # Center each shape in its cell; on a half pixel, // rounds down (toward the top-left)
    for shape_data, (column, row) in zip(focused, cells):
        Shape(shape_data, scene)._set_position(
            left = column_starts[column] + (column_widths[column] - shape_data['width'])  // 2,
            top  = row_starts[row]       + (row_heights[row]      - shape_data['height']) // 2,
        )

    # The camera is exactly the packed cells (+ margins between them), + padding on both sides
    camera_resize(
        width  = sum(column_widths) + margin * (number_of_columns - 1) + 2 * padding,
        height = sum(row_heights)   + margin * (number_of_rows    - 1) + 2 * padding,
    )

def rotate_scene(**kwargs):
    scene = DSL_CONFIG['scene']
    quarters = int(kwargs.get('quarters', 1)) % 4
    if quarters == 0:
        return

    for _ in range(quarters):
        scene_w = scene['width']
        scene_h = scene['height']
        for shape_data in scene['shapes']:
            old_left = shape_data['left']
            old_top = shape_data['top']
            old_h = shape_data['height']
            # rotate position: 90° CW maps (x,y) → (scene_h - y - h, x)
            shape_data['left'] = scene_h - old_top - old_h
            shape_data['top'] = old_left
            # rotate the shape's own pixels
            Shape(shape_data, scene).rotate(quarters=1)
            # rotate() already updated width/height/right/bottom/center relative to new left/top
        scene['width'] = scene_h
        scene['height'] = scene_w
        update_scene_bounds(scene)


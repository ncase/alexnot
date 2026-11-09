# Connect with lines! One top-level call joins every pair of selected shapes that line up EXACTLY (straight across,
# straight up/down, or at exactly 45 degrees), by adding a brand-new 1-pixel-wide line shape in the gap between them.
# Lines are never blocked: they go right through anything in the way (drawn behind the shapes they connect).
import random
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple

from .top_level_methods import DSL_CONFIG
from .rays import make_line_shape # Same new-shape builder as rays, minus the "ray" type
from ...utils import _select_shape_datas

# The 4 kinds of line: horizontal ("lr"), vertical ("ud"), and the two diagonals, "diag_up" (the "/" one: up-right &
# down-left) and "diag_down" (up-left & down-right). Each directions setting allows some of them, named like the rays' shortcuts
LINE_DIRECTION_KINDS:Dict[str, FrozenSet[str]] = {
    'lr': frozenset({'lr'}),
    'ud': frozenset({'ud'}),
    'diag_up': frozenset({'diag_up'}),
    'diag_down': frozenset({'diag_down'}),
    '+': frozenset({'lr', 'ud'}),
    'x': frozenset({'diag_up', 'diag_down'}),
    'all': frozenset({'lr', 'ud', 'diag_up', 'diag_down'}),
}


def _doubled_center(shape_data:Dict) -> Tuple[int, int]:
    """The center of the shape's bounding box, in doubled coords (so an even-sized box's half-pixel center stays an integer)."""
    return (2 * int(shape_data['left']) + int(shape_data['_shadow'].shape[1]) - 1,
            2 * int(shape_data['top']) + int(shape_data['_shadow'].shape[0]) - 1)


def _covers_pixel(shape_data:Dict, x:int, y:int) -> bool:
    """Does this shape fill that scene pixel?"""
    local_x:int = x - int(shape_data['left'])
    local_y:int = y - int(shape_data['top'])
    shadow = shape_data['_shadow']
    return 0 <= local_y < shadow.shape[0] and 0 <= local_x < shadow.shape[1] and shadow[local_y, local_x] == 1


def line_kind(center_a:Tuple[int, int], center_b:Tuple[int, int]) -> Optional[str]:
    """Which kind of line joins these two centers: "lr", "ud", "diag_up", "diag_down", or None if they don't line up exactly."""
    delta_x:int = center_b[0] - center_a[0]
    delta_y:int = center_b[1] - center_a[1]
    # Exactly lined up, or no line at all (no +/- 1s): straight = same center row/column, diagonal = exactly 45 degrees
    if delta_y == 0 and delta_x != 0:
        return 'lr'
    if delta_x == 0 and delta_y != 0:
        return 'ud'
    if delta_x != 0 and abs(delta_x) == abs(delta_y):
        # y goes DOWN, so going right while going up (opposite signs) is the "/" diagonal
        return 'diag_up' if (delta_x > 0) != (delta_y > 0) else 'diag_down'
    return None


def _gap_pixels(shape_a:Dict, shape_b:Dict, center_a:Tuple[int, int], center_b:Tuple[int, int],
                allowed_kinds:FrozenSet[str]) -> Tuple[List[int], List[int]]:
    """The pixels of the center-to-center line that sit in the gap between the two shapes (xs, ys). Empty = not connectable."""
    # Not lined up, or not a kind of line we're allowed to draw? No line
    if line_kind(center_a, center_b) not in allowed_kinds:
        return [], []
    center_a_x, center_a_y = center_a
    center_b_x, center_b_y = center_b
    delta_x:int = center_b_x - center_a_x
    delta_y:int = center_b_y - center_a_y
    # One step at a time from A's center to B's, in doubled coords. Only the spots where BOTH coords are even are real
    # pixels. (A line through two 2-wide shapes' centers runs between pixel columns, so it has none, and draws nothing)
    step_x:int = (delta_x > 0) - (delta_x < 0)
    step_y:int = (delta_y > 0) - (delta_y < 0)
    line_pixels:List[Tuple[int, int]] = []
    for step in range(max(abs(delta_x), abs(delta_y)) + 1):
        doubled_x:int = center_a_x + step * step_x
        doubled_y:int = center_a_y + step * step_y
        if doubled_x % 2 == 0 and doubled_y % 2 == 0:
            line_pixels.append((doubled_x // 2, doubled_y // 2))
    # Both shapes need a pixel ON the line. The gap = everything after A's last line pixel, up to B's first one
    # (so a hollow shape's inside never gets drawn in, and shapes that touch or overlap get no line)
    pixels_in_a:List[int] = [index for index, (x, y) in enumerate(line_pixels) if _covers_pixel(shape_a, x, y)]
    pixels_in_b:List[int] = [index for index, (x, y) in enumerate(line_pixels) if _covers_pixel(shape_b, x, y)]
    if not pixels_in_a or not pixels_in_b or pixels_in_a[-1] >= pixels_in_b[0]:
        return [], []
    gap:List[Tuple[int, int]] = line_pixels[pixels_in_a[-1] + 1:pixels_in_b[0]]
    return [x for x, _ in gap], [y for _, y in gap]


def connect_with_lines(line_color:Optional[str] = None, directions:Optional[str] = None, same_color:bool = True, **selectors) -> None:
    """
    connect_with_lines( directions:str, line_color:str=<the shapes' color>, same_color:bool=True, **shape selectors )
    Draw a line between every pair of selected shapes that line up exactly. A top-level call: it does the whole scene at once
    directions (required): "all", "+" (straight), "x" (diagonal), "lr", "ud", "diag_up" (the "/" diagonal), "diag_down" (the other one)
    """
    scene = DSL_CONFIG['scene']
    # No directions (it's required), or an unknown one? Connect nothing (so a typo'd program just scores badly)
    if directions not in LINE_DIRECTION_KINDS:
        return
    allowed_kinds:FrozenSet[str] = LINE_DIRECTION_KINDS[directions]
    # Who may be connected? (Picked once, BEFORE adding any lines, so the new lines never get connected themselves.
    # Hidden shapes aren't drawn, so they don't count either)
    selected_shapes:List[Dict] = [shape for shape in _select_shape_datas(scene['shapes'], selectors) if not shape.get('hidden')]
    # Each shape's center, worked out once (not once per pair)
    centers:List[Tuple[int, int]] = [_doubled_center(shape) for shape in selected_shapes]
    # Every pair exactly once (so no pair gets two lines)
    for index_a, index_b in combinations(range(len(selected_shapes)), 2):
        shape_a:Dict = selected_shapes[index_a]
        shape_b:Dict = selected_shapes[index_b]
        # same_color=True: only join a 3 to a 3, a 4 to a 4, etc
        if same_color and shape_a['color'] != shape_b['color']:
            continue
        line_xs, line_ys = _gap_pixels(shape_a, shape_b, centers[index_a], centers[index_b], allowed_kinds)
        if not line_xs:
            continue
        # No line_color? Use the shapes' color (two different colors: pick one randomly).
        # A multicolor shape has no one color to give, so it draws nothing
        this_line_color:str
        if line_color is not None:
            this_line_color = line_color
        elif not (shape_a['monochrome'] and shape_b['monochrome']):
            continue
        else:
            this_line_color = random.choice([shape_a['color'], shape_b['color']])
        # A missing color does nothing, just like paint()
        if this_line_color == 'MISSING' or this_line_color == 'c-1':
            continue
        # The line is a NEW shape (no existing shape's pixels get edited). (Never None: the gap has at least one pixel)
        line_shape:Dict = make_line_shape(scene, line_xs, line_ys, int(this_line_color[1:])) # [1:] to convert c0 => 0, etc
        # It goes BEHIND the two shapes it connects: one z below the lower of them. (So it also goes behind any other
        # shape at their depth that it crosses. If a level wants it on top, move(z=...) can lift it)
        line_shape['z'] = min(shape_a.get('z', 0), shape_b.get('z', 0)) - 1
        scene['shapes'].append(line_shape)

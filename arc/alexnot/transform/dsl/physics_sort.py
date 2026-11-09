# Sorting! Same flag-now-act-later pattern as fall() & apply_gravity: shape.sort() marks a shape,
# then apply_sort() lines all the marked shapes up in a stack, ordered by one of their properties.
from typing import Any, Dict, List, Optional
from .top_level_methods import all
from .shape import Shape


def apply_sort(key:Optional[str] = None, direction:Optional[str] = None, descending:bool = False, spacing:int = 0) -> None:
    """Stack every shape.sort()-marked shape along `direction` ("u"/"d"/"l"/"r"), ordered by its `key` property."""

    # Who's been marked? (Consume the marks, like apply_gravity does)
    sorting_shapes:List[Shape] = []
    for shape in all():
        if shape.data.pop('_sort', False):
            sorting_shapes.append(shape)

    # No key, no direction, or nobody to sort? Do nothing (both key and direction are required)
    if key is None or direction not in ('u', 'd', 'l', 'r') or not sorting_shapes:
        return

    # Ties keep their ORIGINAL order along the stacking direction, not the parse's (shuffled) shape-list
    # order, so the result doesn't depend on the shuffle. "Original order" = whoever was first in the old stack
    # comes first again: for "u" the bottom-most shape, "d" the top-most, "r" the left-most, "l" the right-most.
    position_in_old_stack = {
        'u': lambda shape: -shape.data['bottom'],
        'd': lambda shape: shape.data['top'],
        'r': lambda shape: shape.data['left'],
        'l': lambda shape: -shape.data['right'],
    }[direction]
    sorting_shapes.sort(key=position_in_old_stack)

    # Now the real sort. Python's sort is stable (even with reverse=True), so the ties stay as above.
    # A missing property counts as 0 (Shape returns None for those), so the program doesn't crash on it.
    def sort_value(shape:Shape) -> Any:
        value:Any = getattr(shape, key)
        return 0 if value is None else value
    sorting_shapes.sort(key=sort_value, reverse=descending)

    # The stack starts at the old bounding box's edge on the "from" side: "u" keeps the same bottom,
    # "d" the same top, "r" the same left, "l" the same right. Then each shape goes in, one after the other.
    # place() only moves along the sort axis, so every shape keeps its position on the other axis.
    # (place() also counts ONE edit + move per shape that actually moved, just like a fall)
    if direction == 'u':
        next_bottom:int = max(shape.data['bottom'] for shape in sorting_shapes)
        for shape in sorting_shapes:
            shape.place(bottom=next_bottom)
            next_bottom -= shape.data['height'] + spacing
    elif direction == 'd':
        next_top:int = min(shape.data['top'] for shape in sorting_shapes)
        for shape in sorting_shapes:
            shape.place(top=next_top)
            next_top += shape.data['height'] + spacing
    elif direction == 'r':
        next_left:int = min(shape.data['left'] for shape in sorting_shapes)
        for shape in sorting_shapes:
            shape.place(left=next_left)
            next_left += shape.data['width'] + spacing
    else: # 'l'
        next_right:int = max(shape.data['right'] for shape in sorting_shapes)
        for shape in sorting_shapes:
            shape.place(right=next_right)
            next_right -= shape.data['width'] + spacing

# The "Order" monkey: when 3+ shapes move and their new order along an axis lines up perfectly with one of
# their number props, propose sorting them by it: shape.sort() + apply_sort(key, direction, ...).
from collections import Counter
from typing import Any, Optional

from .monkey_helpers import ShapeActionBatcher
from .shape_change import _get_moved_shape_pairs
from ...utils.selector_helpers import _is_number

# Per axis: which prop is "where along the axis" (start & end), and which apply_sort direction keeps the
# start edge in place ("d"/"r": same top/left) vs the end edge ("u"/"l": same bottom/right)
_AXES:dict[str, dict[str, str]] = {
    'y': {'start': 'top', 'end': 'bottom', 'keeps_start': 'd', 'keeps_end': 'u'},
    'x': {'start': 'left', 'end': 'right', 'keeps_start': 'r', 'keeps_end': 'l'},
}


def shapes_sort(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """Notice 3+ shapes moving into an order that matches one of their props, propose sort() + apply_sort()."""

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # For each demo: the moved shapes, as one (before, after) pair per OUTPUT shape.
        # A pure recolor (same spot) isn't a move, so skip those. If a before matches several identical afters, each
        # after keeps its first before (identical shapes have the same key anyway, for the shape-y props at least)
        moved_pairs_per_demo:list[list[tuple[dict, dict]]] = []
        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            before_by_after:dict[int, tuple[dict, dict]] = {}
            for before, after in _get_moved_shape_pairs(predicted_scene, pair['output']):
                if before['left'] == after['left'] and before['top'] == after['top']:
                    continue
                before_by_after.setdefault(id(after), (before, after))
            moved_pairs_per_demo.append(list(before_by_after.values()))

        # Early exit: no demo with 3+ moved shapes? Then it's not a sort (this keeps it cheap on other levels)
        if max(len({id(before) for before, after in moved_pairs}) for moved_pairs in moved_pairs_per_demo) < 3:
            continue

        # Every apply_sort(...) line that fits all demos, over both axes
        apply_sort_lines:set[str] = set()
        for axis in _AXES.values():
            apply_sort_lines |= _propose_apply_sorts(moved_pairs_per_demo, axis)
        if not apply_sort_lines:
            continue

        # The ingroup = every moved shape (all demos). The batcher's selectors may generalize it, e.g. to
        # all(), which also brings along the unmoved shapes in the stack (they matter for where everybody lands)
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )
        for moved_pairs in moved_pairs_per_demo:
            for before, after in moved_pairs:
                shape_actions.add(before, 'sort()')

        # Flush: every selector x every apply_sort line
        for code in shape_actions.flush():
            for apply_sort_line in apply_sort_lines:
                candidates.add( program_result['program'] + '\n' + code.strip() + '\n' + apply_sort_line )

    # All candidates!
    return candidates


def _propose_apply_sorts(moved_pairs_per_demo:list[list[tuple[dict, dict]]], axis:dict[str, str]) -> set[str]:
    """All apply_sort(...) lines along one axis that fit every demo: key x direction x spacing."""
    start_prop:str = axis['start']
    end_prop:str = axis['end']

    # Only demos where 2+ shapes moved can tell us anything about order
    informative_demos:list[list[tuple[dict, dict]]] = [moved_pairs for moved_pairs in moved_pairs_per_demo if len(moved_pairs) >= 2]

    # Sort each demo's moved shapes by where they ENDED UP along the axis. If two end up at the same spot on
    # this axis (side by side), they're not stacked along it, so this axis is out
    ordered_demos:list[list[tuple[dict, dict]]] = []
    for moved_pairs in informative_demos:
        ordered_pairs:list[tuple[dict, dict]] = sorted(moved_pairs, key=lambda pair: pair[1][start_prop])
        after_starts:list[int] = [after[start_prop] for before, after in ordered_pairs]
        if len(set(after_starts)) < len(after_starts):
            return set()
        ordered_demos.append(ordered_pairs)

    # Direction: which edge of the moved shapes' bounding box stayed put in EVERY demo? "Suggest all ties",
    # so if both edges stayed put, propose both. (Neither? Then apply_sort can't anchor the stack, skip the axis)
    keeps_start:bool = all(min(before[start_prop] for before, after in pairs) == min(after[start_prop] for before, after in pairs) for pairs in ordered_demos)
    keeps_end:bool = all(max(before[end_prop] for before, after in pairs) == max(after[end_prop] for before, after in pairs) for pairs in ordered_demos)
    if not (keeps_start or keeps_end):
        return set()

    # Spacing: the gaps between neighbors (in their new order), all demos pooled. The most common gap wins,
    # and all gaps tied for most common get proposed. (Negative = overlapping along the axis, not a gap: skipped)
    gaps:list[int] = []
    for ordered_pairs in ordered_demos:
        for (previous_before, previous_after), (next_before, next_after) in zip(ordered_pairs, ordered_pairs[1:]):
            gap:int = next_after[start_prop] - previous_after[end_prop]
            if gap >= 0:
                gaps.append(gap)
    if not gaps:
        return set()
    gap_counts:Counter = Counter(gaps)
    most_common_count:int = max(gap_counts.values())
    spacings:list[int] = [gap for gap, count in gap_counts.items() if count == most_common_count]

    # Key: any number prop whose values, in the new order, only go up in every demo (or only go down in every
    # demo). Ties along the way are OK, but all-the-same isn't: then it doesn't explain the order.
    # Positions count too (their BEFORE values), so a re-sort like "left-to-right becomes top-to-bottom" can be found
    apply_sort_lines:set[str] = set()
    reference_before:dict = ordered_demos[0][0][0]
    for key in reference_before:
        if key.startswith('_') or not _is_number(reference_before[key]):
            continue
        key_goes_up:Optional[bool] = _monotonic_direction(ordered_demos, key)
        if key_goes_up is None:
            continue

        # "d"/"r" stack from the start edge, so the first shape (smallest unless descending) goes at the start.
        # "u"/"l" stack from the end edge, so it's flipped: key going up along the axis = biggest first = descending
        directions:list[tuple[str, bool]] = []
        if keeps_start:
            directions.append((axis['keeps_start'], not key_goes_up))
        if keeps_end:
            directions.append((axis['keeps_end'], key_goes_up))
        for direction, descending in directions:
            for spacing in spacings:
                # Only spell out the non-default args (descending=False & spacing=0 are apply_sort's defaults)
                arguments:str = f'key="{key}", direction="{direction}"'
                if descending:
                    arguments += ', descending=True'
                if spacing != 0:
                    arguments += f', spacing={spacing}'
                apply_sort_lines.add(f'apply_sort({arguments})')

    return apply_sort_lines


def _monotonic_direction(ordered_demos:list[list[tuple[dict, dict]]], key:str) -> Optional[bool]:
    """True if the key only goes up (in new order) in every demo, False if only down, None if neither."""
    goes_up:Optional[bool] = None
    for ordered_pairs in ordered_demos:
        values:list[Any] = [before.get(key) for before, after in ordered_pairs]
        if not all(_is_number(value) for value in values):
            return None

        # Which way does this demo go? (All-equal explains nothing, and mixed up-and-down isn't monotonic)
        steps:list[Any] = [next_value - previous_value for previous_value, next_value in zip(values, values[1:])]
        if all(step >= 0 for step in steps) and any(step > 0 for step in steps):
            this_demo_goes_up:bool = True
        elif all(step <= 0 for step in steps) and any(step < 0 for step in steps):
            this_demo_goes_up = False
        else:
            return None

        # ...and it has to be the same way in every demo
        if goes_up is not None and goes_up != this_demo_goes_up:
            return None
        goes_up = this_demo_goes_up
    return goes_up

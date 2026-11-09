# Gravity: apply_gravity, and everything only it uses
from typing import Dict, List, Tuple
from .top_level_methods import DSL_CONFIG, all
from .shape import Shape

from ...grid import _shapes_hit_test, _shapes_in_line_of_sight
from ...utils import _select_shape_datas

from .physics_utils import _pixel_gap, _direction_from_bounds, _direction_away


def apply_gravity():
    """
    Gravity! Everything that had _fall, _fall_towards or _fall_away data, it's time to actually move it.

    Example 1:

    for shape in all(color=red):
        shape.fall(x=1) # data stored; shape['_fall'] = {x:1}
    apply_gravity() # NOW all red shapes fall right

    Example 2:

    for shape in all(color=cyan):
        shape.fall_towards(color=gray) # data stored; shape['_fall_towards'] = {color:"gray"}
    apply_gravity() # NOW all cyan shapes fall towards the CHEBYSHEV-CLOSEST gray,
                    # _if it's in horizontal/vertical/diagonal line of sight_.
                    # If it's not, then do NOT try to fall towards it!

    Example 3:

    for shape in all(color=cyan):
        shape.fall_away(color=gray) # data stored; shape['_fall_away'] = {color:"gray"}
    apply_gravity() # NOW all cyan shapes fall AWAY from the closest gray (smallest pixel gap)
                    # (away from the side they touch it on, if they touch it straight on)
    """

    scene:Dict = DSL_CONFIG['scene']
    max_steps:int = max(scene['width'], scene['height']) # can't fall farther than the scene

    # Collect all shapes that have been tagged with _fall, _fall_towards or _fall_away,
    # and figure out their (dx, dy) gravity direction.
    falling:List[Tuple[Shape, int, int]] = [] # (shape, dx, dy)
    # Fix 2: who's falling which way, via plain fall() only? (Keyed by id() of the shape's data dict)
    plain_fall_direction_of:Dict[int, Tuple[int, int]] = {}
    all_shapes = all()
    for shape in all_shapes:
        d:Dict = shape.data

        if '_fall' in d:
            fall_data:Dict = d.pop('_fall') # consume it
            dx:int = max(-1, min(int(fall_data.get('x', 0)), 1))
            dy:int = max(-1, min(int(fall_data.get('y', 0)), 1))
            if dx != 0 or dy != 0:
                falling.append((shape, dx, dy))
                plain_fall_direction_of[id(d)] = (dx, dy) # track it

        elif '_fall_towards' in d:
            selector:Dict = d.pop('_fall_towards') # consume it
            # Find all matching targets
            targets:List[Dict] = _select_shape_datas(scene['shapes'], selector)
            if not targets:
                continue
            # Compute bounds-based direction to each target, grouped by direction.
            # Then we call _shapes_in_line_of_sight once per unique direction (not once per target),
            # and check which targets appear in the result. Much fewer sweeps.
            targets_by_direction:Dict[Tuple[int,int], List[Dict]] = {}
            for target in targets:
                tdx, tdy = _direction_from_bounds(d, target)
                if tdx == 0 and tdy == 0:
                    continue # we're already on top of it
                direction_key:Tuple[int,int] = (tdx, tdy)
                if direction_key not in targets_by_direction:
                    targets_by_direction[direction_key] = []
                targets_by_direction[direction_key].append(target)
            # For each unique direction, do ONE line-of-sight sweep
            candidates:List[Tuple[Dict, int, int]] = []
            for (tdx, tdy), direction_targets in targets_by_direction.items():
                los_shapes:List[Dict] = _shapes_in_line_of_sight(scene, d, tdx, tdy)
                # Check which of our targets appear in line of sight
                los_set = set(id(s) for s in los_shapes) # Use id() for fast membership
                for target in direction_targets:
                    if id(target) in los_set:
                        candidates.append((target, tdx, tdy))
            if not candidates:
                continue
            # Pick the closest one (smallest pixel gap)
            closest:Tuple[Dict, int, int] = min(candidates, key=lambda c: _pixel_gap(d, c[0]))
            falling.append((shape, closest[1], closest[2]))

        elif '_fall_away' in d:
            selector:Dict = d.pop('_fall_away') # consume it
            # Every matching target but ourselves (no line-of-sight check: we're leaving, not arriving)
            targets:List[Dict] = [target for target in _select_shape_datas(scene['shapes'], selector) if target is not d]
            if not targets:
                continue
            # Fall away from the sides we touch targets on (else away from the closest one), like move_away
            dx, dy = _direction_away(d, targets)
            if dx != 0 or dy != 0:
                falling.append((shape, dx, dy))

    # None are falling? Quit!
    if not falling:
        return

    # Snapshot each falling shape's _edits_cost BEFORE the gravity loop, so we can
    # collapse all per-step move() increments into a SINGLE EDIT afterward.
    # (**A fall is ONE action, no matter how many pixels it travels.**)
    edits_before_gravity:List[int] = [shape.data.get('_edits_cost', 0) for shape, _, _ in falling]
    positions_before_gravity:List[Tuple[int, int]] = [(shape.data['left'], shape.data['top']) for shape, _, _ in falling]
    # Same for 'edited': each step's move() marks it, even a step that bumps into something and gets undone
    edited_before_gravity:List[bool] = [shape.data.get('edited', False) for shape, _, _ in falling]

    # Whatever a shape is ALREADY overlapping when the fall starts, it pretends isn't there
    # (so it falls right through it). Snapshot those now, before anybody moves.
    # (Keyed by id() of the shape's data dict, since dicts aren't hashable)
    ignored_at_start:List[set] = [set(id(other) for other in _shapes_hit_test(shape.data, scene['shapes'])) for shape, _, _ in falling]

    # Keep looping until nobody moves (shapes can stack on each other, so order matters)
    for _ in range(len(falling)):
        anybody_moved:bool = False

        for shape_index, (shape, dx, dy) in enumerate(falling): # getting index, to look up its ignored_at_start
            # Step one pixel at a time in the gravity direction
            for _ in range(max_steps):
                old_left:int = shape.data['left']
                old_top:int = shape.data['top']

                # Try moving! (calls method so it auto-updates left, top, etc)
                shape.move(x=dx, y=dy)

                # move() might be a no-op (e.g. static constraint)
                if shape.data['left'] == old_left and shape.data['top'] == old_top:
                    break

                # Who are we hitting that actually blocks us? Skip shapes we overlapped at the start (Fix 1).
                # And if we AND they both fall() in different directions, we pass through each other (Fix 2).
                # Everyone else blocks us: still-standing shapes, same-way fallers, and fall_towards shapes.
                # (fall_towards is left out of Fix 2 on purpose: two shapes falling towards EACH OTHER are falling
                # opposite ways, and they should meet in the middle, not swap places.)
                my_plain_direction:Tuple[int, int] | None = plain_fall_direction_of.get(id(shape.data))
                blockers:List[Dict] = []
                for other in _shapes_hit_test(shape.data, scene['shapes']):
                    if id(other) in ignored_at_start[shape_index]:
                        continue # was already overlapping at the start, pretend it's not there
                    their_plain_direction:Tuple[int, int] | None = plain_fall_direction_of.get(id(other))
                    if my_plain_direction is not None and their_plain_direction is not None and their_plain_direction != my_plain_direction:
                        continue # both fall(), different ways: pass through
                    blockers.append(other)

                # Did we hit scene border or real blockers? Undo and stop.
                if shape._am_i_hitting_scene_boundaries() or len(blockers) > 0:
                    shape.move(x=-dx, y=-dy)
                    break

                # Otherwise yes, this counts as a move
                anybody_moved = True

        if not anybody_moved:
            break # everything settled

    # Collapse all the per-step move() edits into ONE edit per falling shape.
    # If the shape actually moved, count it as saved + 1. If it didn't move, restore saved.
    for i, (shape, _, _) in enumerate(falling):
        original_left, original_top = positions_before_gravity[i]
        actually_moved:bool = (shape.data['left'] != original_left or shape.data['top'] != original_top)
        shape.data['_edits_cost'] = edits_before_gravity[i] + (1 if actually_moved else 0)
        # Edited if it was already, or if the fall actually moved it
        shape.data['edited'] = edited_before_gravity[i] or actually_moved


import random
import numpy as np
from ...grid import _flip_grid, _rotate_grid, _shapes_in_line_of_sight, _shapes_hitbox_test # (Hitbox test is for shapes_fit)

from .monkey_helpers import ShapeActionBatcher
from ..dsl.physics_utils import _closest_targets # move_away/fall_away proposals: the same "closest" the DSL uses
from .monkey_helpers import _original_footprint
from ..dsl.shape import _COMPASS_DIRECTIONS, _direction_to_unit_step, _sign

# Every direction name move() understands: the 8 compass ones, plus towards / away from the scene's center
_NAMED_DIRECTIONS:list[str] = list(_COMPASS_DIRECTIONS) + ['in', 'out']


###################################################
### BASIC SHAPE CHANGES:
### Paint, Hide, Move, Gravity, Transform
###################################################

def shapes_repaint(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that get recolored, propose recoloring them
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )
        # A 2nd batcher for paint(color=shape.<prop>), so shapes from ALL demos that take "their own prop's color"
        # land in ONE batch, even when the actual colors differ (demo 1 goes red, demo 2 goes blue). Separate, so these
        # don't push the plain hard-coded paints out of the batcher's top N
        shape_actions_from_own_prop = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, compare Predicted vs Final shapes
        for pair_index, pair in enumerate(train_pairs):
            predicted_shapes:list[dict] = program_result['predicted_scenes'][pair_index]['shapes']
            final_shapes:list[dict] = pair['output']['shapes']

            # A recolored shape can MERGE with a same-colored neighbor in the output (e.g. a segment painted
            # the color of the dot it sits on), so no output shape has its exact shadow. So also check pixel-wise:
            # which output pixels are covered by some output shape? (Not-covered = background, whatever its color.)
            output_grid:np.ndarray = pair['output']['_raw']
            output_covered:np.ndarray = np.zeros(output_grid.shape, dtype=bool)
            for after in final_shapes:
                after_height, after_width = after['_shadow'].shape
                # Clip to the grid first (same as annotate_touching_shapes): a chunk can be parsed hanging partly
                # OFF the canvas, and a negative slice start would wrap around & crash the monkey
                clip_top:int = max(after['top'], 0)
                clip_left:int = max(after['left'], 0)
                clip_bottom:int = min(after['top'] + after_height, output_grid.shape[0])
                clip_right:int = min(after['left'] + after_width, output_grid.shape[1])
                if clip_bottom <= clip_top or clip_right <= clip_left:
                    continue # entirely off the canvas: covers nothing
                clipped_shadow:np.ndarray = after['_shadow'][clip_top-after['top'] : clip_bottom-after['top'], clip_left-after['left'] : clip_right-after['left']]
                output_covered[clip_top:clip_bottom, clip_left:clip_right] |= (clipped_shadow != 0) # (Shadows are int8 masks)

            for before in predicted_shapes:
                # Compare the shape's ORIGINAL pixels (what paint() recolors), not its whole current shadow,
                # so a shape that's already been outlined can still be proposed a paint
                original_left, original_top, original_shadow = _original_footprint(before)
                was_proposed:bool = False # So the pixel-wise check below doesn't propose the same paint twice
                for after in final_shapes:

                    # Only if SAME SHADOW & DIFFERENT COLOR *AND IN THE SAME SPOT*
                    # (like shapes_hide checks). Otherwise every 1px dot "matches"
                    # every other-colored 1px dot anywhere in the output, & floods the batcher with bogus paints.
                    if (np.array_equal(original_shadow, after['_shadow']) and before['color']!=after['color']
                            and original_left==after['left'] and original_top==after['top']):
                        # Suggest re-paint
                        shape_actions.add(before, f'paint(color="{after["color"]}")', pair_index=pair_index)
                        _add_paints_from_own_props(shape_actions_from_own_prop, before, after["color"], pair_index)
                        was_proposed = True

                # No exact match? Then look at the output pixels under the shape's original footprint: if they're
                # ALL covered by output shapes, and ALL one (new) color, it got repainted (and maybe merged with a neighbor)
                if was_proposed:
                    continue
                footprint_height, footprint_width = original_shadow.shape
                footprint_ys, footprint_xs = np.nonzero(original_shadow)
                footprint_ys = footprint_ys + original_top
                footprint_xs = footprint_xs + original_left

                # Skip if the footprint falls off the output grid, e.g. the shape moved off it, or the output's a different size
                if footprint_ys.size == 0 or original_top < 0 or original_left < 0 or original_top + footprint_height > output_grid.shape[0] or original_left + footprint_width > output_grid.shape[1]:
                    continue
                if not output_covered[footprint_ys, footprint_xs].all():
                    continue
                output_colors_under_footprint:np.ndarray = np.unique(output_grid[footprint_ys, footprint_xs])
                new_color:str = f'c{int(output_colors_under_footprint[0])}'
                if output_colors_under_footprint.size == 1 and new_color != before['color']:
                    shape_actions.add(before, f'paint(color="{new_color}")', pair_index=pair_index)
                    _add_paints_from_own_props(shape_actions_from_own_prop, before, new_color, pair_index)

        # Flush, try append all suggestions!
        code_blocks:list[str] = shape_actions.flush() + shape_actions_from_own_prop.flush() # Both batchers
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates


# For each of the shape's own props that holds the new color (touching_color_most, touching_bottom_color...),
# batch a paint(color=shape.<that prop>). The causality monkeys can only generalize a hard-coded paint ONE demo at a
# time, so they miss rules where different shapes need different colors in the same demo.
def _add_paints_from_own_props(batcher:ShapeActionBatcher, shape:dict, new_color:str, pair_index:int) -> None:
    """Batch paint(color=shape.<prop>) for every (public) prop of the shape whose value is the new color."""
    for prop_name, prop_value in shape.items():
        # Skip private stuff like _pixels, and non-strings, so numpy arrays never get compared
        if prop_name.startswith('_') or not isinstance(prop_value, str):
            continue
        if prop_value == new_color:
            batcher.add(shape, f'paint(color=shape.{prop_name})', pair_index=pair_index)


def shapes_hide(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that disappear (no shadow match in output), propose hiding them
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, compare Predicted vs Final shapes
        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            final_shapes:list[dict] = pair['output']['shapes']
            for before in predicted_scene['shapes']:

                # Already hidden? SKIP
                if before.get('hidden'):
                    continue

                # Cropped out of the camera entirely? Hiding it wouldn't change anything you can see, so SKIP
                # (else, after a crop, all the off-camera shapes crowd the visible ones out of the batcher's top N)
                if (before['left'] >= predicted_scene['width'] or before['left'] + before['width'] <= 0
                        or before['top'] >= predicted_scene['height'] or before['top'] + before['height'] <= 0):
                    continue

                # If the SAME SHADOW is not in the SAME SPOT, suggest hide!
                # (TODO: how to handle MOVES?)
                if not any(
                    np.array_equal(before['_shadow'], after['_shadow'])
                    and before['left'] == after['left']
                    and before['top'] == after['top']
                    for after in final_shapes
                ):
                    shape_actions.add(before, f'hide()')

        # Flush, try append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates


# Which scene properties each place() parameter can land on, only along its own axis.
# (No scene.width for left, or scene.height for top: in the scene, those would be its left/top, which is just 0)
_SCENE_PROPERTIES_PER_PLACE_PARAMETER:dict[str, tuple[str, ...]] = {
    'left': ('center_x',),
    'right': ('width', 'center_x'),
    'center_x': ('center_x',),
    'top': ('center_y',),
    'bottom': ('height', 'center_y'),
    'center_y': ('center_y',),
}

def _scene_expressions_equal_to(place_parameter:str, value:float, scene:dict) -> list[str]:
    """The scene.<property> expressions (same axis as place_parameter) whose value is exactly this value."""
    # Exact equality only: scene centers are halves (width/2), and a near miss isn't a coincidence worth proposing
    return [f'scene.{scene_property}' for scene_property in _SCENE_PROPERTIES_PER_PLACE_PARAMETER[place_parameter] if scene[scene_property] == value]


def shapes_move(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that moved (same shadow, different position), propose move/place actions
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # Two-axis placements (and one center axis alone) get their own batcher: they're usually different
        # in every demo, so each one is a small batch that the plain suggestions (which often line up across demos)
        # would crowd out of the top N
        center_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # ...and so does move_towards(color=shape.color): shapes that step towards something of their OWN color.
        # (Same idea as shapes_gravity's own-color batcher: with several colors moving, each to its own color,
        # the pooled targets can't say "my color", so here the target selector only has to say WHICH one)
        own_color_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # Moves by the shape's OWN property (like "move down by your own area") get their own batcher,
        # so their many coincidental matches can't crowd the plain suggestions out of the batcher's top N
        own_property_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, find moved shapes (predicted vs final)
        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            final_scene:dict = pair['output']
            line_of_sight_cache:dict[tuple[int, int, int], list[dict]] = {} # One sweep per shape & direction, in this scene
            for before, after in _get_moved_shape_pairs(predicted_scene, final_scene):

                # Suggest: absolute placement, by each edge
                for edge in ('left', 'top', 'right', 'bottom'):
                    shape_actions.add(before, f'place({edge}={after[edge]})', pair_index=pair_index)

                    # ...and by the scene's size or center too, if the edge lands exactly on it.
                    # Those read the same in every demo, so they can batch across demos even when the number differs
                    for scene_expression in _scene_expressions_equal_to(edge, after[edge], predicted_scene):
                        shape_actions.add(before, f'place({edge}={scene_expression})', pair_index=pair_index)

                # Each place() parameter's options: its number, or a scene property if it lands exactly there.
                # (Centers can be halves, like 2.5. ":g" writes a whole one as "3", not "3.0")
                place_options:dict[str, list[str]] = {
                    parameter: [f'{after[parameter]:g}'] + _scene_expressions_equal_to(parameter, after[parameter], predicted_scene)
                    for parameter in ('left', 'right', 'center_x', 'top', 'bottom', 'center_y')
                }

                # Suggest: placement on both axes at once, one horizontal anchor (left, right or center) and one
                # vertical (top, bottom or center), every combination of their options. So a coincidence monkey can
                # swap just one of the two numbers for another shape's property (like "sit on the floor, centered
                # under that shape"), and center_coincidence can read a whole center off one other shape in one go
                for horizontal_parameter in ('left', 'right', 'center_x'):
                    for vertical_parameter in ('top', 'bottom', 'center_y'):
                        for horizontal_option in place_options[horizontal_parameter]:
                            for vertical_option in place_options[vertical_parameter]:
                                center_actions.add(before, f'place({horizontal_parameter}={horizontal_option},{vertical_parameter}={vertical_option})', pair_index=pair_index)

                # ...and each center axis alone, for shapes centered along one axis only (the other axis set some
                # other way). Same batcher as the two-axis placements
                for center_x_option in place_options['center_x']:
                    center_actions.add(before, f'place(center_x={center_x_option})', pair_index=pair_index)
                for center_y_option in place_options['center_y']:
                    center_actions.add(before, f'place(center_y={center_y_option})', pair_index=pair_index)

                # Suggest: relative move. Only name the axes it actually moves along
                # (move() defaults a missing x or y to 0, so "y=0" would just be extra tokens)
                dx:int = after['left'] - before['left']
                dy:int = after['top'] - before['top']
                move_arguments:list[str] = []
                if dx != 0:
                    move_arguments.append(f'x={dx}')
                if dy != 0:
                    move_arguments.append(f'y={dy}')
                # (A color-only change doesn't move at all: no move to suggest)
                if move_arguments:
                    shape_actions.add(before, f'move({",".join(move_arguments)})', pair_index=pair_index)

                # Suggest: the same move by a named direction ("u", "dr", "in"...), if it's a straight line along one:
                # one axis still, or both axes by the same amount. ("in" and "out" read the same in every demo and
                # for every shape, so they can batch shapes that head different compass ways towards the middle)
                steps:int = max(abs(dx), abs(dy)) # In Chebyshev distance
                if move_arguments and (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
                    for direction in _NAMED_DIRECTIONS:
                        if _direction_to_unit_step(direction, before, predicted_scene) == (_sign(dx), _sign(dy)):
                            shape_actions.add(before, f'move(direction="{direction}",steps={steps})', pair_index=pair_index)

                # Suggest: relative move by one of the shape's own whole-number properties that equals the step.
                # (self_coincidence can find these too, but only from a parent program, and a move that's only
                # right for some shapes rarely scores well enough to become one.) Bools are skipped: True isn't 1 here.
                # Only for steps of 2+: a 1-step move matching some property is usually a coincidence,
                # and the plain literal move above already covers it.
                for key, value in before.items():
                    if key.startswith('_') or type(value) is not int:
                        continue
                    # The property equals the step, or its negative (like "move UP by your own area")
                    if abs(dx) >= 2 and value == dx:
                        own_property_actions.add(before, f'move(x=shape.{key},y={dy})', pair_index=pair_index)
                    if abs(dx) >= 2 and value == -dx:
                        own_property_actions.add(before, f'move(x=-shape.{key},y={dy})', pair_index=pair_index)
                    if abs(dy) >= 2 and value == dy:
                        own_property_actions.add(before, f'move(x={dx},y=shape.{key})', pair_index=pair_index)
                    if abs(dy) >= 2 and value == -dy:
                        own_property_actions.add(before, f'move(x={dx},y=-shape.{key})', pair_index=pair_index)

                # Suggest: move towards what shapes? (Same Chebyshev steps as above)
                for target in _cached_line_of_sight(line_of_sight_cache, predicted_scene, before, dx, dy):
                    shape_actions.add(before, f'move_towards(steps={steps})', target, pair_index=pair_index)
                    # Same color as the mover? Then it may be heading for "its own color"
                    if target['color'] == before['color']:
                        own_color_actions.add(before, f'move_towards(steps={steps},color=shape.color)', target, pair_index=pair_index)

                # Suggest: move away from what shapes? Look BEHIND the shape (opposite the way it moved), and keep
                # only the closest ones there, same as shapes_gravity's fall_away. (Same batcher as move_towards)
                shapes_behind:list[dict] = _cached_line_of_sight(line_of_sight_cache, predicted_scene, before, -dx, -dy)
                if shapes_behind:
                    for target in _closest_targets(before, shapes_behind):
                        shape_actions.add(before, f'move_away(steps={steps})', target, pair_index=pair_index)

        # Flush, append all suggestions! (from all four batchers)
        code_blocks:list[str] = shape_actions.flush() + center_actions.flush() + own_color_actions.flush() + own_property_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates


def shapes_restack(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that stay put but change place in the stacking order, propose move(z=...) / move(max_z=...)
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, find shapes that are the same in both (spot, color, shadow) but sit elsewhere in the stack
        for pair_index, pair in enumerate(train_pairs):
            predicted_shapes:list[dict] = program_result['predicted_scenes'][pair_index]['shapes']
            final_shapes:list[dict] = pair['output']['shapes']
            for before in predicted_shapes:
                if before.get('hidden'):
                    continue
                # A shape the program made (like a ray) was never annotated, so it has no place in the input's stack
                if 'rank_z' not in before:
                    continue
                for after in final_shapes:

                    # Cheapest checks first: same spot & color, then the same shadow
                    if (before['left'] != after['left'] or before['top'] != after['top']
                            or before['color'] != after['color']
                            or not np.array_equal(before['_shadow'], after['_shadow'])):
                        continue

                    # Compare stack RANKS, not raw z: the input's and output's stacks are numbered separately.
                    # rank_z = 0 is the frontmost, so a smaller rank_z (or a bigger rank_z_reversed) = it came forward.
                    # (rank_z is annotated once, so after an earlier z move in the program it's the input's rank still)
                    if after['rank_z'] < before['rank_z'] or after['rank_z_reversed'] > before['rank_z_reversed']:
                        actions:list[str] = ['move(z=1)', 'move(z=2)', 'move(max_z=1)']
                    elif after['rank_z'] > before['rank_z'] or after['rank_z_reversed'] < before['rank_z_reversed']:
                        actions = ['move(z=-1)', 'move(z=-2)', 'move(max_z=-1)']
                    else:
                        continue # Same place in the stack: nothing to propose
                    for action in actions:
                        shape_actions.add(before, action)

        # Flush, append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates


def shapes_bring_to_front(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Look at a few wrong pixels: if a shape under one has the output's color there, propose move(max_z=1) to bring it to the front
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            predicted_grid:np.ndarray = predicted_scene['_raw']
            output_grid:np.ndarray = pair['output']['_raw']

            # Only when the grid sizes match: then a wrong pixel is the same spot in both grids
            if predicted_grid.shape != output_grid.shape:
                continue
            wrong_ys, wrong_xs = np.nonzero(predicted_grid != output_grid)
            if len(wrong_ys) == 0:
                continue

            # Up to 3 randomly chosen wrong pixels (a sample is enough, and it keeps big diffs cheap)
            for wrong_index in random.sample(range(len(wrong_ys)), min(3, len(wrong_ys))):
                pixel_y, pixel_x = int(wrong_ys[wrong_index]), int(wrong_xs[wrong_index])
                wanted_color:int = int(output_grid[pixel_y, pixel_x])
                for shape in predicted_scene['shapes']:
                    if shape.get('hidden'):
                        continue
                    # Is this spot inside the shape's pixel grid? (Shapes can be multicolor, so read the pixel itself)
                    local_y:int = pixel_y - int(shape['top'])
                    local_x:int = pixel_x - int(shape['left'])
                    shape_height, shape_width = shape['_pixels'].shape
                    if not (0 <= local_y < shape_height and 0 <= local_x < shape_width):
                        continue
                    # It has the right color here but the pixel's still wrong, so something covers it: bring it to the front.
                    # (Transparent pixels are -1, so they never match a real color)
                    if shape['_pixels'][local_y, local_x] == wanted_color:
                        shape_actions.add(shape, 'move(max_z=1)')

        # Flush, append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates


def shapes_gravity(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that moved, propose fall/fall_towards/fall_away (with apply_gravity afterwards)
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )
        # fall_away gets its own batcher, so it never pushes fall/fall_towards out of the top-N cut (or vice versa)
        fall_away_actions = ShapeActionBatcher( program_result['predicted_scenes'] )
        # ...and so does fall_towards(color=shape.color): shapes that fall towards something of their OWN color.
        # The plain batch pools every actor's targets, so with several colors falling (each to its own color)
        # its target selector can never say "my color". Batched here, the target selector only has to say WHICH one
        own_color_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, find moved shapes (predicted vs final)
        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            final_scene:dict = pair['output']
            line_of_sight_cache:dict[tuple[int, int, int], list[dict]] = {} # One sweep per shape & direction, in this scene
            for before, after in _get_moved_shape_pairs(predicted_scene, final_scene):

                # Suggest: fall (direction clamped to -1/0/1)
                dx:int = after['left'] - before['left']
                dy:int = after['top'] - before['top']
                clamped_dx:int = int( max(-1, min(dx, 1)) )
                clamped_dy:int = int( max(-1, min(dy, 1)) )
                shape_actions.add(before, f'fall(x={clamped_dx},y={clamped_dy})', pair_index=pair_index)

                # Suggest: fall towards what shapes?
                for target in _cached_line_of_sight(line_of_sight_cache, predicted_scene, before, dx, dy):
                    shape_actions.add(before, f'fall_towards()', target, pair_index=pair_index)
                    # Same color as the faller? Then it may be falling towards "its own color"
                    if target['color'] == before['color']:
                        own_color_actions.add(before, 'fall_towards(color=shape.color)', target, pair_index=pair_index)

                # Suggest: fall away from what shapes? Look BEHIND the shape (opposite the way it moved),
                # and keep only the closest ones there (smallest pixel gap): fall_away leaves its closest target, not just any
                shapes_behind:list[dict] = _cached_line_of_sight(line_of_sight_cache, predicted_scene, before, -dx, -dy)
                if shapes_behind:
                    for target in _closest_targets(before, shapes_behind):
                        fall_away_actions.add(before, f'fall_away()', target, pair_index=pair_index)

        # Flush, append all suggestions! (from all three batchers, with apply_gravity() afterwards)
        code_blocks:list[str] = shape_actions.flush() + fall_away_actions.flush() + own_color_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() + '\napply_gravity()' )

    # All candidates!
    return candidates


# The jigsaw monkey: a shape that moved INTO somebody else's bounding box (and wasn't in anyone's before)
# probably got fit into their gaps. Propose fit() for those (with apply_fit afterwards), like shapes_gravity.
def shapes_fit(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that moved into someone else's bounding box, propose fit (with apply_fit afterwards)
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, find moved shapes (predicted vs final)
        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            final_scene:dict = pair['output']

            # Scene size doesn't match the output yet? Then propose nothing for this pair.
            # A crop/resize (e.g. focus) should come first; fitting pieces into gaps comes after.
            # (Uncropped fit searches can blow up: ~60x more search nodes than after a focus, on one jigsaw level)
            if predicted_scene['width'] != final_scene['width'] or predicted_scene['height'] != final_scene['height']:
                continue

            visible_before:list[dict] = [shape for shape in predicted_scene['shapes'] if not shape.get('hidden')]
            for before, after in _get_moved_shape_pairs(predicted_scene, final_scene):

                # Was it already overlapping someone's box? Then it's not fitting INTO anything new
                # ("inside a box" = overlapping it at all, not fully inside: a piece only half-plugs into a gap)
                if before.get('hidden') or _shapes_hitbox_test(before, visible_before):
                    continue

                # Now it IS in someone's box? Suggest: fit!
                if _shapes_hitbox_test(after, final_scene['shapes']):
                    shape_actions.add(before, 'fit()')

        # Flush, append all suggestions! (with apply_fit() afterwards)
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            # (Just the one variant: apply_fit's alignment bonus is always on, for area >= 5)
            candidates.add( program_result['program'] + '\n' + code.strip() + '\napply_fit()' )

    # All candidates!
    return candidates


# The attraction monkey: a shape that moved and now sits ON TOP of another shape, with every overlapping pixel
# the same color as the pixel under it (like a dot snapping onto a same-colored dot), probably got attracted there.
# Propose attract(overlap=True) for those (with apply_attract afterwards), like shapes_fit.
def shapes_attract(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes that moved onto another shape with an exact color-matched overlap, propose attract (with apply_attract afterwards)
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, find moved shapes (predicted vs final)
        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            final_scene:dict = pair['output']

            # Unlike shapes_fit, we DON'T skip pairs whose scene size differs from the output's:
            # the attraction can happen BEFORE a crop, in which case every piece just looks "moved" by the crop offset.
            for before, after in _get_moved_shape_pairs(predicted_scene, final_scene):

                # At its output spot, does it overlap some other output shape with an exact color match? Suggest: attract!
                if any(_overlap_colors_match(after, other_shape) for other_shape in _shapes_hitbox_test(after, final_scene['shapes'])):
                    shape_actions.add(before, 'attract(overlap=True)')

        # Flush, append all suggestions! (with apply_attract() afterwards)
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() + '\napply_attract()' )

    # All candidates!
    return candidates


# The stamping monkey: an output shape that looks EXACTLY like a predicted shape (same pixels, same colors), but sits
# somewhere else, right on top of some other predicted shape with an exact color-matched overlap (like a stamp's
# colored mark landing on a same-colored marker dot), is probably a copy stamped onto that shape. Propose copy_to(target).
def shapes_copy(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice output shapes that are copies of a predicted shape sitting on a color-matched target, propose copy_to
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # Execution-guided progsys: batch actions against *predicted* shapes
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, look for copies (predicted vs final)
        for pair_index, pair in enumerate(train_pairs):
            predicted_scene:dict = program_result['predicted_scenes'][pair_index]
            visible_before:list[dict] = [shape for shape in predicted_scene['shapes'] if not shape.get('hidden')]

            # Output shapes grouped by their exact pixels (colors included), so each predicted shape finds its
            # look-alikes in one lookup, instead of comparing against every output shape
            outputs_by_pixels:dict[tuple, list[dict]] = {}
            for after in pair['output']['shapes']:
                outputs_by_pixels.setdefault((after['_pixels'].shape, after['_pixels'].tobytes()), []).append(after)

            for before in visible_before:
                for after in outputs_by_pixels.get((before['_pixels'].shape, before['_pixels'].tobytes()), []):

                    # A copy is somewhere ELSE (the same look-alike in the same spot is just the shape staying put)
                    if (after['left'], after['top']) == (before['left'], before['top']):
                        continue

                    # Which predicted shapes does that copy sit on, color for color? Those are its targets
                    for target in _shapes_hitbox_test(after, visible_before):
                        if target is not before and _overlap_colors_match(after, target):
                            shape_actions.add(before, 'copy_to()', target, pair_index=pair_index)

        # Flush, append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add( program_result['program'] + '\n' + code.strip() )

    # All candidates!
    return candidates


# Do two shapes' real (non-gap) pixels overlap somewhere, and is EVERY overlapping pixel the same color in both?
# (Colors are compared coordinate for coordinate, so two checkerboards only match if their squares line up exactly.)
def _overlap_colors_match(shape:dict, other_shape:dict) -> bool:
    """True if the two shapes share at least 1 filled pixel, and all shared filled pixels have the same color"""

    # The overlap rectangle, in grid coordinates (their boxes already overlap, from the hitbox test)
    overlap_left:int = max(shape['left'], other_shape['left'])
    overlap_right:int = min(shape['right'], other_shape['right'])
    overlap_top:int = max(shape['top'], other_shape['top'])
    overlap_bottom:int = min(shape['bottom'], other_shape['bottom'])

    # Each shape's pixels inside that rectangle (shifted into each shape's own coordinates; -1 = gap)
    shape_pixels:np.ndarray = shape['_pixels'][overlap_top - shape['top']:overlap_bottom - shape['top'],
                                               overlap_left - shape['left']:overlap_right - shape['left']]
    other_pixels:np.ndarray = other_shape['_pixels'][overlap_top - other_shape['top']:overlap_bottom - other_shape['top'],
                                                     overlap_left - other_shape['left']:overlap_right - other_shape['left']]

    # Where are BOTH filled? Need at least one such pixel, and they all have to agree
    both_filled:np.ndarray = (shape_pixels >= 0) & (other_pixels >= 0)
    return bool(both_filled.any()) and bool(np.array_equal(shape_pixels[both_filled], other_pixels[both_filled]))


def shapes_transform(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice shapes whose shadow is a non-identity D4 transform of the input shadow.
    D4 = the dihedral group of the square: 3 rotations + 4 reflections = all 7 non-identity elements.
    Every rotation+flip composition is already one of the 4 reflections, so checking them separately covers everything.
    """

    # For each program result
    candidates:set = set()
    for program_result in parent_program_results:

        # All shapes to compare against: all *predicted scenes' shapes* only,
        # since that's what we're selecting from!
        shape_actions = ShapeActionBatcher( program_result['predicted_scenes'] )

        # For each pair, compare Predicted vs Final shapes
        for pair_index, pair in enumerate(train_pairs):
            predicted_shapes:list[dict] = program_result['predicted_scenes'][pair_index]['shapes']
            final_shapes:list[dict] = pair['output']['shapes']
            for before in predicted_shapes:

                # Skip this shape if it appears UNCHANGED in the output (same shadow AND same position).
                # (Position matters: different shapes can share a silhouette, e.g. two 2x2 L-shapes at *different* positions)
                if any(
                    np.array_equal(before['_shadow'], after['_shadow'])
                    and before['left'] == after['left']
                    and before['top'] == after['top']
                    for after in final_shapes
                ):
                    continue

                # Otherwise, batch-add all After non-indentity transformations
                for after in final_shapes:
                    for action in _detect_d4_transforms(before, after):
                        shape_actions.add(before, action)

        # Flush, append all suggestions!
        code_blocks:list[str] = shape_actions.flush()
        for code in code_blocks:
            candidates.add(program_result['program'] + '\n' + code.strip()) # append        

    # All candidates!
    return candidates






###################################################
### SHARED HELPERS ################################
###################################################

def _cached_line_of_sight(cache:dict[tuple[int, int, int], list[dict]], scene:dict, shape:dict, dx:int, dy:int) -> list[dict]:
    """_shapes_in_line_of_sight, but each shape gets swept just once per direction (pass one cache per scene)."""

    # The sweep only cares which WAY the shape moved (dx, dy's signs), not how far. And one shape can "move" to several
    # same-shadow output shapes, often in the same direction: on a scene with hundreds of shapes, that's lots of repeat sweeps
    cache_key:tuple[int, int, int] = (id(shape), (dx > 0) - (dx < 0), (dy > 0) - (dy < 0))
    if cache_key not in cache:
        cache[cache_key] = _shapes_in_line_of_sight(scene, shape, dx, dy)
    return cache[cache_key]

def _get_moved_shape_pairs(before_scene:dict, after_scene:dict) -> list[tuple[dict, dict]]:
    """
    Find shapes in before that changed: same shadow in after, but different x/y/color.
    Score = number of edits (0-3): did x change? did y change? did color change?
    Skip 0-change (identical) shapes. Return all tied-for-lowest-score matches.
    """
    moved_pairs:list[tuple[dict,dict]] = []

    # Spots already filled by an identical, visible shape in before: nothing else should "move" into those.
    # (E.g. after a crop, every off-camera copy of an identical box would otherwise "move" onto the boxes already in frame)
    filled_spots:set[tuple] = {(b['left'], b['top'], b['_pixels'].shape, b['_pixels'].tobytes())
                               for b in before_scene['shapes'] if not b.get('hidden', False)}

    for b in before_scene['shapes']:

        # Score each same-shadow match by number of edits (x, y, color)
        scored:list[tuple[int, dict]] = []
        for a in after_scene['shapes']:
            if np.array_equal(b['_shadow'], a['_shadow']):
                # An identical shape's already sitting in that spot? Then it's not a move target
                if (a['left'], a['top'], a['_pixels'].shape, a['_pixels'].tobytes()) in filled_spots and (a['left'], a['top']) != (b['left'], b['top']):
                    continue
                # Multicolor shapes: the colors INSIDE are part of what the shape is, so they have to match too.
                # (E.g. when every box has the same solid square shadow, without this every box "moved"
                # to every output box, and the move monkey's selectors grabbed all of them instead of just the odd ones)
                if not b['monochrome'] and not np.array_equal(b['_pixels'], a['_pixels']):
                    continue
                edits:int = (b['left'] != a['left']) + (b['top'] != a['top']) + (b['color'] != a['color'])
                scored.append((edits, a))

        # Skip if no shadow match, OR if any match has 0 edits (shape already in place).
        # Otherwise correctly-placed shapes would still make spurious move proposals against other
        # same-shadow shapes at different positions, crowding out the batcher's budget
        # (BATCHER_TOP_N_PIXELS) for shapes that actually need to move.
        if not scored or any(edits == 0 for edits, a in scored):
            continue

        # Keep all tied-for-lowest-score matches
        best_score:int = min(edits for edits, a in scored)
        for edits, a in scored:
            if edits == best_score:
                moved_pairs.append((b, a))

    return moved_pairs


def _detect_d4_transforms(before:dict, after:dict) -> list[str]:
    """
    Check if after's shadow/pixels are a non-identity D4 transform of before's.
    Returns list of matching transform actions, or empty list if no match.
    (Note that YES two different transformations can produce the same shape!
    For example, | => — could be 90° rotation or 270° rotation or D1 or D2 flip)
    """
    actions:set = set()

    before_shadow = before['_shadow']
    before_pixels = before['_pixels']
    after_shadow = after['_shadow']
    after_pixels = after['_pixels']

    # Try both shadow (silhouette) and pixel (colored) grids
    for is_shadow in (True, False):
        before_grid = before_shadow if is_shadow else before_pixels
        after_grid = after_shadow if is_shadow else after_pixels

        # Rotations (90, 180, 270)
        for quarters in (1, 2, 3):
            rotated_grid = _rotate_grid(before_grid, quarters)
            if not np.array_equal(before_grid, rotated_grid) and np.array_equal(rotated_grid, after_grid):
                actions.add(f'rotate( quarters={quarters} )')
                actions.add(f'rotate( quarters={quarters-4} )') # counterclockwise equivalent

        # Flips (horizontal, vertical, diagonal_1, diagonal_2)
        for flip_mode in ('horizontal', 'vertical', 'diagonal_1', 'diagonal_2'):
            flipped_grid = _flip_grid(before_grid, flip_mode)
            if not np.array_equal(before_grid, flipped_grid) and np.array_equal(flipped_grid, after_grid):
                actions.add(f'flip( mode="{flip_mode}" )')

    return list(actions)

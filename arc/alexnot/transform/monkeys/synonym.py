"""
SYNONYMS FOR SELECTORS
- Keep a program as it is, except HOW one all(...) / one(...) / action's target picks its shapes
- Try other ways to describe the same shapes ("the tall ones" -> "the multicolor ones"), the scorer keeps the simplest
"""

import re
from typing import Any, Optional

from ...utils import _propose_selectors, _dict_to_argstring, _select_shape_datas, _clean_arg_value
from .monkey_helpers import scenes_before_block

# CONFIG KNOBS
from ...config import BATCHER_TOP_N_ACTIONS

# An all(...) or one(...) call, and its arguments (selector args never hold parentheses)
SELECTOR_CALL_PATTERN:re.Pattern = re.compile(r'\b(all|one)\(([^()]*)\)')
# The actions that take a target selector, and their NON-selector params (kept as they are, everything else gets re-rolled)
ACTION_NON_SELECTOR_PARAMS:dict[str, tuple[str, ...]] = {
    'move_towards': ('steps',),
    'fall_towards': (),
    'fall_away': (),
    'attract': ('overlap',),
    'add_ray_towards': ('cut_corners', 'ray_color'),
    'copy_to': (),
    'focus': ('padding', 'hide_non_matches', 'one_match', 'hug', 'margin'),
    'connect_with_lines': ('line_color', 'directions', 'same_color'),
}
# A call to one of those actions, and its arguments (shape methods like .copy_to(...), or top-level ones like focus(...))
ACTION_CALL_PATTERN:re.Pattern = re.compile(r'\b(' + '|'.join(ACTION_NON_SELECTOR_PARAMS) + r')\(([^()]*)\)')
# The loop a one(...) join sits in, e.g. "for shape in all(color="c5"):"
LOOP_PATTERN:re.Pattern = re.compile(r'for shape in all\(([^()]*)\):')
# A join to the loop's shape, e.g. the "shape.holes" in one(holes=shape.holes)
JOIN_VALUE_PATTERN:re.Pattern = re.compile(r'^shape\.(\w+)$')
# A per-demo scene value, e.g. the "scene.count_color_c1" in all(rank_area=scene.count_color_c1)
SCENE_VALUE_PATTERN:re.Pattern = re.compile(r'^scene\.(\w+)$')


def reroll_selectors(train_pairs:list[dict], parent_program_results:list[dict]) -> set[str]:
    """For each parent, re-roll ONE all()/one()/action target selector at a time (same shapes, other descriptions)."""

    # Twice the batcher's count: this monkey only re-rolls, so it can afford more synonyms per selector
    synonyms_per_selector:int = BATCHER_TOP_N_ACTIONS * 2

    candidates:set[str] = set()
    for program_result in parent_program_results:
        program:str = program_result['program']

        # The for-loops, so a one(...) with a join knows which loop's shapes it joins to
        loop_matches:list[re.Match] = list(LOOP_PATTERN.finditer(program))

        # Every all()/one(), and every action's target selector
        call_matches:list[re.Match] = list(SELECTOR_CALL_PATTERN.finditer(program)) + list(ACTION_CALL_PATTERN.finditer(program))
        for selector_match in call_matches:

            # An action's non-selector params (like padding=1) aren't part of its selector: set them aside, as they are
            non_selector_params:tuple[str, ...] = ACTION_NON_SELECTOR_PARAMS.get(selector_match.group(1), ())
            selector_args:list[str] = []
            kept_args:list[str] = []
            for arg in selector_match.group(2).split(','):
                if arg.split('=', 1)[0].strip() in non_selector_params:
                    kept_args.append(arg.strip())
                elif arg.strip():
                    selector_args.append(arg)

            # An action with no target selector at all? Nothing to re-roll (and for some actions "no selector" means "do nothing")
            if not selector_args and selector_match.group(1) in ACTION_NON_SELECTOR_PARAMS:
                continue

            # Split the args into plain values + scene values (both re-rolled) and joins like holes=shape.holes (kept as they are)
            split_args:Optional[tuple[dict, dict[str, str], dict[str, str]]] = _split_selector_args(','.join(selector_args))
            if split_args is None:
                continue # Some other kind of value we don't re-roll around
            literal_selector, scene_values, joins = split_args

            # Pick the shapes on the scenes just before this selector's block runs (not the raw input, not the final output)
            scenes:list = scenes_before_block(program, selector_match.start())

            # Every shape in every demo, pooled (the same pooling the batcher does)
            all_shapes:list[dict] = [shape for scene in scenes for shape in scene['shapes']]

            # Which shapes does this selector pick, across all demos?
            ingroup:Optional[list[dict]]
            if len(joins) == 0:
                ingroup = [shape for scene in scenes
                           for shape in _select_shape_datas(scene['shapes'], _fill_scene_values(literal_selector, scene_values, scene))]
            else:
                ingroup = _joined_ingroup(scenes, literal_selector, scene_values, joins, loop_matches, selector_match.start())

            # Picks nothing? Then there's nothing to describe (and the empty selector would pick everything instead)
            if not ingroup:
                continue

            # Other ways to pick the same shapes. A join's key is taken already, so leave it out
            new_selectors:list[dict] = _propose_selectors(ingroup, all_shapes, N=synonyms_per_selector,
                                                          props_to_exclude=list(joins.keys()))

            for new_selector in new_selectors:
                # The parent's own selector again? Skip it (one with scene values never matches: those get replaced)
                if new_selector == literal_selector and not scene_values:
                    continue

                # Swap just this one selector in: new args first, then the joins and non-selector params as they were
                new_args:list[str] = [_dict_to_argstring(new_selector)] if new_selector else []
                new_args += [f'{key}={value}' for key, value in joins.items()]
                new_args += kept_args
                new_call:str = f'{selector_match.group(1)}({",".join(new_args)})'
                candidates.add(program[:selector_match.start()] + new_call + program[selector_match.end():])

    return candidates


def _split_selector_args(argstring:str) -> Optional[tuple[dict, dict[str, str], dict[str, str]]]:
    """(plain args as a selector dict, scene values as {key: scene prop}, joins as {key: "shape.prop"}), or None if an arg is none of those."""
    literal_selector:dict = {}
    scene_values:dict[str, str] = {}
    joins:dict[str, str] = {}
    for arg in argstring.split(','):
        if not arg.strip():
            continue # all() with no args
        key, value = [part.strip() for part in arg.split('=', 1)]
        cleaned_value:Any = _clean_arg_value(value)
        scene_value_match:Optional[re.Match] = SCENE_VALUE_PATTERN.match(value)
        if cleaned_value is not None:
            literal_selector[key] = cleaned_value
        elif JOIN_VALUE_PATTERN.match(value):
            joins[key] = value
        elif scene_value_match:
            scene_values[key] = scene_value_match.group(1) # Filled in per demo later, see _fill_scene_values
        else:
            return None
    return literal_selector, scene_values, joins


def _fill_scene_values(literal_selector:dict, scene_values:dict[str, str], scene:dict) -> dict:
    """A copy of the selector with each scene.X value filled in from this demo's scene."""
    # Read it the way the DSL does at runtime (a missing scene prop is 0), so we pick what the program would pick
    return {**literal_selector, **{key: scene.get(prop_name, 0) for key, prop_name in scene_values.items()}}


def _joined_ingroup(scenes:list[dict], literal_selector:dict, scene_values:dict[str, str], joins:dict[str, str],
                    loop_matches:list[re.Match], selector_position:int) -> Optional[list[dict]]:
    """The shapes a joined one(...) can pick for any of its loop's shapes, across all demos."""

    # The loop it sits in = the last for-loop above it (a join to "shape." only works inside one)
    loops_above:list[re.Match] = [loop for loop in loop_matches if loop.start() < selector_position]
    if len(loops_above) == 0:
        return None
    loop_split:Optional[tuple[dict, dict[str, str], dict[str, str]]] = _split_selector_args(loops_above[-1].group(1))
    if loop_split is None or loop_split[2]:
        return None # The loop's own selector has joins (or values we can't read), so we can't tell which shapes it loops over
    loop_selector, loop_scene_values, _ = loop_split

    # For each shape the loop picks, which shapes match the one(...) with that shape's values filled in?
    ingroup:list[dict] = []
    for scene in scenes:
        picked_ids:set[int] = set() # (A shape picked by several loop shapes only counts once)
        for loop_shape in _select_shape_datas(scene['shapes'], _fill_scene_values(loop_selector, loop_scene_values, scene)):
            filled_selector:dict = _fill_scene_values(literal_selector, scene_values, scene)
            for key, value in joins.items():
                filled_selector[key] = loop_shape.get(JOIN_VALUE_PATTERN.match(value).group(1)) # pyright: ignore[reportOptionalMemberAccess]
            for picked_shape in _select_shape_datas(scene['shapes'], filled_selector):
                if id(picked_shape) not in picked_ids:
                    picked_ids.add(id(picked_shape))
                    ingroup.append(picked_shape)
    return ingroup

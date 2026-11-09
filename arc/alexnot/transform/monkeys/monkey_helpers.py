import random # To break ties randomly
from typing import Dict, List, Optional
import numpy as np # For _original_footprint
from ...utils import _propose_selectors, _dict_to_argstring, _expand_template
from ..apply_transform import apply_transform_one # To re-run a program prefix the search hasn't scored

# CONFIG KNOBS
from ...config import BATCHER_TOP_N_PIXELS, BATCHER_TOP_N_ACTIONS, BATCHER_TOP_N_ACTORS_WITH_TARGET, BATCHER_TOP_N_TARGETS, BATCHER_PER_DEMO_PAIR

class ShapeActionBatcher:
    """
    A helper object: as a code-monkey goes through all the possible actions that
    could happen, they could dump the relevant shapes & action into a Batcher.
    Upon request, the Batcher then runs the _propose_selectors helper on
    each group of shapes/targets for each action, and returns a list of code-strings.

    Each action gets batched across ALL demos (selectors picked against every demo's shapes), and, if the
    monkey passes pair_index AND the BATCHER_PER_DEMO_PAIR knob is on, once more per demo pair (selectors picked
    against just that demo's shapes). The knob is off for now, so only the all-demos batches get proposed.
    The per-demo batches catch rules where the action's value comes from a different shape in each demo, so no one
    hard-coded action fits every demo. (The search keeps a best program per demo, so those can survive.)

    Methods:
    - init(scenes) # to get all the shapes we'll be _propose_selecting against for actors/targets
    - add(actor_shape, "action(params=vals)", target_shape, pair_index=i)
    - flush() # gives all code proposals
    """

    def __init__(self, scenes:list[dict]):

        # Flattens all scenes' shapes to a single .all_shapes property
        self.all_shapes = []
        for scene in scenes:
            self.all_shapes += scene['shapes'] # extends list

        # ...and keeps each demo's shapes on their own too, for the per-demo selectors
        self.shapes_per_pair:list[list[dict]] = [scene['shapes'] for scene in scenes]

        # "Batches" of actor.action(target) to use later!
        # Each batch tracks actors, targets, and how many pixels it accounts for
        self.batches:dict[str, dict] = {} # { "action_string":{ actors:[shapes], targets:[shapes], pixels_affected:int }

        # The same kind of batches, but one set per demo pair
        self.batches_per_pair:list[dict[str, dict]] = [{} for scene in scenes]

    def add(self, shape:dict, action:str, target:Optional[dict]=None, pair_index:Optional[int]=None):

        # Into the all-demos batches, AND (if the monkey says which demo it's from) into that demo's own batches.
        # Per-demo selectors are slow-ish (one more _propose_selectors run per action per demo), so only
        # the monkeys that need them pass pair_index
        # (The per-demo ones only while the BATCHER_PER_DEMO_PAIR knob is on; off, flush() has nothing per-demo to propose)
        _add_to_batches(self.batches, shape, action, target)
        if pair_index is not None and BATCHER_PER_DEMO_PAIR:
            _add_to_batches(self.batches_per_pair[pair_index], shape, action, target)

    def flush(self) -> list[str]:

        # All-demos proposals, then each demo's own (selectors against just that demo's shapes)
        code_blocks:set[str] = _batches_to_code(self.batches, self.all_shapes)
        for pair_batches, pair_shapes in zip(self.batches_per_pair, self.shapes_per_pair):
            code_blocks |= _batches_to_code(pair_batches, pair_shapes)

        # Gimme deduplicated code blocks
        return list(code_blocks)


def _add_to_batches(batches:dict[str, dict], shape:dict, action:str, target:Optional[dict]) -> None:
    """Add one actor (and maybe its target) to its action's batch, creating the batch if it's new."""

    # Create if new
    if action not in batches:
        batches[action] = {
            "actors": [],
            "targets": [],
            "pixels_affected": 0
        }

    # Add to the dicts
    # TODO MAYBE: de-dupe actors/targets by id(shape) to avoid redundant selector work
    batches[action]["actors"].append(shape)
    if target:
        batches[action]["targets"].append(target)

    # Track how many pixels this action accounts for (actor's area only, not target's)
    batches[action]["pixels_affected"] += shape.get("area", 0)


def _batches_to_code(batches:dict[str, dict], all_shapes:list[dict]) -> set[str]:
    """Code proposals for the top batches, with selectors picked against all_shapes."""

    code_blocks:set[str] = set()
    code_template = "for shape in all(%SELECT%):\n\tshape.%ACTION%"
    # Actions that read a property off "other" (like place(top=other.top)) pick that other shape with one(...) first
    other_code_template = "for shape in all(%SELECT%):\n\tother = one(%TARGET%)\n\tshape.%ACTION%"

    # Only generate code for the top N most-pixels-affecting actions
    # Shuffle first, so actions TIED on pixels at the top-N cap get cut randomly (sorted() is stable),
    # instead of by dict order (= the order the monkey happened to add them)
    shuffled_batches:list[tuple[str, dict]] = list(batches.items())
    random.shuffle(shuffled_batches)
    sorted_batches = sorted(
        shuffled_batches,
        key=lambda item: item[1]["pixels_affected"],
        reverse=True
    )
    top_batches = sorted_batches[:BATCHER_TOP_N_PIXELS]

    # For each action, propose top selectors for the actors/targets
    # (If no target, get the top BATCHER_TOP_N_ACTIONS actors. If target, top
    # BATCHER_TOP_N_ACTORS_WITH_TARGET actors x BATCHER_TOP_N_TARGETS targets, all config knobs)
    for action, actor_targets in top_batches:
        actors:list[dict] = actor_targets["actors"]
        targets:list[dict] = actor_targets["targets"]
        if len(targets)==0:
            # No target, just swap in. (A config knob, so double_time can propose more)
            actor_selectors:list[dict] = _propose_selectors(actors, all_shapes, N=BATCHER_TOP_N_ACTIONS)
            for actor_selector in actor_selectors:
                code:str = _expand_template(code_template,{
                    "%SELECT%": _dict_to_argstring(actor_selector),
                    "%ACTION%": action
                })
                code_blocks.add(code)
        else:
            # A target? Do nested-loop: actor-selectors x target-selectors (separate knobs for each)
            actor_selectors:list[dict] = _propose_selectors(actors, all_shapes, N=BATCHER_TOP_N_ACTORS_WITH_TARGET)
            target_selectors:list[dict] = _propose_selectors(targets, all_shapes, N=BATCHER_TOP_N_TARGETS)
            for actor_selector in actor_selectors:
                for target_selector in target_selectors:

                    # Reads a property off "other"? Then the target selector gets its own one(...) line
                    if "other." in action:
                        code:str = _expand_template(other_code_template,{
                            "%SELECT%": _dict_to_argstring(actor_selector),
                            "%TARGET%": _dict_to_argstring(target_selector),
                            "%ACTION%": action
                        })
                        code_blocks.add(code)
                        continue

                    # Splicing in the target-selector into the action's parameters
                    # (with comma-logic to deal with existing vs not-pre-existing args)
                    # Splice target selector as flat kwargs into the action string (assumes action ends with ")")
                    # If action has no existing args (e.g. "fall_towards()"), skip the leading comma
                    flat_target_args:str = _dict_to_argstring(target_selector)
                    action_prefix:str = action[:-1] # strip trailing ")"
                    if not action_prefix.endswith("("): # there ARE other args
                        flat_target_args = ", " + flat_target_args # comma to separate them
                    spliced_action:str = action_prefix + flat_target_args + ")" # put paren back in
                    code:str = _expand_template(code_template,{
                        "%SELECT%": _dict_to_argstring(actor_selector),
                        "%ACTION%": spliced_action
                    })
                    code_blocks.add(code)

    return code_blocks




# Where a shape's ORIGINALLY-parsed pixels sit, ignoring anything it grew since (an outline ring, filled holes...).
# That's what paint() recolors, so the repaint/fill_holes monkeys can still spot "same shape, new color" after an outline.
def _original_footprint(shape:dict) -> tuple[int, int, np.ndarray]:
    """(left, top, shadow) of just the shape's original pixels, trimmed to their own bounding box."""
    original_shadow:np.ndarray = shape['_original_shadow']

    # Nothing grew? Then it's just the shape itself (the common case, so skip the trimming)
    if np.array_equal(original_shadow, shape['_shadow']):
        return shape['left'], shape['top'], shape['_shadow']

    # Trim the mask down to the original pixels' bounding box...
    ys, xs = np.nonzero(original_shadow)
    if ys.size == 0:
        return shape['left'], shape['top'], original_shadow # (An inset ate every original pixel: all-zero, matches no shape)
    top_offset:int = int(ys.min())
    left_offset:int = int(xs.min())
    trimmed_shadow:np.ndarray = original_shadow[top_offset:int(ys.max()) + 1, left_offset:int(xs.max()) + 1]

    # ...and shift its corner to where it really is in the grid
    return shape['left'] + left_offset, shape['top'] + top_offset, trimmed_shadow


# The scenes each program PREFIX leaves behind, for the monkeys that rewrite a block in the middle of a program.
# search_transform resets this at the start of every transform search (one per parse), see reset_prefix_scenes.
# Shared by every monkey, so they must only READ these scenes, never edit their shapes in place.
PREFIX_SCENES:dict = {
    'train_pairs': [], # The demo pairs (the empty prefix's scenes are just their inputs)
    'program_results': {}, # search_transform's all_program_results: every scored program's predicted scenes
    'scenes_by_prefix': {}, # Prefixes the search never scored, re-run once and remembered here
}

def reset_prefix_scenes(train_pairs:list[dict], all_program_results:dict) -> None:
    """Start fresh for a new transform search (a new parse)."""
    PREFIX_SCENES['train_pairs'] = train_pairs
    PREFIX_SCENES['program_results'] = all_program_results # The search's own dict, so new results show up as they're scored
    PREFIX_SCENES['scenes_by_prefix'] = {}

def scenes_before_block(program:str, position:int) -> list:
    """The scenes (one per demo) just before the top-level block holding program[position] runs."""

    # Walk back to the top-level (unindented) line this block starts on
    block_start:int = program.rfind('\n', 0, position) + 1
    while block_start > 0 and program[block_start] in ' \t\n':
        block_start = program.rfind('\n', 0, block_start - 1) + 1

    # The prefix: every block before this one. Nothing before it? Then it's the input scenes
    prefix:str = program[:block_start].strip()
    if prefix == '':
        return [pair['input'] for pair in PREFIX_SCENES['train_pairs']]

    # Already scored by the search? (Often: a parent's prefix is usually its own parent)
    prefix_result:Optional[dict] = PREFIX_SCENES['program_results'].get(prefix)
    if prefix_result is not None:
        return prefix_result['predicted_scenes']

    # Else re-run it on each demo's input once, and remember it for the next monkey
    scenes_by_prefix:dict = PREFIX_SCENES['scenes_by_prefix']
    if prefix not in scenes_by_prefix:
        scenes_by_prefix[prefix] = [apply_transform_one(prefix, pair['input']) for pair in PREFIX_SCENES['train_pairs']]
    return scenes_by_prefix[prefix]

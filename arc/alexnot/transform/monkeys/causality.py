import re
import numpy as np
from typing import Dict, List, Any, Tuple

from ...utils import _propose_selectors, _dict_to_argstring, _argstring_to_dict, _select_shape_datas, _clean_arg_value
from .monkey_helpers import scenes_before_block

# Compiled once, reused everywhere (DRY > perf -- re already caches, so the speedup is ~nil)
_FOR_ALL_PATTERN = re.compile(r'for shape in all\(([^)]*)\):\s*\n\s+shape\.(\w+)\(([^)]*)\)')
_HARDCODED_VALUE_PATTERN = re.compile(r'"[^"]*"|-?\d+(?:\.\d+)?') # "quoted string" or int/float (incl 0 & negatives)

def join_coincidence(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    """Turn a hard-coded action value into one read off a matching "instruction" shape, joined on an equal property."""

    # A concrete example. This:
    #
    # for shape in all( color=gray ):
    #     shape.paint( color=red )
    #
    # asks WHY gray shapes get painted red. If every gray shape has some red, unselected shape
    # in the same row (same rank_bottom), that red shape may be the "instruction" for it, so propose:
    #
    # for shape in all( color=gray ):                                 # same selector as before
    #     other = one( left=0, rank_bottom=shape.rank_bottom )        # the joined instruction shape
    #     shape.paint( color=other.color )                            # read the value off it

    candidates:set = set()
    for program_result in parent_program_results: # usually ~5
        program:str = program_result['program']

        # Which pair did this program do the best in?
        pair_scores:List[float] = program_result['pair_scores']
        best_pair_index:int = pair_scores.index( min(pair_scores) )

        # For each match of that for-all-shapes pattern...
        for match in _FOR_ALL_PATTERN.finditer(program): # usually ~3

            # That pair's shapes just before this block runs (earlier blocks may have changed them)
            best_scene_shapes:List[Dict] = scenes_before_block(program, match.start())[best_pair_index]['shapes']

            shape_selectors:Dict = _argstring_to_dict( match.group(1) )
            action_method:str = match.group(2)
            action_parameters:Dict = _argstring_to_dict( match.group(3) )

            # Targets: the shapes this block acts on, in that pair's scene
            target_shapes:List[Dict] = _select_shape_datas( best_scene_shapes, shape_selectors )
            if not target_shapes:
                continue # Nothing to join to

            # Holders can't be targets themselves (id()-set: shapes are dicts, so unhashable)
            target_ids:set = { id(shape) for shape in target_shapes }
            unselected_shapes:List[Dict] = [shape for shape in best_scene_shapes if id(shape) not in target_ids]

            # Join candidates: any public property of the targets
            target_properties:List[str] = [key for key in target_shapes[0].keys() if not key.startswith('_')]

            for action_param_k,action_param_v in action_parameters.items():
                if not _is_hardcoded_value(action_param_v):
                    continue

                # Holders: unselected shapes holding the action value (or its negative) in any property.
                # Remember WHICH properties hold it (other.color, other.start_color, -other.area...), reused below.
                references_by_holder:Dict[int,List[str]] = {}
                for shape in unselected_shapes:
                    references:List[str] = _references_to_value(shape, action_param_v, 'other')
                    if references:
                        references_by_holder[id(shape)] = references
                holder_shapes:List[Dict] = [shape for shape in unselected_shapes if id(shape) in references_by_holder]
                if not holder_shapes:
                    continue

                # TODO MAYBE: also try P = shape.P + 1 / - 1
                for join_property in target_properties:

                    # Joining on the action key itself would make other.KEY == shape.KEY, so the action's a no-op
                    if join_property == action_param_k:
                        continue

                    # Every target needs at least one holder with an EQUAL join property (each target may have its own)
                    matched_holders:Dict[int,Dict] = {}
                    every_target_matched:bool = True
                    for target in target_shapes:
                        # Some properties are optional (e.g. only lines have a subtype): a target without it can't match
                        if join_property not in target:
                            every_target_matched = False
                            break
                        target_value:Any = target[join_property]
                        holders_for_target:List[Dict] = [holder for holder in holder_shapes if join_property in holder and _is_equal_value(holder[join_property], target_value)]
                        if not holders_for_target:
                            every_target_matched = False
                            break
                        for holder in holders_for_target:
                            matched_holders[id(holder)] = holder
                    if not every_target_matched:
                        continue

                    # Which of the holders' properties hold the action value?
                    # Union over all matched holders, deduped but in order, so the proposals stay deterministic.
                    other_action_references:List[str] = list(dict.fromkeys(
                        reference for holder_id in matched_holders for reference in references_by_holder[holder_id]
                    ))

                    # Select the holders, REQUIRING the join property (it becomes shape.P below),
                    # but never by the action key: one(color="c3").color is just the hard-coded value again
                    other_selectors:List[Dict] = _propose_selectors(list(matched_holders.values()), best_scene_shapes, props_to_exclude=[action_param_k], props_required=[join_property], N=10)

                    # Which shapes get the joined action? With the block's own selector as-is, the join paints the
                    # same pixels as its parent for more tokens, so it'd lose before delete_arguments could generalize it.
                    # So also propose the generalized versions right away: minus the join property (all(top=3) joined
                    # on top), and all() (the join explains EVERY shape). Deduped, in order, so proposals stay deterministic.
                    shape_selectors_strings:List[str] = [match.group(1)]
                    if join_property in shape_selectors:
                        shape_selectors_strings.append( _dict_to_argstring({key:value for key,value in shape_selectors.items() if key != join_property}) )
                    shape_selectors_strings.append('')
                    shape_selectors_strings = list(dict.fromkeys(shape_selectors_strings))

                    for other_selector in other_selectors: # Usually ~10?

                        # The join: the holder's P value becomes the target's own shape.P
                        other_selectors_string:str = _selector_to_string_with_refs(other_selector, {join_property: f'shape.{join_property}'})

                        # One candidate per property holding the action value
                        for other_action_reference in other_action_references:
                            action_parameters_string:str = _selector_to_string_with_refs(action_parameters, {action_param_k: other_action_reference})

                            # FINALLY, suggest the replacement! (Splice into THIS match's span only)
                            for shape_selectors_string in shape_selectors_strings:
                                the_edited_loop:str = ''
                                the_edited_loop += f'for shape in all({shape_selectors_string}):\n'
                                the_edited_loop += f'\tother = one({other_selectors_string})\n'
                                the_edited_loop += f'\tshape.{action_method}({action_parameters_string})'
                                new_program = (program[:match.start()] + the_edited_loop + program[match.end():]).strip()
                                candidates.add(new_program)

    # Gimme
    return candidates

# (TODO: IS THIS FRAGILE???) ⚠️ TO ACTUALLY REVIEW
def any_coincidence(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    
    # Same as above, but doesn't worry about joining at all.
    #
    # For example, this:
    # 
    # for shape in all():
    #     shape.place(top=5)
    #
    # Becomes: (if there's a blue shape with top=5)
    #
    # for shape in all():
    #     other = one(color=blue)
    #     shape.place(top=other.top)
    
    candidates:set = set()
    for program_result in parent_program_results:
        program:str = program_result['program']

        # Which pair did this program do the best in?
        pair_scores:List[float] = program_result['pair_scores']
        best_pair_index:int = pair_scores.index( min(pair_scores) )

        # For each match of that for-all-shapes pattern...
        for match in _FOR_ALL_PATTERN.finditer(program): # usually ~3

            # That pair's shapes just before this block runs (earlier blocks may have changed them)
            best_scene_shapes:List[Dict] = scenes_before_block(program, match.start())[best_pair_index]['shapes']

            action_method:str = match.group(2)
            action_parameters:Dict = _argstring_to_dict( match.group(3) )

            # For each hardcoded value in action-values...
            action_key_values:List[Tuple] = []
            for action_param_k,action_param_v in action_parameters.items():
                if _is_hardcoded_value(action_param_v):
                    action_key_values.append( (action_param_k,action_param_v) )
            for akv in action_key_values:
                action_param_k,action_param_v = akv # unpack

                # Find candidates: has hardcoded value (or its negative) in *any* key.
                candidate_others:List[Dict] = []
                for shape in best_scene_shapes:
                    if _references_to_value(shape, action_param_v, 'other'):
                        candidate_others.append(shape)

                for other in candidate_others: # usually ~5 ?

                    # ACTION-params: the targeted hard-coded value becomes other.[key holding it] (or -other.[key]).
                    # Depends only on `other` (not other_selector), so hoisted out of the inner loop.
                    # Try EVERY key holding the value, not just the first. E.g. if the other shape has right=7 AND
                    # bottom=7, place(bottom=7) shouldn't only become place(bottom=other.right), which can be wrong on other pairs.
                    other_references:List[str] = _references_to_value(other, action_param_v, 'other')

                    # MAYBE-TODO: pass props_to_exclude=[<key holding action_param_v>] here.
                    # Without it, propose_selectors can pick that very key as the selector, giving
                    # circular candidates like  other = one(top=5) ... shape.place(top=other.top)
                    # (selects other BY top=5, then reads other.top back = 5: a no-op, just noise).
                    # Leaving it off for now; will add if that noise hurts performance.
                    other_selectors:List[Dict] = _propose_selectors([other], best_scene_shapes, N=10)
                    for other_selector in other_selectors: # Usually ~10?

                        # One candidate per key holding the value (other.bottom, other.right, -other.area, ...)
                        for other_reference in other_references:
                            action_parameters_string:str = _selector_to_string_with_refs(action_parameters, {action_param_k: other_reference})

                            # Make the substitution, e.g.:
                            # for shape in all():                                 # same selectors
                            #     other = one(color=blue)                         # just select
                            #     shape.place(top=other.top)                      # REPLACE hard-coded value, [action_value] becomes other.[key holding it]

                            # Also try it for ALL shapes: if the other shape explains the value, the parent's
                            # selector was probably overfit too (e.g. an all(area=6) that only fits one pair).
                            # Without this, the search early-stops before delete_arguments gets to drop it.
                            # (OK for now, but it ~doubles this monkey's proposals. May address later!)
                            for shape_selectors_string in {match.group(1), ''}:

                                # FINALLY, suggest the replacement! (splice into THIS match's span only)
                                the_edited_loop:str = ''
                                the_edited_loop += f'for shape in all({shape_selectors_string}):\n'
                                the_edited_loop += f'\tother = one({ _dict_to_argstring(other_selector) })\n'
                                the_edited_loop += f'\tshape.{action_method}({action_parameters_string})'
                                new_program = (program[:match.start()] + the_edited_loop + program[match.end():]).strip()
                                candidates.add(new_program)
    
    # Gimme
    return candidates


def self_coincidence(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    
    # Same as above, but maybe the secret was within you the whole time?
    #
    # For example, this:
    # 
    # for shape in all(area=5):     # BTW, TODO FOR ALL: what if a multiple of a value, like twice?
    #     shape.move(y=5)           # or +/- 1,2,3?
    #
    # Becomes:
    #
    # for shape in all():
    #     shape.move(y=shape.area)
    #
    # (And a move(y=-5) becomes move(y=-shape.area): negatives count too.)
    
    candidates:set = set()
    for program_result in parent_program_results:
        program:str = program_result['program']

        # Which pair did this program do the best in?
        pair_scores:List[float] = program_result['pair_scores']
        best_pair_index:int = pair_scores.index( min(pair_scores) )

        # For each match of that for-all-shapes pattern...
        for match in _FOR_ALL_PATTERN.finditer(program): # usually ~3

            # That pair's shapes just before this block runs (earlier blocks may have changed them)
            best_scene_shapes:List[Dict] = scenes_before_block(program, match.start())[best_pair_index]['shapes']

            shape_selectors:Dict = _argstring_to_dict( match.group(1) )
            action_method:str = match.group(2)
            action_parameters:Dict = _argstring_to_dict( match.group(3) )

            # Get shapes affected by this
            selected_shapes:List[Dict] = _select_shape_datas( best_scene_shapes, shape_selectors )

            # Get hardcoded values in action params...
            action_key_values:List[Tuple] = []
            for action_param_k,action_param_v in action_parameters.items():
                if _is_hardcoded_value(action_param_v):
                    action_key_values.append( (action_param_k,action_param_v) )
            
            # For each shape & hard-coded value
            for self_shape in selected_shapes:
                for akv in action_key_values:
                    action_param_k,action_param_v = akv # unpack

                    # Find candidate keys in self: all keys in self that hold the hardcoded value (or its negative)
                    candidate_references:List[str] = _references_to_value(self_shape, action_param_v, 'shape')
                    for candidate_reference in candidate_references:
                        candidate_key:str = candidate_reference.split('.')[-1]

                        # REPLACE hard-coded value: [action_value] becomes shape.[candidate_key] (or -shape.[candidate_key])
                        action_parameters_string:str = _selector_to_string_with_refs(action_parameters, {action_param_k: candidate_reference})

                        # If the coincidence is ON a selector key, that selector now just
                        # restates what we read back (a no-op) -- so DROP it, to generalize.
                        # e.g. all(area=5): move(y=5)  ->  all(): move(y=shape.area)
                        # (When candidate_key is a selector key, a *selected* shape's value for it
                        #  always equals the selector's value, so "in shape_selectors" is enough.)
                        if candidate_key in shape_selectors:
                            generalized:Dict = {k:v for k,v in shape_selectors.items() if k!=candidate_key}
                            selectors_string:str = _dict_to_argstring(generalized)
                        else:
                            selectors_string = match.group(1) # not a selector key -> keep as-is

                        # FINALLY, suggest the replacement! (splice into THIS match's span only)
                        the_edited_loop:str = ''
                        the_edited_loop += f'for shape in all({selectors_string}):\n'
                        the_edited_loop += f'\tshape.{action_method}({action_parameters_string})'
                        new_program = (program[:match.start()] + the_edited_loop + program[match.end():]).strip()
                        candidates.add(new_program)
    
    # Gimme
    return candidates

def center_coincidence(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    """Turn a hard-coded place(center_x=X, center_y=Y) into the exact center of some other shape, or of the scene."""

    # For example, this:
    #
    # for shape in all(area=1):
    #     shape.place(center_x=6.5,center_y=6.5)
    #
    # Becomes: (if the big frame's center is exactly 6.5, 6.5)
    #
    # for shape in all(area=1):
    #     other = one(rank_area=0)
    #     shape.place(center_x=other.center_x,center_y=other.center_y)
    #
    # (any_coincidence only swaps ONE value at a time, and a block with an "other" line can't be swapped again,
    # so it could never move a shape onto another shape's center on both axes. This swaps both together.)

    candidates:set = set()
    for program_result in parent_program_results:
        program:str = program_result['program']

        # Which pair did this program do the best in?
        pair_scores:List[float] = program_result['pair_scores']
        best_pair_index:int = pair_scores.index( min(pair_scores) )

        # For each place() block with a hard-coded center on both axes...
        for match in _FOR_ALL_PATTERN.finditer(program):
            if match.group(2) != 'place':
                continue
            action_parameters:Dict = _argstring_to_dict( match.group(3) )
            center_x:Any = action_parameters.get('center_x')
            center_y:Any = action_parameters.get('center_y')
            if not (_is_number(center_x) and _is_number(center_y)):
                continue

            # That pair's scene just before this block runs (earlier blocks may have changed it)
            best_scene:Dict = scenes_before_block(program, match.start())[best_pair_index]

            # The scene's own center? Then read it off the scene
            if _is_equal_value(best_scene['center_x'], center_x) and _is_equal_value(best_scene['center_y'], center_y):
                action_parameters_string:str = _selector_to_string_with_refs(action_parameters, {'center_x': 'scene.center_x', 'center_y': 'scene.center_y'})
                the_edited_loop:str = f'for shape in all({match.group(1)}):\n\tshape.place({action_parameters_string})'
                candidates.add( (program[:match.start()] + the_edited_loop + program[match.end():]).strip() )

            # Some shape's center? Then select that shape as "other" and read both off it (any_coincidence-style)
            action_parameters_string = _selector_to_string_with_refs(action_parameters, {'center_x': 'other.center_x', 'center_y': 'other.center_y'})
            for other in best_scene['shapes']:
                if not (_is_equal_value(other['center_x'], center_x) and _is_equal_value(other['center_y'], center_y)):
                    continue
                for other_selector in _propose_selectors([other], best_scene['shapes'], N=10):

                    # Also try it for ALL shapes, like any_coincidence: if the other shape explains the
                    # center, the parent's selector was probably overfit too
                    for shape_selectors_string in {match.group(1), ''}:
                        the_edited_loop = ''
                        the_edited_loop += f'for shape in all({shape_selectors_string}):\n'
                        the_edited_loop += f'\tother = one({ _dict_to_argstring(other_selector) })\n'
                        the_edited_loop += f'\tshape.place({action_parameters_string})'
                        candidates.add( (program[:match.start()] + the_edited_loop + program[match.end():]).strip() )

    # Gimme
    return candidates

def scene_prop_coincidence(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:

    # Just find all hardcoded values, and replace them with scene.property

    candidates:set = set()
    for program_result in parent_program_results:
        program:str = program_result['program']

        # Which pair did this program do the best in?
        pair_scores:List[float] = program_result['pair_scores']
        best_pair_index:int = pair_scores.index( min(pair_scores) )

        # For each match a hard-coded value ("quoted string" or int or float, incl negatives)
        for match in _HARDCODED_VALUE_PATTERN.finditer(program):
            hard_coded_value:Any = _clean_arg_value( match.group(0) )

            # Skip numbers inside an array/list literal [...] (any depth, any call),
            # e.g. the colors in replace_pixels' pixel grids. Those are grid data, not params,
            # and swapping a 5 in there for scene.height is just a coincidence.
            # (Bracket depth = how many [ minus ] come before this match.)
            text_before_match:str = program[:match.start()]
            bracket_depth:int = text_before_match.count('[') - text_before_match.count(']')
            if bracket_depth > 0:
                continue

            if hard_coded_value is not None and _is_hardcoded_value(hard_coded_value):

                # Every pair's scene just before this value's block runs (earlier blocks may have changed them)
                scenes:List[Dict] = scenes_before_block(program, match.start())
                best_scene:Dict = scenes[best_pair_index]

                # Search scene props that'd match (or match its negative)...
                for scene_reference in _references_to_value(best_scene, hard_coded_value, 'scene'):

                    # Not a number (a color, say)? Then every pair's scene must hold that kind of value too. Some scene
                    # props only exist on some scenes, and a missing one reads as 0: not a color, so the program crashes
                    if not _is_number(hard_coded_value):
                        prop_name:str = scene_reference.removeprefix('scene.')
                        if not all(type(scene.get(prop_name)) == type(hard_coded_value) for scene in scenes):
                            continue

                    # A match? Propose an edit! (splice into THIS match's span only)
                    new_program = (program[:match.start()] + scene_reference + program[match.end():]).strip()
                    candidates.add(new_program)

    # Gimme
    return candidates





########################################################################
# Tiny helpers #########################################################
########################################################################

def _is_number(value:Any)->bool:
    # type(), not isinstance(), because isinstance(True, int) is True
    return type(value) in (int,float)

def _is_hardcoded_value(value:Any)->bool:
    # Should this value get swapped for a reference? Numbers only when |value| >= 2:
    # a 0 or 1 matching some property is nearly always a coincidence. Everything else: yes.
    if _is_number(value):
        return abs(value) >= 2
    return True

def _references_to_value(properties:Dict, value:Any, owner_name:str) -> List[str]:
    # Every property holding the value, as a reference string like 'other.top'
    references:List[str] = [f'{owner_name}.{key}' for key,property_value in properties.items() if _is_equal_value(property_value,value)]

    # Numbers: also properties holding its NEGATIVE, e.g. move(y=-3) with area=3 -> '-shape.area'
    if _is_number(value):
        references += [f'-{owner_name}.{key}' for key,property_value in properties.items() if _is_equal_value(property_value,-value)]
    return references

def _is_equal_value(a:Any,b:Any)->bool:
    # Because we do NOT want 1==True, but we DO want 1.0==1.
    # Conditionals, in this order, to short-circuit
    # And not isinstance becase isinstance(True, int) is True. Gah, Python.
    # Skip numpy arrays: they're grids, not scalar values to compare as coincidences.
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return False
    return a==b and (type(a)==type(b) or (type(a)==int and type(b)==float) or (type(b)==int and type(a)==float))

def _selector_to_string_with_refs(selector:Dict, refs:Dict[str,str]) -> str:
    """
    Render a selector dict to a 'k=v,k=v' string, BUT for any key present in `refs`,
    emit its raw reference expression (UNquoted) instead of the literal value. e.g.
        selector={'type':'line', 'color':'green'}, refs={'color':'shape.color'}
        ->  'type="line",color=shape.color'
    Used to turn a hard-coded value into a live reference (shape.X / other.X), without
    the quoting/substring hazards of a raw string .replace().
    """
    parts:List[str] = []
    for k,v in selector.items():
        if k in refs:
            parts.append( f'{k}={refs[k]}' )            # raw reference, no quotes
        else:
            parts.append( f'{k}={_add_quotes_if_string(v)}' ) # literal value
    return ','.join(parts)

def _add_quotes_if_string(value:Any)->str:
    if isinstance(value,str):
        return '"' + value + '"'
    else:
        return str(value)

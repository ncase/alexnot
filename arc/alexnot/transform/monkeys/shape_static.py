# (NOT ACTUALLY CURRENTLY USED; WIP): the whole file is one string for now, so none of it runs
"""
from typing import Dict, List, Optional
from ...utils import _propose_selectors, _dict_to_argstring
from .monkey_helpers import metric

###################################################
### SHAPE STATIC (pin unchanged properties) #######
###################################################

_STATIC_PROP_CHECKS = [
    ('left', lambda inp, out, pred: inp['left'] == out['left'] and pred['left'] != inp['left']),
    ('top',  lambda inp, out, pred: inp['top'] == out['top'] and pred['top'] != inp['top']),
    ('color', lambda inp, out, pred: inp.get('color') == out.get('color') and pred.get('color') != inp.get('color')),
]

def shapes_static(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates:set = set()

    for program_result in parent_program_results:
        program:str = program_result['program']
        if not program.strip():
            continue

        # For each training pair, find shapes whose props are same in input/output
        # but different in predicted (the program wrongly changed them)
        shape_static_info:list = []  # (pair_index, input_shape, static_props)

        for pair_index, pair in enumerate(train_pairs):
            input_shapes:List[Dict] = pair['input']['shapes']
            predicted_shapes:List[Dict] = program_result['predicted_scenes'][pair_index]['shapes']
            output_shapes:List[Dict] = pair['output']['shapes']

            if not output_shapes or not predicted_shapes:
                continue

            for input_shape in input_shapes:
                if input_shape.get('hidden'):
                    continue

                output_match = min(output_shapes, key=lambda s: metric(input_shape, s))
                predicted_match = min(predicted_shapes, key=lambda s: metric(input_shape, s))

                static_props:set = set()
                for prop, check in _STATIC_PROP_CHECKS:
                    if check(input_shape, output_match, predicted_match):
                        static_props.add(prop)

                if static_props:
                    shape_static_info.append((pair_index, input_shape, static_props))

        if not shape_static_info:
            continue

        # Collect all unique prop-sets to try: each individual prop + the full combination
        all_prop_sets:set = set()
        for _, _, props in shape_static_info:
            for p in props:
                all_prop_sets.add(frozenset([p]))
            if len(props) > 1:
                all_prop_sets.add(frozenset(props))

        for prop_set in all_prop_sets:
            # Shapes that have ALL these props static
            pair_ingroups:Dict[int, List[Dict]] = {}
            for pair_index, shape, props in shape_static_info:
                if prop_set <= props:
                    pair_ingroups.setdefault(pair_index, []).append(shape)

            if not pair_ingroups:
                continue

            static_arg:str = ','.join(f'{p}=True' for p in sorted(prop_set))

            # Propose selectors (within each pair, and across all pairs)
            selectors:List[Dict] = []
            for pair_index, ingroup in pair_ingroups.items():
                all_shapes:List[Dict] = train_pairs[pair_index]['input']['shapes']
                selectors += _propose_selectors(ingroup, all_shapes, N=100)

            if len(pair_ingroups) >= 2:
                combined_ingroup:List[Dict] = [s for shapes in pair_ingroups.values() for s in shapes]
                combined_all:List[Dict] = [s for pi in pair_ingroups for s in train_pairs[pi]['input']['shapes']]
                selectors += _propose_selectors(combined_ingroup, combined_all, N=100)

            unique_selectors:Dict[str,Dict] = { _dict_to_argstring(s): s for s in selectors }
            for selector_string in unique_selectors:
                code:str = f'for shape in all({selector_string}):\n\tshape.static({static_arg})\n{program}'
                candidates.add(code.strip())

    return candidates

"""
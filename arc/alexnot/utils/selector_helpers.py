# from __future__ import annotations
import json
from typing import List, Dict, Optional, Any
import random
import statistics
from collections import OrderedDict
import numpy as np

# CONFIG KNOBS
from ..config import SELECTOR_KEY_COST
from ..config import SELECTOR_MAX_NOT_PROPOSALS
# SINGLET_PROPS:List[str] = ["start_color","color","section"]


#################################################################################
#################################################################################


# The filter rows of the last few (ingroup, outgroup) splits, so repeat calls on the SAME shapes skip rebuilding them.
# Keyed by the shapes' ids, in order. Each entry also holds the ingroup & outgroup lists themselves:
# while a shape is referenced here it can't be freed, so its id can't get reused by a different shape.
# (Only the last few are kept, so memory and old shapes can't pile up)
_FilterMaskCacheEntry = tuple[list[dict], list[dict], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]
_FILTER_MASK_CACHE:OrderedDict[tuple[tuple[int, ...], tuple[int, ...]], _FilterMaskCacheEntry] = OrderedDict()
_FILTER_MASK_CACHE_SIZE:int = 8


def _propose_selectors(ingroup:list[dict], all_shapes:list[dict], N:int, props_to_exclude:list[str]|None=None, props_required:list[str]|None=None) -> list[dict]:
    """

    ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
    ✨ THE BONGARD FUNCTION ✨
    ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨

    Rewrite! Here's the plan:
    - go through all possible selectors
      (w.l.o.g., take 1st shape from ingroup, test empty set + all single-prop + all double-props)
      (use props_to_exclude & props_required if passed into parameters)
    - score them all by # ingroup selected - # outgroup selected
    - take top N
    - plus up to N single-prop selectors that grab the whole ingroup, ignoring the outgroup
    """

    # No ingroup? Just return empty selector
    if len(ingroup) == 0:
        return [{}]

    # The ingroup can list the same shape many times over (a batcher adds an actor once per move it could make,
    # tens of thousands of times on a big scene). Every check below runs on each DIFFERENT shape just once,
    # then counts it as many times as it's listed (its weight), so the scores come out exactly the same, just faster
    ingroup_weight_by_id:dict[int, int] = {}
    unique_ingroup:list[dict] = []
    for shape in ingroup:
        if id(shape) not in ingroup_weight_by_id:
            ingroup_weight_by_id[id(shape)] = 0
            unique_ingroup.append(shape)
        ingroup_weight_by_id[id(shape)] += 1
    ingroup_weights:np.ndarray = np.array([ingroup_weight_by_id[id(shape)] for shape in unique_ingroup], dtype=np.int32)

    # Split into ingroup vs outgroup (by identity, since numpy arrays make == ambiguous)
    outgroup:list[dict] = [s for s in all_shapes if id(s) not in ingroup_weight_by_id]

    # Pick random shape instead of always first (forces more explicit selectors)
    the_reference_ingroup_shape:dict = random.choice(ingroup)

    # Get all public properties that aren't excluded (a helper, so the NOT selectors can reuse it on an OUTGROUP shape)
    public_props:list[str] = _public_props(the_reference_ingroup_shape, props_to_exclude)

    # The candidates are: empty + all singles + all doubles of the reference shape's props (scored all at once, below),
    # then the range & NOT selectors (gathered here, one key each)
    extra_selectors:list[dict] = []

    ########################################################
    # Range selectors
    ########################################################

    # First, to find all numerical properties, look at the_reference_ingroup_shape's keys and find the numerical ones
    # (SKIP bools! In Python, True/False count as ints. Also check per value: e.g. touching_color_* can mix -1 and 'cN')
    # Then, for each numerical property, go through its values for all ingroup shapes & outgroup shapes,
    # And propose a _GTE xor _LTE that best separates them.
    # NOTE that it doesn't have to be a perfect division (the below 'scoring' mechanism handles that)
    # So ideally, if all ingroup values < all outgroup values (ingroup is the SMALL side),
    # then propose _LTE = the ingroup's max (e.g. area_LTE=2)
    # If the median ingroup < median outgroup *but the biggest ingroup is larger than smallest outgroup*,
    # *go ahead and propose _LTE = the ingroup's max *anyway*.
    # (reverse the above for _GTE: if ingroup is the BIG side, propose _GTE = the ingroup's min, e.g. area_GTE=4)
    # (Use the ingroup's own min/max, NOT a rounded midpoint: Python's round() is banker's rounding,
    # and a midpoint can cross into the outgroup or misgeneralize, e.g. "four or more" becoming area_GTE=3)
    # NOTE: Range selectors are single-prop only, and skipped when props_required is set
    # (a "holes_GTE" key wouldn't count as having the required "holes" key, same as the filter above)

    if not props_required:
        for prop in public_props:
            # Only numbers (bools are ints in Python, so skip those explicitly)
            if not _is_number(the_reference_ingroup_shape[prop]):
                continue

            # Gather this prop's numerical values for ingroup/outgroup (each ingroup value repeated by its weight, for the median)
            ingroup_values:list[float] = []
            for shape, weight in zip(unique_ingroup, ingroup_weights.tolist()):
                if _is_number(shape.get(prop)):
                    ingroup_values.extend([shape[prop]] * weight)
            outgroup_values:list[float] = [shape[prop] for shape in outgroup if _is_number(shape.get(prop))]
            # Nothing to separate from? Then a range selector makes no sense, skip
            if len(outgroup_values) == 0:
                continue

            # Ingroup's the SMALL side: everything up to the ingroup's biggest value
            ingroup_median:float = statistics.median(ingroup_values)
            outgroup_median:float = statistics.median(outgroup_values)
            if ingroup_median < outgroup_median:
                extra_selectors.append({f'{prop}_LTE': max(ingroup_values)})
            # Ingroup's the BIG side: everything from the ingroup's smallest value up
            elif ingroup_median > outgroup_median:
                extra_selectors.append({f'{prop}_GTE': min(ingroup_values)})
            # (Same medians? no clear side, so no range selector for this prop)

    ########################################################
    # NOT selectors
    ########################################################

    # Flip ingroup & outgroup: a single-prop selector prop=v that grabs EVERY outgroup shape and NO ingroup
    # shape means the ingroup is exactly "everyone whose prop isn't v", so propose prop_NOT=v (e.g. color_NOT="c4").
    # (Any outgroup shape works as the reference: a prop=v matching ALL of the outgroup has to come from each one.)
    # Not a recursive _propose_selectors call, just its single-prop step, so no NOTs-of-NOTs & no infinite recursion.
    # Single-prop only and skipped with props_required, same as the range selectors above.
    if not props_required and len(outgroup) > 0:
        the_reference_outgroup_shape:dict = outgroup[0]
        not_selectors_added:int = 0
        for prop in _public_props(the_reference_outgroup_shape, props_to_exclude):
            # At most SELECTOR_MAX_NOT_PROPOSALS of these per call
            if not_selectors_added >= SELECTOR_MAX_NOT_PROPOSALS:
                break
            value:Any = the_reference_outgroup_shape[prop]
            # Does prop=value grab every outgroup shape...
            if not all(_shape_passes_filter(shape, prop, value) for shape in outgroup):
                continue
            # ...and no ingroup shape?
            if any(_shape_passes_filter(shape, prop, value) for shape in unique_ingroup):
                continue
            extra_selectors.append({f'{prop}_NOT': value})
            not_selectors_added += 1

    # Which shapes pass each single (key, value) filter, as 0/1 rows (entry i = the i-th unique ingroup / outgroup shape).
    # Each filter gets checked against every shape just ONCE, then all the singles & doubles get counted from those rows.
    # Seen these exact shapes recently? Then reuse their rows (only "who passes what", never the weights or scores)
    split_key:tuple[tuple[int, ...], tuple[int, ...]] = (tuple(id(shape) for shape in unique_ingroup), tuple(id(shape) for shape in outgroup))
    # (Pop + re-insert moves it to the newest end. Single dict ops only, so a leftover timed-out attempt thread
    # sharing this cache can't make a lookup crash halfway)
    cache_entry:_FilterMaskCacheEntry|None = _FILTER_MASK_CACHE.pop(split_key, None)
    if cache_entry is None:
        # Copies of the lists, so the shapes stay referenced even if the caller empties its own list later
        cache_entry = (list(unique_ingroup), list(outgroup), {})
    _FILTER_MASK_CACHE[split_key] = cache_entry
    # Too many? Drop the least recently used split
    if len(_FILTER_MASK_CACHE) > _FILTER_MASK_CACHE_SIZE:
        _FILTER_MASK_CACHE.popitem(last=False)
    filter_rows:dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = cache_entry[2]
    def _filter_rows(key:str, value:Any) -> tuple[np.ndarray, np.ndarray]:
        # repr() so unhashable or mixed-type values (lists, -1 vs 'c3'...) still make a safe dict key
        cache_key:tuple[str, str] = (key, repr(value))
        if cache_key not in filter_rows:
            # int32 0/1s: the counts below never go past the number of shapes, so no overflow and no floats
            in_row:np.ndarray = np.array([_shape_passes_filter(shape, key, value) for shape in unique_ingroup], dtype=np.int32)
            out_row:np.ndarray = np.array([_shape_passes_filter(shape, key, value) for shape in outgroup], dtype=np.int32)
            filter_rows[cache_key] = (in_row, out_row)
        return filter_rows[cache_key]

    # Stack the reference shape's single filters into matrices (rows = filters, columns = shapes)
    prop_count:int = len(public_props)
    single_rows:list[tuple[np.ndarray, np.ndarray]] = [_filter_rows(prop, the_reference_ingroup_shape[prop]) for prop in public_props]
    in_matrix:np.ndarray = np.array([in_row for in_row, _ in single_rows], dtype=np.int32).reshape(prop_count, len(unique_ingroup))
    out_matrix:np.ndarray = np.array([out_row for _, out_row in single_rows], dtype=np.int32).reshape(prop_count, len(outgroup))

    # One matrix multiply counts EVERY pair at once: entry (i, j) = how many shapes pass both filter i and filter j.
    # (The diagonal, i = j, is just the singles. Ingroup shapes count as many times as they're listed: their weights)
    in_pair_counts:np.ndarray = (in_matrix * ingroup_weights) @ in_matrix.T
    out_pair_counts:np.ndarray = out_matrix @ out_matrix.T
    single_props:np.ndarray = np.arange(prop_count)
    # Each pair once, in the same order itertools.combinations gives: (0,1), (0,2)... (1,2)...
    double_firsts, double_seconds = np.triu_indices(prop_count, k=1)

    # The empty selector comes first, and selects everyone
    empty_count:int = 1
    # If props_required is set, keep only selectors that include ALL required props (and so never the empty one).
    # A selector's props are all different, so "has all of them" = "has as many required props as there are"
    if props_required:
        required_props:set[str] = set(props_required)
        is_required:np.ndarray = np.array([prop in required_props for prop in public_props], dtype=np.int32).reshape(prop_count)
        empty_count = 0
        single_props = single_props[is_required == len(required_props)]
        keep_doubles:np.ndarray = (is_required[double_firsts] + is_required[double_seconds]) == len(required_props)
        double_firsts = double_firsts[keep_doubles]
        double_seconds = double_seconds[keep_doubles]

    # The range & NOT selectors are one key each, so they count straight from their own rows
    extra_rows:list[tuple[np.ndarray, np.ndarray]] = [_filter_rows(*next(iter(selector.items()))) for selector in extra_selectors]

    # Every candidate in order (empty, singles, doubles, range & NOT): # ingroup matched, # outgroup matched, # keys
    in_hits:np.ndarray = np.concatenate([
        np.full(empty_count, len(ingroup), dtype=np.int64),
        np.diagonal(in_pair_counts)[single_props],
        in_pair_counts[double_firsts, double_seconds],
        np.array([in_row @ ingroup_weights for in_row, _ in extra_rows], dtype=np.int64),
    ]).astype(np.int64)
    out_hits:np.ndarray = np.concatenate([
        np.full(empty_count, len(outgroup), dtype=np.int64),
        np.diagonal(out_pair_counts)[single_props],
        out_pair_counts[double_firsts, double_seconds],
        np.array([out_row.sum() for _, out_row in extra_rows], dtype=np.int64),
    ]).astype(np.int64)
    key_counts:np.ndarray = np.concatenate([
        np.zeros(empty_count, dtype=np.int64),
        np.ones(len(single_props), dtype=np.int64),
        np.full(len(double_firsts), 2, dtype=np.int64),
        np.ones(len(extra_selectors), dtype=np.int64),
    ])

    # Score each candidate: # ingroup matched - # outgroup matched,
    # with a small regularization penalty preferring fewer keys (0.2 per key)
    # Random jitter for tiebreakers (scores are integers minus small penalty, so ±0.01 never swaps ranks).
    # One draw per candidate, in candidate order, with the same math as random.uniform(-0.01, 0.01): low + (high - low) * random()
    jitter_low:float = -0.01
    jitter_high:float = 0.01
    random_draws:np.ndarray = np.array([random.random() for _ in range(len(key_counts))], dtype=np.float64)
    scores:np.ndarray = (in_hits - out_hits).astype(np.float64) - SELECTOR_KEY_COST * key_counts + (jitter_low + (jitter_high - jitter_low) * random_draws)

    # Sort descending by score, take top N. (Stable, so exact ties keep candidate order, same as Python's sort)
    top_indices:list[int] = np.argsort(-scores, kind='stable')[:N].tolist()
    singles_end:int = empty_count + len(single_props)
    doubles_end:int = singles_end + len(double_firsts)

    # Plus up to N more single-prop selectors that grab the WHOLE ingroup, no matter how much outgroup they also grab.
    # (E.g. only some vertical lines change because the rest are already right: type="line" still explains them all,
    # but the score above sinks it for grabbing the no-op ones.) Picked at random, since the outgroup doesn't count here.
    # Skip singles that also grab the WHOLE outgroup: they grab every shape, so they're just a longer way to write all()
    already_proposed:set[int] = set(top_indices)
    single_in_hits:np.ndarray = in_hits[empty_count:singles_end]
    single_out_hits:np.ndarray = out_hits[empty_count:singles_end]
    grabs_whole_ingroup_only:np.ndarray = (single_in_hits == len(ingroup)) & (single_out_hits < len(outgroup))
    whole_ingroup_indices:list[int] = (np.flatnonzero(grabs_whole_ingroup_only) + empty_count).tolist()
    whole_ingroup_singles:list[int] = [index for index in whole_ingroup_indices if index not in already_proposed]
    random.shuffle(whole_ingroup_singles)
    top_indices += whole_ingroup_singles[:N]

    # Only now build the actual selector dicts, just for the winners
    top_selectors:list[dict] = []
    for index in top_indices:
        if index < empty_count:
            top_selectors.append({})
        elif index < singles_end:
            prop:str = public_props[single_props[index - empty_count]]
            top_selectors.append({prop: the_reference_ingroup_shape[prop]})
        elif index < doubles_end:
            first_prop:str = public_props[double_firsts[index - singles_end]]
            second_prop:str = public_props[double_seconds[index - singles_end]]
            top_selectors.append({first_prop: the_reference_ingroup_shape[first_prop], second_prop: the_reference_ingroup_shape[second_prop]})
        else:
            top_selectors.append(extra_selectors[index - doubles_end])
    return top_selectors




#################################################################################
#################################################################################
#################################################################################



def _dict_to_dictstring(selector:Dict) -> str:
    # Build Python-valid dict string instead of json.dumps,
    # so booleans come out as True/False (not true/false) and strings use double quotes
    parts:List[str] = []
    for key, value in selector.items():
        if isinstance(value, bool):
            # Python booleans: True/False, not json's true/false
            parts.append(f'"{key}": {value}')
        elif isinstance(value, str):
            # Double quotes
            parts.append(f'"{key}": "{value}"')
        else:
            parts.append(f'"{key}": {value}')
    return "{" + ", ".join(parts) + "}"

def _dict_to_argstring(selector:Dict) -> str:
    """
    Convert a selector-dict into a selector-string, e.g.
        {'color':'c2', 'width':1, 'area':5}  ->  'color="c2",width=1,area=5'
    (An empty dict -> '', which means "select everything".)
    """
    parts:List[str] = []
    for k,v in selector.items():
        if isinstance(v,str):
            parts.append(f'{k}="{v}"') # String value, wrap in quotes
        else:
            parts.append(f'{k}={v}') # otherwise, raw int/float
    return ','.join(parts) # TODO: Wait what about bools?

def _argstring_to_dict(args:str) -> Dict:
    """
    Converts "prop1=val1, prop2=val2" => {prop1:val1, prop2:val2}
    """
    selector:Dict = {}
    args_split:List[str] = args.split(',')
    for arg_split in args_split:
        arg_split = arg_split.strip()
        arg_split_split = arg_split.split('=')
        if len(arg_split_split)==2:
            prop:str = arg_split_split[0].strip()
            value:str = arg_split_split[1].strip()
            cleaned_value:Any = _clean_arg_value(value)
            if cleaned_value is not None:
                selector[prop] = cleaned_value
    # Gimme
    return selector

def _select_shape_datas(shapes:List[Dict], selector_dict:Dict) -> List[Dict]:
    selected_shapes:List[Dict] = []
    # Just test if they pass all filters
    for shape_data in shapes:
        passes_all_filters = True
        for k,v in selector_dict.items():
            # _shape_passes_filter also handles _GTE/_LTE ranges, etc
            if not _shape_passes_filter(shape_data, k, v):
                passes_all_filters = False
                break
        if passes_all_filters:
            selected_shapes.append(shape_data)
    # Gimme
    return selected_shapes

def _public_props(shape:dict, props_to_exclude:list[str]|None) -> list[str]:
    """All of a shape's public properties that aren't excluded (shared by _propose_selectors and its NOT selectors)."""
    public_props:list[str] = []
    for prop in shape.keys():
        if prop.startswith('_'):  # skip _private variables
            continue
        if props_to_exclude and prop in props_to_exclude:
            continue
        public_props.append(prop)
    return public_props

def _is_number(value:Any) -> bool:
    """True for ints & floats, but NOT bools (Python counts True/False as ints)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _shape_passes_filter(shape_data:Dict, key:str, value:Any) -> bool:
    """Does one shape pass one selector filter? Exact match, or a range for (prop)_GTE / (prop)_LTE keys."""

    # Range filter: shape needs (prop), it has to be a number, and it's >= (or <=) the value
    if key.endswith('_GTE') or key.endswith('_LTE'):
        prop:str = key[:-4] # four chars behind
        shape_value:Any = shape_data.get(prop)
        if not _is_number(shape_value):
            return False
        if key.endswith('_GTE'):
            return shape_value >= value
        else:
            return shape_value <= value

    # NOT filter: shape needs (prop) too (same as GTE/LTE), and its value has to be anything BUT this one
    if key.endswith('_NOT'):
        prop:str = key[:-4] # four chars behind, again
        return prop in shape_data and shape_data[prop] != value

    # Otherwise, the standard "does it match exactly"
    return key in shape_data and shape_data[key] == value

def _clean_arg_value(value:str)->Any:
    cleaned_value:Any
    if '"' in value or "'" in value:
        cleaned_value = value[1:-1]  # Remove quotes
    elif value=='True' or value=='False':
        cleaned_value = (value=='True') # NOTE: bool('False') is True, so compare!
    else:
        # Try number: int if it's whole (aesthetics), else float.
        try:
            cleaned_value = float(value)
            if cleaned_value == int(cleaned_value):
                cleaned_value = int(cleaned_value)
        except (ValueError, OverflowError):
            # Not a finite literal (e.g. "shape.right", "inf") -- skip it, don't add to dict.
            return None
    return cleaned_value

def _expand_template(template:str, substitutions:Dict[str,str]) -> str:
    """Substitute all %TOKEN%s in a template. Single pass, no recursion."""
    code:str = template
    for token, value in substitutions.items():
        code = code.replace(token, value)
    return code
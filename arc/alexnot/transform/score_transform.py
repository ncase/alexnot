from typing import Dict, List, Optional
import random
import re
import numpy as np

from .. import parse
from .apply_transform import apply_transform_one, _NOTHING_SCENE # (_NOTHING_SCENE = what a crashed program predicts)
from ..utils.deadline import start_deadline, clear_deadline # Hard per-program time cap
# Shape matching keys, shared with the parse scorer (shadow/pixels as a hashable key, and its flips/rotations)
from ..parse.score_parse import _match_grid, _shadow_key, _cached_d4_keys

# CONFIG KNOBS
# (No separate hard-coded-value penalty: value tokens carry their own cost, via VALUE_BIASES)
from ..config import (SCORE_COEFF_WRONG_PIXELS, SCORE_COEFF_TOKEN_COST, SCORE_COEFF_UNIQUE_TOKEN_COST, SCORE_COEFF_UNIQUE_LINE_COST,
                      SCORE_COEFF_EDIT_COST, SCORE_COEFF_OVERALL_IN_PAIR, SCORE_COEFF_SHAPE_DIFF,
                      SHAPE_DIFF_UNMATCHED_OUTPUT_SHAPE_COST, TOKEN_BIASES, VALUE_BIASES, VALUE_DEFAULT_BIAS)
from ..config import TIME_LIMIT_PER_PROGRAM # Hard per-program time cap
from ..config import ABLATION_STUDY_MODE

# Compile these now, for a tiny gain in efficiency
# Tokens: numbers, single- or double-quoted strings, and words. Words stop at underscores ("rank_area" -> "rank", "area").
# [^\W_] is "word character, but not underscore".
_TOKEN_PATTERN = re.compile(r'-?\d+\.\d+|-?\d+|"[^"]*"|\'[^\']*\'|[^\W_]+')
# Words that aren't new concepts: comparisons (GTE/LTE/NOT, as in color_NOT="c4"), and modifiers that tweak
# a property (rank_area = area, ranked; transformed_pixels_twins = twins up to rotation/flip). So e.g.
# focus(rank_area=0) doesn't pay a whole extra concept for "rank", and rank_shape_color_commonness doesn't pay for
# "commonness" (it's how the color gets ranked). (They still pay their TOKEN_BIASES as words.)
# 'nonbg' (scene.most_common_nonbg_color) just narrows which colors count, so it's a plain 1.0 word, not a new concept.
# The axis letters h/v/d (symm_h, symm_v, symm_d) only say WHICH symmetry, so all the axes together are one concept, "symm".
_NOT_CONCEPT_WORDS = frozenset({'GTE', 'LTE', 'NOT', 'rank', 'most', 'least', 'reversed', 'start', 'transformed', 'commonness',
                                'nonbg', 'h', 'v', 'd'})
_VALUE_WORDS = frozenset({'True', 'False'}) # The only value tokens that don't start with a digit, minus sign, or quote
# Words that never count towards the unique-concept cost: values + comparisons
_NO_EXTRA_COST_WORDS = _VALUE_WORDS | _NOT_CONCEPT_WORDS

# For unique token counting: strip quoted strings, then grab alphabetic substrings.
# Single-quoted too, so 'c8' doesn't leak a "c" into the unique words.
_QUOTED_STRING_PATTERN = re.compile(r'"[^"]*"|\'[^\']*\'')
_ALPHA_WORD_PATTERN = re.compile(r'[a-zA-Z]+') # words *not counting digits & underscores*

# For unique line counting: the literal values (quoted strings, numbers, True/False) that get swapped for X.
# Numbers only when they're not part of a word or a decimal (so "rank_2" or "x2" stay as they are)
_LITERAL_PATTERN = re.compile(r'"[^"]*"|\'[^\']*\'|(?<![\w.])-?\d+(?:\.\d+)?|\b(?:True|False)\b')

def get_program_simplicity(program:str) -> tuple[float, int, int]:
    """A program's token cost, unique-concept count and unique-line count, from its string alone (no need to run it)."""

    tokens = _TOKEN_PATTERN.findall(program)

    # It's a float not int, that we add to, so we can bias towards some tokens...
    token_cost:float = 0
    # Every token is exactly one of two kinds: a VALUE (number, quoted string, True/False)
    # looked up in VALUE_BIASES, or a WORD (identifier) looked up in TOKEN_BIASES. Default word = 1,
    # default value = VALUE_DEFAULT_BIAS (a "magic number"), so no separate hard-coded penalty is needed.
    for t in tokens:
        if t[0].isdigit() or t[0] in '-"\'' or t in _VALUE_WORDS:
            # A value: strip the quotes so "c8" and 'c8' are the same key
            token_cost += VALUE_BIASES.get(t.strip('"\''), VALUE_DEFAULT_BIAS)
        else:
            # A word: a specific human-prior? give it the custom cost! Otherwise 1
            token_cost += TOKEN_BIASES.get(t, 1.0)

    # Count unique alphabetic words, ignoring numbers, quoted strings, and punctuation.
    # Also splits by underscore (e.g. "rank_area" -> {"rank", "area"}).
    # Regularizes towards programs that reuse the same concepts.
    program_without_quotes:str = _QUOTED_STRING_PATTERN.sub('', program)
    unique_tokens:set = set(_ALPHA_WORD_PATTERN.findall(program_without_quotes))
    unique_tokens -= _NO_EXTRA_COST_WORDS # Values (True/False), comparisons and modifiers aren't new concepts
    unique_token_count:int = len(unique_tokens)

    # Count unique lines, with every literal value swapped for X (and indentation stripped).
    # Regularizes towards programs that repeat one idea with different values, e.g. "by holes" in every block.
    # (References like shape.area aren't literals, so they stay as they are)
    unique_lines:set = {_LITERAL_PATTERN.sub('X', line.strip()) for line in program.split('\n')}
    unique_lines.discard('') # Blank lines aren't lines of code
    unique_line_count:int = len(unique_lines)
    return token_cost, unique_token_count, unique_line_count


def get_simplicity_score(program:str) -> float:
    """The simplicity part of every score (total, shape-diff, each pair's): a lower bound on all of them, since the rest is >= 0."""
    # Random scores (an ablation) are all in [0, 1), so the only safe lower bound is 0: nothing gets pruned
    if ABLATION_STUDY_MODE == "RANDOM SCORES":
        return 0.0
    token_cost, unique_token_count, unique_line_count = get_program_simplicity(program)
    return (SCORE_COEFF_TOKEN_COST * token_cost + SCORE_COEFF_UNIQUE_TOKEN_COST * unique_token_count
            + SCORE_COEFF_UNIQUE_LINE_COST * unique_line_count)


def get_transform_program_result(program:str, parsed_train_pairs:List) -> Optional[Dict]:
    """
    Scores a program on a list of example pairs, returns program results (None if it crashed on any pair).

    Args:
        program (str): The program
        parsed_train_pairs (List): List of (parsed) training pairs (TODO TAKE RAW)

    Returns:
        Dict: Program results (see types.py for explanation)
    """

    ###################################
    # ACCURACY: Count wrong pixels ####
    ###################################

    total_wrong_pixels:int = 0 # Wrong px overall
    pair_wrong_pixels:List[int] = [] # Wrong px per train pair

    # Shape diff: how far the predicted shapes are from the output's shapes (see _shape_diff)
    total_shape_diff:float = 0.0
    pair_shape_diffs:List[float] = []

    # Remember predictions, so monkeys can do execution-guided progsys!
    pr_predicted_scenes:List[Dict] = []
    pr_predicted_pixels:List[np.ndarray] = []

    # Hard time cap for this whole program, all pairs (heavy loops check it). Out of time? apply_transform
    # catches the ProgramTimedOut like any crash, so a timed-out program gets dropped just like a crashed one
    start_deadline(TIME_LIMIT_PER_PROGRAM)

    # For each pair...
    for pair in parsed_train_pairs:

        # Run the program to get the prediction
        # ...and clear the deadline if something crashes, so it can't fire on a later
        # unrelated run (like main_solver running the winners on the test inputs)
        try:
            predicted_scene:Dict = apply_transform_one(program, pair['input'])
        except BaseException:
            clear_deadline()
            raise

        # Crashed (or timed out) on this pair? Drop the whole program: no point scoring the other pairs
        if predicted_scene is _NOTHING_SCENE:
            clear_deadline()
            return None

        pr_predicted_scenes.append(predicted_scene) # remember
        predicted_pixels:np.ndarray = parse.reverse_parse_one(predicted_scene)
        pr_predicted_pixels.append(predicted_pixels)

        # Store raw pixel grid in the predicted scene so monkeys can compare
        # predicted vs output at the pixel level (e.g. shapes_fill_holes)
        predicted_scene['_raw'] = predicted_pixels # "_raw" is Numpy

        # Actual pixels
        actual_pixels:np.ndarray = pair['output']['_raw'] # "_raw" is Numpy now

        #####################################
        # NOW LET'S COUNT YOUR WRONG PIXELS #
        #####################################

        this_pairs_wrong_pixels = 0

        # Predicted, Actual, and Overlap (minimum)'s pixel grid dimensions
        p_height, p_width = predicted_pixels.shape
        a_height, a_width = actual_pixels.shape
        min_width, min_height = min(p_width,a_width), min(p_height,a_height)

        # Different dimensions? Count up non-overlap
        if p_width!=a_width or p_height!=a_height:
            p_area, a_area = p_width*p_height, a_width*a_height
            overlap_area = min_width * min_height
            # Total pixels NOT in the overlap = A + B - 2*Overlap (coz double-count, so double-subtract)
            leftover_pixels = p_area + a_area - 2*overlap_area
            this_pairs_wrong_pixels += leftover_pixels

        # NOW count wrong pixels *within* their overlap (numpy vectorized, way faster than row loop)
        this_pairs_wrong_pixels += int(np.count_nonzero(
            predicted_pixels[:min_height, :min_width] != actual_pixels[:min_height, :min_width]
        ))

        # COUNT 'EM UP!
        total_wrong_pixels += this_pairs_wrong_pixels
        pair_wrong_pixels.append(this_pairs_wrong_pixels)

        # And how far off are its shapes? (Only used to pick parents, not the winner)
        this_pairs_shape_diff:float = _shape_diff(predicted_scene, pair['output'])
        total_shape_diff += this_pairs_shape_diff
        pair_shape_diffs.append(this_pairs_shape_diff)

        # TODO: also count gaussian blur, edges, etc

    # Program's done (every pair ran), so no more deadline
    clear_deadline()

    ###################################
    # EDIT COUNT ######################
    ###################################

    # Sum up _edits_cost across all shapes in all predicted scenes, plus each scene's own (e.g. its replace_pixels
    # spots, added by reverse_parse). Regularizes against programs that "undo and redo" — prefer fewer actual actions.
    total_edits_cost:int = 0
    for predicted_scene in pr_predicted_scenes:
        total_edits_cost += predicted_scene.get('_edits_cost', 0)
        for shape_data in predicted_scene['shapes']:
            total_edits_cost += shape_data.get('_edits_cost', 0)

    ###################################
    # SIMPLICITY ######################
    ###################################

    # Token cost + unique concepts + unique lines: from the program string alone (see get_program_simplicity)
    token_cost, unique_token_count, unique_line_count = get_program_simplicity(program)

    ###################################
    # SCORING EACH PROGRAM! ###########
    ###################################

    # Weighted sum of Accuracy & Simplicity (lower is better)
    total_score = 0
    total_score += SCORE_COEFF_WRONG_PIXELS * total_wrong_pixels
    total_score += SCORE_COEFF_TOKEN_COST * token_cost
    total_score += SCORE_COEFF_UNIQUE_TOKEN_COST * unique_token_count
    total_score += SCORE_COEFF_UNIQUE_LINE_COST * unique_line_count
    total_score += SCORE_COEFF_EDIT_COST * total_edits_cost

    # Shape-diff score: the same simplicity terms, but shape differences instead of wrong pixels.
    # Only used to pick some extra parents in search_transform (the winner is still picked by total_score)
    shape_diff_score:float = SCORE_COEFF_SHAPE_DIFF * total_shape_diff
    shape_diff_score += SCORE_COEFF_TOKEN_COST * token_cost
    shape_diff_score += SCORE_COEFF_UNIQUE_TOKEN_COST * unique_token_count
    shape_diff_score += SCORE_COEFF_UNIQUE_LINE_COST * unique_line_count
    shape_diff_score += SCORE_COEFF_EDIT_COST * total_edits_cost

    # Pair Scores
    pair_scores = []
    for wrong_pixels in pair_wrong_pixels:
        score = 0
        score += SCORE_COEFF_WRONG_PIXELS * wrong_pixels
        score += SCORE_COEFF_TOKEN_COST * token_cost
        score += SCORE_COEFF_UNIQUE_TOKEN_COST * unique_token_count
        score += SCORE_COEFF_UNIQUE_LINE_COST * unique_line_count
        score += SCORE_COEFF_EDIT_COST * total_edits_cost
        score += SCORE_COEFF_OVERALL_IN_PAIR * total_wrong_pixels # a pair's solution is also better if it does better on overall pairs!
        pair_scores.append(score)

    # The dumbest possible selection, for the paper's ablations: every score is a coin flip, so the winner,
    # every parent, and attempt 2 are all random picks from whatever the monkeys proposed
    if ABLATION_STUDY_MODE == "RANDOM SCORES":
        total_score = random.random()
        shape_diff_score = random.random()
        pair_scores = [random.random() for _ in pair_scores]

    ###################################
    # Return... #######################
    ###################################

    return {
        'program': program, # Storing itself again for convenience
        'predicted_scenes': pr_predicted_scenes,
        'predicted_pixels': pr_predicted_pixels,

        'token_cost': token_cost,
        'unique_token_count': unique_token_count,
        'unique_line_count': unique_line_count,
        'total_edits_cost': total_edits_cost,

        'pair_wrong_pixels': pair_wrong_pixels,
        'pair_scores': pair_scores,

        'total_wrong_pixels': total_wrong_pixels,
        'total_score': total_score,

        'pair_shape_diffs': pair_shape_diffs,
        'total_shape_diff': total_shape_diff,
        'shape_diff_score': shape_diff_score,
    }


def _shape_diff(predicted_scene:dict, output_scene:dict) -> float:
    """How far are the predicted shapes from the output's shapes? 0 = exactly the same shapes, lower = closer."""

    # Hidden shapes aren't drawn, so they don't count as predicted. Neither do shapes the camera cropped out entirely
    # (focus() just moves them outside the frame, it doesn't hide them)
    camera_width:int = int(predicted_scene['width'])
    camera_height:int = int(predicted_scene['height'])
    predicted_shapes:list[dict] = [shape for shape in predicted_scene['shapes'] if not shape.get('hidden', False)
                                   and shape['left'] < camera_width and shape['left'] + shape['width'] > 0
                                   and shape['top'] < camera_height and shape['top'] + shape['height'] > 0]
    output_shapes:list[dict] = output_scene['shapes']
    output_width:int = output_scene['width']
    output_height:int = output_scene['height']

    # Match keys: the shadow for one-color shapes, the pixels for multicolor ones (same as the parse scorer,
    # so a multicolor shape with its insides recolored is a different shape, not a "wrong color")
    predicted_keys:list[tuple] = [_shadow_key(_match_grid(shape)) for shape in predicted_shapes]
    output_keys:list[tuple] = [_shadow_key(_match_grid(shape)) for shape in output_shapes]

    # Speedup: match the exactly-right shapes first (same key, color and spot), they cost 0.
    # Usually most shapes are untouched, so this leaves only a few for the slower pairwise matching below
    unmatched_output_indices_by_exact:dict[tuple, list[int]] = {}
    for output_index, (output_shape, output_key) in enumerate(zip(output_shapes, output_keys)):
        exact_key:tuple = (output_key, output_shape['color'], output_shape['left'], output_shape['top'])
        unmatched_output_indices_by_exact.setdefault(exact_key, []).append(output_index)
    leftover_predicted_indices:list[int] = []
    for predicted_index, (predicted_shape, predicted_key) in enumerate(zip(predicted_shapes, predicted_keys)):
        exact_key = (predicted_key, predicted_shape['color'], predicted_shape['left'], predicted_shape['top'])
        same_output_indices:list[int]|None = unmatched_output_indices_by_exact.get(exact_key)
        if same_output_indices:
            same_output_indices.pop() # Matched (one-to-one), for free
        else:
            leftover_predicted_indices.append(predicted_index)
    leftover_output_indices:list[int] = [output_index for output_indices in unmatched_output_indices_by_exact.values()
                                         for output_index in output_indices]

    # Every leftover output shape, by its match key (to find same-shape candidates fast)
    leftover_output_indices_by_key:dict[tuple, list[int]] = {}
    for output_index in leftover_output_indices:
        leftover_output_indices_by_key.setdefault(output_keys[output_index], []).append(output_index)

    # Every possible match between a leftover predicted shape and a leftover output shape of the same shape
    # (as-is, or flipped/rotated), and what it costs
    candidate_matches:list[tuple[float, int, int]] = []
    for predicted_index in leftover_predicted_indices:
        predicted_shape:dict = predicted_shapes[predicted_index]
        predicted_key:tuple = predicted_keys[predicted_index]
        transformed_keys:set[tuple] = _cached_d4_keys(_match_grid(predicted_shape), predicted_key)
        for matching_key in {predicted_key} | transformed_keys:
            for output_index in leftover_output_indices_by_key.get(matching_key, ()):
                output_shape:dict = output_shapes[output_index]
                # Wrong color: +1. Flipped/rotated (checked as-is first, so symmetrical shapes aren't "flipped"): +1
                cost:float = float(predicted_shape['color'] != output_shape['color'])
                cost += float(predicted_key != output_keys[output_index])
                # Wrong spot: how far, as a fraction of the output grid (so 0 to 2, and closer is cheaper)
                cost += abs(predicted_shape['left'] - output_shape['left']) / output_width
                cost += abs(predicted_shape['top'] - output_shape['top']) / output_height
                candidate_matches.append((cost, predicted_index, output_index))

    # One-to-one, cheapest match first: each shape gets matched at most once
    diff:float = 0.0
    matched_predicted_indices:set[int] = set()
    matched_output_indices:set[int] = set()
    for cost, predicted_index, output_index in sorted(candidate_matches):
        if predicted_index in matched_predicted_indices or output_index in matched_output_indices:
            continue
        matched_predicted_indices.add(predicted_index)
        matched_output_indices.add(output_index)
        diff += cost

    # A predicted shape the output doesn't have: +1 each
    diff += len(leftover_predicted_indices) - len(matched_predicted_indices)
    # An output shape nothing predicted matches costs more (it appears out of nowhere), so "delete it" isn't a free pass
    diff += SHAPE_DIFF_UNMATCHED_OUTPUT_SHAPE_COST * (len(leftover_output_indices) - len(matched_output_indices))
    return diff

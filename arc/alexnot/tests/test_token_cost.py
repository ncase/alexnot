import sys
from pathlib import Path

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Re-use score_transform's OWN token patterns & word lists, so this can't drift from the real scoring
import alexnot.transform  # (the package exports functions with the modules' names, so grab the module from sys.modules)
score_transform_module = sys.modules['alexnot.transform.score_transform']
from alexnot.config import (SCORE_COEFF_TOKEN_COST, SCORE_COEFF_UNIQUE_TOKEN_COST, SCORE_COEFF_UNIQUE_LINE_COST,
                            TOKEN_BIASES, VALUE_BIASES, VALUE_DEFAULT_BIAS)

####################################################################################
## What does a program cost, simplicity-wise? Token by token, plus its unique-concept cost.
## This is ONLY the simplicity part of score_transform: no wrong pixels, no edit cost (those need a level).
####################################################################################

def token_cost_breakdown(program:str) -> dict:
    """Token-by-token simplicity cost of a program, the same way score_transform counts it."""

    # Each token is a VALUE (number, quoted string, True/False) or a WORD, same split as score_transform
    token_costs:list[tuple[str, float]] = []
    for token in score_transform_module._TOKEN_PATTERN.findall(program):
        if token[0].isdigit() or token[0] in '-"\'' or token in score_transform_module._VALUE_WORDS:
            token_costs.append((token, VALUE_BIASES.get(token.strip('"\''), VALUE_DEFAULT_BIAS)))
        else:
            token_costs.append((token, TOKEN_BIASES.get(token, 1.0)))

    # Unique concepts: alphabetic words outside quotes, minus values & "not really a new concept" words (rank, GTE...)
    program_without_quotes:str = score_transform_module._QUOTED_STRING_PATTERN.sub('', program)
    unique_words:set = set(score_transform_module._ALPHA_WORD_PATTERN.findall(program_without_quotes))
    unique_words -= score_transform_module._NO_EXTRA_COST_WORDS

    token_total:float = SCORE_COEFF_TOKEN_COST * sum(cost for _, cost in token_costs)
    unique_total:float = SCORE_COEFF_UNIQUE_TOKEN_COST * len(unique_words)

    # Unique lines: each line stripped, with its literal values swapped for X (same as score_transform)
    unique_lines:set = {score_transform_module._LITERAL_PATTERN.sub('X', line.strip()) for line in program.split('\n')}
    unique_lines.discard('')
    unique_lines_total:float = SCORE_COEFF_UNIQUE_LINE_COST * len(unique_lines)
    return {'token_costs': token_costs, 'token_total': token_total,
            'unique_words': sorted(unique_words), 'unique_total': unique_total,
            'unique_lines': sorted(unique_lines), 'unique_lines_total': unique_lines_total,
            'total': token_total + unique_total + unique_lines_total}


if __name__ == "__main__":

    #######################################################################
    ## CHOOSE THE PROGRAMS (each one is scored on its own)
    #######################################################################

    programs:list[str] = [
        'camera_resize(width=7)',
        'focus(rank_left=0, padding=1, one_match=False)',
        # one_match=True is the non-default alternative the monkeys propose
        'focus(rank_left=0, padding=1, one_match=True)',
        'focus(rank_left=0, padding=1)',
        # "transformed" should cost exactly what "rank" costs (0.2, not a new concept)
        'all(pixels_twins=0)',
        'all(transformed_pixels_twins=0)',
        'all(rank_pixels_twins=0)',
        # "NOT" should cost what GTE/LTE cost (0.1, not a new concept)
        'all(color_NOT="c4")',
        # Direction words cost 1 each (like "lr"/"ud"), so a direction can beat its x/y spelling
        'fall(direction="d")',
        'fall(y=1)',
        'fall(direction="dl")',
        'fall(x=-1,y=1)',
        'move(direction="out")',
    ]

    #######################################################################
    ## NOW print each program's breakdown
    #######################################################################

    for program in programs:
        breakdown:dict = token_cost_breakdown(program)
        print(f"\n{program}")
        print("  tokens: " + ", ".join(f"{token}={cost:g}" for token, cost in breakdown['token_costs']))
        print(f"  token cost {breakdown['token_total']:g}  +  unique concepts {breakdown['unique_total']:g} "
              f"({SCORE_COEFF_UNIQUE_TOKEN_COST:g} x {len(breakdown['unique_words'])}: {', '.join(breakdown['unique_words'])})"
              f"  +  unique lines {breakdown['unique_lines_total']:g} "
              f"({SCORE_COEFF_UNIQUE_LINE_COST:g} x {len(breakdown['unique_lines'])}: {' | '.join(breakdown['unique_lines'])})"
              f"  =  {breakdown['total']:g}")

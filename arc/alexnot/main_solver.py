"""
This machine solves ARC puzzles!
"""

from typing import Dict, List, Optional

import gc
import time
import re
import json
import random
import traceback
import numpy as np

from . import parse
from . import annotate
from . import transform
from . import similarize
from .bayes_posterior import approximate_posterior

from .utils.reasoning_trace import ReasoningTrace
from func_timeout import func_timeout, FunctionTimedOut

# CONFIG KNOBS
from .config import TOTAL_ATTEMPTS, TIME_LIMIT_PER_ATTEMPT

# For "you messed up"
RED_CROSS = [
    [2,0,0,0,0,0,2],
    [0,2,0,0,0,2,0],
    [0,0,2,0,2,0,0],
    [0,0,0,2,0,0,0],
    [0,0,2,0,2,0,0],
    [0,2,0,0,0,2,0],
    [2,0,0,0,0,0,2]
]

def parse_config_for_trace(parse_config:Dict) -> str:
    """A parse config as indented JSON for the reasoning trace, with denoise, autocomplete_rects and pattern_background always spelled out."""
    # These switches are only in the config when they're on, so fill in False to show "off" too.
    # (Chunks are numpy arrays, so they print as nested lists)
    config_json:str = json.dumps({**parse_config,
                                  'denoise': parse_config.get('denoise', False),
                                  'autocomplete_rects': parse_config.get('autocomplete_rects', False),
                                  'pattern_background': parse_config.get('pattern_background', False)},
                                 indent=4, default=lambda value: value.tolist())

    # indent=4 puts every number on its own line, so re-dump each innermost list of numbers
    # (a chunk's row) compactly onto one line, to make chunks read like 2D grids
    return re.sub(r'\[[-\d,\s]*\]', lambda match: json.dumps(json.loads(match.group(0))), config_json)

def solve(level_data:Dict, show_reasoning=True, profiling=False, random_seed:int=-1, bayes:bool=False) -> Dict:

    """
    == UNDERSTAND ==
    1. Parse: search for best parser config & chunked shapes, apply to all
    2. Transform: search for best transformation programs
    (note that Annotations are applied to scenes right after Parse, and after every DSL action)
    // TODO wait do I actually re-annotate?
    == APPLY ==
    3. (TODO) Similarize: change test cases to be most similar to train cases
    4. SOLVE! apply best parser + transform(s) to similarized test cases # try other augmentations?
    5. Finally, return best solutions, with reasoning trace
    (TODO will real contest also want reasoning trace?)
    """

    #####################################################################
    ## Preamble: Random seed, copy-and-numpy data, remove test answers
    #####################################################################

    # Random seed for debugging purposes...
    if random_seed<0:
        random_seed = random.randrange(2**16)
    random.seed(random_seed)

    # Start the reasoning trace...
    start_time:float = time.time()
    train_trace_string:str = ""
    trace = ReasoningTrace()

    # Make a deep copy just in case
    level_data = json.loads(json.dumps(level_data))

    # Convert all grids to numpy arrays right away (int8 fits ARC colors 0-9 and empty -1)
    for pair in level_data.get('train', []):
        pair['input'] = np.array(pair['input'], dtype=np.int8)
        pair['output'] = np.array(pair['output'], dtype=np.int8)
    for pair in level_data.get('test', []):
        pair['input'] = np.array(pair['input'], dtype=np.int8)
        pair['output'] = None # DESTROY THIS INFO (Test Answers)

    # All pairs
    raw_train_pairs:List[Dict] = level_data['train']
    raw_test_pairs:List[Dict] = level_data['test']
    all_pairs:list[dict] = raw_train_pairs + raw_test_pairs

    #####################################################################
    ## Search for best Parse Configs
    #####################################################################

    # Try ALL parse configs, score & sort them from best-to-worst. Already de-duplicated: no two give the same
    # bg & shapes on every demo, so each attempt below just takes the next one in order
    # Each entry: { 'config': {...}, 'score': float, 'duplicates': [twin configs it beat] }
    parse_results:List[Dict] = parse.search_parse(all_pairs, trace=trace)

    # Fewer different parses than attempts? Then just do fewer attempts: re-running a parse we already tried
    # would only retry the exact same scenes
    number_of_attempts:int = min(TOTAL_ATTEMPTS, len(parse_results))

    # Phase 1 done: everything so far (preamble + parse search) counts as "search parse"
    seconds_search_parse:float = time.time() - start_time

    #####################################################################
    ## Search for Transforms (1 attempt for each of N best Parse Configs)
    #####################################################################

    # Make a single solve attempt its own function, so we can use func_timeout to stop it
    # Start it (and "attempt") at the first attempt's values, so the type checker knows they're bound after
    # the loop. (The loop sets them on its first pass anyway)
    currently_attempted_parse_result:Dict = parse_results[0] # Define outside of fn, so it can be accessed outside of it
    attempt:int = 0
    # This attempt's sorted program ranking (None if it crashed, or timed out before ranking anything)
    transform_program_results:Optional[List[Dict]] = None

    # The attempt we'll answer with: the one whose best program scored best so far. Keeping only the running best
    # (not every attempt's whole ranking) holds at most two rankings in memory at once.
    # Demo-perfect beats not-demo-perfect, then lower total_score (the score the search sorts by). Ties: the earlier attempt
    chosen_parse_result:Optional[Dict] = None
    chosen_program_results:Optional[List[Dict]] = None
    def ranking_key(program_result:Dict) -> tuple:
        """How good an attempt's best program is: demo-perfect first, then lower score."""
        return (program_result['total_wrong_pixels']>0, program_result['total_score'])

    # Tracebacks of any attempts (or single monkeys) that crashed, so they show up in the trace AND the result (not just stdout)
    crash_tracebacks:List[str] = []
    # search_transform refills this with its ranking every round, so a timed-out attempt still has something to show.
    # A FRESH list per attempt (rebound below), so a stopped-but-not-quite-dead attempt thread can't write into the next one's.
    best_so_far_program_results:List[Dict] = []
    # Each attempt's full wall time (re-parse, annotate, search, even if it timed out or crashed), for the Summary
    seconds_per_attempt:List[float] = []
    def one_attempt() -> None:
        nonlocal transform_program_results
        # Re-parse train pairs on the fly (cheap enough, saves memory vs caching)
        parsed_train_pairs:List[Dict] = parse.apply_parse(currently_attempted_parse_result['config'], raw_train_pairs)
        # A crashing monkey doesn't stop the search, but its traceback still lands in crash_tracebacks
        transform_program_results = transform.search_transform(parsed_train_pairs, trace=trace, best_so_far=best_so_far_program_results,
                                                               crashes=crash_tracebacks)

    # Try the top parse configs, each with a time limit
    for attempt in range(number_of_attempts):

        # Try one attempt with TIME LIMIT
        start_solve_time = time.time()
        ran_out_of_time:bool = False
        crash_traceback:Optional[str] = None # Set if this attempt crashed

        # Use this attempt's parse (the de-duped ranking's #attempt)
        currently_attempted_parse_result = parse_results[attempt]
        best_so_far_program_results = [] # Fresh list for this attempt (one_attempt reads the name at call time)
        transform_program_results = None # So a crash can't leave the previous attempt's ranking here

        # Try searching for the Transform
        try:
            if profiling:
                one_attempt() # For cProfile purposes - NO threading
            else:
                # Normal execution: limit time per attempt
                func_timeout(TIME_LIMIT_PER_ATTEMPT, one_attempt)
        except FunctionTimedOut:
            ran_out_of_time = True
            # Out of time, but did the search already rank some programs (i.e. finish at least one round)?
            # Then keep them, exactly as if the attempt had finished. (Otherwise we'd throw away even a demo-perfect program)
            if best_so_far_program_results:
                transform_program_results = list(best_so_far_program_results)
        except Exception as e:
            print(f"Error in attempt: {e}")
            # Keep the traceback, so it goes in the trace (below) and the result's "crashes" list
            crash_traceback = traceback.format_exc()
            crash_tracebacks.append(crash_traceback)
            print(crash_traceback)

        # Did this attempt's best program beat the chosen attempt's? Then answer with this attempt instead.
        # (Strictly better only, so ties keep the earlier attempt)
        this_attempt_best:Optional[Dict] = transform_program_results[0] if transform_program_results else None
        if this_attempt_best and (chosen_program_results is None or ranking_key(this_attempt_best) < ranking_key(chosen_program_results[0])):
            chosen_parse_result = currently_attempted_parse_result
            chosen_program_results = transform_program_results

        # Free any leftover objects from the attempt thread (func_timeout spawns threads
        # that can hold refs to big numpy arrays / shape lists). Without this, running
        # 100s of levels in a row causes a slow memory leak that eventually gets OOM-killed.
        gc.collect()

        # Flush out the reasoning trace to a string, append
        elapsed_time = time.time() - start_solve_time
        seconds_per_attempt.append(elapsed_time)

        # A crash says CRASHED, with the traceback right there.
        # The "Using this Parse Config" header also says that parse's rank in the de-duped ranking
        crash_trace_string:str = f"\n\n### 💀 CRASHED! Traceback:\n\n{crash_traceback}" if crash_traceback else ""
        train_trace_string += f"""
## Search Transform, Attempt {attempt+1}/{number_of_attempts} ({elapsed_time:.2f}s) {"FAIL" if ran_out_of_time else ""}{"💀 CRASHED" if crash_traceback else ""}

### Using this Parse Config: (#{attempt+1} of {len(parse_results)} in the parse ranking)

{parse_config_for_trace(currently_attempted_parse_result['config'])}

### The Search: ({len(transform_program_results) if transform_program_results else '(FAIL)'} programs tested)

Best program's score: {f"{this_attempt_best['total_score']:.2f} ({this_attempt_best['total_wrong_pixels']} wrong pixels)" if this_attempt_best else '(none)'}

{trace.get_log('search_transform')}{crash_trace_string}
""".strip()
        train_trace_string += "\n\n"
        trace.clear() # clear, for next attempt

        # Did we find a winning parse + transform?
        if this_attempt_best and this_attempt_best['total_wrong_pixels']==0:
            break # If so, we're done!

    # Phase 2 done: "search transform" = all the attempts together. Phase 3 ("run on tests") starts now
    seconds_search_transform:float = sum(seconds_per_attempt)
    start_run_on_tests_time:float = time.time()

    # 4. (TODO) Similarize the test cases to be similar to train cases, upon running parse + annotate + transform

    #####################################################################
    ## Now apply best Parse & Config to give 2 solutions!
    #####################################################################

    # What if LITERALLY NO SOLUTION FOUND? (Every attempt crashed, or timed out before ranking anything)
    if chosen_parse_result is None or chosen_program_results is None:
        reasoning_trace = "we ain't found shiz" + "\n\n# The Yap Before We Failed:\n\n" + train_trace_string
        return {
            "test_solutions": [ {'attempt_1':RED_CROSS, 'attempt_2':RED_CROSS} for i in range(len(raw_test_pairs)) ],
            'reasoning_trace': reasoning_trace,
            'crashes': crash_tracebacks, # So callers can see crashes without digging through the trace
            'best_program': None, 
            'best_parse': None,
            'solved_all_demos': False,
            # Phase times, same as the normal return below (BLAST_EM sums them). Tests never ran, so 0
            'seconds_search_parse': seconds_search_parse,
            'seconds_search_transform': seconds_search_transform,
            'seconds_run_on_tests': 0.0,
            'bayes_posterior': None # No programs to weigh (the webapp then just skips the posterior sections)
        }

    # 5a. SOLVE! Apply the chosen attempt's Parse + its best Transform.
    best_transform_program_result:Dict = chosen_program_results[0]
    # Re-parse test pairs on the fly (cheap enough, saves memory vs caching)
    parsed_tests:List[Dict] = parse.apply_parse(chosen_parse_result['config'], raw_test_pairs)
    annotated_tests:List[Dict] = annotate.apply_annotate(parsed_tests) # remember to annotate first! (TODO less jank?)
    best_solution_scenes:List[Dict] = transform.apply_transform(best_transform_program_result['program'], [pair['input'] for pair in parsed_tests])
    best_solutions:List[np.ndarray] = parse.reverse_parse(best_solution_scenes) # reverse_parse gives numpy grids
    
    # 5b. Second-best answer: same Parse + Annotate, next-best Transform with *different* pixel results
    # (eh, forget traces)
    second_best_solutions:List[np.ndarray|List[List[int]]] = [] # Numpy grids, or the RED_CROSS list
    # For each test: (bc we're picking *different* runner-up programs for each test)
    runner_up_transforms:List[Dict] = []
    for i, annotated_test in enumerate(annotated_tests):
        best_solution = best_solutions[i]
        found_other_answer = False
        # Try best 99 runner-up programs (from the same attempt, so they fit the same parse)
        for runner_up_transform in chosen_program_results[1:100]:
            runner_up_solution_scene:Dict = transform.apply_transform_one(runner_up_transform['program'], annotated_test['input'])
            runner_up_solution:np.ndarray = parse.reverse_parse_one(runner_up_solution_scene) # A numpy grid
            # Does this return the same answer as the best? Next!
            if np.array_equal(best_solution, runner_up_solution):
                continue
            else:
                # Otherwise, we found our second answer!
                found_other_answer = True
                second_best_solutions.append(runner_up_solution)
                runner_up_transforms.append(runner_up_transform)
                break
        # Didn't find a different answer? Give up, return a big red X.
        if not found_other_answer:
            second_best_solutions.append( RED_CROSS )
            runner_up_transforms.append({ 'program': 'no runner-up found' })
    
    # 5c. Output in ARC JSON Format (convert numpy arrays back to plain lists for JSON).
    # Zipped instead of indexed: the type checker can't narrow "list[i]" with isinstance, only a plain name.
    # (Both lists have one entry per test)
    test_solutions:List[Dict] = [
        {
            'attempt_1': best_solution.tolist() if isinstance(best_solution, np.ndarray) else best_solution,
            'attempt_2': second_best_solution.tolist() if isinstance(second_best_solution, np.ndarray) else second_best_solution
        }
        for best_solution, second_best_solution in zip(best_solutions, second_best_solutions)
    ]

    # Cache out the reasoning trace to a string
    # (TODO)
    trace.clear()

    #####################################################################
    ## WRAP UP Chain of Thought results, and return
    #####################################################################

    # We're done! Stop the timer:
    end_time:float = time.time()
    elapsed_time:float = end_time - start_time
    
    # Phase 3 done: parsing the tests, running the program, reverse parse, runner-ups
    seconds_run_on_tests:float = end_time - start_run_on_tests_time
    # Each attempt's own time (attempt N always uses the de-duped ranking's parse #N), like "1) 20.10s, 2) 3.00s"
    attempt_times_string:str = ", ".join(f"{attempt_number+1}) {seconds:.2f}s"
                                         for attempt_number, seconds in enumerate(seconds_per_attempt))

    # Summary
    # TODO: more meta-summary stats
    # The total, split into its 3 phases (the per-attempt times also show how many attempts there were)
    summary:str = f"""
# Summary

Random seed: {random_seed}

This took {elapsed_time:.2f} seconds!
- search parse: {seconds_search_parse:.2f}s
- search transform: {seconds_search_transform:.2f}s // attempts: {attempt_times_string}
- run on tests: {seconds_run_on_tests:.2f}s

## Best parse

{parse_config_for_trace(chosen_parse_result['config'])}

## Best transform (score {best_transform_program_result['total_score']:.2f}, {best_transform_program_result['total_wrong_pixels']} wrong pixels)

{best_transform_program_result['program']}
""".strip()
    
    # Runner-up Transforms
    for i,runner_up_transform in enumerate(runner_up_transforms):
        summary += "\n\n"
        summary += f"""
## Runner-up transform for test #{i}

{runner_up_transform['program']}
""".strip()

    # Combine summary with mid-reasoning trace during Train Pairs
    reasoning_trace = summary + "\n\n# Big Ol' Reasoning Yap:\n\n" + train_trace_string

    # TODO (Nicky / a subagent): also return a "confidence" float, 0.0 to 1.0.
    # 0.0 if the best program doesn't get ALL demo pairs right (total_wrong_pixels > 0),
    # otherwise higher the more robust / less "weird" the solution looks (formula TBD).
    # tests/BLAST_EM.py already reads result.get("confidence") to measure precision,
    # and the Kaggle notebook will use it to decide which puzzles skip the neural fallback.

    # The chosen parse config on one line, every switch spelled out (BLAST_EM logs it per level)
    best_parse_string:str = re.sub(r'\s+', ' ', parse_config_for_trace(chosen_parse_result['config']))

    # Did the best program get EVERY demo pair 100% right? (BLAST_EM & the webapp report this per level)
    solved_all_demos:bool = best_transform_program_result['total_wrong_pixels']==0

    # Webapp only: the approximate posterior over programs + test outputs (display only, attempts 1/2 don't change).
    # It runs after the timer stopped, so it never counts towards the solver's timings
    bayes_posterior:Optional[Dict] = approximate_posterior(chosen_program_results, annotated_tests) if bayes else None

    # Finally, tell me the results & reasoning! <3
    if show_reasoning:
        return {
            "test_solutions": test_solutions,
            "reasoning_trace": reasoning_trace,
            "crashes": crash_tracebacks, # (empty list = no crashes)
            "best_program": best_transform_program_result['program'],
            "best_parse": best_parse_string,
            "solved_all_demos": solved_all_demos,
            # The 3 phase times, as numbers (BLAST_EM sums them over all levels)
            "seconds_search_parse": seconds_search_parse,
            "seconds_search_transform": seconds_search_transform,
            "seconds_run_on_tests": seconds_run_on_tests,
            "bayes_posterior": bayes_posterior # None unless bayes=True
        }
    else: # For Kaggle contest, just spit out the solutions
        return {
            "test_solutions": test_solutions,

            # These small extras (no big trace), so BLAST_EM can log crashes & programs
            "crashes": crash_tracebacks,
            "best_program": best_transform_program_result['program'],
            "best_parse": best_parse_string,
            "solved_all_demos": solved_all_demos,
            # The 3 phase times, as numbers (BLAST_EM sums them over all levels)
            "seconds_search_parse": seconds_search_parse,
            "seconds_search_transform": seconds_search_transform,
            "seconds_run_on_tests": seconds_run_on_tests,
            "bayes_posterior": bayes_posterior # None unless bayes=True
        }


#####################################################################
## (BATCH SOLVE: moved to main_solver_batch.py)
#####################################################################

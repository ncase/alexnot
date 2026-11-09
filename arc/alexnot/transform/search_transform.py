import bisect
import random
import time
import traceback
from typing import Optional
from typing import Iterable

from .. import annotate
from .score_transform import get_transform_program_result, get_simplicity_score
# apply_fit and apply_attract cache their searches, so we clear them every round
from .dsl.physics_fitting import clear_fit_cache
from .dsl.physics_attract import clear_attract_cache

# CONFIG KNOBS
from ..config import (EVO_MAX_GENERATIONS, EVO_EARLY_STOP_COUNTDOWN, PARENTS_TOP_K_BY_SCORE, PARENTS_TOP_K_BY_SHAPE_DIFF,
                      MONKEY_CAP_START, MONKEY_CAP_DISCOUNT, MONKEY_CAP_MIN, CLOCK_BUDGET_PER_MONKEY,
                      CLOCK_BUDGET_PER_BATCHER_MONKEY, ABLATION_STUDY_MODE)

################################################
# ADD MONKEYS!
################################################

from .monkeys import causality, shape_change, shape_change_fancy, scene_change, camera_change, compression, pixels_change
from .monkeys import ray_change # The ray monkey
from .monkeys import line_change # The connect-with-lines monkey
from .monkeys import sort_change # The "Order" (sort) monkey
from .monkeys import pathfind_change # Winding through the gaps to a target: pathfind
from .monkeys import synonym # Re-rolls one selector at a time
from .monkeys.monkey_helpers import reset_prefix_scenes # The scenes before a program's middle block, for the monkeys that rewrite one

MONKEYS:list = [

    shape_change.shapes_repaint,
    shape_change.shapes_hide,
    shape_change.shapes_move,
    shape_change.shapes_restack, # Stacking order: move(z=...) + move(max_z=...)
    shape_change.shapes_bring_to_front, # Stacking order from wrong pixels: move(max_z=1) for a covered shape with the right color
    shape_change.shapes_gravity,
    shape_change.shapes_fit, # Jigsaw: fit() + apply_fit()
    shape_change.shapes_attract, # Magnets: attract(overlap=True) + apply_attract()
    shape_change.shapes_copy, # Stamps: copy_to(target) onto color-matched targets
    shape_change.shapes_transform,
    # shape_change.shapes_static, -- deactivate for now, before we refactor it to make it cleaner & fit the rest...

    shape_change_fancy.shapes_outline,
    shape_change_fancy.shapes_fill_holes,
    ray_change.shapes_add_ray, # Rays: add_ray + add_ray_towards
    line_change.shapes_connect_with_lines, # Lines between lined-up shapes: connect_with_lines
    sort_change.shapes_sort, # Order: sort() + apply_sort(key, direction, descending, spacing)
    pathfind_change.shapes_pathfind, # Bent paths to a target: pathfind(passthru=...)
    # shape_change_fancy.shapes_inline, # Off: inline is disabled in the DSL for now, we'll deal with it later

    scene_change.change_background,

    pixels_change.replace_pixels, # Raw-pixel find-and-replace, for 2+ identical added patches

    camera_change.camera_resize,
    camera_change.focus,
    camera_change.focus_by_sliding_window, # Focus, but the ingroup is the shapes inside the best-matching crop window
    camera_change.focus_on_window_edge, # Same crop windows, but the ingroup is only the inside shapes touching the window's edge
    camera_change.focus_by_output_size, # Focus on the shapes whose box (padded up to 3 px a side) is the output grid's size

    compression.delete_lines,
    compression.delete_blocks,
    compression.delete_arguments,
    compression.permute_code_blocks, # Draw order: reorder the top-level blocks, so later ones land on top
    synonym.reroll_selectors, # Same program, one selector described another way

    causality.any_coincidence,
    causality.join_coincidence,
    causality.self_coincidence,
    causality.center_coincidence, # place(center_x, center_y) onto another shape's center, or the scene's
    causality.scene_prop_coincidence

    #MONKEYS.append(compression.mashup) # TODO should be a different module?

]

# The monkeys whose ShapeActionBatcher also batches per demo (they pass pair_index). They propose for all demos AND for each demo alone (lots more
# programs), so they get their own, bigger scoring budget: CLOCK_BUDGET_PER_BATCHER_MONKEY
BATCHER_MONKEY_NAMES:set[str] = {monkey.__name__ for monkey in [
    shape_change.shapes_repaint, shape_change.shapes_move, shape_change.shapes_gravity,
]}

################################################
# Ranking programs, with random tie-breaks
################################################

# Callers pass dict.values(), not a list (the list() below copies it either way)
# score_key: which score to rank by. 'total_score' (pixels + simplicity) picks the winner; 'shape_diff_score' only picks some parents
def _rank_program_results(program_results:Iterable[dict], score_key:str='total_score') -> list[dict]:
    """Program results sorted best-first by a score (total_score by default). Ties: random order (not "earlier round wins")."""

    # Shuffle a copy first, then sort: sorted() is stable, so tied programs keep their (random) shuffled order
    shuffled_results:list[dict] = list(program_results)
    random.shuffle(shuffled_results)
    return sorted(shuffled_results, key=lambda item:item[score_key])


class _ParentCutoffs:
    """The score a program has to beat to make each parent island, updated as programs get scored."""

    def __init__(self, pair_count:int) -> None:
        # The best few total scores and shape-diff scores so far (sorted, at most K each)
        self.best_total_scores:list[float] = []
        self.best_shape_diff_scores:list[float] = []
        # The best score on each demo pair so far
        self.best_pair_scores:list[float] = [float('inf')] * pair_count

    def add(self, program_result:dict) -> None:
        """Remember a scored program, so the cutoffs can only get better."""
        bisect.insort(self.best_total_scores, program_result['total_score'])
        del self.best_total_scores[PARENTS_TOP_K_BY_SCORE:]
        bisect.insort(self.best_shape_diff_scores, program_result['shape_diff_score'])
        del self.best_shape_diff_scores[PARENTS_TOP_K_BY_SHAPE_DIFF:]
        self.best_pair_scores = [min(best, score) for best, score in zip(self.best_pair_scores, program_result['pair_scores'])]

    def could_be_parent(self, simplicity_score:float) -> bool:
        """Could a program this simple still make any island? (Every score it gets is >= its simplicity score)"""

        # Fewer than K programs on an island so far? Then anything still fits on it
        if len(self.best_total_scores) < PARENTS_TOP_K_BY_SCORE or len(self.best_shape_diff_scores) < PARENTS_TOP_K_BY_SHAPE_DIFF:
            return True

        # Strictly worse than every island's cutoff = it can't even tie its way in (ties are broken randomly).
        # An island that's switched off (K = 0, or no demo islands in an ablation) has no cutoff
        cutoffs:list[float] = self.best_total_scores[-1:] + self.best_shape_diff_scores[-1:]
        if ABLATION_STUDY_MODE != "NO DEMO ISLANDS":
            cutoffs += self.best_pair_scores
        return any(simplicity_score <= cutoff for cutoff in cutoffs)

################################################
# *NOW* LET'S DO AN EVOLUTIONARY SEARCH...
################################################

# Optional best_so_far: a list the CALLER owns. Every round we refill it with the current ranking, so if
# the attempt's clock runs out mid-search (func_timeout in main_solver), the caller still has the best programs
# found so far instead of nothing. (Some levels find their program in round 0, but round 1 alone takes >10s.)
# Optional crashes: another caller-owned list. A monkey that crashes just proposes nothing that round (instead of
# killing the whole attempt), and its traceback goes in here, so the bug still shows up in the solver's result.
def search_transform(train_pairs:list[dict], hyperparams:int=0, trace=None, best_so_far:Optional[list[dict]]=None,
                     crashes:Optional[list[str]]=None)->list[dict]:

    # TODO: Don't re-parent already expanded programs?

    # TODO: Annotate all scenes (TODO: also do this again after applying ANY action in the DSL)
    # oh wait no wouldn't that screw up order? eh. could always do start_ ... or forget about it,
    # it's unlikely a rule in ARC would do that
    train_pairs = annotate.apply_annotate(train_pairs)

    # Program results: all (already tested) and parents for monkeys to evolve on
    all_program_results:dict = {}
    parent_program_results:list[dict] = []

    # Monkeys that rewrite a block mid-program look up the scenes just before it here (this parse's scenes, so reset)
    reset_prefix_scenes(train_pairs, all_program_results)

    # Programs that crashed (or timed out) on some pair: dropped from the search, but remembered so they're not re-run
    crashed_programs:set[str] = set()

    # Start with the blank program (it does nothing, so it can't crash: the population is never empty)
    root_program_result:dict = get_transform_program_result('', train_pairs) # pyright: ignore[reportAssignmentType]
    root_program_result['monkey'] = '(root)'
    all_program_results[''] = root_program_result
    parent_program_results.append(root_program_result)

    # Each island's cutoff so far: a program whose simplicity alone is worse than all of them can't become a parent
    # (or the winner), so it doesn't need to run at all. Cutoffs only get better, so this never drops a later parent
    parent_cutoffs:_ParentCutoffs = _ParentCutoffs(len(train_pairs))
    parent_cutoffs.add(root_program_result)

    # Run til outta rounds, or no improvement for a while
    previous_best_total_score:float
    current_best_total_score:float = float('inf')
    no_improvement_countdown:int = EVO_EARLY_STOP_COUNTDOWN

    # Every monkey crash reported so far (see crashes above)
    seen_monkey_tracebacks:set[str] = set()

    # For N generations (or no improvement for a while)
    current_monkey_cap:int = MONKEY_CAP_START
    for generation in range(EVO_MAX_GENERATIONS):

        # Wall-clock timing for this round
        round_start_time:float = time.time()

        # Fresh round, fresh apply_fit cache (so it can't grow forever across a whole level)
        clear_fit_cache()
        clear_attract_cache() # ...and apply_attract's

        #########################################
        # 1. Get a bunch of descendants (run parent_program_results thru all Code Monkeys)
        #########################################

        programs_by_monkey:dict[str, list[str]] = {}
        count_monkey_suggestions:dict = {} # for debug
        crashed_monkey_names:list[str] = [] # Monkeys that crashed this round, for the trace
        for monkey in MONKEYS:

            # Log the monkey's name, for debug
            monkey_name = monkey.__name__

            # Monkey returns a SET of unique program strings.
            # One buggy monkey shouldn't throw away everything the others found, so a crash = no proposals this round.
            # (The try costs nothing until something actually raises)
            try:
                suggested_programs:set = monkey(train_pairs, parent_program_results)
            except Exception:
                crashed_monkey_names.append(monkey_name)
                monkey_traceback:str = traceback.format_exc()
                # Each different crash gets reported once, so a monkey that crashes every round doesn't flood anything
                if monkey_traceback not in seen_monkey_tracebacks:
                    seen_monkey_tracebacks.add(monkey_traceback)
                    print(f"Monkey {monkey_name} crashed:\n{monkey_traceback}")
                    if crashes is not None:
                        crashes.append(monkey_traceback)
                    if trace:
                        trace.log("search_transform", f"### 💀 Monkey {monkey_name} CRASHED! Traceback:\n\n{monkey_traceback}")
                suggested_programs = set()

            # Filter out the ones we've already tested:
            # TODO what if two monkeys recommend same program before they're scored?... eh whatever...
            suggested_programs_new:list[str] = [p.strip() for p in suggested_programs
                                                if p.strip() not in all_program_results and p.strip() not in crashed_programs]

            # Cap to a maximum amount of suggestions:
            how_many_monkeys = min( len(suggested_programs_new), current_monkey_cap )
            suggested_programs_new = random.sample(suggested_programs_new, how_many_monkeys)

            # Save programs to score, grouped by monkey
            programs_by_monkey[monkey_name] = suggested_programs_new
            count_monkey_suggestions[monkey_name] = len(suggested_programs_new)
        
        #for pr in parent_program_results:
        #    pr['had_kids'] = True # prevent revisit? force more explore?
        
        # Aside: lowering the monkey cap next round...
        current_monkey_cap = round( current_monkey_cap * MONKEY_CAP_DISCOUNT )
        current_monkey_cap = max(MONKEY_CAP_MIN, current_monkey_cap) # floor

        #########################################
        # 2. Score all the new suggested programs, remember their results
        #########################################

        # Wall-clock budget: score each monkey's programs until time runs out,
        # so expensive monkeys on huge levels don't eat the whole budget
        count_monkey_actually_tested:dict[str, int] = {}  # How many programs each monkey actually scored
        count_skipped_by_simplicity:int = 0 # How many were skipped instead, by the simplicity bound
        count_crashed_this_round:int = 0 # How many crashed (or timed out), so got dropped
        for monkey_name, programs in programs_by_monkey.items():
            random.shuffle(programs) # randomize which ones get scored within the budget
            tested_count:int = 0
            clock_budget:float = CLOCK_BUDGET_PER_BATCHER_MONKEY if monkey_name in BATCHER_MONKEY_NAMES else CLOCK_BUDGET_PER_MONKEY
            t0:float = time.time()
            for program in programs:
                if time.time() - t0 > clock_budget:
                    break
                # Too complicated to make any island, even with zero errors? Skip it (not saved, so it can be re-checked later)
                if not parent_cutoffs.could_be_parent(get_simplicity_score(program)):
                    count_skipped_by_simplicity += 1
                    continue
                new_program_result:Optional[dict] = get_transform_program_result(program, train_pairs)
                tested_count += 1
                # Crashed on some pair? Drop it: never a parent, the winner, or a runner-up
                if new_program_result is None:
                    crashed_programs.add(program)
                    count_crashed_this_round += 1
                    continue
                new_program_result['monkey'] = monkey_name
                all_program_results[program] = new_program_result # Save it: remember what's been tested!
                parent_cutoffs.add(new_program_result)
            count_monkey_actually_tested[monkey_name] = tested_count

        #########################################
        # 3. Select new parent programs (islands: the best by total score, the best by shape diff, the best on each pair)
        #########################################

        # Ranked by total score (pixels + simplicity): the real ranking, the winner comes from here
        all_program_results_sorted = _rank_program_results(all_program_results.values()) # Ties: random, not earlier round
        overall_best:dict = all_program_results_sorted[0]

        # Publish this round's ranking for the caller (see best_so_far above). Slice-assign, so it's the SAME list object
        if best_so_far is not None:
            best_so_far[:] = all_program_results_sorted
        
        # Score for each train pair: Best 1 (unless an ablation switches these islands off)
        pair_bests:list[dict] = [] if ABLATION_STUDY_MODE == "NO DEMO ISLANDS" else \
            [ min(all_program_results.values(), key=lambda item:item['pair_scores'][i]) for i in range(len(train_pairs)) ]

        # The top K by total score (the overall best included; slicing past the end is fine if there are fewer)
        top_by_score:list[dict] = all_program_results_sorted[:PARENTS_TOP_K_BY_SCORE]

        # The top K by shape-diff score: right shapes in roughly the right spots, wrong pixels or not.
        # (So a "close, but shapes in the wrong place" program can still be a parent, even when e.g. "hide everything"
        # gets more pixels right.) Only picks parents: the winner is still picked by total score
        top_by_shape_diff:list[dict] = _rank_program_results(all_program_results.values(), 'shape_diff_score')[:PARENTS_TOP_K_BY_SHAPE_DIFF]

        # list of potential next parents (not yet de-duplicated)
        bests:list[dict] = top_by_score + top_by_shape_diff + pair_bests

        # De-duplicate, and they're the parents now
        parent_program_results = []
        for program_result in bests:
            if program_result not in parent_program_results:
                parent_program_results.append(program_result)

        #########################################
        # 4. Early stop?
        #########################################
        
        _LETS_QUIT:bool = False
        previous_best_total_score = current_best_total_score
        current_best_total_score = min(current_best_total_score, overall_best['total_score'])
        if previous_best_total_score==current_best_total_score:
            # No improvement? Count down...
            no_improvement_countdown -= 1
            if no_improvement_countdown==0:
                _LETS_QUIT = True # Early stop! Not doing break /yet/, so we can trace CoT
        else:
            # Improvement? Reset countdown!
            no_improvement_countdown = EVO_EARLY_STOP_COUNTDOWN

        #########################################
        # Log all this in the Chain of Thought! Then early stop if LETS_QUIT
        #########################################

        if trace:

            this_round_trace:str = ""

            # Round # with wall-clock time and best score
            round_elapsed:float = time.time() - round_start_time
            this_round_trace += f'--- Round {generation} | {round_elapsed:.1f}s | best total score: {current_best_total_score} '
            if previous_best_total_score==current_best_total_score:
                this_round_trace += '(stable) ' # lack of improvement?
            this_round_trace += '-----------------\n\n'

            # Counts for what each monkey suggested vs actually tested
            total_added = sum( x for x in count_monkey_suggestions.values() )
            total_actually_tested = sum( x for x in count_monkey_actually_tested.values() )
            this_round_trace += f"+{total_added} programs (+{total_actually_tested} actually tested)" # total suggestions
            total_checked:int = total_actually_tested + count_skipped_by_simplicity
            this_round_trace += f" (skipped {count_skipped_by_simplicity} of {total_checked} checked, by the simplicity bound)"
            this_round_trace += f" (dropped {count_crashed_this_round} that crashed, {len(crashed_programs)} so far)"
            for name,added in count_monkey_suggestions.items():
                tested:int = count_monkey_actually_tested.get(name, 0)
                crashed_note:str = " 💀 CRASHED" if name in crashed_monkey_names else ""
                if added > 0:
                    this_round_trace += f"\n* {name}: +{added} ({tested}){crashed_note}"
                else:
                    this_round_trace += f"\n* {name}: +{added}{crashed_note}"

            # The best programs this round
            for program_result in parent_program_results:
                this_round_trace += "\n\n"
                this_round_trace += f"""
-- score: {program_result['total_score']} --
train wrong px: { ','.join([str(wrong) for wrong in program_result['pair_wrong_pixels']]) } (total: {program_result['total_wrong_pixels']})
shape diff: { ','.join([f"{diff:.2f}" for diff in program_result['pair_shape_diffs']]) } (shape-diff score: {program_result['shape_diff_score']:.2f})
suggested by: {program_result['monkey']}

{program_result['program']}
""".strip()

            # Log in Reasoning Trace
            trace.log("search_transform", this_round_trace)
        
        # Early stop
        if _LETS_QUIT:
            break


    # Return sorted list of best program_results!
    best_program_results:list[dict] = _rank_program_results(all_program_results.values()) # Just values (ties: random)
    return best_program_results

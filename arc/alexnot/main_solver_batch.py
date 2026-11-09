"""
Batch solve: lots of ARC levels in parallel, on all our CPUs!
(Moved here from the bottom of main_solver.py, so that file is just the solver itself.)
Used by tests/BLAST_EM.py, the webapp, and (soon) the Kaggle notebook.
"""

from typing import Callable, Dict, List, Optional

import os
import time
import json
import threading
import traceback
import multiprocessing
from pathlib import Path
from multiprocessing.connection import Connection

from .main_solver import solve, RED_CROSS
from .config import CONFIG_NAME

#####################################################################
## BATCH SOLVE: lots of levels in parallel, on all our CPUs
#####################################################################

# The parallel engine, here in alexnot/ (which must never import from tests/), so BLAST_EM, the webapp,
# and the Kaggle notebook all share ONE engine. blast()'s results_path is optional, and its stop_event
# lets the webapp cancel a batch. solve_batch() is the wrapper that returns a Kaggle submission.

# How many CPUs may we use? os.process_cpu_count() only exists in Python 3.13+, and Kaggle's
# notebooks may run an older Python (where it'd crash). Nothing's hardcoded to one machine: on a 40-CPU
# Kaggle box this gives 40 (or whatever share of CPUs the container lets us use).
# (Kaggle's 4xL4 box is 48 hyperthreads with a cgroup quota of 46, and this gives 46.)
def available_cpu_count() -> int:
    """The number of CPUs this process may use (NOT the whole host's count, in a container)."""
    # Python 3.13+: respects CPU affinity & the -X cpu_count option
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    # Older Python on Linux (Kaggle, RunPod): the CPUs we're actually allowed to run on
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) # pyright: ignore[reportAttributeAccessIssue] (Linux-only, and the hasattr() above guards it, so Pylance on a Mac is just being cautious)
    # Older Python elsewhere: best guess
    return os.cpu_count() or 1


#####################################################################
## The worker: solve ONE level, send back the answers
#####################################################################

def solve_one_level(scrubbed_level: dict, connection: Connection, random_seed: int = -1, bayes: bool = False) -> None:
    """Runs in its own process. The level it gets has NO test answers in it. bayes=True: the webapp's posterior too."""
    start_time: float = time.time()
    try:
        # show_reasoning=True, so the chain of thought comes back too (solve() builds it either way, so it's free).
        # (Pass the seed on!) The Kaggle submission strips the non-standard fields later, in solve_batch()
        result: dict = solve(scrubbed_level, show_reasoning=True, random_seed=random_seed, bayes=bayes)
        connection.send({
            "status": "ok",
            "test_solutions": result["test_solutions"],
            "confidence": result.get("confidence"),  # None until solve() reports it (see the TODO in main_solver.py)
            "seconds": time.time() - start_time,

            # The winning program, its parse, and any caught crashes (so all three are visible in results.jsonl)
            "best_program": result["best_program"],
            "best_parse": result["best_parse"],
            "crashes": result["crashes"],

            # Did it find a program that gets ALL demos right? And the chain of thought itself
            "solved_all_demos": result["solved_all_demos"],
            "reasoning_trace": result["reasoning_trace"],
            "bayes_posterior": result.get("bayes_posterior"),  # None unless bayes=True (and solve() found programs)

            # How long each phase took (BLAST_EM's summary.md sums them over all levels)
            "seconds_search_parse": result["seconds_search_parse"],
            "seconds_search_transform": result["seconds_search_transform"],
            "seconds_run_on_tests": result["seconds_run_on_tests"],
        })
    except Exception:
        connection.send({"status": "error", "error": traceback.format_exc(), "seconds": time.time() - start_time})
    connection.close()


#####################################################################
## The boss: hand out levels, collect answers
#####################################################################

def blast(level_keys: list[str], scrubbed_levels: dict[str, dict], workers: int, timeout_seconds: float,
          results_path: Optional[Path] = None, stop_event: Optional[threading.Event] = None,
          random_seed: int = -1, on_level_done: Optional[Callable[[str, dict], None]] = None,
          bayes: bool = False) -> dict[str, dict]:
    """Keep `workers` levels running at once. Each level gets a fresh process, so a hung or crashed
    level can be killed without taking anything else down. Returns {level_key: outcome}."""
    
    # results_path=None means "don't log to a file" (the webapp doesn't need one).
    # Setting stop_event (from another thread) stops the batch early: running workers get killed by
    # the `finally` below, and only the levels that already finished are returned.
    outcomes: dict[str, dict] = {}
    waiting_keys: list[str] = list(level_keys)

    # level_key -> (process, the boss's end of the pipe, start time)
    running: dict[str, tuple[multiprocessing.Process, Connection, float]] = {}

    def finish(level_key: str, outcome: dict) -> None:

        # Timed out / crashed / errored levels never got to report these, so fill them in:
        # no program that solves the demos, and a one-line note instead of a chain of thought
        outcome.setdefault("solved_all_demos", False)
        outcome.setdefault("reasoning_trace", f"(No reasoning trace: this level's status was '{outcome['status']}')")

        # Record it, and append it to results.jsonl RIGHT NOW (crash-safety)
        outcomes[level_key] = outcome
        if results_path is not None:  # Only if we were given a log file
            with open(results_path, "a") as file:
                # which config this run used, on every line (so paper runs say which config they used)
                file.write(json.dumps({"level": level_key, "config": CONFIG_NAME, **outcome}, default=lambda value: value.tolist()) + "\n")  # .tolist() for stray numpy ints
        status: str = outcome["status"]
        emoji: str = {"ok": "💥", "error": "💀", "timeout": "⏰", "crashed": "💀"}[status]
        
        # Also shout about attempts that crashed inside solve() (they're caught, so the status is still "ok")
        crash_count: int = len(outcome.get("crashes", []))
        crash_note: str = f" 💀 {crash_count} ATTEMPT(S) CRASHED" if crash_count else ""
        print(f"[{len(outcomes)}/{len(level_keys)}] {emoji} {level_key} ({outcome['seconds']:.1f}s){'' if status == 'ok' else ' ' + status.upper()}{crash_note}")

        # Tell whoever's listening (e.g. the webapp) right away, instead of only at the very end
        if on_level_done is not None:
            on_level_done(level_key, outcome)

    try:
        while waiting_keys or running:

            # Cancelled? Stop now (the `finally` kills whoever's still running)
            if stop_event is not None and stop_event.is_set():
                print(f"🛑 Batch cancelled after {len(outcomes)}/{len(level_keys)} levels")
                break

            # Hand out levels until every worker is busy
            while waiting_keys and len(running) < workers:
                level_key: str = waiting_keys.pop(0)
                boss_end, worker_end = multiprocessing.Pipe(duplex=False)
                process = multiprocessing.Process(target=solve_one_level, args=(scrubbed_levels[level_key], worker_end, random_seed, bayes))
                process.start()
                worker_end.close()  # The boss only listens
                running[level_key] = (process, boss_end, time.time())

            # Check on everyone who's running
            for level_key, (process, boss_end, start_time) in list(running.items()):
                if boss_end.poll():
                    # Got an answer (or an error report)!
                    try:
                        outcome: dict = boss_end.recv()
                    except EOFError:
                        outcome = {"status": "crashed", "seconds": time.time() - start_time}
                    # It already answered, so a worker that hangs on its way out mustn't keep the boss waiting forever
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                elif not process.is_alive():
                    # Died without saying anything (e.g. out of memory)
                    outcome = {"status": "crashed", "seconds": time.time() - start_time}
                elif time.time() - start_time > timeout_seconds:
                    # Took too long: no answer for this one
                    process.kill()
                    process.join()
                    outcome = {"status": "timeout", "seconds": time.time() - start_time}
                else:
                    continue
                boss_end.close()
                del running[level_key]
                finish(level_key, outcome)

            time.sleep(0.05)  # The boss mostly naps
    finally:
        # Ctrl-C (or anything else) → don't leave orphan workers behind
        for process, _, _ in running.values():
            process.kill()

    return outcomes


#####################################################################
## The wrapper: many levels in → Kaggle submission out
#####################################################################

def solve_batch(levels: Dict[str, Dict], workers: Optional[int] = None, timeout_seconds: float = 120.0,
                stop_event: Optional[threading.Event] = None, random_seed: int = -1,
                on_level_done: Optional[Callable[[str, Dict], None]] = None, bayes: bool = False) -> Dict[str, List[Dict]]:
    """
    Solve lots of levels in parallel, using all the CPUs we're allowed to use.
    In: {level_id: level_data}. Out: a Kaggle submission.json dict,
    {level_id: [{"attempt_1": grid, "attempt_2": grid}, ...one per test input]}.
    EVERY level_id gets an answer (Kaggle requires it): levels that time out / crash / error
    (or never ran, because of stop_event) get the same RED_CROSS guess that solve() gives up with.
    """

    # process_cpu_count, NOT cpu_count: in a container (RunPod, Kaggle) cpu_count() is the whole host,
    # and that many workers would fight over our real share of CPUs (same reasoning as BLAST_EM's --workers)
    if workers is None:
        workers = available_cpu_count()  #  Same thing, but also works on Python < 3.13 (Kaggle?)

    # No need to scrub the test answers here: solve() already throws them away first thing
    level_ids: List[str] = list(levels.keys())
    outcomes: Dict[str, Dict] = blast(level_ids, levels, workers, timeout_seconds, stop_event=stop_event, random_seed=random_seed,
                                      on_level_done=on_level_done, bayes=bayes)

    # Kaggle format: the solver's answers, or a red cross per test input if there aren't any
    # NOTHING ELSE, NO COT.
    submission: Dict[str, List[Dict]] = {}
    for level_id in level_ids:
        outcome: Optional[Dict] = outcomes.get(level_id)
        if outcome is not None and outcome["status"] == "ok":
            submission[level_id] = outcome["test_solutions"]
        else:
            submission[level_id] = [{"attempt_1": RED_CROSS, "attempt_2": RED_CROSS} for _ in levels[level_id]["test"]]
    return submission

"""
A HARD time cap per program, without a thread per program.

CLOCK_BUDGET_PER_MONKEY is only checked BETWEEN programs, so one heavy program can blow right through it
(one apply_fit search once took 35s). So instead: start a deadline before scoring a program, and have the
heavy loops (like apply_fit's search) call check_deadline() as they go. Past the deadline,
it raises ProgramTimedOut, and apply_transform's usual "program crashed" handling makes that program just fail.

Usage:
    start_deadline(TIME_LIMIT_PER_PROGRAM)
    try:
        ... run the program ...
    finally:
        clear_deadline() # else a stale deadline would fire on the NEXT, unrelated run (e.g. the test inputs)
"""

import time
from typing import Optional

class ProgramTimedOut(Exception):
    """A program ran past its deadline."""

# When the current program has to be done by (time.time() seconds), or None = no deadline
# Heads up: one global per process. Fine for now: each process scores one program at a time.
_deadline:Optional[float] = None

def start_deadline(seconds:float) -> None:
    """The current program has this many seconds, from now."""
    global _deadline
    _deadline = time.time() + seconds

def clear_deadline() -> None:
    """No more deadline (the program's done)."""
    global _deadline
    _deadline = None

def check_deadline() -> None:
    """Call this inside heavy loops: raises ProgramTimedOut if the current program's out of time."""
    if _deadline is not None and time.time() > _deadline:
        raise ProgramTimedOut()

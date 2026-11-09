"""A crashing monkey shouldn't kill the whole attempt: it proposes nothing, and its traceback lands in the result's crashes."""
import json
import sys
from typing import Any
from pathlib import Path

# So alexnot imports as a package (it lives in arc/)
_root = str(Path(__file__).resolve().parent.parent / "arc")
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.main_solver import solve
import alexnot.transform.search_transform
# (The package re-exports the search_transform FUNCTION under the same name, so grab the module itself)
# (Typed Any: a plain ModuleType doesn't know it has MONKEYS, so the type checker would flag the swap below)
search_transform_module:Any = sys.modules['alexnot.transform.search_transform']

ROOT = Path(__file__).resolve().parent.parent
LEVELS_DIR = ROOT / "arc" / "levels"


def always_crashing_monkey(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """A deliberately broken monkey: blows up every single time it's called."""
    raise KeyError('deliberate_test_crash')


def test_crashing_monkey_does_not_kill_the_attempt() -> None:
    # An easy level (shift 1 down), so the other monkeys can solve it in round 0 or 1
    with open(LEVELS_DIR / "arc-1-train" / "25ff71a9.json", "r") as file:
        level:dict = json.load(file)

    # Put the broken monkey FIRST, so it runs before every other monkey, every round
    original_monkeys:list = search_transform_module.MONKEYS
    search_transform_module.MONKEYS = [always_crashing_monkey] + original_monkeys
    try:
        result:dict = solve(level, show_reasoning=False, profiling=True) # profiling=True: no thread, easier to debug
    finally:
        search_transform_module.MONKEYS = original_monkeys

    # The other monkeys still solved every demo...
    assert result['solved_all_demos'], "a crashing monkey killed the attempt"
    # ...and the crash is still visible, not swallowed silently
    assert any('deliberate_test_crash' in crash for crash in result['crashes']), "the monkey's crash isn't in the crashes list"
    print("✅ The crashing monkey got caught, and its traceback is in the crashes list")


if __name__ == "__main__":
    test_crashing_monkey_does_not_kill_the_attempt()

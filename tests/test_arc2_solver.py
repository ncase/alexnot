import json
import sys
from pathlib import Path

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent / "arc")  # alexnot lives in arc/
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.main_solver import solve

ROOT = Path(__file__).resolve().parent.parent
LEVELS_DIR = ROOT / "arc" / "levels"

####################################################################################
####################################################################################

if __name__ == "__main__":

    #######################################################################
    ## CHOOSE THE LEVEL
    #######################################################################

    # path = LEVELS_DIR / "arc-2-eval/62593bfd.json" # up twin else down
    # path = LEVELS_DIR / "arc-1-eval/16b78196.json" # fit pieces into the band's gaps, jigsaw-style
    # path = LEVELS_DIR / "arc-2-eval/e8686506.json" # crop to the frame(s), fit pieces into the gaps
    path = LEVELS_DIR / "mini-arc/mini_aeugn2pfna6pvwdt.json" # recolor pink, then inline yellow

    #######################################################################
    ## NOW run the full solver on that level
    #######################################################################

    # Load level
    with open(path, "r") as file:
        level = json.load(file)

    # Solve it!
    print(f"Solving {path.name}...")
    result = solve(level, show_reasoning=True, profiling=True) # profiling=True to allow debug-step-through

    # Show what happened
    print("\n" + "=" * 60)
    print("REASONING TRACE:")
    print("=" * 60)
    print(result['reasoning_trace'])

    print("\n" + "=" * 60)
    print("TEST SOLUTIONS:")
    print("=" * 60)
    for i, sol in enumerate(result['test_solutions']):
        print(f"\nTest #{i} attempt_1:")
        for row in sol['attempt_1']:
            print("  ", row)
        print(f"Test #{i} attempt_2:")
        for row in sol['attempt_2']:
            print("  ", row)

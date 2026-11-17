# arc/ — the AlexNot symbolic solver

AlexNot solves ARC puzzles by **program induction**: it searches for a short,
readable Python program that turns every demo input into its demo output, then
runs that program on the test input.

- `alexnot/` — the solver. `alexnot/main_solver.py` has the entry point,
  `solve(level_data)`, which returns predicted test outputs + a reasoning trace.
- `alexnot/tests/` — test scripts for the engine's parts (parse, monkeys, DSL).
- `webapp/` — a Flask app for watching the solver work on a level.
- `levels/` — puzzle JSONs: `arc-1-train/`, `arc-1-eval/`, `arc-2-train-unique/`,
  `arc-2-eval/`, `mini-arc/`, `concept-arc/`, `custom/`.
  (`arc-2-train-unique/` only has the levels *unique* to ARC-AGI-2's training set:
  the original also contains all the ARC-AGI-1 training & evaluation levels, which
  already live in `arc-1-train/` and `arc-1-eval/`.)

## Setup

Uses the repo's venv (`../.venv`, Python 3.13). If you need to rebuild it:

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt   # from the repo root (any Python 3.11+ works)
```

## How to run things

Always run from **this folder** (`arc/`), so `alexnot` imports as a package:

```bash
cd arc

# The webapp: pick a level, solve it, eyeball the grids → http://127.0.0.1:9999/
../.venv/bin/python webapp/webapp.py
../.venv/bin/python webapp/webapp.py --port 8888   # ...or pick your own port → http://127.0.0.1:8888/

# Engine tests
../.venv/bin/python -m alexnot.tests.test_parse_annotate    # parse + annotate a level, print the shapes
../.venv/bin/python -m alexnot.tests.test_apply_transform   # run a hand-written DSL program (exit 0 = all demos pass)
../.venv/bin/python -m alexnot.tests.test_code_monkey       # what does one monkey propose? top 10 by score
../.venv/bin/python -m alexnot.tests.test_search_parse      # rank parser settings for a level
```

The full-solver test and the benchmark live in the repo-level `tests/` folder.
Run those from the **repo root** instead:

```bash
.venv/bin/python -m tests.test_arc2_solver        # the full solver on one level
.venv/bin/python -m tests.BLAST_EM --datasets mini-arc     # parallel, Kaggle-style scoring (see its docstring)

# ...or the same, but with 2x search time, in case you have a slower CPU (see below)
.venv/bin/python -m tests.BLAST_EM --datasets mini-arc --config double_time
```

**Why double time?** The search runs on a clock (e.g. 30 seconds per attempt), so
a slower CPU gets less search done in the same seconds, and scores lower. CPU speeds
vary a lot, and we only tuned & tested the default time limits on one local machine.
So for replicable scores, use `--config double_time` (`alexnot/config/double_time.py`):
it doubles every time limit, so the search can reach its maximum performance even on
a slower CPU.

Each test script has a **"CHOOSE"** section at the top: edit it to pick the
level, parse config, DSL program, or monkey. Add `ARC_DEBUG=1` before the
command for debug logging.

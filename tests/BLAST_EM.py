"""
BLAST 'EM! 💥 Run the symbolic solver on LOTS of levels in parallel, Kaggle-style.

One "boss" process hands out levels (test answers scrubbed!) to worker processes,
one level at a time, and collects their answers into a Kaggle-format submission.json.
Only AFTER everything's done does the boss score the answers against the ground truth.

A "dataset" is any level list in arc/levels/*.txt (one id per line, "id - description"),
so arc-2-eval.txt gets all 120 (incl. the 6 that live in the arc-1-eval/ folder),
and custom lists like symbolic_only.txt work too.

Everything is written as it happens, so even if the run gets killed halfway,
you keep everything up to that point. Output goes in tests/blast_logs/<timestamp>/:
    results.jsonl    one line per level, appended as each one finishes
                     (incl. its reasoning_trace and solved_all_demos: True if a program got ALL demos right)
    submission.json  Kaggle format: {"<list>/<id>": [{"attempt_1": grid, "attempt_2": grid}, ...]}
    scores.csv       per-level score (pass@2, partial credit per test input)
    summary.md       the scoreboard

Usage (from the repo root):
    .venv/bin/python -m tests.BLAST_EM                                      # all of arc-2-eval, all CPUs
    .venv/bin/python -m tests.BLAST_EM --datasets concept-arc mini-arc      # any combination of lists
    .venv/bin/python -m tests.BLAST_EM --datasets all -n 20 --workers 8     # 20 random levels per list
    .venv/bin/python -m tests.BLAST_EM --datasets mini-arc --seed 1         # one fixed seed for every level (default -1 = fresh random seeds)
    .venv/bin/python -m tests.BLAST_EM --datasets arc-1-train --first 100   # the first 100 in the list (same ones every time)
    .venv/bin/python -m tests.BLAST_EM --datasets mini-arc arc-1-train:100  # all of mini-arc, + the first 100 of arc-1-train
    .venv/bin/python -m tests.BLAST_EM --datasets mini-arc --config double_time     # any arc/alexnot/config/<name>.py
"""

import os
import statistics
import csv
import copy
import json
import sys
import time
import random
import argparse
from pathlib import Path

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent / "arc")  # alexnot lives in arc/
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.main_solver_batch import blast, RED_CROSS, available_cpu_count
from alexnot.config import CONFIG_NAME

ROOT = Path(__file__).resolve().parent.parent
LEVELS_DIR = ROOT / "arc" / "levels"
LOGS_DIR = ROOT / "tests" / "blast_logs"

# What "--datasets all" means: the big public sets (not arc-3-test, not custom lists)
ALL_DATASETS: list[str] = ["arc-1-train", "arc-1-eval", "arc-2-train", "arc-2-eval", "mini-arc", "concept-arc"]


#####################################################################
## Finding levels
#####################################################################

def read_level_list(list_name: str) -> list[str]:
    """Read the level ids from arc/levels/<list_name>.txt (same format the webapp reads: 'id - description')."""
    lines: list[str] = (LEVELS_DIR / f"{list_name}.txt").read_text().splitlines()
    # Skip blank lines & comment lines, keep just the id before the " - "
    return [line.split("-")[0].strip() for line in lines if line.strip() and not line.startswith("#")]


def find_level_file(list_name: str, level_id: str) -> Path:
    """The level's JSON: the same-named folder first, else whichever folder has it (e.g. arc-2-eval's arc-1 repeats)."""
    same_named_folder_path: Path = LEVELS_DIR / list_name / f"{level_id}.json"
    if same_named_folder_path.exists():
        return same_named_folder_path
    matches: list[Path] = sorted(LEVELS_DIR.glob(f"*/{level_id}.json"))
    if not matches:
        raise FileNotFoundError(f"No JSON for level '{level_id}' (from {list_name}.txt) in any arc/levels/ folder")
    return matches[0]


#####################################################################
## Scoring (only after everyone's done!)
#####################################################################

def score_level(answers: list, true_outputs: list) -> float:
    """Kaggle-style: a test input counts if attempt_1 OR attempt_2 is exactly right.
    The level's score = fraction of its test inputs that count."""
    if not answers:
        return 0.0  # timed out / crashed / errored
    correct: int = 0
    for attempts, true_output in zip(answers, true_outputs):
        if attempts["attempt_1"] == true_output or attempts["attempt_2"] == true_output:
            correct += 1
    return correct / len(true_outputs)


def median_of(values: list[float]) -> float:
    """The middle value (0 when there are none, e.g. every level timed out)."""
    return statistics.median(values) if values else 0.0


def write_summary(run_dir: Path, level_keys: list[str], outcomes: dict[str, dict], scores: dict[str, float],
                  workers: int, wall_seconds: float, seed: int) -> str:
    """The scoreboard: per list, total, timing, and precision (once solve() reports confidence)."""
    list_names: list[str] = list(dict.fromkeys(key.split("/")[0] for key in level_keys))
    lines: list[str] = [
        f"# BLAST 'EM 💥 {run_dir.name}",
        "",
        f"- Random seed: {seed} (PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')})",
        f"- Config: {CONFIG_NAME} (arc/alexnot/config/{CONFIG_NAME}.py)",
        f"- Workers: {workers} (we may use {available_cpu_count()} of this machine's {os.cpu_count()} CPUs)", 
        f"- Wall time: {wall_seconds:.0f}s for {len(level_keys)} levels",
        f"- Solver time per level: median {median_of([o['seconds'] for o in outcomes.values()]):.1f}s, "
        f"mean {sum(o['seconds'] for o in outcomes.values()) / max(len(outcomes), 1):.1f}s, "
        f"max {max((o['seconds'] for o in outcomes.values()), default=0):.1f}s",
        # Compare runs by these medians: one slow outlier level can swing a total or a mean, but not a median
        # (timed-out / errored levels don't report phases, so they're left out here)
        f"- Phase medians per level: search parse {median_of([o['seconds_search_parse'] for o in outcomes.values() if 'seconds_search_parse' in o]):.2f}s, "
        f"search transform {median_of([o['seconds_search_transform'] for o in outcomes.values() if 'seconds_search_transform' in o]):.2f}s",
        # Where the time went, summed over all levels (timed-out / errored levels don't report phases, so they add 0)
        f"- Phase totals: search parse {sum(o.get('seconds_search_parse', 0) for o in outcomes.values()):.1f}s, "
        f"search transform {sum(o.get('seconds_search_transform', 0) for o in outcomes.values()):.1f}s, "
        f"run on tests {sum(o.get('seconds_run_on_tests', 0) for o in outcomes.values()):.1f}s",
        "",
        "| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |",
        "|---|---|---|---|---|---|",
    ]
    for list_name in list_names + ["**TOTAL**"]:
        keys: list[str] = level_keys if list_name == "**TOTAL**" else [k for k in level_keys if k.startswith(list_name + "/")]
        total_score: float = sum(scores[k] for k in keys)
        timeouts: int = sum(1 for k in keys if outcomes[k]["status"] == "timeout")
        errors: int = sum(1 for k in keys if outcomes[k]["status"] in ("error", "crashed"))
        lines.append(f"| {list_name} | {len(keys)} | {total_score:.2f} | {100 * total_score / len(keys):.1f}% | {timeouts} | {errors} |")

    # Precision: when the solver says it's confident, how often is it actually right?
    confidences: dict[str, float] = {k: o["confidence"] for k, o in outcomes.items() if o.get("confidence") is not None}
    lines.append("")

    # Crashes that solve() caught (status still "ok", so the table above doesn't count them).
    # The tracebacks are in results.jsonl, under "crashes".
    crashed_keys: list[str] = [k for k in level_keys if outcomes[k].get("crashes")]
    lines.append(f"- **Caught solver crashes:** {len(crashed_keys)} levels had attempts crash"
                 + (f" (tracebacks in results.jsonl): {', '.join(crashed_keys)}" if crashed_keys else ""))
    if confidences:
        confident_keys: list[str] = [k for k, c in confidences.items() if c > 0]
        right: float = sum(scores[k] for k in confident_keys)
        lines.append(f"- **Precision:** {len(confident_keys)} levels had confidence > 0, "
                     f"and they scored {right:.2f} ({100 * right / max(len(confident_keys), 1):.1f}%)")
        lucky: int = sum(1 for k in level_keys if scores[k] > 0 and confidences.get(k, 0) == 0)
        lines.append(f"- Right anyway with confidence 0: {lucky} levels")
    else:
        lines.append("- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)")

    summary: str = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(summary)
    return summary


#####################################################################
## Main
#####################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="BLAST 'EM: run the solver on many levels in parallel")
    parser.add_argument("--datasets", nargs="+", default=["arc-2-eval"],
                        help="level lists from arc/levels/*.txt, or 'all' (default: arc-2-eval)")
    parser.add_argument("-n", type=int, default=None, help="random levels per list (default: every level)")

    # The FIRST n levels of each list, in file order (unlike -n's random pick), so daily runs stay comparable
    parser.add_argument("--first", type=int, default=None, help="first N levels per list, in file order (default: every level)")

    # Where the run folder goes, so tests/daily_blast.py can save into the git-tracked tests/daily-blasts/
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="folder to put the run's log folder in (default: tests/blast_logs)")
    # process_cpu_count, NOT cpu_count: in a container (RunPod, Kaggle) cpu_count() is the whole host
    # (192 on a 32-vCPU RunPod pod!), and that many workers would fight over our real share of CPUs
    # parser.add_argument("--workers", type=int, default=os.process_cpu_count(), help="parallel workers (default: all CPUs we may use)")
    
    # Same idea, via a helper that also works on Python < 3.13 (no os.process_cpu_count() there)
    parser.add_argument("--workers", type=int, default=available_cpu_count(), help="parallel workers (default: all CPUs we may use)")
    parser.add_argument("--timeout", type=float, default=120.0, help="max seconds per level (default: 120)")

    # Fresh random seeds by default: the same seed every run HIDES fragile code (a level that only passes on one
    # lucky tie-break). Each level's seed is in its reasoning_trace ("Random seed: N"), to replay it.
    # (Or pass one seed for every level's solve(), so two runs give the same answers)
    parser.add_argument("--seed", type=int, default=-1, help="random seed passed to every solve() (default: -1 = fresh per level)")

    # Which solver config to use: any arc/alexnot/config/<name>.py (default: $ALEXNOT_CONFIG if set, else "default")
    parser.add_argument("--config", default=os.environ.get("ALEXNOT_CONFIG", "default"),
                        help="solver config, a file name in arc/alexnot/config/ (default: default)")
    args = parser.parse_args()

    # HASH & CONFIG
    # if os.environ.get("PYTHONHASHSEED") != "0" or CONFIG_NAME != args.config:
    #     os.environ["PYTHONHASHSEED"] = "0"
    # No fixed PYTHONHASHSEED: a symbolic solver's results shouldn't depend on set/dict ordering,
    # so we WANT hash-order-fragile levels to show up as flaky. Uncomment the two lines above to bring it back.
    # So we only re-run for the config (alexnot/config reads ALEXNOT_CONFIG once, on import).
    if CONFIG_NAME != args.config:
        os.environ["ALEXNOT_CONFIG"] = args.config
        os.execv(sys.executable, [sys.executable, "-m", "tests.BLAST_EM", *sys.argv[1:]])

    list_names: list[str] = list(ALL_DATASETS if args.datasets == ["all"] else args.datasets)
    # A list can also have its own "first N" cap, written "<list>:N" (e.g. arc-1-train:100), so one run can mix
    # whole lists with capped ones (tests/daily_blast.py does). Split those caps off, so list_names stays plain names.
    first_per_list: dict[str, int] = {}
    for index, dataset in enumerate(list_names):
        list_name, _, first_text = dataset.partition(":")
        list_names[index] = list_name
        if first_text:
            first_per_list[list_name] = int(first_text)
    rng: random.Random = random.Random(42)  # deterministic random, plz.

    # Gather the levels. Keys are "<list>/<id>", since the same id can be in two lists
    level_keys: list[str] = []
    scrubbed_levels: dict[str, dict] = {}
    true_outputs: dict[str, list] = {}
    for list_name in list_names:
        level_ids: list[str] = read_level_list(list_name)
        if args.n is not None:
            level_ids = rng.sample(level_ids, min(args.n, len(level_ids)))
        # Just the top of the list (does nothing for lists shorter than --first)
        if args.first is not None:
            level_ids = level_ids[:args.first]
        # Same, for this one list's own "<list>:N" cap
        if list_name in first_per_list:
            level_ids = level_ids[:first_per_list[list_name]]
        for level_id in level_ids:
            level_key: str = f"{list_name}/{level_id}"
            level_data: dict = json.loads(find_level_file(list_name, level_id).read_text())
            # The boss keeps the answers; workers get a copy with the test outputs scrubbed
            true_outputs[level_key] = [pair["output"] for pair in level_data["test"]]
            scrubbed: dict = copy.deepcopy(level_data)
            for pair in scrubbed["test"]:
                pair.pop("output", None)
            level_keys.append(level_key)
            scrubbed_levels[level_key] = scrubbed

    # Where the logs go
    # run_dir: Path = LOGS_DIR / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{'+'.join(list_names)}"[:200]
    # Same name, but under --logs-dir (default: still tests/blast_logs)
    run_dir: Path = args.logs_dir / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{'+'.join(list_names)}"[:200]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"BLASTING {len(level_keys)} levels from {', '.join(list_names)} with {args.workers} workers, config '{CONFIG_NAME}' 💥")
    print(f"Logging to: {run_dir}\n")

    # BLAST 'EM
    start_time: float = time.time()
    # outcomes: dict[str, dict] = blast(level_keys, scrubbed_levels, args.workers, args.timeout, run_dir / "results.jsonl")
    outcomes: dict[str, dict] = blast(level_keys, scrubbed_levels, args.workers, args.timeout, run_dir / "results.jsonl",
                                      random_seed=args.seed)
    wall_seconds: float = time.time() - start_time

    # Kaggle-format submission (levels with no answer are left out, like an empty guess)
    # submission: dict[str, list] = {k: outcomes[k]["test_solutions"] for k in level_keys if outcomes[k]["status"] == "ok"}
    # Leaving levels out isn't Kaggle-valid: EVERY level needs two attempts per test input.
    # So levels that timed out / crashed / errored get red crosses (same as solve_batch() does)
    submission: dict[str, list] = {
        k: outcomes[k]["test_solutions"] if outcomes[k]["status"] == "ok"
        else [{"attempt_1": RED_CROSS, "attempt_2": RED_CROSS} for _ in true_outputs[k]]
        for k in level_keys
    }
    (run_dir / "submission.json").write_text(json.dumps(submission, default=lambda value: value.tolist()))

    # NOW score against the ground truth
    scores: dict[str, float] = {k: score_level(submission.get(k, []), true_outputs[k]) for k in level_keys}
    with open(run_dir / "scores.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["level", "score", "status", "confidence", "seconds"])
        for k in level_keys:
            writer.writerow([k, f"{scores[k]:.3f}", outcomes[k]["status"], outcomes[k].get("confidence"), f"{outcomes[k]['seconds']:.1f}"])

    # Tell me!    
    print("\n" + write_summary(run_dir, level_keys, outcomes, scores, args.workers, wall_seconds, args.seed))


if __name__ == "__main__":
    main()

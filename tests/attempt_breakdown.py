"""
ATTEMPT BREAKDOWN 🔢 For one blast run: which parse attempt (1/2/3) found the demo-perfect program, and how often
did that program ALSO get every test right?

main_solver tries up to 3 parse attempts and stops at the first one whose program gets ALL demos right, so the last
attempt it ran is the "winner". "Tests right" = the level's score is 1.0 (every test right, pass@2).

Usage (from the repo root):
    .venv/bin/python -m tests.attempt_breakdown                                  # the latest daily blast
    .venv/bin/python -m tests.attempt_breakdown tests/blast_logs/<run folder>    # any BLAST_EM run folder
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_BLASTS_DIR = ROOT / "tests" / "daily-blasts"


def latest_daily_blast() -> Path:
    """The newest daily blast run folder (they start with their timestamp, so sorting by name sorts by time)."""
    run_dirs: list[Path] = sorted(path for path in DAILY_BLASTS_DIR.iterdir() if (path / "results.jsonl").exists())
    return run_dirs[-1]


def attempts_run(reasoning_trace: str) -> int:
    """How many parse attempts ran, from the Summary's "// attempts: 1) 0.80s (parse #1), 2) ..." line."""
    attempts_line = re.search(r"// attempts: (.*)$", reasoning_trace, re.M)
    return attempts_line.group(1).count("(parse #") if attempts_line else 0


def breakdown(run_dir: Path) -> str:
    """One row per dataset (+ ALL): per attempt, tests-right / demos-right, and how many never got the demos."""

    # Every level's score (pass@2, 0..1)
    with open(run_dir / "scores.csv") as file:
        scores: dict[str, float] = {row["level"]: float(row["score"]) for row in csv.DictReader(file)}

    # dataset -> counts: "levels", "no demos", and (attempt, "demos") / (attempt, "tests") per winning attempt
    counts_per_dataset: dict[str, Counter] = defaultdict(Counter)
    with open(run_dir / "results.jsonl") as file:
        for line in file:
            result: dict = json.loads(line)
            # Level keys look like "<list>/<id>", same as in BLAST_EM
            counts: Counter = counts_per_dataset[result["level"].split("/")[0]]
            counts["levels"] += 1
            if not result["solved_all_demos"]:
                counts["no demos"] += 1
                continue
            winning_attempt: int = attempts_run(result["reasoning_trace"])
            counts[(winning_attempt, "demos")] += 1
            if scores.get(result["level"], 0.0) >= 0.999: # (scores.csv rounds to 3 decimals)
                counts[(winning_attempt, "tests")] += 1

    # Datasets in the order they first showed up, then everything added together
    all_counts: Counter = Counter()
    for counts in counts_per_dataset.values():
        all_counts.update(counts)
    rows: list[tuple[str, Counter]] = list(counts_per_dataset.items()) + [("ALL", all_counts)]

    # The table: "A1: 31/80 (19.4%)" = 80 levels got the demos right on attempt 1, 31 of those got every test right
    # too, and those 31 are 19.4% of all the dataset's levels
    lines: list[str] = [f"ATTEMPT BREAKDOWN 🔢 {run_dir.name}",
                        "(per attempt: tests right / demos right, and tests right as % of all levels)", ""]
    for dataset, counts in rows:
        total: int = counts["levels"]
        cells: list[str] = []
        for attempt in (1, 2, 3):
            tests_right: int = counts[(attempt, "tests")]
            cells.append(f"A{attempt}: {tests_right}/{counts[(attempt, 'demos')]} ({100 * tests_right / total:.1f}%)")
        lines.append(f"{dataset:12s} n={total:3d} | " + " | ".join(cells)
                     + f" | no demo-perfect attempt: {counts['no demos']} ({100 * counts['no demos'] / total:.0f}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    # A run folder from the command line, or else the latest daily blast
    run_dir: Path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_daily_blast()
    print(breakdown(run_dir))

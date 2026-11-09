"""
DAILY BLAST 📈 One BLAST_EM run over a fixed set of levels, so we can watch progress day by day.

The set: all of concept-arc, mini-arc, arc-2-train & arc-2-eval, plus the FIRST 100 of arc-1-train & arc-1-eval
(first 100 lines of each, so it's the same levels every day). 160 + 149 + 233 + 120 + 2*100 = 862 levels.

Each run's logs go in tests/daily-blasts/<timestamp>_.../ (gitignored, like tests/blast_logs/: ~17 MB a day),
then the run appends its rows to tests/daily_blast_graph.txt: one ASCII bar per run, per dataset.

Usage (from the repo root):
    .venv/bin/python -m tests.daily_blast                  # the daily blast, all CPUs
    .venv/bin/python -m tests.daily_blast --workers 4      # any extra args get passed on to BLAST_EM
"""

import csv
import statistics
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_BLASTS_DIR = ROOT / "tests" / "daily-blasts"
GRAPH_PATH = ROOT / "tests" / "daily_blast_graph.txt"

# The six datasets, in the spec's order (that's also the order of the graph's sections)
# Written in BLAST_EM's "<list>:N" syntax: a list runs WHOLE unless it ends in ":N" (= only its first N levels).
# All 120 of arc-2-eval: it's the one that counts. All of arc-2-train too: it's the closest to arc-2-eval we can tune on
DAILY_DATASETS: list[str] = ["concept-arc", "mini-arc", "arc-1-train:100", "arc-1-eval:100", "arc-2-train", "arc-2-eval"]
# Same lists, without the ":N"s (the logs & the graph only know plain list names)
DAILY_LIST_NAMES: list[str] = [dataset.partition(":")[0] for dataset in DAILY_DATASETS]
# How many "#"s a 100% bar is (so each "#" is 2%)
BAR_WIDTH: int = 50


# The graph is APPENDED to, one row per run (not redrawn from the run folders): those folders are
# gitignored, so deleting one would silently erase its history.
def graph_row(date: str, percent: float, median_seconds: float, mean_seconds: float) -> str:
    """One row of the graph: "2026-10-13 |#########                | 18.2% |  9.1 median CPU-s/level (mean 29.0)" """
    bar: str = "#" * round(percent / 100 * BAR_WIDTH)
    # Each worker solves one level at a time on one CPU, so a level's seconds ≈ its CPU-seconds.
    # The median is the one to compare runs by (one slow outlier level can't drag it around);
    # the mean stays for estimating Kaggle time = mean CPU-s/level * levels / Kaggle's CPUs
    return f"{date} |{bar.ljust(BAR_WIDTH)}| {percent:5.1f}% | {median_seconds:5.1f} median CPU-s/level (mean {mean_seconds:.1f})"


def append_to_graph() -> str:
    """Append the newest run's rows to the ASCII progress graph: one row at the end of each dataset's section."""
    # The newest run folder (run folders start with their timestamp, so the last one by name is the newest)
    run_dir: Path = sorted(path for path in DAILY_BLASTS_DIR.iterdir() if (path / "scores.csv").exists())[-1]
    with open(run_dir / "scores.csv") as file:
        rows: list[dict] = list(csv.DictReader(file))

    graph_lines: list[str] = GRAPH_PATH.read_text().splitlines()
    for dataset in DAILY_LIST_NAMES:
        # Level keys look like "<list>/<id>", same as in BLAST_EM
        scores: list[float] = [float(row["score"]) for row in rows if row["level"].startswith(dataset + "/")]
        if not scores:
            continue
        percent: float = 100 * sum(scores) / len(scores)
        # Solver seconds per level (BLAST_EM's "seconds" column): median and mean
        seconds: list[float] = [float(row["seconds"]) for row in rows if row["level"].startswith(dataset + "/")]
        median_seconds: float = statistics.median(seconds)
        mean_seconds: float = sum(seconds) / len(seconds)

        # Find this dataset's section, then its end: the next blank line (or the end of the file)
        section_end: int = graph_lines.index(f"== {dataset} ==") + 1
        while section_end < len(graph_lines) and graph_lines[section_end] != "":
            section_end += 1
        # The date is the first 10 characters of the run folder's name (YYYY-MM-DD)
        graph_lines.insert(section_end, graph_row(run_dir.name[:10], percent, median_seconds, mean_seconds))

    graph: str = "\n".join(graph_lines) + "\n"
    GRAPH_PATH.write_text(graph)
    return graph


def main() -> None:
    # Let BLAST_EM do all the real work (it re-runs itself for the config, so a subprocess is simplest).
    # Extra command-line args (e.g. --workers 4) go on the end, so they can override ours.
    subprocess.run([sys.executable, "-m", "tests.BLAST_EM",
                    # (Each list's cap is spelled out in DAILY_DATASETS, as "<list>:N")
                    "--datasets", *DAILY_DATASETS,
                    "--logs-dir", str(DAILY_BLASTS_DIR),
                    *sys.argv[1:]],
                   cwd=ROOT, check=True)

    # Then redraw the graph with today's run in it
    # print("\n" + draw_graph())
    # (Now it appends instead, see append_to_graph)
    print("\n" + append_to_graph())


if __name__ == "__main__":
    main()

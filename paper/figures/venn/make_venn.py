"""
make_venn.py -- AlexNot vs NVARC (2025) on the ARC-AGI-2 Public Eval, ONE DOT PER TASK (120 tasks).
A task counts as solved only if EVERY one of its tests is solved, and a test is solved if either of its 2 attempts
is exactly right (pass@2). So partial credit (e.g. 1 of 2 tests) counts as unsolved here.

Data:
  AlexNot: the Nov 1 evening daily blast (16/120), its submission.json scored against the levels' true test outputs
  NVARC:   the locked 4B recipe, seed 0 (38.17/120): its solved test ids, copied into nvarc_solved_tests.txt

Writes venn.png (1400x1400, white background) + venn_counts.txt (the per-region task lists, for checking).
Run: .venv/bin/python paper/figures/venn/make_venn.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE: Path = Path(__file__).resolve().parent
REPO: Path = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "palettes"))
from make_palettes import ARC_COLORS, PALETTES, TITLE_FONT, mix  # the locked paper style

# A copy of that daily blast's submission.json (tests/daily-blasts/2026-11-01_223408_.../), kept here since blast logs aren't in git
SOLVER_SUBMISSION: Path = HERE / "alexnot_submission.json"
# NVARC's solved test ids, copied from the neural fallback's run logs (that repo isn't public)
NVARC_SOLVED: Path = HERE / "nvarc_solved_tests.txt"

STYLE: dict = PALETTES["A_arc_pastel"]
INK: str = STYLE["ink"]
NAME_INK: str = "#000000"  # the region names in full black, counts + outlines stay ink gray
COLORS: dict[str, str] = {
    "alexnot": STYLE["solver"],                       # pastel blue
    "nvarc": mix(ARC_COLORS[4], "#FFFFFF", 0.4),      # pastel yellow
    "both": STYLE["both"],                            # pastel green
}


def eval_task_ids() -> list[str]:
    """The 120 Public Eval task ids (each line of the list is 'id - nickname')."""
    return [line.split()[0] for line in (REPO / "arc" / "levels" / "arc-2-eval.txt").read_text().splitlines() if line.strip()]


def true_outputs(task_id: str) -> list:
    """The task's true test outputs (6 of the 120 live in arc-1-eval/, since they're in both sets)."""
    for folder in ["arc-2-eval", "arc-1-eval"]:
        path: Path = REPO / "arc" / "levels" / folder / f"{task_id}.json"
        if path.exists():
            return [test["output"] for test in json.loads(path.read_text())["test"]]
    raise FileNotFoundError(task_id)


def alexnot_solved_tests(task_ids: list[str]) -> set[str]:
    submission: dict = json.loads(SOLVER_SUBMISSION.read_text())
    solved: set[str] = set()
    for task_id in task_ids:
        for index, (answer, truth) in enumerate(zip(submission[f"arc-2-eval/{task_id}"], true_outputs(task_id))):
            if truth in (answer["attempt_1"], answer["attempt_2"]):
                solved.add(f"{task_id}_{index}")
    return solved


def nvarc_solved_tests() -> set[str]:
    """One test id per line ('135a2760_0'), skipping the # header lines."""
    return {line.strip() for line in NVARC_SOLVED.read_text().splitlines() if line.strip() and not line.startswith("#")}


def task_score(solved: set[str], task_ids: list[str]) -> float:
    """Kaggle-style score: each task's fraction of tests right, summed (sanity check vs the known totals)."""
    return sum(sum(f"{task_id}_{index}" in solved for index in range(len(true_outputs(task_id)))) / len(true_outputs(task_id))
               for task_id in task_ids)


def fully_solved(task_id: str, solved_tests: set[str]) -> bool:
    """True only if every one of the task's tests is in solved_tests."""
    return all(f"{task_id}_{index}" in solved_tests for index in range(len(true_outputs(task_id))))


def points_in_region(region: str, count: int, centers: tuple, radius: float, spacing: float) -> np.ndarray:
    """Evenly spaced points (a hex grid) inside one region, the `count` closest to the region's middle."""
    xs, ys = np.meshgrid(np.arange(-3, 3, spacing), np.arange(-2.5, 2.5, spacing * 0.866) + centers[0][1])  # a row on the center line
    xs = xs + (np.arange(xs.shape[0])[:, None] % 2) * spacing / 2  # offset every other row
    grid: np.ndarray = np.column_stack([xs.ravel(), ys.ravel()])
    distance_left: np.ndarray = np.hypot(*(grid - centers[0]).T)
    distance_right: np.ndarray = np.hypot(*(grid - centers[1]).T)
    margin: float = spacing * 0.75  # keep dots off the circle lines
    mask: np.ndarray = {
        "alexnot": (distance_left < radius - margin) & (distance_right > radius + margin),
        "nvarc": (distance_right < radius - margin) & (distance_left > radius + margin),
        "both": (distance_left < radius - margin) & (distance_right < radius - margin),
    }[region]
    candidates: np.ndarray = grid[mask]
    # The region's middle: its widest point on the horizontal center line, so the dots sit centered
    middle: np.ndarray = np.array([np.median(candidates[:, 0]), centers[0][1]])
    order: np.ndarray = np.argsort(np.hypot(*(candidates - middle).T))
    assert len(candidates) >= count, f"{region}: only room for {len(candidates)} dots, need {count}"
    return candidates[order[:count]]


def main() -> None:
    task_ids: list[str] = eval_task_ids()
    all_tests: list[str] = [f"{task_id}_{index}" for task_id in task_ids for index in range(len(true_outputs(task_id)))]
    alexnot: set[str] = alexnot_solved_tests(task_ids)
    nvarc: set[str] = nvarc_solved_tests()
    # A task is solved only if all of its tests are
    alexnot_tasks: set[str] = {task_id for task_id in task_ids if fully_solved(task_id, alexnot)}
    nvarc_tasks: set[str] = {task_id for task_id in task_ids if fully_solved(task_id, nvarc)}
    regions: dict[str, list[str]] = {
        "alexnot": sorted(alexnot_tasks - nvarc_tasks),
        "both": sorted(alexnot_tasks & nvarc_tasks),
        "nvarc": sorted(nvarc_tasks - alexnot_tasks),
        "neither": sorted(set(task_ids) - alexnot_tasks - nvarc_tasks),
    }
    summary: str = (f"tests: {len(all_tests)} over {len(task_ids)} tasks\n"
                    f"Kaggle-style (partial credit) task scores: AlexNot {task_score(alexnot, task_ids):.2f}/120, "
                    f"NVARC {task_score(nvarc, task_ids):.2f}/120\n"
                    f"Fully solved tasks (every test): AlexNot {len(alexnot_tasks)}, NVARC {len(nvarc_tasks)}, "
                    f"either {len(alexnot_tasks | nvarc_tasks)}\n"
                    + "".join(f"{name}: {len(tasks)} tasks\n  {' '.join(tasks)}\n" for name, tasks in regions.items()))
    (HERE / "venn_counts.txt").write_text(summary)
    print(summary)

    # --- the figure: 7in x 7in at 200 dpi = 1400 x 1400 px ---
    figure, axes = plt.subplots(figsize=(7, 7), dpi=200, facecolor="#FFFFFF")
    axes.set_position([0, 0, 1, 1])  # fill the whole square
    centers: tuple = (np.array([-0.62, 0.03]), np.array([0.62, 0.03]))
    radius: float = 1.25
    left_fill = Circle(centers[0], radius, facecolor=COLORS["alexnot"], alpha=0.3, edgecolor="none")
    right_fill = Circle(centers[1], radius, facecolor=COLORS["nvarc"], alpha=0.3, edgecolor="none")
    axes.add_patch(left_fill)
    axes.add_patch(right_fill)
    # The middle gets its own green shade: an opaque green circle, clipped to the other circle
    overlap = Circle(centers[0], radius, facecolor=mix(COLORS["both"], "#FFFFFF", 0.55), edgecolor="none")
    axes.add_patch(overlap)
    overlap.set_clip_path(Circle(centers[1], radius, transform=axes.transData))
    for center in centers:
        axes.add_patch(Circle(center, radius, facecolor="none", edgecolor=INK, linewidth=2.2))

    # One dot per solved task
    for region in ["alexnot", "both", "nvarc"]:
        points: np.ndarray = points_in_region(region, len(regions[region]), centers, radius, spacing=0.165)
        axes.scatter(points[:, 0], points[:, 1], s=95, color=COLORS[region], edgecolor=INK, linewidth=1.3, zorder=3)

    # Labels: the three regions on top, "neither" underneath
    label_y: float = centers[0][1] + radius + 0.12
    for x, name, region in [(-1.15, "AlexNot", "alexnot"), (0.0, "Both", "both"), (1.15, "NVARC (2025)", "nvarc")]:
        axes.text(x, label_y, f"{name}\n", ha="center", va="bottom", fontfamily=TITLE_FONT, fontsize=17, color=NAME_INK)
        axes.text(x, label_y, f"{len(regions[region])} tasks", ha="center", va="bottom", fontfamily=TITLE_FONT,
                  fontsize=13, color=INK)
    axes.text(0.0, centers[0][1] - radius - 0.14, "Neither", ha="center", va="top", fontfamily=TITLE_FONT, fontsize=17, color=NAME_INK)
    axes.text(0.0, centers[0][1] - radius - 0.38, f"{len(regions['neither'])} tasks", ha="center", va="top",
              fontfamily=TITLE_FONT, fontsize=13, color=INK)

    axes.set_xlim(-2.1, 2.1)
    axes.set_ylim(-2.1, 2.1)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.savefig(HERE / "venn.png", dpi=200, facecolor="#FFFFFF")
    plt.close(figure)
    print("wrote venn.png")


if __name__ == "__main__":
    main()

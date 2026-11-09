"""
Tune the Bayes posterior's temperature T and dummy score D (config: BAYES_TEMPERATURE, BAYES_DUMMY_SCORE)
for the lowest multi-class Brier score, from the webapp's logged ingredients. No solver runs needed.

Written by Claude, from talking with Nicky.

How it's scored, per test: every output grid in the posterior is a bet, plus the dummy's bet "it's none of these grids".
Each bet scores (y - p)^2, with y = 1 if it was right, else 0. Summed over the bets = that test's Brier score
(0 = perfect, 2 = worst). We report the average over tests.

1. Grid-search T and D on the first 200 ARC-1 Eval levels (list order), then check on the other 200.
2. Compare the posterior's top 2 grids with the solver's attempts 1 & 2 (at the tuned T).
3. Re-score any other logged dataset (e.g. ARC-2 Eval) at the tuned T and D, without re-tuning.

Data (next to this file, copied from arc/webapp/bayes_logs/ + the runs' streamed results):
  <dataset>_bayes_log.jsonl            each top program's raw score, and per test, which programs made which grid
  <dataset>_answers_and_grids.jsonl.gz attempts 1 & 2, and each posterior bucket's grid (same order as the log)

Run from the repo root: .venv/bin/python paper/code/experiments/bayes_tuning/tune_bayes.py
"""
import gzip
import json
from pathlib import Path

import numpy as np

HERE: Path = Path(__file__).resolve().parent
LEVELS: Path = HERE.parents[3] / "arc" / "levels"


def level_ids_of(dataset: str) -> list[str]:
    """The dataset's level ids, in list order ("id - nickname" per line)."""
    return [line.split()[0] for line in (LEVELS / f"{dataset}.txt").read_text().splitlines() if line.strip()]


def true_outputs(level_id: str) -> list:
    """A level's true test outputs (a few ARC-2 Eval levels live in arc-1-eval/, since they're in both)."""
    for path in LEVELS.glob(f"*/{level_id}.json"):
        return [test["output"] for test in json.loads(path.read_text())["test"]]
    raise FileNotFoundError(level_id)


def load_log(dataset: str) -> dict[str, dict]:
    return {entry["level_id"]: entry for entry in map(json.loads, (HERE / f"{dataset}_bayes_log.jsonl").read_text().splitlines())}


def flatten(logs: dict[str, dict], level_ids: list[str]) -> dict:
    """Every test's buckets as flat arrays, so scoring one (T, D) pair is a few numpy calls."""
    scores = np.full((len(level_ids), 101), np.inf)  # Padding programs get probability 0
    dummy_column = np.zeros(len(level_ids), dtype=int)
    entry_level, entry_column, entry_bucket = [], [], []  # One entry per (program, bucket it landed in)
    bucket_test, bucket_true, bucket_is_dummy = [], [], []
    test_count: int = 0
    for row, level_id in enumerate(level_ids):
        log: dict = logs[level_id]
        scores[row, :len(log["program_scores"])] = log["program_scores"]
        dummy_column[row] = log["program_is_dummy"].index(True)
        for test in log["tests"]:
            any_grid_right: bool = any(output["is_correct"] for output in test)
            for output in test:
                bucket: int = len(bucket_test)
                bucket_test.append(test_count)
                bucket_is_dummy.append(output["is_dummy"])
                # The dummy's bet ("none of these grids") is right exactly when no grid is right
                bucket_true.append(output["is_correct"] or (output["is_dummy"] and not any_grid_right))
                for index in output["program_indices"]:
                    entry_level.append(row)
                    entry_column.append(index)
                    entry_bucket.append(bucket)
            test_count += 1
    return {"scores": scores, "dummy_column": dummy_column, "rows": np.arange(len(level_ids)),
            "entry_level": np.array(entry_level), "entry_column": np.array(entry_column),
            "entry_bucket": np.array(entry_bucket), "bucket_test": np.array(bucket_test),
            "bucket_true": np.array(bucket_true, dtype=float), "bucket_is_dummy": np.array(bucket_is_dummy),
            "tests": test_count}


def bucket_probabilities(data: dict, temperature: float, dummy_score: float) -> np.ndarray:
    """Softmax over -score/T (with the dummy's score swapped for D), pooled per bucket."""
    scores = data["scores"].copy()
    scores[data["rows"], data["dummy_column"]] = dummy_score
    logits = -scores / temperature
    logits -= logits.max(axis=1, keepdims=True)  # Stable softmax: exp() never overflows
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.bincount(data["entry_bucket"], weights=probabilities[data["entry_level"], data["entry_column"]],
                       minlength=len(data["bucket_test"]))


def mean_brier(data: dict, temperature: float, dummy_score: float) -> float:
    errors = (data["bucket_true"] - bucket_probabilities(data, temperature, dummy_score)) ** 2
    return float(errors.sum() / data["tests"])


def mass_on_truth(data: dict, temperature: float, dummy_score: float) -> str:
    """The intuitive version: the average probability on the right answer, split by whether a grid was right."""
    true_probability = np.zeros(data["tests"])
    np.add.at(true_probability, data["bucket_test"], bucket_probabilities(data, temperature, dummy_score) * data["bucket_true"])
    grid_right = np.zeros(data["tests"], dtype=bool)
    np.logical_or.at(grid_right, data["bucket_test"], (data["bucket_true"] > 0) & ~data["bucket_is_dummy"])
    return (f"a grid was right in {grid_right.sum()}/{data['tests']} tests: mean probability on it {true_probability[grid_right].mean():.2f}; "
            f"no grid right in {(~grid_right).sum()}: mean probability on the dummy {true_probability[~grid_right].mean():.2f}")


def grid_search(data: dict, temperatures: np.ndarray, dummy_scores: np.ndarray) -> tuple[float, float, float]:
    """(Brier, T, D) of the best pair on the grid."""
    return min((mean_brier(data, temperature, dummy_score), temperature, dummy_score)
               for temperature in temperatures for dummy_score in dummy_scores)


def compare_with_attempts(dataset: str, logs: dict[str, dict], temperature: float) -> None:
    """Posterior's top 2 grids vs attempts 1 & 2: how often they differ, and what each would score.
    (D doesn't change the order of the real grids, only how much is left for them)"""
    answers: dict[str, dict] = {}
    with gzip.open(HERE / f"{dataset}_answers_and_grids.jsonl.gz", "rt") as file:
        for line in file:
            row = json.loads(line)
            answers[row["level_id"]] = row
    differ_first = differ_second = differ_set = tests = 0
    solver_score = posterior_score = 0.0
    changes: list[str] = []
    for level_id in level_ids_of(dataset):
        log, row, truths = logs[level_id], answers[level_id], true_outputs(level_id)
        scores = np.array(log["program_scores"])
        weights = np.exp(-(scores - scores.min()) / temperature)
        solver_right = posterior_right = 0
        for test_log, grids, answer, truth in zip(log["tests"], row["test_grids"], row["test_solutions"], truths):
            # The real (non-dummy) grids, most probable first
            real = [(weights[output["program_indices"]].sum(), grid) for output, grid in zip(test_log, grids) if not output["is_dummy"]]
            top_two: list = [grid for _, grid in sorted(real, key=lambda pair: -pair[0])[:2]]
            top_two += [answer["attempt_1"]] * (2 - len(top_two))  # Only one distinct grid? Repeat it
            differ_first += top_two[0] != answer["attempt_1"]
            differ_second += top_two[1] != answer["attempt_2"]
            differ_set += sorted(map(str, top_two)) != sorted(map(str, [answer["attempt_1"], answer["attempt_2"]]))
            solver_right += truth in (answer["attempt_1"], answer["attempt_2"])
            posterior_right += truth in top_two
            tests += 1
        solver_score += solver_right / len(truths)
        posterior_score += posterior_right / len(truths)
        if solver_right != posterior_right:
            changes.append(f"    {level_id}: attempts {solver_right}/{len(truths)} -> posterior {posterior_right}/{len(truths)}")
    levels: int = len(answers)
    print(f"  posterior #1 != attempt 1 in {differ_first}/{tests} tests, #2 != attempt 2 in {differ_second}, the pair differs in {differ_set}")
    print(f"  score: attempts {solver_score:.2f}/{levels} ({100 * solver_score / levels:.1f}%), "
          f"posterior top 2 {posterior_score:.2f}/{levels} ({100 * posterior_score / levels:.1f}%)")
    print("\n".join(changes) if changes else "    no level changes score")


def main() -> None:
    # 1. Tune on the first 200 ARC-1 Eval levels, check on the other 200
    arc1_logs: dict[str, dict] = load_log("arc-1-eval")
    arc1_levels: list[str] = level_ids_of("arc-1-eval")
    tune, check, both = (flatten(arc1_logs, levels) for levels in (arc1_levels[:200], arc1_levels[200:], arc1_levels))
    # A coarse grid, then a finer one around its best point
    _, coarse_temperature, coarse_dummy = grid_search(tune, np.logspace(-2, 4, 61), np.linspace(-500, 3000, 351))
    brier, temperature, dummy_score = grid_search(tune, coarse_temperature * np.logspace(-0.15, 0.15, 31),
                                                  np.linspace(coarse_dummy - 20, coarse_dummy + 20, 81))
    print(f"ARC-1 Eval: tuned on the first 200 levels -> T = {temperature:.3g}, D = {dummy_score:.1f}")
    for name, data in [("first 200 (tuning)", tune), ("other 200 (held out)", check), ("all 400", both)]:
        # The reference forecaster "always 100% on the dummy" = "I never know anything" (T -> 0, D -> -infinity)
        reference: float = mean_brier(data, 1e-9, -1e9)
        tuned: float = mean_brier(data, temperature, dummy_score)
        print(f"  {name}: mean Brier {tuned:.4f} (always-dummy {reference:.4f}, skill score {1 - tuned / reference:.2f}; "
              f"starting guess T=1 D=100: {mean_brier(data, 1, 100):.4f})")
    held_out_best = grid_search(check, np.logspace(-2, 4, 61), np.linspace(-500, 3000, 351))
    print(f"  (tuned on the other 200 directly instead: T = {held_out_best[1]:.3g}, D = {held_out_best[2]:.1f}, Brier {held_out_best[0]:.4f})")
    print(f"  all 400: {mass_on_truth(both, temperature, dummy_score)}")

    # 2. Top 2 posterior grids vs attempts 1 & 2
    compare_with_attempts("arc-1-eval", arc1_logs, temperature)

    # 3. Any other logged dataset, at the tuned values (no re-tuning)
    for dataset in ["arc-2-eval"]:
        if not (HERE / f"{dataset}_bayes_log.jsonl").exists():
            continue
        logs: dict[str, dict] = load_log(dataset)
        data: dict = flatten(logs, level_ids_of(dataset))
        tuned, reference = mean_brier(data, temperature, dummy_score), mean_brier(data, 1e-9, -1e9)
        print(f"\n{dataset} at T = {temperature:.3g}, D = {dummy_score:.1f}: mean Brier {tuned:.4f} "
              f"(always-dummy {reference:.4f}, skill score {1 - tuned / reference:.2f})")
        print(f"  {mass_on_truth(data, temperature, dummy_score)}")
        compare_with_attempts(dataset, logs, temperature)


if __name__ == "__main__":
    main()

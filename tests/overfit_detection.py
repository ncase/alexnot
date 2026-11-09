"""Can program length tell an overfit program (fits every demo, wrong on the test) from a right one? Reads one BLAST_EM run.

Usage (from the repo root):
    .venv/bin/python -m tests.overfit_detection tests/blast_logs/<run folder>
"""
import json
import sys
from pathlib import Path

import numpy as np

# So `alexnot` imports as a package (same trick as the other repo-level tests)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "arc"))
import alexnot.transform  # noqa: F401 (registers the score_transform module)
from alexnot.transform.score_transform import get_simplicity_score, _TOKEN_PATTERN
from tests.BLAST_EM import find_level_file

# Too flaky to label either way, so it's left out entirely
EXCLUDED_LEVELS: set[str] = {"e3721c99"}

# Main rule: "hand to the fallback if feature > threshold". Alphas for the quadratic variant (a negative one bends it, so both tails land on one side)
ALPHAS: list[float] = [-0.05, -0.02, -0.01, -0.005, -0.002, -0.001, 0.0, 0.001, 0.01, 0.1]


def token_length(program: str) -> float:
    """How many tokens the program has (every token counts 1)."""
    return float(len(_TOKEN_PATTERN.findall(program)))


def load_levels(run_dir: Path) -> list[dict]:
    """One row per level: its list, whether it fit every demo, whether attempt_1 / pass@2 got every test input right."""
    submission: dict = json.loads((run_dir / "submission.json").read_text())
    rows: list[dict] = []
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        record: dict = json.loads(line)
        level_key: str = record["level"]
        list_name, level_id = level_key.split("/")
        if level_id in EXCLUDED_LEVELS:
            continue

        # The answers live in the level file (the blast scrubbed them from what the solver saw)
        true_outputs: list = [pair["output"] for pair in json.loads(find_level_file(list_name, level_id).read_text())["test"]]
        answers: list = submission[level_key]
        # attempt_1 is the best program's own answer, which is what "overfit" is about
        attempt_1_right: bool = all(answer["attempt_1"] == truth for answer, truth in zip(answers, true_outputs))
        pass_at_2_right: bool = all(truth in (answer["attempt_1"], answer["attempt_2"]) for answer, truth in zip(answers, true_outputs))
        rows.append({"list": list_name, "id": level_id, "program": record.get("best_program"),
                     "solved_all_demos": record["solved_all_demos"],
                     "attempt_1_right": attempt_1_right, "pass_at_2_right": pass_at_2_right})
    return rows


def best_threshold(values: np.ndarray, is_right: np.ndarray) -> tuple[float, float]:
    """The threshold t for "right if value <= t" with the best accuracy. Returns (t, accuracy)."""
    best: tuple[float, float] = (float("inf"), float(is_right.mean()))
    for threshold in np.unique(values):
        accuracy: float = float(((values <= threshold) == is_right).mean())
        if accuracy > best[1]:
            best = (float(threshold), accuracy)
    return best


def best_either_direction(values: np.ndarray, is_right: np.ndarray) -> str:
    """Best threshold either way round: "long = overfit" (value > t) or "short = overfit" (value < t)."""
    long_threshold, long_accuracy = best_threshold(values, is_right)
    # Flipping the sign turns "short programs are overfit" into the same "value > t" search
    short_threshold, short_accuracy = best_threshold(-values, is_right)
    if short_accuracy > long_accuracy:
        return f"acc {short_accuracy:.1%}, overfit if < {-short_threshold:.2f}"
    return f"acc {long_accuracy:.1%}, overfit if > {long_threshold:.2f}"


def keep_right_threshold(values: np.ndarray, is_right: np.ndarray, keep_fraction: float) -> float:
    """The smallest threshold that still keeps `keep_fraction` of the right programs (value <= t)."""
    right_values: np.ndarray = np.sort(values[is_right])
    return float(right_values[int(np.ceil(keep_fraction * len(right_values))) - 1])


def describe_rule(name: str, values: np.ndarray, is_right: np.ndarray, threshold: float) -> str:
    """How a 'hand off if value > threshold' rule does: overfit caught, right ones lost, total handed off."""
    handed_off: np.ndarray = values > threshold
    overfit_caught: int = int((handed_off & ~is_right).sum())
    right_lost: int = int((handed_off & is_right).sum())
    accuracy: float = float((~handed_off == is_right).mean())
    return (f"{name}: t={threshold:.2f}, acc {accuracy:.1%}, catches {overfit_caught}/{int((~is_right).sum())} overfit, "
            f"loses {right_lost}/{int(is_right.sum())} right, hands off {int(handed_off.sum())}/{len(values)}")


def main() -> None:
    run_dir: Path = Path(sys.argv[1])
    rows: list[dict] = load_levels(run_dir)

    # Check 1: does any level that MISSED a demo still get the test right?
    missed_demo: list[dict] = [row for row in rows if not row["solved_all_demos"]]
    lucky_attempt_1: list[str] = [f"{row['list']}/{row['id']}" for row in missed_demo if row["attempt_1_right"]]
    lucky_pass_at_2: list[str] = [f"{row['list']}/{row['id']}" for row in missed_demo if row["pass_at_2_right"]]
    print(f"Missed a demo: {len(missed_demo)} levels. Test right anyway: attempt_1 {len(lucky_attempt_1)} {lucky_attempt_1}, "
          f"pass@2 {len(lucky_pass_at_2)} {lucky_pass_at_2}\n")

    # The dataset: only levels whose best program fits every demo, labelled by attempt_1 on the test
    fits: list[dict] = [row for row in rows if row["solved_all_demos"]]
    list_names: list[str] = sorted({row["list"] for row in fits})
    is_right: np.ndarray = np.array([row["attempt_1_right"] for row in fits])
    lists: np.ndarray = np.array([row["list"] for row in fits])
    for list_name in list_names:
        in_list: np.ndarray = lists == list_name
        overfit_ids: list[str] = [row["id"] for row in fits if row["list"] == list_name and not row["attempt_1_right"]]
        print(f"{list_name}: {int(is_right[in_list].sum())} right, {int((~is_right[in_list]).sum())} overfit {overfit_ids}")
    print(f"ALL: {int(is_right.sum())} right, {int((~is_right).sum())} overfit\n")

    # The two base features
    base_features: dict[str, np.ndarray] = {
        "token length": np.array([token_length(row["program"]) for row in fits]),
        "simplicity cost": np.array([get_simplicity_score(row["program"]) for row in fits]),
    }

    for feature_name, values in base_features.items():
        print(f"=== {feature_name} ===")
        print(f"  right:   median {np.median(values[is_right]):.1f}, range {values[is_right].min():.1f}-{values[is_right].max():.1f}")
        print(f"  overfit: median {np.median(values[~is_right]):.1f}, range {values[~is_right].min():.1f}-{values[~is_right].max():.1f}")

        # Plain threshold, then x + alpha * x^2 (a negative alpha makes "right" a middle band, so it can also catch tiny programs)
        threshold, accuracy = best_threshold(values, is_right)
        print(f"  linear: best acc {accuracy:.1%} at t={threshold:.2f} (majority baseline {max(is_right.mean(), 1 - is_right.mean()):.1%})")
        print(f"  linear, either direction: {best_either_direction(values, is_right)}")
        for alpha in ALPHAS:
            print(f"  + {alpha}*x^2: {best_either_direction(values + alpha * values ** 2, is_right)}")

        # Thresholds that keep (almost) all the right ones
        for keep_fraction in (1.0, 0.95, 0.9):
            keep_threshold: float = keep_right_threshold(values, is_right, keep_fraction)
            print("  " + describe_rule(f"keep {keep_fraction:.0%} right", values, is_right, keep_threshold))

        # Per list: the best threshold within the list, and the threshold fit on the OTHER lists applied to this one
        for list_name in list_names:
            in_list = lists == list_name
            own_threshold, own_accuracy = best_threshold(values[in_list], is_right[in_list])
            others_threshold, _ = best_threshold(values[~in_list], is_right[~in_list])
            print(f"    {list_name}: own best acc {own_accuracy:.1%} (t={own_threshold:.2f}); "
                  + describe_rule("fit on others", values[in_list], is_right[in_list], others_threshold))
        print()


if __name__ == "__main__":
    main()

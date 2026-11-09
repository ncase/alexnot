"""
An approximate Bayesian posterior over the evolutionary search's programs, and over each test's output grids.
Display only (the webapp turns it on with solve(..., bayes=True)): it never changes attempts 1 and 2.

P(program) ∝ exp(-total_score / T) for the top programs, plus one dummy "out-of-model" program with a constant score,
standing in for "the real answer is something none of our programs can say".
"""

from typing import Dict, List

import numpy as np

from . import parse
from . import transform
from .config import BAYES_TOP_PROGRAMS, BAYES_TEMPERATURE, BAYES_DUMMY_SCORE

DUMMY_PROGRAM_TEXT: str = 'OUT-OF-MODEL / "KNIGHTIAN" UNCERTAINTY DUMMY PROGRAM'


def approximate_posterior(ranked_program_results: List[Dict], annotated_tests: List[Dict]) -> Dict:
    """
    In: the search's programs, best first (the whole population, on the chosen parse), and the annotated test scenes.
    Out: {"temperature", "dummy_score",
          "programs": [{"program", "score", "probability", "is_dummy"}, ...] (most probable first),
          "tests": [[{"grid", "probability", "is_dummy", "program_indices"}, ...] per test] (most probable first)}
    "program_indices" point into "programs", so the raw scores + which output each program made can be logged,
    and the temperature + dummy score re-tuned later without re-running the solver.
    """

    # The top programs, plus the dummy (its constant score is what lets it win when every real program scores badly)
    top_results: List[Dict] = ranked_program_results[:BAYES_TOP_PROGRAMS]
    programs: List[Dict] = [{"program": result["program"], "score": float(result["total_score"]), "is_dummy": False}
                            for result in top_results]
    programs.append({"program": DUMMY_PROGRAM_TEXT, "score": float(BAYES_DUMMY_SCORE), "is_dummy": True})

    # Softmax over -score/T (lower score = better). Subtract the best score first, so exp() never overflows
    scores: np.ndarray = np.array([program["score"] for program in programs])
    weights: np.ndarray = np.exp(-(scores - scores.min()) / BAYES_TEMPERATURE)
    probabilities: np.ndarray = weights / weights.sum()
    for program, probability in zip(programs, probabilities):
        program["probability"] = float(probability)

    # Most probable first (a stable sort, so equal probabilities keep the search's ranking)
    programs.sort(key=lambda program: -program["probability"])

    # Per test: run every real program, and pool the probability of programs that give the exact same grid
    tests: List[List[Dict]] = []
    for annotated_test in annotated_tests:
        buckets: Dict[tuple, Dict] = {}  # (shape, grid bytes) -> bucket
        for index, program in enumerate(programs):
            if program["is_dummy"]:
                continue
            output_scene: Dict = transform.apply_transform_one(program["program"], annotated_test["input"])
            grid: np.ndarray = parse.reverse_parse_one(output_scene)
            key: tuple = (grid.shape, grid.tobytes())
            if key not in buckets:
                buckets[key] = {"grid": grid.tolist(), "probability": 0.0, "is_dummy": False, "program_indices": []}
            buckets[key]["probability"] += program["probability"]
            buckets[key]["program_indices"].append(index)

        # The dummy gets its own bucket (not keyed by a grid, so a real program can never land in it).
        # No grid: the webapp draws it as the big red X
        dummy_index: int = next(index for index, program in enumerate(programs) if program["is_dummy"])
        outputs: List[Dict] = list(buckets.values())
        outputs.append({"grid": None, "probability": programs[dummy_index]["probability"], "is_dummy": True,
                        "program_indices": [dummy_index]})
        outputs.sort(key=lambda output: -output["probability"])
        tests.append(outputs)

    return {"temperature": BAYES_TEMPERATURE, "dummy_score": BAYES_DUMMY_SCORE, "programs": programs, "tests": tests}

"""
make_reliability.py -- reliability diagrams (calibration plots) of the Bayes posterior: ARC-1 Eval (all 400 levels,
where T and D were tuned) on top, ARC-2 Eval (all 120 levels, same T and D, no re-tuning) below.
Written by Claude, from talking with Nicky.

Every bet is one output grid in a test's posterior, or the dummy's "none of these grids". Bets are binned by their
probability (5 bins of 20%); each bar's height is the fraction of that bin's bets that were actually right, and its
label is the bin's number of bets (n).
A perfectly calibrated forecaster's bars would reach the dashed X = Y line.

Data: paper/experiments/bayes_tuning/ (the webapp's logged ingredients), at the tuned T and D.
Writes reliability.png (1400x2800, two square panels, white background) + reliability_counts.txt (per-bin numbers).
Run: .venv/bin/python paper/figures/brier/make_reliability.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE: Path = Path(__file__).resolve().parent
REPO: Path = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "palettes"))
from make_palettes import PALETTES, TITLE_FONT  # the locked paper style
sys.path.insert(0, str(REPO / "paper" / "experiments" / "bayes_tuning"))
from tune_bayes import bucket_probabilities, flatten, level_ids_of, load_log

# The tuned defaults (BAYES_TEMPERATURE, BAYES_DUMMY_SCORE in arc/alexnot/config/default.py)
TEMPERATURE: float = 15.5
DUMMY_SCORE: float = 18.0
BIN_EDGES: np.ndarray = np.linspace(0, 1, 6)  # 0-20%, 20-40%, ..., 80-100%

STYLE: dict = PALETTES["A_arc_pastel"]
INK: str = STYLE["ink"]
BAR_COLOR: str = STYLE["solver"]  # pastel blue


def draw_panel(axes: plt.Axes, dataset: str, title: str) -> str:
    """One reliability diagram; returns its per-bin numbers as text."""
    # Every bet's probability, and whether it was right
    data: dict = flatten(load_log(dataset), level_ids_of(dataset))
    probabilities: np.ndarray = bucket_probabilities(data, TEMPERATURE, DUMMY_SCORE)
    right: np.ndarray = data["bucket_true"]

    # Bin them (a probability of exactly 100% goes in the last bin)
    bins: np.ndarray = np.clip(np.digitize(probabilities, BIN_EDGES) - 1, 0, len(BIN_EDGES) - 2)
    counts: np.ndarray = np.bincount(bins, minlength=5)
    hits: np.ndarray = np.bincount(bins, weights=right, minlength=5)
    mean_predicted: np.ndarray = np.bincount(bins, weights=probabilities, minlength=5) / np.maximum(counts, 1)
    fraction_right: np.ndarray = hits / np.maximum(counts, 1)
    shares: np.ndarray = counts / counts.sum()

    centers: np.ndarray = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    axes.bar(centers, fraction_right, width=0.2, color=BAR_COLOR, edgecolor=INK, linewidth=1.8, zorder=2)

    # Perfect calibration
    axes.plot([0, 1], [0, 1], linestyle="--", color="#000000", linewidth=1.8, zorder=3)

    # Each bin's number of bets (n), just above its bar (on a white backing, so the dashed line can't cross the text)
    for center, height, count in zip(centers, fraction_right, counts):
        axes.text(center, height + 0.02, f"{count:,} grids", ha="center", va="bottom", fontfamily=TITLE_FONT,
                  fontsize=11, color=INK, zorder=5, bbox={"facecolor": STYLE["page"], "edgecolor": "none", "pad": 1.5})

    ticks: np.ndarray = np.linspace(0, 1, 6)
    axes.set_xticks(ticks, [f"{tick:.0%}" for tick in ticks])
    axes.set_yticks(ticks, [f"{tick:.0%}" for tick in ticks])
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.05)
    axes.set_title(title, fontfamily=TITLE_FONT, fontsize=17, color=INK, loc="left", pad=14)
    axes.set_xlabel("How confident the posterior was", fontfamily=TITLE_FONT, fontsize=15, color=INK, labelpad=10)
    axes.set_ylabel("How often it was right", fontfamily=TITLE_FONT, fontsize=15, color=INK, labelpad=10)
    axes.set_facecolor(STYLE["panel"])
    axes.grid(axis="y", color=STYLE["grid"], linewidth=0.8, zorder=0)
    for spine in axes.spines.values():
        spine.set_color(INK)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    axes.tick_params(colors=INK, labelsize=12)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontfamily(TITLE_FONT)

    return (f"{title}: {data['tests']} tests, {len(probabilities)} bets, T={TEMPERATURE}, D={DUMMY_SCORE}\n"
            + "".join(f"  {BIN_EDGES[i]:.0%}-{BIN_EDGES[i + 1]:.0%}: {counts[i]} bets ({shares[i]:.1%} of all), "
                      f"{hits[i]:.0f} right ({fraction_right[i]:.1%}), mean predicted {mean_predicted[i]:.1%}\n"
                      for i in range(5)))


def main() -> None:
    # 7in x 14in at 200 dpi = 1400 x 2800 px. Each panel is 74% of the width (1037 px), and 1037/2800 of the height
    # keeps it square. Both panels sit a little right of center, to leave room for the y-axis label
    figure = plt.figure(figsize=(7, 14), dpi=200, facecolor=STYLE["page"])
    panel_width: float = 0.741
    panel_height: float = panel_width * 1400 / 2800
    top_axes = figure.add_axes([0.18, 0.565, panel_width, panel_height])
    bottom_axes = figure.add_axes([0.18, 0.065, panel_width, panel_height])
    summary: str = draw_panel(top_axes, "arc-1-eval", "ARC-1 Eval")
    summary += draw_panel(bottom_axes, "arc-2-eval", "ARC-2 Eval")
    (HERE / "reliability_counts.txt").write_text(summary)
    print(summary)

    figure.savefig(HERE / "reliability.png", dpi=200, facecolor=STYLE["page"])
    plt.close(figure)
    print("wrote reliability.png")


if __name__ == "__main__":
    main()

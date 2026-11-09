"""
make_ablations.py -- the ablation bar chart: every variant's score on all 400 ARC-1 Eval levels, next to the full solver
(the control, 10 runs: its bar is the mean, its band is the min-max of the 10 runs, the only variant with a band).
Written by Claude, from talking with Nicky.

Colors: pastel gray = control, red = random scores, blue = compression, yellow = search, green = rounds of evolution.
Data: paper/code/experiments/ablations/<config>_runN/scores.csv (a variant that hasn't been run yet is just skipped).
Writes ablations.png (2800px wide, white background) + ablations_scores.txt (the numbers behind every bar).
Run: .venv/bin/python paper/code/figures/ablations/make_ablations.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE: Path = Path(__file__).resolve().parent
REPO: Path = HERE.parents[3]
RUNS: Path = REPO / "paper" / "code" / "experiments" / "ablations"
sys.path.insert(0, str(HERE.parent / "palettes"))
from make_palettes import ARC_COLORS, CODE_FONT, PALETTES, TITLE_FONT, mix  # the locked paper style

LEVEL_COUNT: int = 400
STYLE: dict = PALETTES["A_arc_pastel"]
FIGURE_WIDTH_INCHES: float = 7.5
DPI: float = 2800 / FIGURE_WIDTH_INCHES  # 2800px wide: it goes across the whole page
INK: str = STYLE["ink"]

# Nicky's colors for this figure: ARC's own colors, softened halfway to white
CONTROL_COLOR: str = mix(ARC_COLORS[5], "#FFFFFF", 0.4)  # gray
GROUP_COLORS: dict[str, str] = {
    "Random scores": mix(ARC_COLORS[2], "#FFFFFF", 0.5),  # red
    "Compression": mix(ARC_COLORS[1], "#FFFFFF", 0.5),  # blue
    "Search": mix(ARC_COLORS[4], "#FFFFFF", 0.4),  # yellow
    "Rounds of evolution": mix(ARC_COLORS[3], "#FFFFFF", 0.5),  # green
}

# Every bar, top to bottom: (group, config name, human-readable name), in Nicky's order
VARIANTS: list[tuple[str, str, str]] = [
    ("Random scores", "compression_random_scores", "RANDOM SCORES"),
    ("Compression", "compression_pixels_only", "Wrong px only"),
    ("Compression", "compression_pixels_and_length", "Wrong px + Plain token length"),
    ("Compression", "compression_no_edit_cost", "No \"actions\" cost"),
    ("Compression", "compression_no_unique_cost", "No \"unique concepts/lines\" cost"),
    ("Compression", "compression_flat_token_prices", "No weighted tokens"),
    ("Search", "search_best_1", "Best 1 (no islands)"),
    ("Search", "search_best_5", "Best 5 (no islands)"),
    ("Search", "search_no_demo_islands", "No demo islands"),
    ("Search", "search_no_shape_diff_island", "No shape-diff island"),
    ("Search", "search_double_time", "Double time"),
    ("Rounds of evolution", "search_rounds_1", "1 round"),
    ("Rounds of evolution", "search_rounds_3", "3 rounds"),
    ("Rounds of evolution", "search_rounds_5", "5 rounds"),
    ("Rounds of evolution", "ablation_control", "10 rounds (full solver)"),  # the default: the control's data, in gray
    ("Rounds of evolution", "search_rounds_15", "15 rounds"),
    ("Rounds of evolution", "search_rounds_20", "20 rounds"),
]


def run_scores(config: str) -> list[float]:
    """The total pass@2 score of each saved run of this config (an empty list if it hasn't been run)."""
    scores: list[float] = []
    for run_folder in sorted(RUNS.glob(f"{config}_run*")):
        with open(run_folder / "scores.csv") as scores_file:
            scores.append(sum(float(row["score"]) for row in csv.DictReader(scores_file)))
    return scores


def score_text(score: float) -> str:
    """Whole scores as "72", halves as "70.5", anything else (thirds, from levels with 3 tests) cut to 2 decimals: "0.33", "0.66"."""
    if abs(score - round(score)) < 1e-6:
        return f"{round(score)}"
    if abs(score * 2 - round(score * 2)) < 1e-6:
        return f"{score:.1f}"
    return f"{np.floor(score * 100 + 1e-6) / 100:.2f}"  # cut, not rounded (2/3 -> 0.66, as Nicky asked)


def score_label(score: float) -> str:
    """e.g. "70.5 (17.6%)"."""
    return f"{score_text(score)} ({score / LEVEL_COUNT:.1%})"


def main() -> None:
    control_scores: list[float] = run_scores("ablation_control")
    control_mean: float = float(np.mean(control_scores))
    band_low, band_high = min(control_scores), max(control_scores)

    # The bars: control first, then every variant that has a saved run
    rows: list[dict] = [{"name": f"Full solver, {len(control_scores)} control runs", "config": "ablation_control",
                         "score": control_mean, "color": CONTROL_COLOR, "group": "Control"}]
    for group, config, name in VARIANTS:
        # The control can show up again inside a group (10 rounds is the default), same data and gray as the top bar
        if config == "ablation_control":
            rows.append({"name": name, "config": config, "score": control_mean, "color": CONTROL_COLOR, "group": group})
            continue
        scores: list[float] = run_scores(config)
        if scores:
            rows.append({"name": name, "config": config, "score": scores[0], "color": GROUP_COLORS[group], "group": group})
        else:
            print(f"(skipping {config}: no saved run yet)")

    # Bar positions, top to bottom, with a small gap between groups so they read as blocks
    positions: list[float] = []
    position: float = 0.0
    for index, row in enumerate(rows):
        previous_group: str | None = rows[index - 1]["group"] if index > 0 else None
        if previous_group is not None and previous_group != row["group"]:
            position += 0.5
        positions.append(position)
        position += 1.0

    # Inches: room for the title + legend, plus each bar. Sized to fit an A4 page (about 7.5 x 10 inches of usable space)
    figure_height: float = 1.6 + 0.4 * position
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, figure_height), dpi=DPI, facecolor=STYLE["page"])
    axes_max: float = 0.2 * LEVEL_COUNT  # 20% of the levels, a little past the best bar

    # The control's band, faintly, across the whole chart, so every bar can be compared to seed luck
    axes.axvspan(band_low, band_high, color=CONTROL_COLOR, alpha=0.35, zorder=0, linewidth=0)
    for edge in [band_low, band_high]:
        axes.axvline(edge, color=INK, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)

    for row, y in zip(rows, positions):
        axes.barh(y, row["score"], height=0.74, color=row["color"], edgecolor=INK, linewidth=1.4, zorder=2)
        label: str = score_label(row["score"])
        if row["config"] == "ablation_control" and row["group"] == "Control":
            # The control's own range, as a whisker on its top bar (the repeat in the rounds group skips it: the band shows it)
            axes.errorbar(control_mean, y, xerr=[[control_mean - band_low], [band_high - control_mean]], color=INK,
                          linewidth=1.4, capsize=5, capthick=1.4, zorder=3)
            label = f"{label}, range {score_text(band_low)}-{score_text(band_high)}"
        # Inside the bar, at its left end: every bar is long enough, and it keeps the band's edges uncluttered
        axes.text(1.2, y, label, ha="left", va="center", fontfamily=TITLE_FONT, fontsize=10, color=INK, zorder=4)

    # Y labels: the readable name, with the config name small underneath
    axes.set_yticks(positions, [""] * len(rows))
    for row, y in zip(rows, positions):
        axes.text(-1.2, y - 0.1, row["name"], ha="right", va="center", fontfamily=TITLE_FONT, fontsize=11, color=INK)
        axes.text(-1.2, y + 0.24, row["config"], ha="right", va="center", fontfamily=CODE_FONT, fontsize=7,
                  color="#777777")
    axes.set_ylim(position - 0.4, -0.6)  # top to bottom

    # X axis: score out of 400, with the % in parens
    ticks: np.ndarray = np.arange(0, axes_max + 1, 10)
    axes.set_xticks(ticks, [score_label(tick) if tick % 20 == 0 else "" for tick in ticks])
    axes.set_xlim(0, axes_max)
    axes.set_xlabel(f"ARC-1 Eval levels solved (pass@2, out of {LEVEL_COUNT})", fontfamily=TITLE_FONT, fontsize=12,
                    color=INK, labelpad=10)
    # Title in the figure's top-left corner, with the legend's top lined up with it
    top_line: float = 1 - 0.3 / figure_height
    figure.text(0.03, top_line, "Variants", ha="left", va="top", fontfamily=TITLE_FONT, fontsize=16, color=INK)

    # Legend: one swatch per color group
    handles = [matplotlib.patches.Patch(facecolor=CONTROL_COLOR, edgecolor=INK, label="Full solver (min-max of 10 runs)")]
    legend_names: dict[str, str] = {"Compression": "How we score Accuracy + Simplicity (Compression)"}
    handles += [matplotlib.patches.Patch(facecolor=color, edgecolor=INK, label=legend_names.get(group, group))
                for group, color in GROUP_COLORS.items()]
    legend = figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.25, top_line + 0.04 / figure_height), ncol=2, frameon=False,
                         fontsize=8.5, handlelength=1.2, columnspacing=1.0, borderaxespad=0.3)
    for text in legend.get_texts():
        text.set_fontfamily(TITLE_FONT)
        text.set_color(INK)

    axes.set_facecolor(STYLE["panel"])
    axes.grid(axis="x", color=STYLE["grid"], linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(INK)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    axes.tick_params(colors=INK, labelsize=9, left=False)
    for tick_label in axes.get_xticklabels():
        tick_label.set_fontfamily(TITLE_FONT)

    # The numbers behind the chart, for the paper's text
    summary: str = (f"Control: {len(control_scores)} runs, mean {score_label(control_mean)}, "
                    f"range {score_label(band_low)} to {score_label(band_high)}\n")
    summary += "".join(f"  {row['config']}: {score_label(row['score'])}\n" for row in rows[1:]
                       if row["config"] != "ablation_control")
    (HERE / "ablations_scores.txt").write_text(summary)
    print(summary)

    figure.subplots_adjust(left=0.36, right=0.94, top=1 - 1.15 / figure_height, bottom=0.75 / figure_height)
    figure.savefig(HERE / "ablations.png", dpi=DPI, facecolor=STYLE["page"])
    plt.close(figure)
    print("wrote ablations.png")


if __name__ == "__main__":
    main()

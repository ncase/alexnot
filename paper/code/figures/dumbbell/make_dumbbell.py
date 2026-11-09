"""
make_dumbbell.py -- how much each system's score drops from ARC-AGI-1 to ARC-AGI-2, in log-odds.
One row per system: an arrow from its ARC-1 score (hollow dot) to its ARC-2 score (filled dot), labeled with the drop.
Rows are grouped by family, best to worst within each, in Nicky's order. No title or footnote: the paper's caption explains it.
Written by Claude, from talking with Nicky.

Why log-odds: raw ratios are unfair to high scorers (80% -> 40% is a far bigger loss of skill than 18% -> 9%).
log-odds(p) = ln(p / (1 - p)), so 50% is 0, and each step further out toward 0% or 100% is "harder" to make.
Colors: pastel red = AlexNot, blue = GPT, orange = Claude, green = Gemini, pink = Grok, purple = DeepSeek,
yellow = tiny ARC-trained nets (TRM, HRM). Grok's pink and DeepSeek's purple follow the ARC Prize leaderboard.
Data: hard-coded below, each with its source. Leaderboard numbers are from arcprize.org/leaderboard (fetched Nov 9, 2026),
Semi-Private sets, one effort per model (its highest).
Writes dumbbell.png (2800px wide, 3:2, white background) + dumbbell_scores.txt (every number behind the chart).
Run: .venv/bin/python paper/code/figures/dumbbell/make_dumbbell.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "palettes"))
from make_palettes import ARC_COLORS, CODE_FONT, PALETTES, TITLE_FONT, mix  # the locked paper style

STYLE: dict = PALETTES["A_arc_pastel"]
INK: str = STYLE["ink"]
FADED: str = "#777777"
FIGURE_WIDTH_INCHES: float = 7.5
FIGURE_HEIGHT_INCHES: float = FIGURE_WIDTH_INCHES * 2 / 3  # 3:2, as Nicky asked
DPI: float = 2800 / FIGURE_WIDTH_INCHES  # 2800px wide: it goes across the whole page

# Nicky's colors for this figure: ARC's own colors, softened toward white
FAMILY_COLORS: dict[str, str] = {
    "AlexNot": mix(ARC_COLORS[2], "#FFFFFF", 0.5),  # red
    "GPT": mix(ARC_COLORS[1], "#FFFFFF", 0.5),  # blue
    "Claude": mix(ARC_COLORS[7], "#FFFFFF", 0.5),  # orange
    "Gemini": mix(ARC_COLORS[3], "#FFFFFF", 0.5),  # green
    "Grok": mix(ARC_COLORS[6], "#FFFFFF", 0.5),  # pink (ARC's magenta, near the leaderboard's xAI pink)
    "DeepSeek": mix("#8B5CF6", "#FFFFFF", 0.5),  # purple (not an ARC color: ARC has no purple)
    "Tiny ARC-trained nets": mix(ARC_COLORS[4], "#FFFFFF", 0.4),  # yellow
}

# Every row, top to bottom: (family, name, effort or set, ARC-1 score, ARC-2 score, source)
SYSTEMS: list[tuple[str, str, str, float, float, str]] = [
    ("AlexNot", "AlexNot", "Public Eval", 70.5 / 400, 16 / 120,
     "ARC-1: mean of the 10 ablation control runs on all 400 ARC-1 Eval levels; ARC-2: 16/120 on ARC-2 Eval (Nov 1 daily blast)"),
    ("AlexNot", "AlexNot", "pessimistic estimate", 70.5 / 400, 0.0692,
     "ARC-1: as above; ARC-2: Nicky's pessimistic estimate of the Semi-Private score"),
    ("GPT", "GPT-6.1 Sol", "Max", 0.9650, 0.9417, "arcprize.org/leaderboard"),
    ("GPT", "GPT-5", "High", 0.6567, 0.0986, "arcprize.org/leaderboard"),
    ("Claude", "Claude Fable 5.1", "Max", 0.9750, 0.9000, "arcprize.org/leaderboard"),
    ("Claude", "Claude Opus 5", "Max", 0.9750, 0.9042, "arcprize.org/leaderboard"),
    ("Claude", "Claude Opus 4", "Thinking 16K", 0.3567, 0.0861, "arcprize.org/leaderboard"),
    ("Gemini", "Gemini 4 Argon", "High", 0.9850, 0.9125, "arcprize.org/leaderboard"),
    ("Gemini", "Gemini 3 Deep Think", "2/26", 0.9600, 0.8458, "arcprize.org/leaderboard"),
    ("Grok", "Grok 4.7", "XHigh", 0.8950, 0.6139, "arcprize.org/leaderboard"),
    ("Grok", "Grok 4", "Thinking", 0.6667, 0.1597, "arcprize.org/leaderboard"),
    ("DeepSeek", "DeepSeek V4.1 Flash", "10/17, Max", 0.9350, 0.7806, "arcprize.org/leaderboard"),
    ("DeepSeek", "DeepSeek R1", "1/25", 0.1580, 0.0130, "arcprize.org/leaderboard"),
    ("Tiny ARC-trained nets", "TRM", "Tiny Recursion Model", 0.4000, 0.0625, "arcprize.org/leaderboard"),
    ("Tiny ARC-trained nets", "HRM", "Hierarchical Reasoning Model", 0.3200, 0.0200, "arcprize.org/leaderboard"),
]

# Percent ticks on the log-odds axis (95% and 99% added, since the frontier models score up there)
PERCENT_TICKS: list[float] = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def log_odds(fraction: float) -> float:
    """ln(p / (1 - p)): 0 at 50%, heading to minus/plus infinity at 0%/100%."""
    return math.log(fraction / (1 - fraction))


def main() -> None:
    rows: list[dict] = [{"family": family, "name": name, "effort": effort, "arc1": arc1, "arc2": arc2, "source": source,
                         "delta": log_odds(arc2) - log_odds(arc1)}
                        for family, name, effort, arc1, arc2, source in SYSTEMS]

    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), dpi=DPI, facecolor=STYLE["page"])
    # Margins first: the y labels below are measured in axes coordinates, so the axes must already be in place
    figure.subplots_adjust(left=0.3, right=0.98, top=1 - 0.78 / FIGURE_HEIGHT_INCHES,
                           bottom=0.5 / FIGURE_HEIGHT_INCHES)

    for y, row in enumerate(rows):
        color: str = FAMILY_COLORS[row["family"]]
        start, end = log_odds(row["arc1"]), log_odds(row["arc2"])
        # The arrow: ARC-1 -> ARC-2, stopping just short of the filled dot so the arrowhead stays visible
        axes.annotate("", xy=(end + 0.1, y), xytext=(start - 0.08, y), zorder=2,
                      arrowprops={"arrowstyle": "-|>,head_length=0.4,head_width=0.18", "color": INK, "linewidth": 1.1,
                                  "shrinkA": 0, "shrinkB": 0})
        # Hollow dot = ARC-1, filled (in the family's color) = ARC-2
        axes.scatter(start, y, s=34, facecolor=STYLE["page"], edgecolor=INK, linewidth=1.1, zorder=3)
        axes.scatter(end, y, s=44, facecolor=color, edgecolor=INK, linewidth=1.1, zorder=3)
        # The drop, just past the ARC-1 dot (on the right, where there's always room)
        axes.text(start + 0.18, y, f"Δ = {row['delta']:.2f}".replace("-", "−"), ha="left", va="center",
                  fontfamily=TITLE_FONT, fontsize=7.5, color=INK, zorder=4)

    # Y labels on one line each, to stay compact: the effort small and faded, right-aligned, then the name to its left
    axes.set_yticks(range(len(rows)), [""] * len(rows))
    renderer = figure.canvas.get_renderer()
    for y, row in enumerate(rows):
        effort_text = axes.text(-0.012, y, row["effort"], transform=axes.get_yaxis_transform(), ha="right", va="center",
                                fontfamily=CODE_FONT, fontsize=5.5, color=FADED)
        # Measure the effort label, so the name can sit just left of it
        effort_left: float = axes.transAxes.inverted().transform(effort_text.get_window_extent(renderer))[0][0]
        axes.text(effort_left - 0.008, y, row["name"], transform=axes.get_yaxis_transform(), ha="right", va="center",
                  fontfamily=TITLE_FONT, fontsize=8.5, color=INK)
    axes.set_ylim(len(rows) - 0.5, -0.6)  # top to bottom

    # Top axis (the main one): log-odds, labeled in percent
    axes.set_xlim(log_odds(0.01) - 0.3, log_odds(0.99) + 1.0)
    axes.xaxis.tick_top()
    axes.xaxis.set_label_position("top")
    axes.set_xticks([log_odds(percent / 100) for percent in PERCENT_TICKS], [f"{percent:g}%" for percent in PERCENT_TICKS])
    axes.set_xlabel("Score (pass@2), on a log-odds scale", fontfamily=TITLE_FONT, fontsize=9, color=INK, labelpad=5)
    # Bottom axis, faded: the raw log-odds, for anyone checking the Δs
    bottom_axes = axes.secondary_xaxis("bottom")
    bottom_axes.set_xticks(range(-4, 5))
    bottom_axes.set_xlabel("log-odds = ln(p / (1 − p))", fontfamily=TITLE_FONT, fontsize=7.5, color=FADED, labelpad=3)
    bottom_axes.tick_params(colors=FADED, labelsize=7, pad=2)
    for tick_label in bottom_axes.get_xticklabels():
        tick_label.set_fontfamily(TITLE_FONT)
    bottom_axes.spines["bottom"].set_color("#AAAAAA")

    # Legend: one line across the very top. Hollow vs filled first, then each family's color
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=STYLE["page"],
                          markeredgecolor=INK, label="ARC-AGI-1 (hollow)"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="#BBBBBB",
                          markeredgecolor=INK, label="ARC-AGI-2 (filled)")]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=color, markeredgecolor=INK,
                           label=family) for family, color in FAMILY_COLORS.items()]
    legend = figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                           frameon=False, fontsize=6.5, handletextpad=0.1, columnspacing=0.6)
    for text in legend.get_texts():
        text.set_fontfamily(TITLE_FONT)
        text.set_color(INK)

    axes.set_facecolor(STYLE["panel"])
    axes.grid(axis="x", color=STYLE["grid"], linewidth=0.7, zorder=0)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(INK)
    # The main axis is on top, so the bottom spine is the faded log-odds one
    for side in ["bottom", "right"]:
        axes.spines[side].set_visible(False)
    axes.tick_params(colors=INK, labelsize=7.5, left=False, pad=2)
    for tick_label in axes.get_xticklabels():
        tick_label.set_fontfamily(TITLE_FONT)

    # The numbers behind the chart, with sources, for the paper's text
    summary: str = "".join(f"{row['name']} ({row['effort']}): {row['arc1']:.2%} -> {row['arc2']:.2%}, log-odds "
                           f"{log_odds(row['arc1']):+.2f} -> {log_odds(row['arc2']):+.2f}, delta {row['delta']:+.2f}  "
                           f"[{row['source']}]\n" for row in rows)
    (HERE / "dumbbell_scores.txt").write_text(summary)
    print(summary)

    figure.savefig(HERE / "dumbbell.png", dpi=DPI, facecolor=STYLE["page"])
    plt.close(figure)
    print("wrote dumbbell.png")


if __name__ == "__main__":
    main()

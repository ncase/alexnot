"""
make_palettes.py -- renders the SAME fake-data figure in a few palette variants, so Nicky can pick a style for the paper.
Each PNG: a swatch strip, an ablation bar chart, an evolution-rounds line plot, and a per-test Venn dot diagram.
ALL DATA HERE IS FAKE (it's only for judging colors/fonts).

Run:
  .venv/bin/python paper/code/figures/palettes/make_palettes.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no window, just files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE: Path = Path(__file__).resolve().parent

# The standard ARC colors, 0-9 (black, blue, red, green, yellow, gray, magenta, orange, azure, maroon)
ARC_COLORS: list[str] = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
                         "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]

TITLE_FONT: str = "Merriweather"  # Nicky's text font
CODE_FONT: str = "Monaco"      # Nicky's font for code-like labels (config names, DSL)


def mix(color: str, other: str, amount: float) -> str:
    """Blend `amount` of `other` into `color` (e.g. mix with white = pastel)."""
    a: np.ndarray = np.array(matplotlib.colors.to_rgb(color))
    b: np.ndarray = np.array(matplotlib.colors.to_rgb(other))
    return matplotlib.colors.to_hex(a * (1 - amount) + b * amount)


# Each palette: the colors for the roles every figure needs, plus the page/panel look
PALETTES: dict[str, dict] = {
    "A_arc_pastel": {
        "description": "ARC colors, softened halfway to white, dark ink outlines, white page",
        "solver": mix(ARC_COLORS[1], "#FFFFFF", 0.5), "neural": mix(ARC_COLORS[7], "#FFFFFF", 0.5),
        "both": mix(ARC_COLORS[3], "#FFFFFF", 0.5), "compression": mix(ARC_COLORS[2], "#FFFFFF", 0.5),
        "search": mix(ARC_COLORS[8], "#FFFFFF", 0.4), "highlight": mix(ARC_COLORS[4], "#FFFFFF", 0.4),
        "ink": "#222222", "page": "#FFFFFF", "panel": "#FFFFFF", "grid": "#DDDDDD",
    },
    "B_arc_pure": {
        "description": "the actual ARC colors, full strength, dark ink outlines, white page",
        "solver": ARC_COLORS[1], "neural": ARC_COLORS[7], "both": ARC_COLORS[3],
        "compression": ARC_COLORS[2], "search": ARC_COLORS[8], "highlight": ARC_COLORS[4],
        "ink": "#111111", "page": "#FFFFFF", "panel": "#FFFFFF", "grid": "#DDDDDD",
    },
    "C_gray_boxes": {
        "description": "ARC pastels on light gray panels (like the code boxes in Nicky's figure)",
        "solver": mix(ARC_COLORS[1], "#FFFFFF", 0.45), "neural": mix(ARC_COLORS[7], "#FFFFFF", 0.45),
        "both": mix(ARC_COLORS[3], "#FFFFFF", 0.45), "compression": mix(ARC_COLORS[6], "#FFFFFF", 0.5),
        "search": mix(ARC_COLORS[8], "#FFFFFF", 0.3), "highlight": mix(ARC_COLORS[4], "#FFFFFF", 0.3),
        "ink": "#2A2A2A", "page": "#FFFFFF", "panel": "#EEEEEE", "grid": "#FFFFFF",
    },
    "D_black_grids": {
        "description": "black panels like ARC grids, full ARC colors, light ink",
        "solver": ARC_COLORS[1], "neural": ARC_COLORS[7], "both": ARC_COLORS[3],
        "compression": ARC_COLORS[2], "search": ARC_COLORS[8], "highlight": ARC_COLORS[4],
        "ink": "#EEEEEE", "page": "#FFFFFF", "panel": "#000000", "grid": "#333333",
    },
    "E_two_tone": {
        "description": "just two soft hues (solver blue vs neural orange) + grays, dark ink",
        "solver": mix(ARC_COLORS[1], "#FFFFFF", 0.45), "neural": mix(ARC_COLORS[7], "#FFFFFF", 0.45),
        "both": mix(mix(ARC_COLORS[1], ARC_COLORS[7], 0.5), "#FFFFFF", 0.4), "compression": "#BBBBBB",
        "search": "#DDDDDD", "highlight": mix(ARC_COLORS[1], "#FFFFFF", 0.45),
        "ink": "#222222", "page": "#FFFFFF", "panel": "#FFFFFF", "grid": "#E5E5E5",
    },
}

# --- fake data (made-up numbers, roughly the right ballpark) ---
ABLATIONS: list[tuple[str, str, float, float]] = [  # (name, group, solve %, seed noise)
    ("full solver", "full", 30.0, 1.2),
    ("pixels only", "compression", 14.5, 1.0),
    ("pixels + length", "compression", 24.0, 1.1),
    ("- fewest actions", "compression", 28.2, 1.2),
    ("- unique concepts", "compression", 27.1, 1.2),
    ("- naturalness", "compression", 22.6, 1.1),
    ("1 parent, no islands", "search", 25.3, 1.2),
    ("- shape-diff island", "search", 28.9, 1.2),
    ("- demo islands", "search", 27.6, 1.2),
    ("double_time", "search", 31.0, 1.2),
]
EVO_ROUNDS: list[int] = [1, 2, 3, 5, 10, 15, 20]
EVO_SOLVE_PERCENT: list[float] = [12.0, 19.5, 23.0, 27.0, 30.0, 30.4, 30.5]
VENN_COUNTS: dict[str, int] = {"solver": 10, "both": 9, "neural": 37}  # per TEST
VENN_NEITHER: int = 111


def style_axes(axes: plt.Axes, palette: dict) -> None:
    """Panel background, ink-colored spines and ticks, title font everywhere."""
    axes.set_facecolor(palette["panel"])
    for spine in axes.spines.values():
        spine.set_color(palette["ink"] if palette["panel"] != "#000000" else "#000000")
        spine.set_linewidth(1.2)
    axes.tick_params(colors="#222222", labelsize=9)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontfamily(TITLE_FONT)


def draw_swatches(axes: plt.Axes, palette: dict) -> None:
    roles: list[str] = ["solver", "neural", "both", "compression", "search", "highlight"]
    for index, role in enumerate(roles):
        axes.add_patch(plt.Rectangle((index, 0), 0.9, 0.9, facecolor=palette[role], edgecolor="#222222", linewidth=1.5))
        axes.text(index + 0.45, -0.25, role, ha="center", va="top", fontsize=9, fontfamily=CODE_FONT, color="#222222")
    axes.set_xlim(-0.1, len(roles))
    axes.set_ylim(-0.6, 1)
    axes.axis("off")


def draw_ablation_bars(axes: plt.Axes, palette: dict) -> None:
    style_axes(axes, palette)
    names: list[str] = [name for name, _, _, _ in ABLATIONS][::-1]  # top-to-bottom = list order
    values: list[float] = [value for _, _, value, _ in ABLATIONS][::-1]
    noise: list[float] = [error for _, _, _, error in ABLATIONS][::-1]
    groups: list[str] = [group for _, group, _, _ in ABLATIONS][::-1]
    colors: list[str] = [palette["highlight"] if group == "full" else palette[group] for group in groups]
    edge: str = palette["ink"] if palette["panel"] != "#000000" else "#FFFFFF"
    axes.barh(names, values, xerr=noise, color=colors, edgecolor=edge, linewidth=1.3,
              error_kw={"ecolor": edge, "capsize": 3, "linewidth": 1})
    # The full solver's score as a dashed reference line
    axes.axvline(ABLATIONS[0][2], color=edge, linestyle="--", linewidth=1)
    axes.set_xlabel("ARC-1 Train solved (%)", fontfamily=TITLE_FONT, fontsize=10)
    axes.set_title("Ablations (FAKE DATA)", fontfamily=TITLE_FONT, fontsize=12, loc="left")
    axes.grid(axis="x", color=palette["grid"], linewidth=0.8)
    axes.set_axisbelow(True)
    for label in axes.get_yticklabels():
        label.set_fontfamily(CODE_FONT)
        label.set_fontsize(8.5)


def draw_evolution_line(axes: plt.Axes, palette: dict) -> None:
    style_axes(axes, palette)
    edge: str = palette["ink"] if palette["panel"] != "#000000" else "#FFFFFF"
    values: np.ndarray = np.array(EVO_SOLVE_PERCENT)
    # Shaded band = seed noise across runs
    axes.fill_between(EVO_ROUNDS, values - 1.3, values + 1.3, color=palette["solver"], alpha=0.35, linewidth=0)
    axes.plot(EVO_ROUNDS, values, color=edge, linewidth=1.5, zorder=2)
    axes.scatter(EVO_ROUNDS, values, s=60, color=palette["solver"], edgecolor=edge, linewidth=1.4, zorder=3)
    axes.set_xticks(EVO_ROUNDS)
    axes.set_xlabel("evolution rounds", fontfamily=TITLE_FONT, fontsize=10)
    axes.set_ylabel("solved (%)", fontfamily=TITLE_FONT, fontsize=10)
    axes.set_title("Evolution rounds (FAKE DATA)", fontfamily=TITLE_FONT, fontsize=12, loc="left")
    axes.grid(color=palette["grid"], linewidth=0.8)
    axes.set_axisbelow(True)


def points_in_region(region: str, count: int, centers: tuple, radius: float) -> np.ndarray:
    """Evenly spaced points (a hex grid) inside one Venn region, thinned down to `count`."""
    spacing: float = 0.16
    xs, ys = np.meshgrid(np.arange(-2.2, 2.2, spacing), np.arange(-1.4, 1.4, spacing * 0.866))
    xs = xs + (np.arange(xs.shape[0])[:, None] % 2) * spacing / 2  # offset every other row
    grid: np.ndarray = np.column_stack([xs.ravel(), ys.ravel()])
    # Keep a margin from the circle edges so dots don't sit on the lines
    in_left: np.ndarray = np.hypot(*(grid - centers[0]).T) < radius - 0.1
    in_right: np.ndarray = np.hypot(*(grid - centers[1]).T) < radius - 0.1
    out_left: np.ndarray = np.hypot(*(grid - centers[0]).T) > radius + 0.1
    out_right: np.ndarray = np.hypot(*(grid - centers[1]).T) > radius + 0.1
    mask: np.ndarray = {"solver": in_left & out_right, "neural": in_right & out_left, "both": in_left & in_right}[region]
    candidates: np.ndarray = grid[mask]
    # Take the `count` points closest to the region's middle, so the dots cluster nicely
    middle: np.ndarray = candidates.mean(axis=0)
    order: np.ndarray = np.argsort(np.hypot(*(candidates - middle).T))
    return candidates[order[:count]]


def draw_venn(axes: plt.Axes, palette: dict) -> None:
    style_axes(axes, palette)
    edge: str = palette["ink"] if palette["panel"] != "#000000" else "#FFFFFF"
    centers: tuple = (np.array([-0.55, 0.0]), np.array([0.55, 0.0]))
    radius: float = 1.15
    for center, role in zip(centers, ["solver", "neural"]):
        axes.add_patch(Circle(center, radius, facecolor=palette[role], alpha=0.25, edgecolor="none"))
        axes.add_patch(Circle(center, radius, facecolor="none", edgecolor=edge, linewidth=1.8))
    # One dot per test, colored by which side(s) solved it
    for region, count in VENN_COUNTS.items():
        points: np.ndarray = points_in_region(region, count, centers, radius)
        axes.scatter(points[:, 0], points[:, 1], s=34, color=palette[region], edgecolor=edge, linewidth=1, zorder=3)
    axes.text(-1.0, 1.25, f"the solver\n{VENN_COUNTS['solver'] + VENN_COUNTS['both']} tests", ha="center", va="bottom",
              fontfamily=TITLE_FONT, fontsize=10, color=edge)
    axes.text(1.0, 1.25, f"NVARC 4B\n{VENN_COUNTS['neural'] + VENN_COUNTS['both']} tests", ha="center", va="bottom",
              fontfamily=TITLE_FONT, fontsize=10, color=edge)
    axes.text(0, -1.45, f"neither: {VENN_NEITHER} of 167 tests", ha="center", va="top", fontfamily=CODE_FONT,
              fontsize=8.5, color=edge)
    axes.set_title("Public Eval, per test (FAKE DATA)", fontfamily=TITLE_FONT, fontsize=12, loc="left")
    axes.set_xlim(-2.0, 2.0)
    axes.set_ylim(-1.75, 1.95)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])


def main() -> None:
    for name, palette in PALETTES.items():
        figure = plt.figure(figsize=(15, 5.6), facecolor=palette["page"])
        grid = figure.add_gridspec(2, 3, height_ratios=[0.22, 1], hspace=0.35, wspace=0.35)
        draw_swatches(figure.add_subplot(grid[0, :]), palette)
        draw_ablation_bars(figure.add_subplot(grid[1, 0]), palette)
        draw_evolution_line(figure.add_subplot(grid[1, 1]), palette)
        draw_venn(figure.add_subplot(grid[1, 2]), palette)
        figure.suptitle(f"{name}: {palette['description']}", fontfamily=TITLE_FONT, fontsize=13, x=0.01, ha="left")
        figure.savefig(HERE / f"{name}.png", dpi=130, bbox_inches="tight", facecolor=palette["page"])
        plt.close(figure)
        print(f"wrote {name}.png")


if __name__ == "__main__":
    main()

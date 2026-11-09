# Test helper: print a 2D ARC int-array as colored terminal blocks.
# Import me from other tests: from alexnot.tests.print_grid import print_grid

# ARC's official 10-color palette, as (R,G,B)
ARC_COLORS = {
    0: (0, 0, 0),       # black
    1: (0, 116, 217),   # blue
    2: (255, 65, 54),   # red
    3: (46, 204, 64),   # green
    4: (255, 220, 0),   # yellow
    5: (170, 170, 170), # grey
    6: (240, 18, 190),  # fuchsia
    7: (255, 133, 27),  # orange
    8: (127, 219, 255), # azure
    9: (135, 12, 37),   # maroon
}

def print_grid(grid):
    """Print a 2D ARC int-array as colored blocks in the terminal."""
    for row in grid:
        line = ""
        for cell in row:
            r, g, b = ARC_COLORS.get(cell, (60, 60, 60)) # fallback grey for -1/unknown
            line += f"\x1b[48;2;{r};{g};{b}m  \x1b[0m" # two spaces = one squarish cell, then RESET
        print(line)


# ALTERNATIVE: emoji-square palette. The solid ANSI blocks above merge together, making it
# hard to count exact pixel widths/positions. Emoji squares have visible cell borders, so you
# can eyeball width/left/etc. Caveats: no exact ARC color match (no grey or cyan square emoji),
# so grey~white and azure~blue-circle (kept distinct from blue square on purpose).
ARC_EMOJI = {
    0: "⬛", # black
    1: "🟦", # blue
    2: "🟥", # red
    3: "🟩", # green
    4: "🟨", # yellow
    5: "⬜️", # grey (or, white...)
    6: "🟪", # fuchsia
    7: "🟧", # orange
    8: "🩵", # cyan
    9: "🟫", # maroon (brown)
}

def print_grid_emoji(grid):
    """Print a 2D ARC int-array as emoji squares — easier to count exact pixel positions."""
    for row in grid:
        # two spaces for -1/unknown (empty cell); ~matches the emoji's 2-column width
        line = "".join(ARC_EMOJI.get(cell, "  ") for cell in row)
        print(line)


def print_grid_comparison(predicted, expected, label:str="") -> int:
    """
    Print predicted vs expected grids side-by-side in TWO formats:
    1) Plain int arrays (LLM-friendly, easy for Claude to parse)
    2) Emoji grids (human-friendly, easy to eyeball)
    Returns the number of wrong pixels.
    """
    import numpy as np
    pred = np.asarray(predicted)
    exp  = np.asarray(expected)

    if label:
        print(f"\n--- {label} ---")

    # Count wrong pixels (same logic as score_transform)
    same_dims = (pred.shape == exp.shape)
    min_h = min(pred.shape[0], exp.shape[0])
    min_w = min(pred.shape[1], exp.shape[1])
    overlap_wrong = int(np.count_nonzero(pred[:min_h, :min_w] != exp[:min_h, :min_w]))
    p_area = pred.shape[0] * pred.shape[1]
    e_area = exp.shape[0] * exp.shape[1]
    total_wrong = overlap_wrong + (p_area + e_area - 2 * min_h * min_w)

    if not same_dims:
        print(f"  Dimensions differ! predicted={pred.shape[1]}x{pred.shape[0]}, expected={exp.shape[1]}x{exp.shape[0]}")

    if total_wrong == 0:
        print(f"  PASS — perfect match!")
        return 0

    print(f"  FAIL — {total_wrong} wrong pixel(s)")

    # --- Format 1: Plain int arrays (for LLM consumption) ---
    print(f"\n  Predicted (int):        Expected (int):")
    max_h = max(pred.shape[0], exp.shape[0])
    for y in range(max_h):
        # predicted row as comma-separated ints, or blank if out of bounds
        p_str = str(pred[y].tolist()) if y < pred.shape[0] else ""
        e_str = str(exp[y].tolist())  if y < exp.shape[0]  else ""
        print(f"  {p_str:<30s} {e_str}")

    # --- Format 2: Emoji side-by-side with diff markers (for human eyeballing) ---
    max_w_pred = pred.shape[1]
    max_w_exp  = exp.shape[1]
    print(f"\n  {'Predicted':<{max_w_pred*2}}   {'Expected':<{max_w_exp*2}}   Diff")
    for y in range(max_h):
        p_row = "".join(ARC_EMOJI.get(int(c), "  ") for c in pred[y]) if y < pred.shape[0] else "  " * max_w_pred
        e_row = "".join(ARC_EMOJI.get(int(c), "  ") for c in exp[y])  if y < exp.shape[0]  else "  " * max_w_exp
        # diff column: dot for match, X for mismatch
        diff = ""
        for x in range(max(max_w_pred, max_w_exp)):
            p_val = pred[y, x] if (y < pred.shape[0] and x < pred.shape[1]) else None
            e_val = exp[y, x]  if (y < exp.shape[0]  and x < exp.shape[1]) else None
            if p_val is not None and e_val is not None and p_val == e_val:
                diff += ". "
            else:
                diff += "X "
        print(f"  {p_row} | {e_row} | {diff}")

    return total_wrong

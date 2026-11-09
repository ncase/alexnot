from typing import List, Dict, Tuple
import numpy as np

def _fill_holes(shape:Dict, color:int=-1, in_place=True) -> Dict:

    # If no color specified, use shape's color
    color = int(shape['color'][1:]) if color<0 else color # don't forget to cast back to int!

    # If not in_place, make copy.
    if not in_place:
        shape = {**shape, '_hole_grids': [h.copy() for h in shape['_hole_grids']]}

    for hole_grid in shape['_hole_grids']:

        # Recolor each shape the color of the shape (in place)
        colored_hole_grid = _recolor_pixels(hole_grid, color, in_place=True)

        # Draw it over original (in place)
        _draw_pixels(shape['_pixels'], colored_hole_grid, 0, 0, in_place=True)

    # Update _hole_grids (empty), holes (zero), shadow, area
    shape['_hole_grids'] = []
    shape['holes'] = 0
    shape['_shadow'] = (shape['_pixels'] >= 0).astype(np.int8) # 0 if not in shape, 1 if so
    shape['area'] = int(np.count_nonzero(shape['_shadow'])) # 1 if in shadow, else 0

    # Gimme!
    return shape

def _recolor_pixels(shape_grid:np.ndarray, new_color:int, in_place:bool=True) -> np.ndarray:

    # In place? Modify original or make a copy?
    pixels = shape_grid if in_place else shape_grid.copy()

    # Recolor all non-empty cells, leave -1 (empty) alone
    pixels[pixels >= 0] = new_color

    # Gimme
    return pixels

def _draw_pixels(original_grid:np.ndarray, shape_grid:np.ndarray, x:int, y:int, in_place:bool=True) -> np.ndarray:

    # In place? Modify original or make a copy?
    pixels = original_grid if in_place else original_grid.copy()
    orig_h, orig_w = pixels.shape
    shape_h, shape_w = shape_grid.shape
    x, y = int(x), int(y) # HACK: this SHOULD already be an int

    # Compute overlap between shape and target grid (clamp to both edges)
    src_y0, src_x0 = max(0, -y), max(0, -x)
    src_y1 = min(shape_h, orig_h - y)
    src_x1 = min(shape_w, orig_w - x)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return pixels # no overlap, nothing to draw

    # Extract the overlapping region of the shape
    src_region = shape_grid[src_y0:src_y1, src_x0:src_x1]
    mask = src_region >= 0 # only draw where the shape has a real color (not -1)

    # Write colored shape pixels into destination (view, so modifies pixels in place)
    dst_region = pixels[max(0, y):max(0, y) + (src_y1 - src_y0),
                        max(0, x):max(0, x) + (src_x1 - src_x0)]
    np.copyto(dst_region, src_region, where=mask)

    # Gimme!
    return pixels

def _flip_grid(grid:np.ndarray, mode:str) -> np.ndarray:
    if mode == 'horizontal': # like A, H, I
        return grid[:, ::-1].copy() # flip columns
    elif mode == 'vertical': # like B, C, D
        return grid[::-1].copy() # flip rows
    elif mode == 'diagonal_1': # like %
        # 180° rotation then transpose (same as original: grid[h-1-y][w-1-x] scanned x-then-y)
        return grid[::-1, ::-1].T.copy()
    elif mode == 'diagonal_2': # like Q
        return grid.T.copy() # transpose (swaps rows<->cols)
    else:
        raise ValueError(f"_flip_grid: unknown mode {mode!r}")

def _rotate_grid(grid:np.ndarray, quarters:int) -> np.ndarray: # TODO: 'cw' and 'ccw' instead?
    quarters = quarters % 4
    if quarters == 0:
        return grid
    # np.rot90 rotates COUNTER-clockwise; negate quarters for CLOCKWISE
    return np.rot90(grid, k=-quarters).copy()

# Find-and-replace on raw pixels: wherever `old` matches the grid, stamp `new` on top.
# Both are 2D int grids of the same shape, colors 0-9, with -1 meaning "don't care" in `old`
# (that cell matches anything) and "leave that pixel alone" in `new`.
def _replace_pixels(pixels:np.ndarray, old:np.ndarray, new:np.ndarray) -> Tuple[np.ndarray, int]:
    """Find-and-replace a pixel pattern. Returns the new pixels AND how many spots matched (for edit counting)."""

    grid_height, grid_width = pixels.shape
    pattern_height, pattern_width = old.shape

    # Pad the grid with -1 so the pattern may hang off the edges: an off-grid cell only matches a
    # -1 (don't care) cell of the pattern, never a real color. (This mirrors how the pixels_change
    # monkey reads off-grid context as -1, so a pattern it learned at an edge matches the same way here.)
    pad_y:int = pattern_height - 1
    pad_x:int = pattern_width - 1
    padded:np.ndarray = np.pad(pixels, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=-1)

    # One boolean per window position (top-left corner of the pattern in padded coords)
    windows_height:int = grid_height + pad_y
    windows_width:int = grid_width + pad_x
    matches:np.ndarray = np.ones((windows_height, windows_width), dtype=bool)

    # AND together "this cell of the pattern matches" for every real (non -1) pattern cell. Vectorized:
    # each cell is one shifted comparison over the whole grid, so it's cheap even on 30x30.
    for cell_y, cell_x in zip(*np.nonzero(old >= 0)):
        matches &= padded[cell_y:cell_y+windows_height, cell_x:cell_x+windows_width] == old[cell_y, cell_x]
    match_ys, match_xs = np.nonzero(matches)
    if len(match_ys) == 0:
        return pixels, 0 # nothing matched, nothing to do

    # Find ALL matches first (above, against the untouched grid), THEN write: so overlapping matches
    # don't see each other's edits. Writes go into a copy of the padded grid, so `new` cells that land
    # off-grid just get cropped away below.
    result:np.ndarray = padded.copy()
    for cell_y, cell_x in zip(*np.nonzero(new >= 0)):
        result[match_ys + cell_y, match_xs + cell_x] = new[cell_y, cell_x]

    # Crop the padding back off. Gimme! (Each matched spot is one replacement, so the scorer can count it as one edit)
    return result[pad_y:pad_y+grid_height, pad_x:pad_x+grid_width], len(match_ys)

def _chebyshev_dilate(filled:np.ndarray, radius:int) -> np.ndarray:
    """Grow a boolean mask by `radius` pixels in every direction (square corners). Same size out as in."""
    # Separable trick: a Chebyshev (square) dilation = dilate horizontally THEN vertically.
    # Cost: 2*(2*radius+1) OR ops instead of (2*radius+1)^2, a big win at radius > 1.
    # (Pixels that would land outside the grid just get cut off, so pad first if you need room to grow.)
    grid_height, grid_width = filled.shape
    # Horizontal pass: any neighbor in the row within ±radius is filled
    horizontal_any:np.ndarray = np.zeros((grid_height, grid_width), dtype=bool)
    for shift_x in range(-radius, radius+1):
        source_start:int = max(0, -shift_x)
        source_end:int = grid_width - max(0, shift_x)
        destination_start:int = max(0, shift_x)
        horizontal_any[:, destination_start:destination_start + source_end - source_start] |= filled[:, source_start:source_end]
    # Vertical pass: any neighbor in the column within ±radius got set by the horizontal pass
    dilated:np.ndarray = np.zeros((grid_height, grid_width), dtype=bool)
    for shift_y in range(-radius, radius+1):
        source_start = max(0, -shift_y)
        source_end = grid_height - max(0, shift_y)
        destination_start = max(0, shift_y)
        dilated[destination_start:destination_start + source_end - source_start, :] |= horizontal_any[source_start:source_end, :]
    return dilated

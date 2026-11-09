########################################################
## Pattern backgrounds: a big repeating (tiled) or mirror-symmetric picture behind everything,
## partly covered up by shapes that break its rule
########################################################

from typing import Callable, NamedTuple, Optional
import numpy as np
import scipy.signal # FFT correlation: how well the grid lines up with a shifted / flipped copy of itself, at every shift at once
import scipy.sparse
import scipy.sparse.csgraph # connected components: which pixels the pattern says must be the same color

# A grid has to be at least this big both ways to have a pattern background (smaller = too few repeats to trust)
_PATTERN_MIN_SIDE:int = 6

# Its most common color can't cover more than this fraction of it (else it's a plain bg with a few shapes on it)
_PATTERN_MAX_DOMINANT_FRACTION:float = 0.75

# Stricter for mirrors & turns: a mostly-plain grid with one symmetric shape on it is symmetric too, but it's a shape on
# a plain bg, not a pattern background. (Real mirrored backgrounds measured 0.17-0.31, symmetric shapes on a plain bg 0.42 and up)
_PATTERN_MAX_DOMINANT_FRACTION_MIRROR:float = 0.35

# A tiling needs a repeat at least this long in at least one direction, so 2-color stripes (every other row or
# column) count. (Some things that aren't patterns, like a few boxes or bars in a row, line up at period 2 too:
# those can get found, and then have to lose the parse ranking. The other direction may repeat every 1 or 2)
_PATTERN_MIN_TILE_PERIOD:int = 2

# A direction's smallest passing period gets swapped for a multiple of it that agrees at least this much better.
# (E.g. rows A B C B: shift 2 already lines up the B rows and half of A/C, but the real repeat is 4. Real tiles'
# multiples measured at most +0.07, fakes' at most +0.11, the 4-row tile above +0.25: this sits about halfway between)
_PATTERN_MIN_MULTIPLE_GAIN:float = 0.20

# A repeat or a mirror counts if at least this fraction of the pixels it lines up have the same color...
_PATTERN_MIN_AGREEMENT:float = 0.70

# ...and it disagrees at most this fraction as often as two random pixels of the grid would. (On a grid that's mostly
# one color, everything lines up with everything a lot, so the bar goes up there)
_PATTERN_MAX_RELATIVE_MISMATCH:float = 0.5

# A mirror has to line up at least this fraction of the grid with itself (a mirror line near an edge proves little)
_PATTERN_MIN_MIRROR_OVERLAP:float = 0.5

# A color is "trusted" (part of the pattern, not a cover) if at least this fraction of its links land on the same color.
# (Pattern colors measured 0.69-1.0, covering colors 0.0-0.4)
_PATTERN_MIN_TRUST:float = 0.5

# A partial mirror (only true in places, e.g. one quarter of the picture is mirrored along its diagonal) still has to
# line up at least this fraction of the pixels. It's only ever used to fill pixels the real mirrors can't see
_PATTERN_MIN_PARTIAL_AGREEMENT:float = 0.5


# A map from pixels to the pixels the pattern says match them: (ys, xs) -> (mapped ys, mapped xs), in grid coords
PixelMap = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

class PatternBackground(NamedTuple):
    """The pattern, completed: its pixels (may reach past the grid's edges), where its top-left sits, and which grid pixels break it."""
    pixels:np.ndarray # every pixel of the completed pattern, no gaps
    top:int # can be negative (the pattern goes off the top of the grid)
    left:int # can be negative too
    breaks_pattern:np.ndarray # grid-sized: True where the grid's pixel isn't the pattern's color (= a shape in front)


# Cache: grid -> its pattern background (or None). The same grids get parsed under dozens of configs, and the
# pattern only depends on the pixels. search_parse empties it at the start & end of each search
_pattern_background_cache:dict[tuple, Optional[PatternBackground]] = {}

def find_pattern_background(grid:np.ndarray) -> Optional[PatternBackground]:
    """The grid's big repeating or mirror-symmetric pattern, completed under whatever covers it, or None if it has none. Cached per grid."""
    cache_key:tuple = (grid.shape, grid.tobytes())
    if cache_key not in _pattern_background_cache:
        _pattern_background_cache[cache_key] = _find_pattern_background(grid)
    return _pattern_background_cache[cache_key]

def clear_pattern_background_cache() -> None:
    """Forget every cached pattern (search_parse calls this at the start & end of each search)."""
    _pattern_background_cache.clear()


def _find_pattern_background(grid:np.ndarray) -> Optional[PatternBackground]:
    """find_pattern_background, uncached."""
    height, width = grid.shape

    # Cheapest checks first: too small, or mostly one color? Then no pattern (most grids bail out right here)
    if min(height, width) < _PATTERN_MIN_SIDE:
        return None
    colors, color_counts = np.unique(grid, return_counts=True)
    color_fractions:np.ndarray = color_counts / grid.size
    if color_fractions.max() > _PATTERN_MAX_DOMINANT_FRACTION:
        return None

    # How often would two random pixels match? A repeat or mirror has to do much better than that to count
    random_agreement:float = float((color_fractions ** 2).sum())
    min_agreement:float = max(_PATTERN_MIN_AGREEMENT, 1 - _PATTERN_MAX_RELATIVE_MISMATCH * (1 - random_agreement))

    # One yes/no layer per color, for the FFT correlations
    color_layers:np.ndarray = (grid[None, :, :] == colors[:, None, None]).astype(np.float64)

    # Tiling first: the repeat down the rows and across the columns (at least one of them 2+ long)
    pixel_maps, tile_height, tile_width = _find_repeats(color_layers, min_agreement)
    partial_mirrors:list[PixelMap] = [] # (only mirrors get these)
    canvas_top, canvas_left, canvas_bottom, canvas_right = 0, 0, height, width # (a tiling only gets completed inside the grid)

    # No tiling? Then mirrors & turns, which may complete the picture past the grid's edges.
    # (Only on grids that aren't mostly one color: see _PATTERN_MAX_DOMINANT_FRACTION_MIRROR)
    if not pixel_maps:
        if color_fractions.max() > _PATTERN_MAX_DOMINANT_FRACTION_MIRROR:
            return None
        mirrors:list[tuple[float, PixelMap]] = _find_mirrors(grid, color_layers)
        pixel_maps = [mirror for agreement, mirror in mirrors if agreement >= min_agreement]
        if not pixel_maps:
            return None
        # The ones that only line up in places, best first, for the pixels the real mirrors can't see (see below)
        partial_mirrors = [mirror for agreement, mirror in sorted(mirrors, key=lambda agreement_and_mirror: -agreement_and_mirror[0])
                           if _PATTERN_MIN_PARTIAL_AGREEMENT <= agreement < min_agreement]
        # The canvas = the grid, plus wherever each mirror sends it (a mirror is affine, so its corners give its box)
        corner_ys:np.ndarray = np.array([0, 0, height - 1, height - 1])
        corner_xs:np.ndarray = np.array([0, width - 1, 0, width - 1])
        for pixel_map in pixel_maps:
            mapped_ys, mapped_xs = pixel_map(corner_ys, corner_xs)
            canvas_top, canvas_left = min(canvas_top, int(mapped_ys.min())), min(canvas_left, int(mapped_xs.min()))
            canvas_bottom, canvas_right = max(canvas_bottom, int(mapped_ys.max()) + 1), max(canvas_right, int(mapped_xs.max()) + 1)

    # Which pixels does the pattern say must be the same color? Link every canvas pixel to where each map sends it,
    # then each connected group ("orbit") is one color
    canvas_height:int = canvas_bottom - canvas_top
    canvas_width:int = canvas_right - canvas_left
    canvas_ys, canvas_xs = np.indices((canvas_height, canvas_width)).reshape(2, -1)
    canvas_ys, canvas_xs = canvas_ys + canvas_top, canvas_xs + canvas_left # (in grid coords)
    edge_starts:list[np.ndarray] = []
    edge_ends:list[np.ndarray] = []
    for pixel_map in pixel_maps:
        mapped_ys, mapped_xs = pixel_map(canvas_ys, canvas_xs)
        # (Links that leave the canvas are dropped: the canvas is everything we complete)
        in_canvas:np.ndarray = (mapped_ys >= canvas_top) & (mapped_ys < canvas_bottom) & (mapped_xs >= canvas_left) & (mapped_xs < canvas_right)
        edge_starts.append(((canvas_ys - canvas_top) * canvas_width + (canvas_xs - canvas_left))[in_canvas])
        edge_ends.append(((mapped_ys - canvas_top) * canvas_width + (mapped_xs - canvas_left))[in_canvas])
    starts:np.ndarray = np.concatenate(edge_starts)
    ends:np.ndarray = np.concatenate(edge_ends)
    pixel_count:int = canvas_height * canvas_width
    links = scipy.sparse.coo_matrix((np.ones(len(starts), dtype=np.int8), (starts, ends)), shape=(pixel_count, pixel_count))
    orbit_count, orbit_of_pixel = scipy.sparse.csgraph.connected_components(links, directed=False)

    # How much do we trust each color? The fraction of its links (between two in-grid pixels) that land on the same
    # color. A pattern color nearly always matches its partners; a covering color mostly doesn't
    in_grid:np.ndarray = (canvas_ys >= 0) & (canvas_ys < height) & (canvas_xs >= 0) & (canvas_xs < width)
    color_index_grid:np.ndarray = np.searchsorted(colors, grid) # each pixel's color, as an index into colors
    color_index_of_pixel:np.ndarray = np.full(pixel_count, -1)
    color_index_of_pixel[in_grid] = color_index_grid[canvas_ys[in_grid], canvas_xs[in_grid]]
    start_colors:np.ndarray = color_index_of_pixel[starts]
    end_colors:np.ndarray = color_index_of_pixel[ends]
    both_in_grid:np.ndarray = (start_colors >= 0) & (end_colors >= 0)
    link_colors:np.ndarray = np.concatenate([start_colors[both_in_grid], end_colors[both_in_grid]])
    link_matches:np.ndarray = np.tile((start_colors == end_colors)[both_in_grid], 2).astype(np.float64)
    color_trust:np.ndarray = (np.bincount(link_colors, weights=link_matches, minlength=len(colors))
                              / np.maximum(np.bincount(link_colors, minlength=len(colors)), 1))
    # (A tiny floor, so an orbit whose only visible pixels are untrusted colors still gets one of them)
    color_trust += 1e-6

    # Each orbit's color: the vote of its in-grid pixels, each weighted by its color's trust.
    # (Ties, which need exactly equal trust sums, go to the lowest color number)
    orbit_votes:np.ndarray = np.zeros((orbit_count, len(colors)))
    in_grid_colors:np.ndarray = color_index_of_pixel[in_grid]
    np.add.at(orbit_votes, (orbit_of_pixel[in_grid], in_grid_colors), color_trust[in_grid_colors])
    orbit_color_indices:np.ndarray = orbit_votes.argmax(axis=1)
    pattern_color_indices:np.ndarray = orbit_color_indices[orbit_of_pixel]

    # Orbits only covered-up pixels can see (they won with an untrusted color)? Borrow their color through a partial
    # mirror, wherever it lands on a trusted pixel, best mirror first. (E.g. a cover sitting on pixels whose mirror
    # images are all off the grid or covered too)
    pixel_is_trusted:np.ndarray = (color_trust[orbit_color_indices] >= _PATTERN_MIN_TRUST)[orbit_of_pixel]
    for partial_mirror in partial_mirrors:
        untrusted_pixels:np.ndarray = np.flatnonzero(~pixel_is_trusted)
        if len(untrusted_pixels) == 0:
            break
        mapped_ys, mapped_xs = partial_mirror(canvas_ys[untrusted_pixels], canvas_xs[untrusted_pixels])
        in_canvas:np.ndarray = (mapped_ys >= canvas_top) & (mapped_ys < canvas_bottom) & (mapped_xs >= canvas_left) & (mapped_xs < canvas_right)
        untrusted_pixels, mapped_pixels = untrusted_pixels[in_canvas], ((mapped_ys - canvas_top) * canvas_width + (mapped_xs - canvas_left))[in_canvas]
        borrows:np.ndarray = pixel_is_trusted[mapped_pixels]
        pattern_color_indices[untrusted_pixels[borrows]] = pattern_color_indices[mapped_pixels[borrows]]
        pixel_is_trusted[untrusted_pixels[borrows]] = True
    pattern_pixels:np.ndarray = colors[pattern_color_indices].reshape(canvas_height, canvas_width).astype(grid.dtype)

    # A one-color "pattern" is just a plain background. (For a tiling, one tile is enough to check: a one-color tile
    # tiles into a one-color pattern, and is just a solid block either way. For mirrors the tile is the whole grid)
    pattern_in_grid:np.ndarray = pattern_pixels[-canvas_top:height - canvas_top, -canvas_left:width - canvas_left]
    one_tile:np.ndarray = pattern_in_grid[:tile_height, :tile_width]
    if (one_tile == one_tile.flat[0]).all():
        return None
    return PatternBackground(pattern_pixels, canvas_top, canvas_left, pattern_in_grid != grid)


def _correlate_layers(color_layers:np.ndarray, other_layers:np.ndarray) -> np.ndarray:
    """At every shift (shift_y, shift_x): how many pixels p have grid[p + shift] == other[p] (same color)? Index [shift_y + other_height-1, shift_x + other_width-1]."""
    matches:np.ndarray = scipy.signal.fftconvolve(color_layers, other_layers[:, ::-1, ::-1], mode='full', axes=(1, 2)).sum(axis=0)
    return np.rint(matches) # (FFTs give floats with tiny rounding noise)

def _overlap_counts(height:int, width:int, other_height:int, other_width:int) -> np.ndarray:
    """At every shift, same indexing as _correlate_layers: how many pixels of the other grid land on the grid?"""
    shift_ys:np.ndarray = np.arange(-(other_height - 1), height)
    shift_xs:np.ndarray = np.arange(-(other_width - 1), width)
    # Rows p (0..other_height-1) with p + shift inside 0..height-1, same for columns
    overlap_rows:np.ndarray = np.minimum(other_height, height - shift_ys) - np.maximum(0, -shift_ys)
    overlap_columns:np.ndarray = np.minimum(other_width, width - shift_xs) - np.maximum(0, -shift_xs)
    return np.outer(overlap_rows, overlap_columns)

def _find_repeats(color_layers:np.ndarray, min_agreement:float) -> tuple[list[PixelMap], int, int]:
    """The tiling's pixel maps (none if it doesn't tile), and its tile's height & width (the whole grid's, along a direction that doesn't repeat)."""
    _, height, width = color_layers.shape
    agreement:np.ndarray = _correlate_layers(color_layers, color_layers) / _overlap_counts(height, width, height, width)

    # Each direction's period, on its own (None = no repeat that way). Agreement by shift: down the rows, across the columns
    row_period:Optional[int] = _pick_period(agreement[height - 1:, width - 1], height, min_agreement)
    column_period:Optional[int] = _pick_period(agreement[height - 1, width - 1:], width, min_agreement)

    # A tiling needs a real repeat (2+ long) in at least one direction
    if max(row_period or 0, column_period or 0) < _PATTERN_MIN_TILE_PERIOD:
        return [], height, width

    # Only one direction repeats? Lots of covers can break the repeat both ways, so re-measure the other direction
    # counting only the colors the found repeat trusts (covers don't count against it)
    if row_period is None:
        trusted_layers:np.ndarray = _trusted_layers(color_layers, axis=2, period=column_period)
        row_period = _pick_period(_trusted_agreement_by_shift(trusted_layers, axis=1), height, min_agreement)
    elif column_period is None:
        trusted_layers:np.ndarray = _trusted_layers(color_layers, axis=1, period=row_period)
        column_period = _pick_period(_trusted_agreement_by_shift(trusted_layers, axis=2), width, min_agreement)

    # The other direction may repeat every 1 (a stripe: every row or column is the same), every 2, or not at all.
    # Period 1 still links each pixel to its neighbor along the stripe, so covers on a stripe get its color
    pixel_maps:list[PixelMap] = []
    tile_height, tile_width = height, width
    if row_period is not None:
        pixel_maps.append(lambda ys, xs, period=row_period: (ys + period, xs))
        tile_height = row_period
    if column_period is not None:
        pixel_maps.append(lambda ys, xs, period=column_period: (ys, xs + period))
        tile_width = column_period
    return pixel_maps, tile_height, tile_width

def _trusted_layers(color_layers:np.ndarray, axis:int, period:int) -> np.ndarray:
    """The color layers of just the trusted colors: those whose pixels mostly match their partner one period along the axis (1 = rows, 2 = columns)."""
    # Each pixel paired with the one a period further along (both pixels of a pair are links of their colors)
    size:int = color_layers.shape[axis]
    starts:np.ndarray = color_layers.take(np.arange(size - period), axis=axis)
    ends:np.ndarray = color_layers.take(np.arange(period, size), axis=axis)
    same_color_links:np.ndarray = 2 * (starts * ends).sum(axis=(1, 2))
    all_links:np.ndarray = starts.sum(axis=(1, 2)) + ends.sum(axis=(1, 2))
    # Same trust bar as the pattern completion uses
    color_trust:np.ndarray = same_color_links / np.maximum(all_links, 1)
    return color_layers[color_trust >= _PATTERN_MIN_TRUST]

def _trusted_agreement_by_shift(trusted_layers:np.ndarray, axis:int) -> np.ndarray:
    """Agreement at each shift along the axis (1 = rows, 2 = columns), up to half the grid, among pairs of trusted pixels only. Index = shift."""
    size:int = trusted_layers.shape[axis]
    trusted_mask:np.ndarray = trusted_layers.sum(axis=0) # 1 where the pixel is a trusted color (each pixel has one color)
    agreement_by_shift:np.ndarray = np.zeros(size // 2 + 1)
    for shift in range(1, size // 2 + 1):
        # Pairs where both pixels are trusted, and how many of them match
        trusted_pairs:float = (trusted_mask.take(np.arange(size - shift), axis=axis - 1)
                               * trusted_mask.take(np.arange(shift, size), axis=axis - 1)).sum()
        matching_pairs:float = (trusted_layers.take(np.arange(size - shift), axis=axis)
                                * trusted_layers.take(np.arange(shift, size), axis=axis)).sum()
        agreement_by_shift[shift] = matching_pairs / max(trusted_pairs, 1)
    return agreement_by_shift

def _pick_period(agreement_by_shift:np.ndarray, size:int, min_agreement:float) -> Optional[int]:
    """One direction's period: the smallest that lines up well enough, or a multiple of it that lines up much better (None = no repeat)."""
    # Only up to half the grid, so the tile shows up at least twice, fully
    smallest_period:Optional[int] = next((period for period in range(1, size // 2 + 1)
                                          if agreement_by_shift[period] >= min_agreement), None)

    # Period 1 = a stripe, which lines up at every period, so no multiple can do better
    if smallest_period is None or smallest_period == 1:
        return smallest_period

    # Its best-agreeing multiple (ties go to the smaller one, so a perfect tile's multiples don't win)
    multiples:range = range(2 * smallest_period, size // 2 + 1, smallest_period)
    best_multiple:Optional[int] = max(multiples, key=lambda period: agreement_by_shift[period], default=None)
    if best_multiple is not None and agreement_by_shift[best_multiple] - agreement_by_shift[smallest_period] >= _PATTERN_MIN_MULTIPLE_GAIN:
        return best_multiple
    return smallest_period

def _find_mirrors(grid:np.ndarray, color_layers:np.ndarray) -> list[tuple[float, PixelMap]]:
    """(Agreement, pixel map) for each mirror (left-right, up-down, both diagonals) and turn (half, quarter both ways), at its best-agreeing line/center."""
    height, width = grid.shape

    # Each flip, as (the flipped color layers, the map from a flipped pixel back to the grid pixel it came from,
    # and which shifts make it a true mirror: flipping twice has to land back where it started)
    flips:list[tuple[np.ndarray, PixelMap, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
        (color_layers[:, :, ::-1], lambda ys, xs: (ys, width - 1 - xs), lambda shift_ys, shift_xs: shift_ys == 0), # left-right
        (color_layers[:, ::-1, :], lambda ys, xs: (height - 1 - ys, xs), lambda shift_ys, shift_xs: shift_xs == 0), # up-down
        (color_layers[:, ::-1, ::-1], lambda ys, xs: (height - 1 - ys, width - 1 - xs), lambda shift_ys, shift_xs: np.ones(shift_ys.shape, dtype=bool)), # half-turn (any center)
        # Quarter turns: like the half-turn, any shift is a true turn (about some center), so every shift counts
        (np.rot90(color_layers, 1, axes=(1, 2)), lambda ys, xs: (xs, width - 1 - ys), lambda shift_ys, shift_xs: np.ones(shift_ys.shape, dtype=bool)), # 90 degrees counterclockwise
        (np.rot90(color_layers, 3, axes=(1, 2)), lambda ys, xs: (height - 1 - xs, ys), lambda shift_ys, shift_xs: np.ones(shift_ys.shape, dtype=bool)), # 90 degrees clockwise
        (color_layers.transpose(0, 2, 1), lambda ys, xs: (xs, ys), lambda shift_ys, shift_xs: shift_xs == -shift_ys), # main diagonal
        (color_layers[:, ::-1, ::-1].transpose(0, 2, 1), lambda ys, xs: (height - 1 - xs, width - 1 - ys),
         lambda shift_ys, shift_xs: shift_xs - shift_ys == width - height), # anti-diagonal
    ]

    mirrors:list[tuple[float, PixelMap]] = []
    for flipped_layers, unflip, is_true_mirror in flips:
        _, flipped_height, flipped_width = flipped_layers.shape
        overlap:np.ndarray = _overlap_counts(height, width, flipped_height, flipped_width)
        agreement:np.ndarray = _correlate_layers(color_layers, flipped_layers) / np.maximum(overlap, 1)

        # Only true mirrors that line up enough of the grid
        shift_ys, shift_xs = np.meshgrid(np.arange(-(flipped_height - 1), height), np.arange(-(flipped_width - 1), width), indexing='ij')
        agreement[~is_true_mirror(shift_ys, shift_xs) | (overlap < _PATTERN_MIN_MIRROR_OVERLAP * grid.size)] = 0
        best_index:tuple = np.unravel_index(int(agreement.argmax()), agreement.shape)

        # Grid pixel x lines up with flipped pixel x - shift, which came from grid pixel unflip(x - shift)
        best_shift_y:int = int(shift_ys[best_index])
        best_shift_x:int = int(shift_xs[best_index])
        mirrors.append((float(agreement[best_index]),
                        lambda ys, xs, unflip=unflip, shift_y=best_shift_y, shift_x=best_shift_x: unflip(ys - shift_y, xs - shift_x)))
    return mirrors

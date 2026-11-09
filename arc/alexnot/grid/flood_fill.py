from typing import List, Tuple, Optional
import numpy as np
import scipy.ndimage

def _build_structure(diagonal_pixels:bool, column_pixels:bool, row_pixels:bool) -> np.ndarray:
    """
    Build a 3x3 connectivity structure for scipy.ndimage.label.
    Think of it like a "which neighbors count as touching?" grid:
        [[diag, col, diag],
         [row,  ME,  row ],
         [diag, col, diag]]
    1 = connected, 0 = not connected. Center is always 1.
    """
    struct:np.ndarray = np.zeros((3,3), dtype=int)
    struct[1,1] = 1 # center always connected to itself
    if row_pixels:
        struct[1,0] = struct[1,2] = 1 # left, right
    if column_pixels:
        struct[0,1] = struct[2,1] = 1 # up, down
    if diagonal_pixels:
        struct[0,0] = struct[0,2] = struct[2,0] = struct[2,2] = 1
    return struct


def _extract_components(labels:np.ndarray, num_labels:int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Given a labeled grid (from scipy.ndimage.label), pull out (ys, xs) for each label.
    Returns them sorted in scan order (top-to-bottom, left-to-right).
    """
    components:List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, num_labels + 1):
        # np.where gives us (row_indices, col_indices) where the label matches
        ys:np.ndarray
        xs:np.ndarray
        ys, xs = np.where(labels == i)
        components.append((ys, xs))
    
    return components


# Connection radius: label the mask so pixels up to `radius` apart (Chebyshev distance, i.e. a gap of radius-1
# pixels, in any of the 8 directions) count as the same shape. scipy's label only takes a 3x3 structure, so here's the trick:
# grow every pixel into a radius x radius box, sticking out to its right & down (NOT centered), then label the boxes
# 8-connected. Two boxes touch (8-connected) only when their pixels are <= radius apart on BOTH axes, which is exactly
# "Chebyshev distance <= radius". (A centered box would connect pixels 2*half_width+1 apart, i.e. odd radii only.)
# Radius 1 = a 1x1 box = plain 8-connectivity. Then each ORIGINAL pixel keeps its box's label.
def _label_with_radius(mask:np.ndarray, radius:int) -> Tuple[np.ndarray, int]:
    """Like scipy.ndimage.label 8-connected, but pixels up to `radius` apart (Chebyshev) connect too."""
    height:int
    width:int
    height, width = mask.shape

    # Grow each pixel into its radius x radius box. (Padded right & down by radius-1, so boxes at the edge don't get clipped)
    grown_mask:np.ndarray = np.zeros((height + radius - 1, width + radius - 1), dtype=bool)
    for shift_y in range(radius):
        for shift_x in range(radius):
            grown_mask[shift_y:shift_y + height, shift_x:shift_x + width] |= mask

    # Label the boxes, 8-connected (all 8 directions)
    grown_labels:np.ndarray
    num_labels:int
    grown_labels, num_labels = scipy.ndimage.label(grown_mask, structure=np.ones((3, 3), dtype=int)) # type: ignore

    # Each original pixel takes its box's label (it's its box's top-left corner, so same spot), the gap pixels go back to 0.
    # (Every box has its own pixel in it, so every label still has at least one pixel: no empty labels)
    labels:np.ndarray = grown_labels[:height, :width] * mask
    return labels, num_labels


def _flood_fill_select_all_shapes(grid_arr:np.ndarray,
                         bg_color:int=-1,
                         multicolor_shapes:bool=False,
                         only_select_this_one_color:Optional[int]=None,
                         diagonal_pixels:bool=True,
                         column_pixels:bool=True,
                         row_pixels:bool=True,
                         radius:int=1) -> List[Tuple[np.ndarray, np.ndarray]]: # Connection radius (only meant for all 8 directions)
    """
    Find all connected components ("shapes") in a grid, using scipy's C-optimized labeler.
    Returns a list of (ys, xs) numpy arrays — one pair per shape found.
    """

    # No grid? WTF, no shapes, doofus.
    if grid_arr.size == 0:
        return []
    
    # Dimensions
    h:int
    w:int
    h, w = grid_arr.shape

    # Build the connectivity structure: which neighbors count as "same shape"?
    struct:np.ndarray = _build_structure(diagonal_pixels, column_pixels, row_pixels)

    # Results: List of Shape Coords: (Y's as vector, X's as vector).
    # Yes [y,x] is convention coz row,col order.
    results:List[Tuple[np.ndarray, np.ndarray]] = []

    if multicolor_shapes and only_select_this_one_color is None:
        # Multi-color mode: all non-bg pixels, connected regardless of color
        fg_mask:np.ndarray = grid_arr != bg_color # numpy is fancy with the == overloading
        # scipy.ndimage.label: labels each connected region with a unique int (1, 2, 3...), bg stays 0
        labels:np.ndarray
        n:int
        # Radius > 1? Then pixels up to `radius` apart connect, whatever their colors (radius 1 = the usual label)
        if radius > 1:
            labels, n = _label_with_radius(fg_mask, radius)
        else:
            labels, n = scipy.ndimage.label(fg_mask, structure=struct) # type: ignore
        results = _extract_components(labels, n)
    else:
        # Single-color mode: find connected components separately for each color
        colors:List[int]
        if only_select_this_one_color is not None:
            colors = [only_select_this_one_color] # just this color
        else:
            colors = np.unique(grid_arr).tolist() # all available colors
        for color in colors:
            if color == bg_color:
                continue # skip background
            # Mask: True where this color lives, False everywhere else
            mask:np.ndarray = grid_arr == color # numpy is fancy with the == overloading
            # Radius > 1? Per color: two same-color pixels connect across a gap even if OTHER colors sit in that gap
            # (only this color's mask gets grown, so other colors count as "not this color", just like bg)
            if radius > 1:
                labels, n = _label_with_radius(mask, radius)
            else:
                labels, n = scipy.ndimage.label(mask, structure=struct) # type: ignore
            results.extend(_extract_components(labels, n))

    return results
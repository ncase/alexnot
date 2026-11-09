import numpy as np
from ...grid import _flood_fill_select_all_shapes

###################################################
### RAW PIXEL CHANGES:
### Find-and-replace patterns of pixels, bypassing the shape parse
###################################################

# Minimum number of identical additions (counted across ALL demo pairs together, not per pair)
# before we bother proposing a replacement: one addition alone could be anything.
_MIN_IDENTICAL_ADDITIONS:int = 2

def replace_pixels(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    """
    Notice when 2+ identical patches of pixels got added (predicted vs output), find what those spots
    had in common beforehand, propose replace_pixels(what_they_had_in_common, the_added_patch)
    """

    # "Shape" here means a connected patch of CHANGED PIXELS (predicted grid vs output grid), NOT a
    # parsed shape. The parse can chop one added thing into several shapes (e.g. with a rows-only multicolor
    # parse, a "+" is three shapes: a 3-wide row plus two lone pixels), and a cross attached
    # to other same-colored pixels shows up as a changed shape plus an added one. Diffing raw pixels sidesteps
    # all of that, which is the whole point of a pixel-level find-and-replace.

    candidates:set = set()
    for program_result in parent_program_results:

        # Identical added patches, grouped: patch key -> list of (predicted grid padded with -1, top, left)
        groups:dict[tuple, list[tuple[np.ndarray, int, int]]] = {}
        patches:dict[tuple, np.ndarray] = {} # patch key -> the patch itself (new colors, -1 where unchanged)

        for pair_index, pair in enumerate(train_pairs):

            # The grid we'd be editing (execution-guided: the PARENT program's output, which for the root
            # program is just the input grid) vs the grid we want
            predicted_pixels:np.ndarray = program_result['predicted_scenes'][pair_index]['_raw']
            actual_pixels:np.ndarray = pair['output']['_raw']

            # Pixel edits can't resize the grid, so skip pairs where the sizes differ
            if predicted_pixels.shape != actual_pixels.shape:
                continue

            # Bail early when nothing changed in this pair (the common case for most parents)
            changed_mask:np.ndarray = predicted_pixels != actual_pixels
            if not changed_mask.any():
                continue

            # The changed pixels, with their NEW colors (-1 everywhere else), split into connected patches.
            # (8-connected, any colors: the loosest notion of "one thing got added")
            diff_grid:np.ndarray = np.where(changed_mask, actual_pixels, -1).astype(np.int8)
            padded_predicted:np.ndarray = np.pad(predicted_pixels, 1, constant_values=-1) # off-grid = -1 = don't care, see below
            for patch_ys, patch_xs in _flood_fill_select_all_shapes(diff_grid, bg_color=-1, multicolor_shapes=True, diagonal_pixels=True):

                # Cut the patch out at its bounding box: ONLY this patch's own pixels, -1 elsewhere
                # (a neighboring patch could poke into the box otherwise)
                top:int = int(patch_ys.min())
                left:int = int(patch_xs.min())
                patch:np.ndarray = np.full((int(patch_ys.max()) - top + 1, int(patch_xs.max()) - left + 1), -1, dtype=np.int8)
                patch[patch_ys - top, patch_xs - left] = diff_grid[patch_ys, patch_xs]

                # Identical patches go in the same group
                patch_key:tuple = (patch.shape, patch.tobytes())
                patches[patch_key] = patch
                groups.setdefault(patch_key, []).append((padded_predicted, top, left))

        # For each patch that got added 2+ times: what did those spots look like BEFORE the addition?
        for patch_key, members in groups.items():
            if len(members) < _MIN_IDENTICAL_ADDITIONS:
                continue
            patch:np.ndarray = patches[patch_key]
            patch_height, patch_width = patch.shape

            # Cell-by-cell "what they all had in common" over the bounding box + 1 padding, read from the
            # PREDICTED grid: same color everywhere -> keep it, any disagreement -> -1 (don't care).
            # Off-grid padding cells are -1, so a patch at the grid's edge makes that context don't-care too.
            common_context:np.ndarray | None = None
            for padded_predicted, top, left in members:
                window:np.ndarray = padded_predicted[top:top+patch_height+2, left:left+patch_width+2] # +1 padding each side (indices already shifted by the pad)
                common_context = window.copy() if common_context is None else np.where(common_context == window, common_context, -1)
            assert common_context is not None # (members is non-empty)

            # Two proposals, and let the scorer pick: WITH the 1-pixel context ring (for rules that depend
            # on the surroundings), and WITHOUT it, just the bounding box (cheaper, so it wins when the context
            # doesn't matter; e.g. a test input can have crosses touching other same-colored pixels that no demo cross did).
            new_with_ring:np.ndarray = np.pad(patch, 1, constant_values=-1)
            for old_grid, new_grid in ((common_context, new_with_ring), (common_context[1:-1, 1:-1], patch)):

                # Trim rows/columns that are -1 in BOTH grids: they carry no information, just token cost
                informative:np.ndarray = (old_grid >= 0) | (new_grid >= 0)
                keep_rows:np.ndarray = informative.any(axis=1)
                keep_cols:np.ndarray = informative.any(axis=0)
                old_grid = old_grid[keep_rows][:, keep_cols]
                new_grid = new_grid[keep_rows][:, keep_cols]

                # Nothing in common at all? Then the pattern would match everywhere: skip
                if not (old_grid >= 0).any():
                    continue

                # Propose! (Positional args, and no spaces, to keep the token cost down)
                old_text:str = str(old_grid.tolist()).replace(' ', '')
                new_text:str = str(new_grid.tolist()).replace(' ', '')
                code:str = f'replace_pixels({old_text}, {new_text})'
                candidates.add( (program_result['program'] + '\n' + code).strip() )

    # All candidates!
    return candidates

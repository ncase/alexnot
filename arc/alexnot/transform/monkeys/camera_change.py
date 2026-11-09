from typing import Dict, List, Optional
import numpy as np
from ...utils import _propose_selectors, _dict_to_argstring

def camera_resize(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates = set()
    for program_result in parent_program_results:
        for pair_index, pair in enumerate(train_pairs):
            # Before
            predicted_scene = program_result['predicted_scenes'][pair_index]
            # After
            output_scene = pair['output']
            # Hey, if size changed, suggest resize camera!
            before_w, after_w, before_h, after_h = predicted_scene['width'], output_scene['width'], predicted_scene['height'], output_scene['height']
            # All possible combos:
            resize_args:List[str] = []
            if before_w!=after_w:
                resize_args.append(f'width={after_w}')
            if before_h!=after_h:
                resize_args.append(f'height={after_h}')
            if before_w!=after_w and before_h!=after_h:
                resize_args.append(f'width={after_w},height={after_h}')
            # Propose all combos after current code:
            for a in resize_args:
                candidates.add(f"{program_result['program']}\ncamera_resize({a})".strip())

    return candidates

# Shared by both focus monkeys: the min & max padding for one output scene
def _focus_paddings(after_scene:Dict) -> set:
    """Min & max gap between the output scene's edge and its shapes' bounding box."""
    after_shapes:List[Dict] = after_scene['shapes']
    # Calculate padding: output scene dimensions vs the total bounding box of output shapes
    bbox_left:int   = min(s['left']   for s in after_shapes)
    bbox_top:int    = min(s['top']    for s in after_shapes)
    bbox_right:int  = max(s['right']  for s in after_shapes)
    bbox_bottom:int = max(s['bottom'] for s in after_shapes)
    # padding is the gap between the output scene edge and the shapes' bounding box
    pad_left:int   = bbox_left
    pad_top:int    = bbox_top
    pad_right:int  = after_scene['width'] - bbox_right
    pad_bottom:int = after_scene['height'] - bbox_bottom
    return { min(pad_left, pad_top, pad_right, pad_bottom), # try min & max
             max(pad_left, pad_top, pad_right, pad_bottom) }

# Crop to the inside of a frame: which input shapes is the output exactly the inside of?
def _frames_around_output(before_scene:Dict, after_scene:Dict) -> List[tuple]:
    """(shape, thickness) for every visible shape whose box, shrunk by the same thickness on all 4 sides, is pixel-for-pixel the output."""
    before_pixels:np.ndarray = before_scene['_raw']
    after_pixels:np.ndarray = after_scene['_raw']
    frames:List[tuple] = []
    for shape in before_scene['shapes']:
        if shape.get('hidden'):
            continue
        # The box has to stick out past the output by the same amount on every side (cheap size check first)
        thickness_x, leftover_x = divmod(shape['width'] - after_scene['width'], 2)
        thickness_y, leftover_y = divmod(shape['height'] - after_scene['height'], 2)
        if thickness_x < 1 or thickness_x != thickness_y or leftover_x or leftover_y:
            continue
        # Then the inside of the box has to match the output exactly
        thickness:int = thickness_x
        inside:np.ndarray = before_pixels[shape['top'] + thickness : shape['bottom'] - thickness,
                                          shape['left'] + thickness : shape['right'] - thickness]
        if np.array_equal(inside, after_pixels):
            frames.append((shape, thickness))
    return frames

# Shared by both focus monkeys: the hug="2d" margin one output scene suggests
def _hug_2d_margin(before_scene:Dict, after_scene:Dict) -> Optional[int]:
    """The smallest gap between the output's shapes (on either axis), or None if the output doesn't fit in the input."""
    # A 2D hug only ever packs shapes tighter, so the output has to fit inside the input
    if after_scene['width'] > before_scene['width'] or after_scene['height'] > before_scene['height']:
        return None

    # Per axis, merge the overlapping (start, end) spans of the output's shapes: every break between runs is a gap
    gaps:List[int] = []
    for start_key, end_key in (('left', 'right'), ('top', 'bottom')):
        sorted_spans:list = sorted((shape[start_key], shape[end_key]) for shape in after_scene['shapes'])
        run_end:int = sorted_spans[0][1]
        for start, end in sorted_spans[1:]:
            if start >= run_end:
                gaps.append(start - run_end)
            run_end = max(run_end, end)

    # The tightest gap is the margin (bigger gaps can come from small shapes centered in big cells).
    # No gaps at all = the cells touch, margin 0
    return min(gaps) if gaps else 0

# Shared by both focus monkeys: selector x padding x spotlight-args, one focus(...) program each
def _focus_programs(parent_program:str, proposed_selectors:List[Dict], paddings:set, hug_2d_margins:set) -> set:
    """All focus(...) programs for these selectors & paddings, appended to the parent program."""
    candidates:set = set()

    # Build the list of spotlight arg combos to try
    # (don't list the defaults: one_match=False, hide_non_matches=False)
    spotlight_arg_combos:List[str] = [
        "",
        ", one_match=True",
        ", hide_non_matches=True",
        ", hide_non_matches=True, one_match=True",
        # Squish the matches side by side and crop to them (hug hides the non-matches by itself)
        ", hug=True",
    ]

    # Lay the matches out in a grid of cells instead, but only if every demo agrees on one margin
    # (None = some demo's output didn't fit in its input). margin=0 is the default, so it's left out
    if len(hug_2d_margins) == 1 and None not in hug_2d_margins:
        hug_2d_margin:int = next(iter(hug_2d_margins))
        spotlight_arg_combos.append(', hug="2d"' + (f", margin={hug_2d_margin}" if hug_2d_margin else ""))

    # For each valid selector x padding x spotlight-args combo, propose a program
    padding_values:List[int] = sorted(paddings) if paddings else [0]
    for selector in proposed_selectors:
        selector_string:str = _dict_to_argstring(selector)
        for padding in padding_values:
            for spotlight_args in spotlight_arg_combos:
                # focus(selector, padding, (extra spotlight args)
                program = (
                    f"{parent_program}\n"
                    f"focus({selector_string}, padding={padding}{spotlight_args})"
                ).strip()
                candidates.add(program)

    return candidates

# Proposes one-liner focus(<selector>, ...) programs.
def focus(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates:set = set()

    for program_result in parent_program_results:

        # Collect selectors and paddings across all pairs
        proposed_selectors:List[Dict] = []
        paddings:set = set()
        hug_2d_margins:set = set()
        ingroup_per_pair:Dict[int, List[Dict]] = {}  # pair_index -> ingroup shapes

        for pair_index, pair in enumerate(train_pairs):

            # Exec-guided: Before is Predicted
            before_scene:Dict = program_result['predicted_scenes'][pair_index]
            after_scene:Dict = pair['output']
            before_shapes:List[Dict] = before_scene['shapes']
            after_shapes:List[Dict] = after_scene['shapes']

            # Same dimensions? No camera zoom happened, skip this pair
            if before_scene['width'] == after_scene['width'] and before_scene['height'] == after_scene['height']:
                continue

            # Is the output the inside of a frame? Then focus on that frame, with a negative padding to shave it off.
            # (Checked before the empty-output skip: the inside of a frame can be all background)
            for frame_shape, thickness in _frames_around_output(before_scene, after_scene):
                proposed_selectors.extend(_propose_selectors([frame_shape], before_shapes, N=100))
                paddings.add(-thickness)

            # No shapes in after scene? Kinda impossible, but skip if so
            if not after_shapes:
                continue

            # For each shape in After, find the first shape in Before with same shadow.
            # This will give us the Ingroup (camera's tracking these) vs Outgroup (not)
            # TODO: Non-max suppression of After shapes, or shapes only if within camera.
            # Match each output shape to the closest input shape by pixel content (shadow),
            ingroup:List[Dict] = []
            for after_shape in after_shapes:
                for before_shape in before_shapes:
                    if np.array_equal(before_shape['_shadow'], after_shape['_shadow']):
                        ingroup.append(before_shape)
                        break # next after-shape to test, please

            # No matches in the Before scene? Bye
            if not ingroup:
                continue

            # Stash ingroup for reuse in the second loop (hide_others / show_all checks)
            ingroup_per_pair[pair_index] = ingroup

            # Propose selectors that distinguish spotlighted shapes from the rest
            selectors:List[Dict] = _propose_selectors(ingroup, before_shapes, N=100)
            proposed_selectors.extend(selectors)

            # Padding: gap between the output's edge & its shapes' bounding box (try min & max)
            paddings |= _focus_paddings(after_scene)

            # hug="2d" margin: the tightest gap between the output's shapes
            hug_2d_margins.add(_hug_2d_margin(before_scene, after_scene))

        if not proposed_selectors:
            continue

        # TODO: Remove selectors that don't work in ALL pairs ??
        # Is this a way to cut down on excessive suggestions in other monkeys, too?

        # Build programs: selector x padding x spotlight-args
        candidates |= _focus_programs(program_result['program'], proposed_selectors, paddings, hug_2d_margins)

    return candidates

# How far a size-match focus may pad (or, negative, shave) on each side
SIZE_MATCH_MAX_PADDING:int = 3

# A fourth focus proposer, for "crop to the region under a shape" levels, where the output's pixels match nothing in the input.
# It never looks at the output's shapes or pixels, only its SIZE: the ingroup is the shapes whose box, padded by the same
# amount on every side, is exactly the output grid.
def focus_by_output_size(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates:set = set()

    for program_result in parent_program_results:
        predicted_scenes:List[Dict] = program_result['predicted_scenes']

        # Nothing to crop if no demo's size changes yet
        if all(scene['width'] == pair['output']['width'] and scene['height'] == pair['output']['height']
               for scene, pair in zip(predicted_scenes, train_pairs)):
            continue

        # focus(padding=p) grows the box by p on all 4 sides, so the shape's box is the output minus 2p on each axis
        for padding in range(-SIZE_MATCH_MAX_PADDING, SIZE_MATCH_MAX_PADDING + 1):
            ingroup:List[Dict] = []
            all_shapes:List[Dict] = []
            for scene, pair in zip(predicted_scenes, train_pairs):
                wanted_width:int = pair['output']['width'] - 2 * padding
                wanted_height:int = pair['output']['height'] - 2 * padding
                matches:List[Dict] = [shape for shape in scene['shapes'] if not shape.get('hidden')
                                      and shape['width'] == wanted_width and shape['height'] == wanted_height]
                # EVERY demo needs a match (even ones a parent already resized), so no demo gets skipped
                # and the selectors can't overfit to just the rest
                if not matches:
                    break
                ingroup.extend(matches)
                all_shapes.extend(scene['shapes'])
            else:
                # One selector call across all demos' shapes, so it has to pick the matches in every demo at once
                proposed_selectors:List[Dict] = _propose_selectors(ingroup, all_shapes, N=100)
                candidates |= _focus_programs(program_result['program'], proposed_selectors, {padding}, set())

    return candidates

# Shared by the sliding-window focus monkeys: where in the input could the output have been cropped from?
def _best_crop_windows(before_scene:Dict, after_scene:Dict) -> List[tuple]:
    """Every (top, left, bottom, right) crop window tied for the most pixels matching the output."""
    # Slide the output's raw pixels over the predicted raw pixels, at every offset where it fits.
    # windows[y, x] is the before-patch whose top-left corner is at (x, y)
    before_pixels:np.ndarray = before_scene['_raw']
    after_pixels:np.ndarray = after_scene['_raw']
    window_height, window_width = after_pixels.shape
    windows:np.ndarray = np.lib.stride_tricks.sliding_window_view(before_pixels, (window_height, window_width))

    # How many pixels are identical at each offset? (All pixels, background included)
    matching_pixel_counts:np.ndarray = (windows == after_pixels).sum(axis=(2, 3))

    # Keep EVERY offset tied for the best count: each one's a candidate crop window
    best_offsets:np.ndarray = np.argwhere(matching_pixel_counts == matching_pixel_counts.max())
    return [(int(top), int(left), int(top) + window_height, int(left) + window_width) for top, left in best_offsets]

# Shared by the sliding-window focus monkeys: the visible shapes lying FULLY inside a window
def _shapes_inside_window(shapes:List[Dict], window_top:int, window_left:int, window_bottom:int, window_right:int) -> List[Dict]:
    """The visible shapes whose bounding box fits inside the window."""
    return [
        shape for shape in shapes
        if not shape.get('hidden')
        and shape['left'] >= window_left and shape['right'] <= window_right
        and shape['top'] >= window_top and shape['bottom'] <= window_bottom
    ]

# A second focus proposer: find the crop window by sliding the output's pixels over the input's,
# then the ingroup is whatever shapes sit fully inside the best window(s). The first focus monkey's ingroup
# ("every shape whose shadow shows up in the output") is everything in crop-then-rearrange levels.
def focus_by_sliding_window(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates:set = set()

    for program_result in parent_program_results:

        # Collect selectors and paddings across all pairs (same as the first focus monkey)
        proposed_selectors:List[Dict] = []
        paddings:set = set()
        hug_2d_margins:set = set()

        for pair_index, pair in enumerate(train_pairs):

            # Exec-guided: Before is Predicted
            before_scene:Dict = program_result['predicted_scenes'][pair_index]
            after_scene:Dict = pair['output']
            before_shapes:List[Dict] = before_scene['shapes']
            after_shapes:List[Dict] = after_scene['shapes']

            # Only if the camera changed size, AND the output fits fully inside the before scene
            if before_scene['width'] == after_scene['width'] and before_scene['height'] == after_scene['height']:
                continue
            if after_scene['width'] > before_scene['width'] or after_scene['height'] > before_scene['height']:
                continue

            # No shapes in after scene? Kinda impossible, but skip if so (we need them for the padding below)
            if not after_shapes:
                continue

            for window_top, window_left, window_bottom, window_right in _best_crop_windows(before_scene, after_scene):

                # Ingroup = the visible before-shapes lying FULLY inside this window
                # TODO: later we may also want to try shapes that are only *partly* inside the window
                ingroup:List[Dict] = _shapes_inside_window(before_shapes, window_top, window_left, window_bottom, window_right)

                # Nobody inside this window? Next window
                if not ingroup:
                    continue

                # Propose selectors that pick out the in-window shapes from the rest (one call per window)
                selectors:List[Dict] = _propose_selectors(ingroup, before_shapes, N=100)
                proposed_selectors.extend(selectors)

            # Padding: same as the first focus monkey (via the shared helper)
            paddings |= _focus_paddings(after_scene)

            # hug="2d" margin: the tightest gap between the output's shapes
            hug_2d_margins.add(_hug_2d_margin(before_scene, after_scene))

        if not proposed_selectors:
            continue

        # Build programs exactly like the first focus monkey does (via the shared helper)
        candidates |= _focus_programs(program_result['program'], proposed_selectors, paddings, hug_2d_margins)

    return candidates

# A third focus proposer: same crop windows as focus_by_sliding_window, but the ingroup is ONLY the inside shapes
# touching the window's edge. For "markers show where to crop" levels, where other shapes sit inside the window too
# (and make "everything inside" impossible to select), but the markers alone still span the whole window.
def focus_on_window_edge(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates:set = set()

    for program_result in parent_program_results:

        # Collect selectors and paddings across all pairs (same as the other focus monkeys)
        proposed_selectors:List[Dict] = []
        paddings:set = set()
        hug_2d_margins:set = set()

        for pair_index, pair in enumerate(train_pairs):

            # Exec-guided: Before is Predicted
            before_scene:Dict = program_result['predicted_scenes'][pair_index]
            after_scene:Dict = pair['output']
            before_shapes:List[Dict] = before_scene['shapes']

            # Only if the camera changed size, AND the output fits fully inside the before scene
            if before_scene['width'] == after_scene['width'] and before_scene['height'] == after_scene['height']:
                continue
            if after_scene['width'] > before_scene['width'] or after_scene['height'] > before_scene['height']:
                continue

            # No shapes in after scene? Kinda impossible, but skip if so (we need them for the padding below)
            if not after_scene['shapes']:
                continue

            for window_top, window_left, window_bottom, window_right in _best_crop_windows(before_scene, after_scene):

                # The shapes fully inside the window, then just the ones touching its edge
                # (shapes poking out past the edge don't count: they're not fully inside)
                inside_shapes:List[Dict] = _shapes_inside_window(before_shapes, window_top, window_left, window_bottom, window_right)
                edge_shapes:List[Dict] = [
                    shape for shape in inside_shapes
                    if shape['left'] == window_left or shape['right'] == window_right
                    or shape['top'] == window_top or shape['bottom'] == window_bottom
                ]

                # Nobody on the edge, or EVERYONE inside is on the edge? Then focus_by_sliding_window
                # already proposes this ingroup (or there's none), nothing new here
                if not edge_shapes or len(edge_shapes) == len(inside_shapes):
                    continue

                # Propose selectors that pick out the edge shapes from the rest (one call per window)
                proposed_selectors.extend(_propose_selectors(edge_shapes, before_shapes, N=100))

            # Padding & hug="2d" margin: same as the other focus monkeys (via the shared helpers)
            paddings |= _focus_paddings(after_scene)
            hug_2d_margins.add(_hug_2d_margin(before_scene, after_scene))

        if not proposed_selectors:
            continue

        # Build programs exactly like the other focus monkeys do (via the shared helper)
        candidates |= _focus_programs(program_result['program'], proposed_selectors, paddings, hug_2d_margins)

    return candidates

from typing import Dict, List
import numpy as np

def _shapes_hitbox_test(shape:Dict, all_shapes:List[Dict]) -> List[Dict]:
    # For now, simple whole hitbox, not pixel-perfect accuracy
    overlapped_shapes:List[Dict] = []
    for other_shape in all_shapes:
        # You're me? Forget about it!
        if other_shape is shape:
            continue
        # Our bounds overlap? Add to results!
        if shape['right']>other_shape['left'] and shape['left']<other_shape['right'] and shape['bottom']>other_shape['top'] and shape['top']<other_shape['bottom']:
            overlapped_shapes.append(other_shape)
    return overlapped_shapes

def _shapes_hit_test(shape:Dict, all_shapes:List[Dict]) -> List[Dict]:
    # EXACT pixel-perfect hit test (numpy: extract overlap slices, check with np.any)
    overlapped_shapes = _shapes_hitbox_test(shape,all_shapes) # first, filter only shapes whose bounds overlap
    hit_shapes = []
    for other_shape in overlapped_shapes:

        # Exact coords of OVERLAP rectangle (in global coords):
        my_l, my_r, my_t, my_b = shape['left'], shape['right'], shape['top'], shape['bottom']
        other_l, other_r, other_t, other_b = other_shape['left'], other_shape['right'], other_shape['top'], other_shape['bottom']
        overlap_left   = max(my_l, other_l)
        overlap_right  = min(my_r, other_r)
        overlap_top    = max(my_t, other_t)
        overlap_bottom = min(my_b, other_b)

        # Check SHADOW grid for both shapes: extract the overlap region from each, check if
        # they're BOTH filled (==1) at any point. Numpy slicing + np.any replaces the Python loop.
        my_shadow = shape['_shadow']
        other_shadow = other_shape['_shadow']
        my_overlap = my_shadow[overlap_top - my_t:overlap_bottom - my_t,
                               overlap_left - my_l:overlap_right - my_l]
        other_overlap = other_shadow[overlap_top - other_t:overlap_bottom - other_t,
                                     overlap_left - other_l:overlap_right - other_l]
        found_collision = np.any((my_overlap == 1) & (other_overlap == 1))

        # Did we ACTUALLY collide? Add to the list!
        if found_collision:
            hit_shapes.append(other_shape)

    return hit_shapes

def _shapes_in_line_of_sight(scene:Dict, shape:Dict, dx:int, dy:int) -> List[Dict]:
    """Find all scene shapes in a shape's line of sight, by sweeping its shadow in direction (dx,dy)."""

    # Clamp direction to unit steps
    dx = max(-1, min(int(dx), 1))
    dy = max(-1, min(int(dy), 1))
    if dx == 0 and dy == 0:
        return []

    sw:int = scene['width']
    sh:int = scene['height']
    shadow:np.ndarray = shape['_shadow']
    stamp_h:int = shadow.shape[0]
    stamp_w:int = shadow.shape[1]

    # Compute the bounding box of the entire sweep path, so we can skip shapes
    # outside it without building the full sweep mask. The sweep starts at the
    # shape's current position and extends in direction (dx, dy) to the scene edge.
    sweep_left:int = shape['left']
    sweep_top:int = shape['top']
    sweep_right:int = shape['left'] + stamp_w
    sweep_bottom:int = shape['top'] + stamp_h
    if dx > 0:
        sweep_right = sw
    elif dx < 0:
        sweep_left = 0
    if dy > 0:
        sweep_bottom = sh
    elif dy < 0:
        sweep_top = 0

    # Pre-filter: which shapes even overlap the sweep bounding box?
    # If none do, skip the expensive sweep mask entirely.
    candidate_shapes:List[Dict] = []
    for other in scene['shapes']:
        if other is shape:
            continue
        # Bounds overlap with sweep bounding box?
        if other['right'] > sweep_left and other['left'] < sweep_right and other['bottom'] > sweep_top and other['top'] < sweep_bottom:
            candidate_shapes.append(other)
    if not candidate_shapes:
        return []

    # Build a scene-sized sweep mask: stamp the shape's shadow, then keep moving & stamping
    # until it slides completely off the scene edge.
    sweep:np.ndarray = np.zeros((sh, sw), dtype=np.int8)
    stamp_left:int = shape['left']
    stamp_top:int = shape['top']
    while stamp_left < sw and stamp_top < sh and stamp_left + stamp_w > 0 and stamp_top + stamp_h > 0:
        # Clip the stamp to scene bounds
        src_y0:int = max(0, -stamp_top)
        src_x0:int = max(0, -stamp_left)
        src_y1:int = min(stamp_h, sh - stamp_top)
        src_x1:int = min(stamp_w, sw - stamp_left)
        dst_y0:int = stamp_top + src_y0
        dst_x0:int = stamp_left + src_x0
        # OR the shadow into the sweep mask (numpy broadcast: 1 stays 1)
        sweep[dst_y0:stamp_top + src_y1, dst_x0:stamp_left + src_x1] |= shadow[src_y0:src_y1, src_x0:src_x1]
        # Slide the stamp
        stamp_left += dx
        stamp_top += dy

    # Pixel-test only the pre-filtered candidate shapes against the sweep mask
    results:List[Dict] = []
    for other in candidate_shapes:

        # Get the *onscreen* bounding box of the other shape
        other_left:int = other['left']
        other_top:int = other['top']
        other_right:int = other['right']
        other_bottom:int = other['bottom']
        onscreen_top:int = max(0, other_top) # Clipped to the scene's edges
        onscreen_left:int = max(0, other_left)
        onscreen_bottom:int = min(sh, other_bottom)
        onscreen_right:int = min(sw, other_right)

        # Fixes previous bug: Shapes can sit partly or fully OFF the scene (e.g. camera_spotlight crops
        # scene smaller). Then the on-scene overlap is empty (right <= left or bottom <= top), and slicing with
        # those would make numpy wrap negative indices around, giving mismatched slices & a crash.
        # The sweep mask only covers the scene anyway, so an off-scene shape can't be hit: skip it.
        if onscreen_bottom <= onscreen_top or onscreen_right <= onscreen_left:
            continue

        # Extract the overlap region from both grids and check for any shared pixels
        other_shadow:np.ndarray = other['_shadow']
        sweep_slice:np.ndarray = sweep[onscreen_top:onscreen_bottom, onscreen_left:onscreen_right] # sweep, sliced at their overlap
        other_slice:np.ndarray = other_shadow[onscreen_top - other_top:onscreen_bottom - other_top,
                                              onscreen_left - other_left:onscreen_right - other_left] # other, sliced at overlap
        if np.any((sweep_slice == 1) & (other_slice == 1)): # any(): stop at first hit!
            results.append(other)

    # Gimme
    return results
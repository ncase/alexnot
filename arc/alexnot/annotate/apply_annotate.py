"""
Annotate with extra useful information about this Scene or its Shapes:

Touching:
- [x] shape.touching_border : bool    
- [x] shape.touching_count : int                            # distinct shapes touching me, any side
- [x] shape.touching_(left/right/top/bottom)_count : int    # distinct shapes touching that side
- [x] shape.touching_(left/right/top/bottom)_color : str    # most common color on that side, by area ("c-1" if none)
- [ ] shape.touching_(color)_count : int
- [x] shape.touching_color_most : str 
- [ ] shape.touching_(color)_least : str
- [ ] shape.touching_(chunk)_count : int
- [ ] shape.touching_(chunk)_most : str
- [ ] shape.touching_(chunk)_least : str

Closest:
- [ ] shape.closest_color
- [ ] shape.closest_chunk

Section:
- [ ] shape.section
- [ ] shape.section_reversed
- [ ] shape.section_multicolor
- [ ] shape.section_multicolor_reversed : int

Ranks by Category:
- [x] shape.rank_(prop) : int                   # e.g. (color_twins), (chunk_twins), area, height, width, left, right, top, bottom
- [x] shape.rank_(prop)_reversed : int
- [x] shape.rank_(x|y)(_reversed) : int          # position rank by the shape's CENTER (0 = left-most / top-most)
- [x] shape.rank_(x|y)(_reversed)_mod(2|3|4) : int  # every 2nd/3rd/4th one, e.g. alternating rows
- [x] shape.rank_(shape|pixel)_color_commonness(_reversed) : int  # how common my color is, by # shapes (or # pixels) of that color
- [ ] shape.(color)_rank_(prop)(?_reversed) : int # categories

Twins:
# The shorter (cheaper) name is the simpler idea: plain = EXACT copies, "transformed_" = copies up to
# rotation/flip (D4). (Later, scale could go in transformed_ too.)
- [x] shape.shadow_twins : int                  # exact copies, no transforms
- [x] shape.pixels_twins : int
- [x] shape.transformed_shadow_twins : int      # up to D4 (rotations/flips)
- [x] shape.transformed_pixels_twins : int
- [ ] shape.(prop)_twins : int                  # e.g. color, section, area, width, height ("chunk" already included with shadow/pixels)

Dimensions:
- [x] shape.dim_min : int
- [x] shape.dim_max : int
- [x] shape.wider : int                        # 1 if width > height, -1 if width < height, 0 if square

Halves:
- [x] shape.start_vhalf : int                  # -1 if my whole bounding box is in the top half, 1 if bottom, 0 if it straddles
- [x] shape.start_hhalf : int                  # same, -1 = left half, 1 = right half

Colors: (for multicolor shapes)
- [ ] has_color : bool
- [ ] color % float??? for multicolor shapes

Inside/Outside:
- [ ] shape.inside_color
- [x] shape.within_color : str                  # color of the (innermost) shape whose hole I'm fully inside ("c-1" if none)
- [x] shape.contains_count : int                # how many shapes sit fully in my holes
- [x] shape.contains_pixel_count : int          # monochrome: # pixels of those shapes. Multicolor: # of my pixels not my majority color

Overlaps:
- [ ] shape.overlap_(shape/color/chunk_count/most/least)
- [x] shape.over_color : str                    # color of the overlapping shape just below me in z ("c-1" if none)
- [x] shape.under_color : str                   # color of the overlapping shape just above me in z ("c-1" if none)

Symmetry:
- [x] shape.symm_h : bool
- [x] shape.symm_v : bool
- [x] shape.symm_d1 : bool                      # like %
- [x] shape.symm_d2 : bool                      # like Q

===

Scene: Mostly rankings
- [ ] scene.sections_count : int
- [x] scene.(color)_count : int
- [x] scene.count_color_pixels_(color) : int     # pixels of that color in the shapes
- [x] scene.shape_count : int                   # shapes in the scene (bg isn't a shape)
- [x] scene.color_count : int                   # distinct shape colors in the scene
- [x] scene.most_common_nonbg_color(_2nd/_3rd) : str  # colors in the shapes, by pixel count (bg not counted, ties: random)
- [x] scene.least_common_nonbg_color(_2nd/_3rd) : str # same, least first. (Only as many ranks as colors: 1 color = most = least)
- [ ] scene.(chunk)_count : int
- [ ] scene.chunk_count_most : str
- [ ] scene.chunk_count_least : str

===

More things to annotate:
- color_twins & 0 for unique
- FLOATS like color density, ratios
- How to handle _relationships?_ Simplest: (within/closest_to/overlap/right_of)_(color/chunk)
	- subshape in... // green for biggest, then the subshapes are red... HARD ARC-1 EVAL!
	- "could fit snugly in gap of" ?... subshadow in...
- Modulo
- Rank left right top down too! (DONE)
- Convexity/Gaps
- annotate dividers

===

# TODO: Shape In Line of Sight of?
"""

import random # to break ties randomly
from typing import List, Dict
import numpy as np
from scipy.ndimage import binary_dilation
from ..grid import _flip_grid, _rotate_grid

def apply_annotate(pairs:List[Dict])->List[Dict]:
    for pair in pairs:
        pair['input'] = apply_annotate_one_scene(pair['input'])
        if pair.get('output') is not None: # THIS IS OPTIONAL, REMEMBER # TODO: when testing for real, OMIT/DELETE ANSWER IN TEST.
            pair['output'] = apply_annotate_one_scene(pair['output'])
    return pairs

def apply_annotate_one_scene(scene:Dict)->Dict:

    #######################
    # SHAPE PROPERTIES...
    #######################

    # (Sections are off for now: they were implemented wrong, so shapes get no 'section' prop)

    # All twins
    annotate_twins_grid(scene, 'shadow')
    annotate_twins_grid(scene, 'pixels')
    annotate_twins_prop(scene)

    # All ranks for numerical properties
    # TODO: we also do 'color_twins' and 'chunk_twins' to get most/least common colors/chunks
    for prop_name in ("area","width","height","left","right","top","bottom"):
        annotate_ranks(scene, prop_name)

    # Position ranks by the shape's CENTER: rank_x = 0 for the left-most, rank_y = 0 for the top-most
    # (same tie rule: shapes in the same row/column share a rank). Plus their mod 2/3/4, for "every other row" rules
    annotate_ranks(scene, 'center_x', rank_name='rank_x')
    annotate_ranks(scene, 'center_y', rank_name='rank_y')
    annotate_rank_mods(scene)

    # Stacking order: only autocompleted parses stack shapes, so everyone else is 0 (nothing on top of anything).
    # rank_z = 0 for the frontmost (highest z): biggest first, like rank_area, so it's the opposite way round from
    # "lowest z = 0" on purpose. rank_z_reversed = 0 for the backmost
    for shape in scene['shapes']:
        shape.setdefault('z', 0)
    normalize_z(scene)
    annotate_ranks(scene, 'z')

    # Which shape's color do I sit just over / just under? (Only stacked shapes overlap, so mostly "c-1")
    annotate_over_under(scene)

    # Touching
    # NOTE: we only annotate ONCE, at the start. So after an action (move, paint...), a shape's
    # touching_* values can go stale (e.g. it used to be touching pink, but now it's not). We MAY want to
    # recalculate annotations after each action someday, but for now, annotate once & leave it that way.
    annotate_touching_border(scene)
    annotate_touching_shapes(scene)

    # Within: which shape's hole am I sitting inside? (+ the flip side: what's inside MY holes?)
    annotate_within(scene)

    # Dimensions
    annotate_dimensions(scene)

    # Halves: is the shape's whole box in the top/bottom (left/right) half of the scene, or across the middle?
    # Annotated once at the start, like the rest, so it stays the STARTING half even after a move
    annotate_halves(scene)

    # Symmetry
    annotate_symmetry(scene)

    #######################
    # SCENE PROPERTIES...
    #######################

    annotate_scene_counts(scene)

    # Color ranks (most/least common color), by shapes & by pixels. Must come AFTER the scene counts, since they use them
    annotate_color_ranks(scene)

    return scene


######################################################################
## HELPERS
######################################################################


# numpy arrays can't use plain `in` on a list (== is element-wise, ambiguous truth value)
def _array_in_list(arr:np.ndarray, arr_list:List[np.ndarray]) -> bool:
    return any(np.array_equal(arr, a) for a in arr_list)

def annotate_scene_counts(scene:Dict):

    # Count how many shapes of what color
    for i in range(10): # all ARC colors
        color:str = 'c'+str(i)
        count:int = 0
        for shape in scene['shapes']:
            if shape['color']==color:
                count += 1
        scene['count_color_'+color] = count

    # Also count how many PIXELS of each color are in the scene's shapes (not just how many shapes).
    # Uses each shape's actual pixels, so a multicolor shape counts towards ALL its colors. (-1 = not part of the shape)
    pixel_counts:np.ndarray = np.zeros(10, dtype=int)
    for shape in scene['shapes']:
        shape_pixels:np.ndarray = shape['_pixels']
        pixel_counts += np.bincount(shape_pixels[shape_pixels >= 0], minlength=10)[:10]
    for i in range(10): # all ARC colors
        scene['count_color_pixels_c'+str(i)] = int(pixel_counts[i])

    # How many shapes are in the scene, total? (The background isn't a shape, so it never counts)
    scene['shape_count'] = len(scene['shapes'])

    # How many DISTINCT colors do the shapes have? Uses each shape's main 'color', same as count_color_cN above
    # (so a multicolor shape only counts its one 'color', not all the colors in its pixels)
    scene['color_count'] = len(set(shape['color'] for shape in scene['shapes']))

    # Most & least common colors, by PIXEL (so every color inside a multicolor shape counts too). Background-colored
    # shapes count (like bg-colored noise), the background itself doesn't, since it isn't a shape
    present_color_digits:List[int] = [i for i in range(10) if pixel_counts[i] > 0]

    # Rank biggest first. Shuffle first so ties come out in a random order (the sort keeps that order for ties)
    random.shuffle(present_color_digits)
    present_color_digits.sort(key=lambda digit: -pixel_counts[digit])
    present_colors:List[str] = ['c'+str(digit) for digit in present_color_digits]

    # Fill in as many ranks as there are colors: 1 color = most & least are the same one, 2 colors = most is
    # also least_2nd (and vice versa), 3+ colors = all six (with fewer than 6, most and least overlap). 0 = none
    for rank_index, suffix in enumerate(('', '_2nd', '_3rd')[:len(present_colors)]):
        scene['most_common_nonbg_color'+suffix] = present_colors[rank_index]
        scene['least_common_nonbg_color'+suffix] = present_colors[-1-rank_index]


def annotate_twins_grid(scene:Dict, grid_name:str):
    for shape in scene['shapes']:

        # Generate all possible transformations of the grids (the full D4 group)
        grid:np.ndarray = shape['_'+grid_name] # It's a private var, so underscore it
        transformed_grids = [grid] # +original

        # Rotations
        for quarters in [1, 2, 3]:
            rotated = _rotate_grid(grid, quarters)
            if not _array_in_list(rotated, transformed_grids):
                transformed_grids.append(rotated)
        # Flips
        for mode in ('horizontal', 'vertical', 'diagonal_1', 'diagonal_2'):
            flipped = _flip_grid(grid, mode)
            if not _array_in_list(flipped, transformed_grids):
                transformed_grids.append(flipped)

        # Count how many OTHER shapes have any of these shadows
        twin_count:int = 0
        exact_twin_count:int = 0 # with no transform
        for other_shape in scene['shapes']:
            if other_shape is shape:  # skip self
                continue
            if np.array_equal(other_shape['_'+grid_name], grid):
                exact_twin_count += 1
            if _array_in_list(other_shape['_'+grid_name], transformed_grids):
                twin_count += 1

        # Annotate! Plain name = exact copies (simplest, cheapest), "transformed_" = up to D4
        shape[grid_name+'_twins'] = exact_twin_count
        shape['transformed_'+grid_name+'_twins'] = twin_count

def annotate_twins_prop(scene:Dict) -> None:
    # For each property, count how many OTHER shapes share the same value
    # e.g. if 3 shapes are red, each gets color_twins=2 (the two others)
    twin_props:List[str] = ["color"] # TODO: expand with more props
    shapes:List[Dict] = scene['shapes']
    for prop in twin_props:
        # First pass: count how many shapes have each value
        value_counts:Dict = {}
        for shape in shapes:
            val = shape.get(prop)
            if val is not None:
                value_counts[val] = value_counts.get(val, 0) + 1
        # Second pass: annotate each shape with (count - 1) = number of OTHER twins
        for shape in shapes:
            val = shape.get(prop)
            if val is not None:
                shape[prop + '_twins'] = value_counts[val] - 1


def normalize_z(scene:Dict) -> None:
    """Renumber the scene's z values to 0, 1, 2... (backmost = 0, gaps of 1, ties kept), so only the stacking ORDER is left."""
    # Parses write raw z values (e.g. a pattern background sits at -1), so the same stacking can come with different
    # numbers. Each scene's renumbered on its own, so an input and its output can number the same shape differently
    sorted_z_values:List[int] = sorted({shape['z'] for shape in scene['shapes']})
    rank_per_z_value:Dict[int, int] = {z_value: rank for rank, z_value in enumerate(sorted_z_values)}
    for shape in scene['shapes']:
        shape['z'] = rank_per_z_value[shape['z']]


def annotate_over_under(scene:Dict) -> None:
    """over_color = color of the overlapping shape just below me in z, under_color = just above me ("c-1" = none)."""
    shapes:List[Dict] = scene['shapes']
    for shape in shapes:
        shape['over_color'] = 'c-1'
        shape['under_color'] = 'c-1'

    # Everyone on the same z? Then nothing's stacked, so nothing sits over anything (the usual case, so skip the work)
    if len({shape['z'] for shape in shapes}) <= 1:
        return

    # For each shape: the closest z below / above it among the shapes it overlaps, plus how many pixels they share
    # (to break ties between same-z shapes, e.g. a dot straddling two side-by-side rects: the bigger overlap wins)
    closest_below:Dict[int, tuple] = {} # shape index -> (z, overlap pixels, color)
    closest_above:Dict[int, tuple] = {}
    for index, shape in enumerate(shapes):
        for other_index in range(index+1, len(shapes)):
            other_shape:Dict = shapes[other_index]

            # Same z = side by side, not stacked
            if shape['z'] == other_shape['z']:
                continue

            # Cheap check first: do our bounding boxes even overlap? (right & bottom are one past the last pixel)
            overlap_top:int = max(shape['top'], other_shape['top'])
            overlap_bottom:int = min(shape['bottom'], other_shape['bottom'])
            overlap_left:int = max(shape['left'], other_shape['left'])
            overlap_right:int = min(shape['right'], other_shape['right'])
            if overlap_top >= overlap_bottom or overlap_left >= overlap_right:
                continue

            # How many pixels do our shadows share, inside that box? (Each shadow is relative to its own top-left)
            my_window:np.ndarray = shape['_shadow'][overlap_top-shape['top']:overlap_bottom-shape['top'],
                                                    overlap_left-shape['left']:overlap_right-shape['left']]
            other_window:np.ndarray = other_shape['_shadow'][overlap_top-other_shape['top']:overlap_bottom-other_shape['top'],
                                                             overlap_left-other_shape['left']:overlap_right-other_shape['left']]
            overlap_pixel_count:int = int(np.count_nonzero((my_window == 1) & (other_window == 1)))
            if overlap_pixel_count == 0:
                continue

            # Who's on top? (Higher z = drawn on top)
            lower_index, upper_index = (index, other_index) if shape['z'] < other_shape['z'] else (other_index, index)
            lower_shape, upper_shape = shapes[lower_index], shapes[upper_index]

            # The upper shape is over the lower one: keep it if it's the closest below so far (higher z, then more overlap)
            candidate_below:tuple = (lower_shape['z'], overlap_pixel_count, lower_shape['color'])
            if upper_index not in closest_below or candidate_below[:2] > closest_below[upper_index][:2]:
                closest_below[upper_index] = candidate_below

            # And the lower shape is under the upper one: closest above = LOWER z, then more overlap
            candidate_above:tuple = (-upper_shape['z'], overlap_pixel_count, upper_shape['color'])
            if lower_index not in closest_above or candidate_above[:2] > closest_above[lower_index][:2]:
                closest_above[lower_index] = candidate_above

    # Write down the winners' colors
    for index, (_, _, color) in closest_below.items():
        shapes[index]['over_color'] = color
    for index, (_, _, color) in closest_above.items():
        shapes[index]['under_color'] = color


def annotate_ranks(scene:Dict, numerical_prop_name:str, rank_name:str|None=None):

    # Area rank: largest shape = rank 0, ties share the same rank
    # Picks smallest & second-largest in no-bg parse: http://localhost:9999/#62ab2642

    # Unique (set) of numerical props, then make a sorted list from biggest to smallest...
    # ...EXCEPT top & left (and the centers), which sort smallest-first: the top-most shape has the smallest top, so it
    # gets rank_top=0 (same for left-most). The right-most / bottom-most shapes already have the biggest right / bottom.
    biggest_first:bool = numerical_prop_name not in ('top', 'left', 'center_x', 'center_y')
    unique_num_prop_values:List[int] = sorted(set(s[numerical_prop_name] for s in scene['shapes']), reverse=biggest_first)
    N_values:int = len(unique_num_prop_values)
    
    # Then, convert that list to value->rank Dict
    value_to_rank:Dict[int,int] = {area: rank for rank, area in enumerate(unique_num_prop_values)}
    
    # Use that Dict to add 'rank_area' to all shapes! (Or a custom name, e.g. 'rank_y' for center_y)
    if rank_name is None:
        rank_name = 'rank_'+numerical_prop_name
    for shape in scene['shapes']:
        value:int = shape[numerical_prop_name]
        shape[rank_name] = value_to_rank[value]
        shape[rank_name+'_reversed'] = (N_values-1) - value_to_rank[value]

# Every 2nd / 3rd / 4th shape along x or y, e.g. rank_y_mod2=1 = every other row, starting from the 2nd one.
# Only for the center position ranks: mods of everything else would be too many extra props for the selector search
def annotate_rank_mods(scene:Dict) -> None:
    for shape in scene['shapes']:
        for rank_name in ('rank_x', 'rank_x_reversed', 'rank_y', 'rank_y_reversed'):
            rank:int = shape[rank_name]
            for modulus in (2, 3, 4):
                shape[f'{rank_name}_mod{modulus}'] = rank % modulus

# How common is each shape's color? rank_shape_color_commonness = by number of SHAPES of that color (count_color_cN),
# rank_pixels_color_commonness = by number of PIXELS of that color (count_color_pixels_cN).
# Same conventions as annotate_ranks: 0-based, biggest count = rank 0, ties share a rank, _reversed = rank 0 for the rarest.
# (Only the colors the shapes actually have get ranked)
def annotate_color_ranks(scene:Dict) -> None:
    for rank_name, count_prefix in (('rank_shape_color_commonness', 'count_color_'), ('rank_pixels_color_commonness', 'count_color_pixels_')):

        # Unique counts of the shapes' colors, sorted from biggest to smallest
        unique_counts:List[int] = sorted(set(scene[count_prefix+shape['color']] for shape in scene['shapes']), reverse=True)
        number_of_counts:int = len(unique_counts)

        # Then, convert that list to count->rank Dict
        count_to_rank:Dict[int,int] = {count: rank for rank, count in enumerate(unique_counts)}

        # Use that Dict to rank every shape by its color's count
        for shape in scene['shapes']:
            rank:int = count_to_rank[scene[count_prefix+shape['color']]]
            shape[rank_name] = rank
            shape[rank_name+'_reversed'] = (number_of_counts-1) - rank

def annotate_dimensions(scene:Dict):
    for shape in scene['shapes']:
        shape['dim_min'] = min(shape['width'],shape['height'])
        shape['dim_max'] = max(shape['width'],shape['height'])
        # Wider than tall? 1 = wide, -1 = tall, 0 = square (bounding box)
        shape['wider'] = int(np.sign(shape['width'] - shape['height']))

def annotate_halves(scene:Dict):
    """Which half of the scene is my whole bounding box in? -1 = top/left, 1 = bottom/right, 0 = straddles the middle."""
    # Top half = rows before height//2, bottom half = rows from (height+1)//2 on. With an odd height, the middle
    # row is in neither half, so a box touching it straddles. (bottom & right are exclusive, one past the last pixel)
    scene_height:int = scene['height']
    scene_width:int = scene['width']
    for shape in scene['shapes']:
        if shape['bottom'] <= scene_height // 2:
            shape['start_vhalf'] = -1
        elif shape['top'] >= (scene_height + 1) // 2:
            shape['start_vhalf'] = 1
        else:
            shape['start_vhalf'] = 0
        # Same thing sideways: left half = -1, right half = 1
        if shape['right'] <= scene_width // 2:
            shape['start_hhalf'] = -1
        elif shape['left'] >= (scene_width + 1) // 2:
            shape['start_hhalf'] = 1
        else:
            shape['start_hhalf'] = 0

def annotate_touching_border(scene:Dict):
    for shape in scene['shapes']:
        touching_border:bool = False
        if shape['top']==0 or shape['left']==0 or shape['right']==scene['width'] or shape['bottom']==scene['height']:
            touching_border = True
        shape['touching_border'] = touching_border

def annotate_symmetry(scene:Dict):
    for shape in scene['shapes']:
        shadow = shape['_shadow']

        # Flip symmetries: compare shadow to its flipped version
        symm_h  = np.array_equal(shadow, _flip_grid(shadow, 'horizontal'))  # like A, H, M
        symm_v  = np.array_equal(shadow, _flip_grid(shadow, 'vertical'))    # like B, C, D
        symm_d1 = np.array_equal(shadow, _flip_grid(shadow, 'diagonal_1')) # like %
        symm_d2 = np.array_equal(shadow, _flip_grid(shadow, 'diagonal_2')) # like Q

        # Rotational symmetry: 180° rotation (quarters=2)
        # Note we DON'T have to check quarters=1 or =3, because if there's 90° rotational symmetry,
        # We get 180° symmetry. So, contrapositively, if there's NOT 180° rot-symm, then there
        # CAN'T be 90° rot-symm.
        symm_rot = np.array_equal(shadow, _rotate_grid(shadow, 2))

        shape['symm_h']    = symm_h
        shape['symm_v']    = symm_v
        shape['symm_d']    = symm_d1 or symm_d2 # solves http://localhost:9999/#a09f6c25 ???
        shape['symm_d_1']  = symm_d1 # extra underscore => extra token => lower-prior
        shape['symm_d_2']  = symm_d2
        shape['symm_rot']  = symm_rot
        shape['symm_flip'] = symm_h or symm_v or symm_d1 or symm_d2
        shape['symm_any']  = symm_h or symm_v or symm_d1 or symm_d2 or symm_rot


def annotate_touching_shapes(scene:Dict):
    """Which color is each shape touching the most/least? (Weighted by the touched shapes' areas)"""

    # "Touching" = within 1px horizontally or vertically (4-connected, a plus-shaped neighborhood).
    # Not diagonal: no puzzle seen so far counts diagonal contact as "touching", and it caused false touches.
    # MAYBE: separate touching_8_* props later.
    shapes:List[Dict] = scene['shapes']
    neighborhood:np.ndarray = np.array([[0,1,0],
                                        [1,1,1],
                                        [0,1,0]], dtype=bool)

    # Paint every shape's index onto one scene-sized grid (-1 = no shape there),
    # so we can look up "which shapes are next to me?" with a quick slice
    shape_index_grid:np.ndarray = np.full((scene['height'], scene['width']), -1, dtype=np.int32)
    # (Back to front, like reverse_parse, so where stacked shapes overlap, the one on top owns the pixel)
    for shape_index, shape in sorted(enumerate(shapes), key=lambda index_and_shape: index_and_shape[1]['z']):
        shadow_mask:np.ndarray = shape['_shadow'] == 1
        # Clip to the scene first: a chunk can be parsed hanging partly OFF the canvas (negative top/left,
        # or right/bottom past the edge), and a negative slice start would wrap around & crash
        clip_top:int = max(shape['top'], 0)
        clip_left:int = max(shape['left'], 0)
        clip_bottom:int = min(shape['bottom'], scene['height'])
        clip_right:int = min(shape['right'], scene['width'])
        if clip_bottom <= clip_top or clip_right <= clip_left:
            continue # entirely off the canvas: nothing to paint
        shadow_mask = shadow_mask[clip_top-shape['top'] : clip_bottom-shape['top'], clip_left-shape['left'] : clip_right-shape['left']]
        shape_index_grid[clip_top:clip_bottom, clip_left:clip_right][shadow_mask] = shape_index

    # Each shape's color (as a digit) & area, as arrays, so we can add up areas per color in one go
    # (Multicolor shapes count as their majority color, same as shape['color'])
    shape_color_digits:np.ndarray = np.array([int(shape['color'][1:]) for shape in shapes], dtype=np.int32)
    shape_areas:np.ndarray = np.array([shape['area'] for shape in shapes], dtype=np.int32)

    # Directional touching: who touches me on my left/right/top/bottom side? Pixel-exact, so a shape
    # tucked into a concavity counts too. Shift the shape-index grid by 1px & see which (me, neighbor) pairs line up.
    # Each direction: (my pixels, the pixels 1px over on that side), as two same-sized slices of the grid
    side_slices:Dict[str, tuple[np.ndarray,np.ndarray]] = {
        'left':   (shape_index_grid[:, 1:],  shape_index_grid[:, :-1]),
        'right':  (shape_index_grid[:, :-1], shape_index_grid[:, 1:]),
        'top':    (shape_index_grid[1:, :],  shape_index_grid[:-1, :]),
        'bottom': (shape_index_grid[:-1, :], shape_index_grid[1:, :]),
    }
    side_pairs:Dict[str, np.ndarray] = {}
    for side_name, (my_pixels, neighbor_pixels) in side_slices.items():
        # Both pixels are shapes, and they're two DIFFERENT shapes
        is_contact:np.ndarray = (my_pixels != -1) & (neighbor_pixels != -1) & (my_pixels != neighbor_pixels)
        # Unique (me, neighbor) pairs, sorted by "me" first, so each neighbor's counted once
        side_pairs[side_name] = np.unique(np.stack([my_pixels[is_contact], neighbor_pixels[is_contact]], axis=1), axis=0)

    for shape_index, shape in enumerate(shapes):

        # Grow the shape's shadow by 1px in every direction (pad first, so it has room to grow)
        padded_shadow:np.ndarray = np.pad(shape['_shadow'] == 1, 1)
        grown_shadow:np.ndarray = binary_dilation(padded_shadow, structure=neighborhood)

        # Where does that grown shadow sit in the scene? (Clip it at the scene's borders)
        grid_top:int = shape['top'] - 1
        grid_left:int = shape['left'] - 1
        clip_top:int = max(grid_top, 0)
        clip_left:int = max(grid_left, 0)
        clip_bottom:int = min(grid_top + padded_shadow.shape[0], scene['height'])
        clip_right:int = min(grid_left + padded_shadow.shape[1], scene['width'])
        grown_shadow = grown_shadow[clip_top-grid_top : clip_bottom-grid_top, clip_left-grid_left : clip_right-grid_left]

        # Which OTHER shapes are under the grown shadow? (Each shape counted once, no matter how many pixels touch)
        nearby_indices:np.ndarray = np.unique(shape_index_grid[clip_top:clip_bottom, clip_left:clip_right][grown_shadow])
        touching_indices:np.ndarray = nearby_indices[(nearby_indices != -1) & (nearby_indices != shape_index)]

        # Two flavors: all touching shapes, & only the ones of a DIFFERENT color than me ("other")
        my_color_digit:int = shape_color_digits[shape_index]
        other_color_indices:np.ndarray = touching_indices[shape_color_digits[touching_indices] != my_color_digit]
        for name_prefix, indices in (('touching_color', touching_indices), ('touching_color_other', other_color_indices)):
            most_color, least_color = _most_and_least_touching_color(indices, shape_color_digits, shape_areas)
            shape[name_prefix+'_most'] = most_color
            shape[name_prefix+'_least'] = least_color

        # How many distinct shapes touch me, on any side?
        shape['touching_count'] = len(touching_indices)

        # Per side: how many distinct shapes touch me there, & their most common color (by area, "c-1" if none)
        for side_name, pairs in side_pairs.items():
            # The pairs are sorted by "me", so my neighbors are one contiguous run of rows
            first_row, last_row = np.searchsorted(pairs[:, 0], [shape_index, shape_index + 1])
            side_neighbor_indices:np.ndarray = pairs[first_row:last_row, 1]
            shape[f'touching_{side_name}_count'] = len(side_neighbor_indices)
            most_color, _ = _most_and_least_touching_color(side_neighbor_indices, shape_color_digits, shape_areas)
            shape[f'touching_{side_name}_color'] = most_color

# within_color = the color of the shape whose hole I'm FULLY inside (all my pixels in one of its holes).
# Nested? Then the innermost one wins, i.e. the containing shape with the smallest bounding-box area.
# Uses the hole masks parse already made (_hole_grids, always 4-connected), so it's cheap. Not inside anything => "c-1"
# Same pass also fills in contains_count / contains_pixel_count, the flip side of within (see below)
def annotate_within(scene:Dict) -> None:
    shapes:List[Dict] = scene['shapes']

    # Everyone starts out containing nothing. (Shapes without holes stay at 0)
    for shape in shapes:
        shape['contains_count'] = 0
        shape['contains_pixel_count'] = 0

    for shape in shapes:
        best_container_area:int|None = None
        within_color:str = 'c-1'
        my_shadow_mask:np.ndarray = shape['_shadow'] == 1

        for other_shape in shapes:
            if other_shape is shape or not other_shape['_hole_grids']:
                continue

            # Cheap check first: to be inside its hole, my bounding box has to fit inside its bounding box
            # (All in scene coords, which works fine for shapes hanging off the canvas too: nothing gets sliced by the scene)
            if (shape['left'] < other_shape['left'] or shape['right'] > other_shape['right'] or
                shape['top'] < other_shape['top'] or shape['bottom'] > other_shape['bottom']):
                continue

            # (No skipping bigger containers early: even if they lose the within_color tie-break,
            # they still need to count me for contains_count. The tie-break happens below, once I'm found inside)
            other_bounding_box_area:int = other_shape['width'] * other_shape['height']

            # Where do I sit, in the other shape's LOCAL coords (its hole grids are relative to its top-left)
            local_top:int = shape['top'] - other_shape['top']
            local_left:int = shape['left'] - other_shape['left']
            for hole_grid in other_shape['_hole_grids']:
                # The bit of the hole grid under my bounding box: are ALL my pixels hole pixels (1)?
                hole_under_me:np.ndarray = hole_grid[local_top:local_top+shape['height'], local_left:local_left+shape['width']]
                if np.all(hole_under_me[my_shadow_mask] == 1):
                    # Tie-break: innermost = smallest bounding box wins
                    if best_container_area is None or other_bounding_box_area < best_container_area:
                        best_container_area = other_bounding_box_area
                        within_color = other_shape['color']

                    # And the container counts me: one more shape in its holes, plus my pixels
                    # (Only used for monochrome containers: multicolor ones get overwritten below)
                    other_shape['contains_count'] += 1
                    other_shape['contains_pixel_count'] += shape['area']
                    break # (One hole is enough)

        shape['within_color'] = within_color

    # Multicolor shapes: contains_pixel_count = how many of my pixels AREN'T my most common color
    # (The "stuff drawn inside me", when parse glued it all into one shape.) Decided per shape, by its 'monochrome' flag
    for shape in shapes:
        if shape['monochrome']:
            continue
        # My pixel colors (-1 = not part of me)
        my_colors:np.ndarray = shape['_pixels'][shape['_pixels'] >= 0]
        majority_color_count:int = int(np.bincount(my_colors).max())
        shape['contains_pixel_count'] = len(my_colors) - majority_color_count

def _most_and_least_touching_color(touching_indices:np.ndarray, shape_color_digits:np.ndarray, shape_areas:np.ndarray) -> tuple[str,str]:
    """Add up the touched shapes' areas per color, then return the (most, least) touched colors."""

    # Not touching anything? Then "c-1", what no-bg uses, for both (not None)
    if len(touching_indices) == 0:
        return 'c-1', 'c-1'

    # Total touched area for each of the 10 ARC colors (np.add.at, so repeated colors add up)
    area_per_color:np.ndarray = np.zeros(10, dtype=np.int32)
    np.add.at(area_per_color, shape_color_digits[touching_indices], shape_areas[touching_indices])

    # Most = biggest total. Least = smallest total, among colors we're actually touching.
    # Ties go to a RANDOM one of the tied colors (so we don't always favor the lowest color number)
    most_color_digit:int = int(random.choice(np.flatnonzero(area_per_color == area_per_color.max())))
    # (Least: only among colors we're actually touching, so the untouched colors' 0s don't count)
    touched_areas:np.ndarray = np.where(area_per_color > 0, area_per_color, np.iinfo(np.int32).max)
    least_color_digit:int = int(random.choice(np.flatnonzero(touched_areas == touched_areas.min())))
    return 'c'+str(most_color_digit), 'c'+str(least_color_digit)
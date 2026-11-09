from typing import List, Dict, Tuple
from typing import Optional # For add_ray's ray_color default
import numpy as np
import random
from ...grid import _flip_grid, _rotate_grid, _chebyshev_dilate
from ...utils import _select_shape_datas
# Ray tracing helpers, for add_ray() and add_ray_towards()
from .rays import (ray_directions, blocker_index_grid, trace_ray, make_ray_shape,
                   STRAIGHT_DIRECTIONS, DIAGONAL_DIRECTIONS)
from .rays import leading_edge_pixels # Which pixels of a non-dot origin shoot rays
from .rays import center_edge_pixel # ...or just the one in line with its center (from_center, and always for "all")
from ...config import DSL_MAX_OUTLINE_LINE_WIDTH # outline()'s line_width cap
from .copying import copy_to_targets # copy_to()'s stamping search
# move_towards/away() & extend_towards/away() head for (or away from) their targets
from .physics_utils import _closest_target, _direction_towards, _direction_away
from .physics_pathfind import pathfind_offset # pathfind() winds through the gaps between shapes

_BLANK_SHAPE = {
    'type':'MISSING',
    'top':0,
    'right':0,
    'bottom':0,
    'width':0,
    'height':0,
    'color':'MISSING',
    '_pixels':'MISSING',
    '_shadow':'MISSING'
}
_EMPTY_SCENE = {}
_X_PROPS = frozenset({'left', 'right', 'center_x'})
_Y_PROPS = frozenset({'top', 'bottom', 'center_y'})

# Named directions for move() and fall(), as (x, y) unit steps. y grows downwards, like the grid
_COMPASS_DIRECTIONS:Dict[str, Tuple[int, int]] = {
    'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0),
    'ul': (-1, -1), 'ur': (1, -1), 'dl': (-1, 1), 'dr': (1, 1),
}

def _sign(value:float) -> int:
    """-1, 0 or 1."""
    return (value > 0) - (value < 0)

def _direction_to_unit_step(direction:str, shape_data:Dict, scene_data:Dict) -> Tuple[int, int]:
    """Turn a named direction ("u", "dr", "in", "out"...) into an (x, y) unit step for this shape."""
    if direction in _COMPASS_DIRECTIONS:
        return _COMPASS_DIRECTIONS[direction]
    # "in" heads towards the scene's center, per axis. An axis the shape is already centered on stays put (0)
    towards_center_x:int = _sign(scene_data['width']/2 - shape_data['center_x'])
    towards_center_y:int = _sign(scene_data['height']/2 - shape_data['center_y'])
    if direction == 'in':
        return towards_center_x, towards_center_y
    # "out" is just the opposite of "in". Anything else isn't a direction, so the program errors out
    if direction == 'out':
        return -towards_center_x, -towards_center_y
    raise ValueError(f"Unknown direction: {direction}")

# TODO much later on: use Mixins to split up all these Shape methods?...
class Shape:

    ########################################################################
    ## HELLO! I AM SHAPE!
    ########################################################################

    __slots__ = ('data', 'scene')

    def __init__(self, shape_data=None, scene_data=None):
        """
        Assign data/scene, with blank/failure defaults
        """
        self.data = shape_data if shape_data is not None else _BLANK_SHAPE.copy()
        self.scene = scene_data if scene_data is not None else _EMPTY_SCENE

    def __getattr__(self, prop_name):
        """
        Wrapper to access data's props
        """
        return self.data[prop_name] if prop_name in self.data else None

    ########################################################################
    ## PAINTING
    ########################################################################

    def paint(self, **kwargs):
        """
        Recolor the whole shape
        """
        d = self.data
        if 'color' in kwargs:
            if kwargs['color'] != "MISSING" and kwargs['color'] != 'c-1': # Only paint if not missing, else do nothing
                d['color'] = kwargs['color']
                color_digit = int(kwargs['color'][1:]) # [1:] to convert c0 => 0, etc
                # Recolor ONLY the originally-parsed pixels, and leave everything else (an outline ring, filled
                # holes, glued-on pieces) as it is. So outline-then-paint gives the same result as paint-then-outline.
                # (Want the whole thing recolored, ring and all? That's an outline in the new color, then paint.)
                old_pixels:np.ndarray = d['_pixels']
                d['_pixels'] = np.where(d['_original_shadow'] == 1, np.int8(color_digit), d['_pixels'])
                # Only counts as edited if a pixel actually changed color (painting a yellow shape yellow doesn't)
                if not np.array_equal(d['_pixels'], old_pixels):
                    d['edited'] = True
                d['_edits_cost'] = d.get('_edits_cost', 0) + 1  # Every action = 1 edit
        #self._enforce_static()

    ########################################################################
    ## HIDING
    ########################################################################

    def hide(self):
        # (Hiding doesn't mark the shape edited: it's just gone)
        self.data['hidden'] = True
        self.data['_edits_cost'] = self.data.get('_edits_cost', 0) + 1  # Every action = 1 edit
        #self._enforce_static()

    ########################################################################
    ## MOVEMENTS
    ########################################################################

    def place(self, **kwargs):
        """
        Move the shape to an absolute position
        """
        d = self.data
        w, h = d['width'], d['height'] # local aliases
        old_left, old_top = d['left'], d['top']

        # Can NOT just set properties & forget them: figure out the resulting left/top
        # from whichever anchor was given, then let _set_position reconcile the rest.
        if 'top' in kwargs:
            self._set_position(top=kwargs['top'])
        if 'bottom' in kwargs:
            self._set_position(top=kwargs['bottom'] - h)
        if 'left' in kwargs:
            self._set_position(left=kwargs['left'])
        if 'right' in kwargs:
            self._set_position(left=kwargs['right'] - w)
        if 'center_x' in kwargs:
            left = int( float(kwargs['center_x']) - w/2 )
            self._set_position(left=left)
        if 'center_y' in kwargs:
            top = int( float(kwargs['center_y']) - h/2 )
            self._set_position(top=top)

        #self._enforce_static()
        if d['left'] != old_left or d['top'] != old_top:
            d['_edits_cost'] = d.get('_edits_cost', 0) + 1
            d['_moves'] = d.get('_moves', 0) + 1 # track moves separately, for draw-order (most-recently-moved on top)
            d['edited'] = True # it actually moved

    def move(self, **kwargs):
        """
        Move the shape by a relative amount.
        Supports both absolute deltas (x, y) and directional moves (ux, uy, steps, or direction="u"/"dr"/"in"... and steps),
        plus moves along the stacking order (z, max_z).
        """
        d = self.data
        old_left, old_top = d['left'], d['top']

        # A named direction is just another way to say the unit vector (ux, uy)
        if 'direction' in kwargs:
            kwargs['ux'], kwargs['uy'] = _direction_to_unit_step(kwargs['direction'], d, self.scene)

        # Directional move: unit vector (ux, uy) * steps
        if 'ux' in kwargs or 'uy' in kwargs:
            ux: int = kwargs.get('ux', 0)
            uy: int = kwargs.get('uy', 0)
            steps: int = kwargs.get('steps', 1)
            kwargs['x'] = ux * steps
            kwargs['y'] = uy * steps

        # _set_position int-casts: a move delta can be a float (e.g. referencing a
        # center_x/center_y property), but grid coords must be ints. Truncating is fine
        # for evo-search — a slightly-off move just scores poorly and gets out-competed.
        self._set_position(left = old_left + kwargs.get('x', 0),
                           top  = old_top  + kwargs.get('y', 0))
        #self._enforce_static()
        moved:bool = d['left'] != old_left or d['top'] != old_top

        # Stacking order: z=+1/-1 steps forward/back, max_z=1 jumps to the very front, max_z=-1 to the very back.
        # (Shapes without a z count as 0, like in reverse_parse. z may go below 0 or leave gaps, that's fine)
        if 'z' in kwargs or 'max_z' in kwargs:
            old_z:int = d.get('z', 0)
            new_z:int = old_z + kwargs.get('z', 0)
            if kwargs.get('max_z', 0) > 0:
                new_z = max(shape.get('z', 0) for shape in self.scene['shapes']) + 1 # 1 above the frontmost
            elif kwargs.get('max_z', 0) < 0:
                new_z = min(shape.get('z', 0) for shape in self.scene['shapes']) - 1 # 1 below the backmost
            d['z'] = new_z
            moved = moved or new_z != old_z

        # A z-change counts as a move too, so a shape that lands on another shape's z gets drawn in front of it
        if moved:
            d['_edits_cost'] = d.get('_edits_cost', 0) + 1
            d['_moves'] = d.get('_moves', 0) + 1 # track moves separately, for draw-order (most-recently-moved on top)
            d['edited'] = True # it actually moved

    def move_towards(self, **kwargs):

        # Steps: 1 multiplier by default
        steps:int = kwargs.pop('steps', 1) # also pop() gets rid of .steps property in kwargs

        # Target selector is flat kwargs — after popping '.steps', remaining kwargs ARE the selector
        targets:List[Dict] = _select_shape_datas(self.scene['shapes'], kwargs)
        if not targets:
            return # nobody to move towards
        # Head for the closest target by pixel gap (like fall_towards), random among equally close ones
        target:Dict = _closest_target(self.data, targets)

        # Direction: gridworld movement (horizontal/vertical/diagonal only)
        x, y = _direction_towards(self.data, target)

        # Multiply by steps
        x *= steps
        y *= steps

        # Finally, move relatively towards!
        self.move(x=x, y=y)

    def pathfind(self, passthru:Optional[str] = None, **kwargs):
        """shape.pathfind(passthru=None, **target selectors): wind through the gaps, as close as it can get to the nearest target."""
        # Every kwarg except passthru is a target selector, like move_towards (and I'm never my own target)
        targets:List[Dict] = [target for target in _select_shape_datas(self.scene['shapes'], kwargs) if target is not self.data]
        if not targets:
            return # nobody to head for
        # The passthru color ("c2" => 2) is one the shape may walk onto, like the background
        passthru_digit:Optional[int] = int(passthru[1:]) if passthru is not None else None
        x, y = pathfind_offset(self.data, targets, self.scene, passthru_digit)
        self.move(x=x, y=y)

    def move_away(self, **kwargs):
        """shape.move_away(steps=1, **target selectors): the opposite of move_towards, a straight step away from the targets."""
        steps:int = kwargs.pop('steps', 1)
        # Every selected shape but me (I'd always be touching myself)
        targets:List[Dict] = [target for target in _select_shape_datas(self.scene['shapes'], kwargs) if target is not self.data]
        if not targets:
            return # nobody to move away from
        # fall_away's rule: away from the sides we touch targets on (added up), else opposite the closest one's bounds
        x, y = _direction_away(self.data, targets)
        self.move(x=x * steps, y=y * steps)

    def _am_i_hitting_scene_boundaries(self) -> bool:
        if self.data['left']<0 or self.data['top']<0 or self.data['right']>self.scene['width'] or self.data['bottom']>self.scene['height']:
            return True
        return False

    def _set_position(self, left=None, top=None):
        """
        The single source of truth for repositioning. Given a new left and/or top,
        int-cast it (grid coords MUST be ints) and re-derive every dependent box prop
        (right/bottom + center_x/center_y) so they all stay consistent. Centralizing the
        int() cast here means no caller can ever leak a float into grid coordinates again.
        """
        d = self.data
        w, h = d['width'], d['height'] # local aliases
        if left is not None:
            d['left'] = int(left)
            d['right'] = d['left'] + w
            d['center_x'] = d['left'] + w/2
        if top is not None:
            d['top'] = int(top)
            d['bottom'] = d['top'] + h
            d['center_y'] = d['top'] + h/2

    ########################################################################
    ## TRANSFORMS
    ########################################################################

    def flip(self, **kwargs):
        """Flip the shape's grid: 'horizontal', 'vertical', 'diagonal_1', or 'diagonal_2'. Pivots around center."""
        d = self.data
        mode = kwargs.get('mode', 'horizontal')

        # Remember the center BEFORE mutating (diagonal flips swap w<->h)
        center_x = d['left'] + d['width']/2
        center_y = d['top'] + d['height']/2
        # Remember what it looked like, so a flip that changes nothing (a symmetric shape) doesn't count as an edit
        old_pixels:np.ndarray = d['_pixels']
        old_position:Tuple[int, int] = (d['left'], d['top'])

        d['_pixels'] = _flip_grid(d['_pixels'], mode)
        d['_shadow'] = _flip_grid(d['_shadow'], mode)
        d['_original_shadow'] = _flip_grid(d['_original_shadow'], mode) # The original pixels flip along, too

        # Refresh dims, then pivot around the original center
        d['height'], d['width'] = d['_pixels'].shape
        self._recenter(center_x, center_y)
        self._mark_edited_if_changed(old_pixels, old_position)
        d['_edits_cost'] = d.get('_edits_cost', 0) + 1  # Every action = 1 edit
        #self._enforce_static()

    def rotate(self, **kwargs):
        """Rotate the shape clockwise by 90-degree (quarter-turn) increments. Pivots around center."""
        d = self.data
        quarters = int(kwargs.get('quarters', 1))  # 1=90°CW, 2=180°, 3=270°CW

        # Remember the center BEFORE mutating dimensions (odd quarter-turns swap w<->h)
        center_x = d['left'] + d['width']/2
        center_y = d['top'] + d['height']/2
        # Remember what it looked like, so a rotation that changes nothing (a symmetric shape) doesn't count as an edit
        old_pixels:np.ndarray = d['_pixels']
        old_position:Tuple[int, int] = (d['left'], d['top'])

        # Rotate 'em
        d['_pixels'] = _rotate_grid(d['_pixels'], quarters)
        d['_shadow'] = _rotate_grid(d['_shadow'], quarters)
        d['_original_shadow'] = _rotate_grid(d['_original_shadow'], quarters) # The original pixels rotate along, too

        # Refresh dims, then pivot around the original center
        d['height'], d['width'] = d['_pixels'].shape
        self._recenter(center_x, center_y)
        self._mark_edited_if_changed(old_pixels, old_position)
        d['_edits_cost'] = d.get('_edits_cost', 0) + 1  # Every action = 1 edit
        #self._enforce_static()

    def _mark_edited_if_changed(self, old_pixels:np.ndarray, old_position:Tuple[int, int]) -> None:
        """Mark the shape edited if its pixels (size included) or its position differ from before."""
        d = self.data
        # (Pixels outside the shape are always -1, so comparing the whole grid compares just the shape's pixels)
        if (d['left'], d['top']) != old_position or not np.array_equal(d['_pixels'], old_pixels):
            d['edited'] = True

    def _recenter(self, center_x, center_y):
        """
        Reposition the shape so its center sits at (center_x, center_y), keeping the
        CURRENT width/height. Recomputes left/top/right/bottom and center_x/center_y.
        (Force-rounds left/top to nearest pixel, then derives center from that so all fields stay consistent.)
        """
        d = self.data
        w, h = d['width'], d['height'] # local aliases
        # round (not truncate) to NEAREST pixel, then _set_position reconciles the rest
        self._set_position(left=round(center_x - w/2),
                           top=round(center_y - h/2))

    ########################################################################
    ## FANCY: FILLING HOLES
    ########################################################################

    def fill_holes(self, **kwargs):
        """
        Fill all interior holes of the shape with a given color.
        shape.fill_holes( color:str )  -- color in "cN" format, e.g. "c4"
        """
        d = self.data
        color:str = kwargs.get('color', 'MISSING')
        if color == "MISSING":
            return
        color_digit:int = int(color[1:])  # "c4" -> 4

        hole_grids:list = d.get('_hole_grids', [])
        if not hole_grids:
            return  # No holes to fill

        # For each hole grid, paint hole pixels into _pixels and _shadow.
        # Copy _pixels first to avoid mutating the shared parsed array (same
        # "rebind, never mutate in place" convention as outline/paint).
        new_pixels = d['_pixels'].copy()
        new_shadow = d['_shadow'].copy()
        for hole_grid in hole_grids:
            mask = hole_grid == 1
            new_pixels[mask] = np.int8(color_digit)
            new_shadow[mask] = np.int8(1)

        d['_pixels'] = new_pixels
        d['_shadow'] = new_shadow
        d['_hole_grids'] = []  # Holes are filled now
        d['holes'] = 0
        # Recompute area since we added pixels
        d['area'] = int(np.count_nonzero(new_shadow))
        d['_edits_cost'] = d.get('_edits_cost', 0) + 1  # Every action = 1 edit
        # Sure, this counts as a 'move' (for redraw-on-top purposes)
        d['_moves'] = d.get('_moves', 0) + 1

    ########################################################################
    ## FANCY: OUTLINING
    ########################################################################

    def _filled_mask_with_holes_filled(self, filled:np.ndarray) -> np.ndarray:
        """For ignore_holes=True: the shape's filled-pixel mask, as if its holes were filled in too."""
        # Cheap skip: no holes (check the annotation first), nothing to fill
        if self.data.get('holes', 0) == 0:
            return filled
        # A scratch copy, never the real pixel grid! The hole masks are in the same local coords as _pixels
        # (outline() pads them right along with the grid), so they line up 1:1
        filled_with_holes:np.ndarray = filled.copy()
        for hole_grid in self.data.get('_hole_grids', []):
            filled_with_holes |= (hole_grid == 1)
        return filled_with_holes

    def outline(self, **kwargs):
        """
        shape.outline( color:str, line_width:int=1, ignore_holes:bool=False )
        (There's no inset=True any more: it just gets ignored)
        """

        # The plan: (TODO: corner styles other than Chebyshev?, /patterned/ outlines?)
        # Two modes, both using line_width Chebyshev-radius (crisp square corners):
        # OUTLINE (default): grow a ring AROUND the shape.
        #   1) Pad the _pixels grid with -1 (not-in-shape), line_width amount on all sides
        #   2) Scan for every -1 cell within line_width of the ORIGINAL non-(-1) cells. Stash 'em
        #      in a scratchpad list FIRST, so outline cells we add don't seed MORE outline (no cascade!). Then paint.
        #   3) Recalculate _shadow, area, top (-= line_width), left (-= line_width),
        #      width (+= 2*line_width), height (+= 2*line_width)
        # (INSET, recoloring the shape's OWN edge pixels, lives in its own method, shape.inline(), below. Disabled for now)

        # The parameters...
        d = self.data
        color = kwargs.get('color', 'MISSING')
        if color == "MISSING": # no color to outline with? do nothing, just like paint()
            return
        color_digit = int(color[1:]) # [1:] to convert c0 => 0, etc
        line_width = int(kwargs.get('line_width', 1))
        # Clamp runaway widths (see DSL_MAX_OUTLINE_LINE_WIDTH in config/default.py): a huge line_width
        # from a property reference like line_width=other.area would allocate an enormous padded grid below.
        line_width = min(line_width, DSL_MAX_OUTLINE_LINE_WIDTH)
        # ignore_holes=True: draw the ring as if the holes were filled in, so nothing gets painted inside them
        ignore_holes:bool = bool(kwargs.get('ignore_holes', False))


        ###############################################################
        # OUTLINE branch (default): grow a ring around the shape.
        ###############################################################

        old_pixels = d['_pixels']
        old_h, old_w = old_pixels.shape
        new_w, new_h = old_w + 2*line_width, old_h + 2*line_width

        # 1) New padded grid, all -1, with the original pixels dropped in at (line_width, line_width)
        padded = np.full((new_h, new_w), -1, dtype=np.int8)
        padded[line_width:line_width+old_h, line_width:line_width+old_w] = old_pixels

        # 2) Morphological dilation: a cell should be outlined if it's empty
        #    AND any neighbor in the Chebyshev square is filled. (score_parse uses the same dilation
        #    to spot outlines, so it lives in grid/ as _chebyshev_dilate)
        filled = padded != -1
        # ignore_holes: pretend the holes are filled (scratch mask only), so they neither get outlined
        # nor seed any outline. (Filled in on the unpadded grid, then padded to match.)
        if ignore_holes:
            filled = np.pad(self._filled_mask_with_holes_filled(old_pixels != -1), line_width, constant_values=False)
        near_filled = _chebyshev_dilate(filled, line_width)

        # Cells to outline: empty AND near a filled cell (safe: reads pre-fill, no cascade)
        outline_mask = (~filled) & near_filled
        padded[outline_mask] = color_digit

        # 3) Commit the new grid, then refresh shadow/area/dims/position
        d['_pixels'] = padded
        d['_shadow'] = (padded != -1).astype(np.int8) # 0 = empty, 1 = filled

        # The hole masks in _hole_grids live in the shape's LOCAL coordinates, and we just grew the pixel
        # grid by line_width on every side (and nudged top-left up-left by line_width, below). Pad each hole mask
        # the same way, so every hole still points at the same real pixels. (Otherwise fill_holes() after outline()
        # would crash on a shape-mismatched mask, or paint the holes line_width too far up-left.)
        # The holes themselves are left alone by an outline: the ring is drawn on the OUTSIDE of the shape. (Yes,
        # the dilation also paints a hole's inner rim, since those cells are empty-and-near-filled. That's fine:
        # a later fill_holes paints the original hole mask right over it, ignoring that the outline was ever there.)
        # Padding value -1 = "not a hole", matching the hole grids' own convention (-1 for no, 1 for yes, see
        # apply_parse.py). fill_holes() only looks for 1s anyway.
        d['_hole_grids'] = [np.pad(hole_grid, line_width, constant_values=-1) for hole_grid in d.get('_hole_grids', [])]

        # Same deal for the original pixels' mask (padding 0 = "not an original pixel"), so a later paint()
        # recolors just the shape inside the ring, not the ring itself
        d['_original_shadow'] = np.pad(d['_original_shadow'], line_width, constant_values=0)

        d['area'] = int(np.count_nonzero(padded != -1)) # every non-(-1) pixel counts
        d['width'], d['height'] = new_w, new_h

        # grew by line_width on every side, so nudge top-left back; _set_position re-derives the rest
        self._set_position(left=d['left']-line_width, top=d['top']-line_width)
        d['_edits_cost'] = d.get('_edits_cost', 0) + 1  # Every action = 1 edit
        #self._enforce_static()


    # inline() is DISABLED for now (we'll deal with in-lining later). Commented out, not deleted. (NOT ACTUALLY CURRENTLY USED; WIP)
    # def inline(self, **kwargs):
    #     """
    #     shape.inline( color:str, line_width:int=1, ignore_holes:bool=False )
    #     """
    #
    #     # IN-lining, split out of outline(inset=True): recolor the shape's OWN edge pixels (no grow).
    #     # A filled pixel is on the edge if ANY cell within line_width (Chebyshev) is empty (-1) or off-grid.
    #     # Only recolors existing pixels, so -1/filled topology (& thus shadow/area/dims) is unchanged.
    #
    #     # The parameters... (same as outline's, minus inset)
    #     d = self.data
    #     color = kwargs.get('color', 'MISSING')
    #     if color == "MISSING": # no color to inline with? do nothing, just like paint()
    #         return
    #     color_digit = int(color[1:]) # [1:] to convert c0 => 0, etc
    #     line_width = int(kwargs.get('line_width', 1))
    #     # Same runaway-width clamp as outline() (DSL_MAX_OUTLINE_LINE_WIDTH in config/default.py)
    #     line_width = min(line_width, DSL_MAX_OUTLINE_LINE_WIDTH)
    #     # ignore_holes=True: find the edge as if the holes were filled in, so hole edges don't get traced
    #     ignore_holes:bool = bool(kwargs.get('ignore_holes', False))
    #
    #     pixels = d['_pixels']
    #     h, w = pixels.shape
    #     # Write into a FRESH grid, never the shared one! shape.copy() in apply_transform is
    #     # shallow, so d['_pixels'] is the SAME array as the original parsed scene's. The whole
    #     # DSL plays by "rebind, never mutate in place" -- so we .copy() here, write into the
    #     # copy, then rebind at the end. (Read edges off the original; recoloring keeps topology.)
    #     new_pixels = pixels.copy()
    #
    #     # Numpy morphological erosion (SEPARABLE): a filled pixel is on the edge if ANY
    #     # neighbor in the Chebyshev square is empty or off-grid. Equivalently: NOT all filled.
    #     # Separable trick: Chebyshev erosion = erode horizontally THEN vertically.
    #     # Cost: 2*(2*lw+1) AND ops instead of (2*lw+1)^2 — big win at line_width > 1.
    #     filled = pixels != -1
    #     # ignore_holes: erode a scratch mask with the holes filled in instead (hole pixels themselves
    #     # still never get recolored, 'coz the edge mask below is ANDed with the REAL filled pixels)
    #     filled_for_erosion:np.ndarray = self._filled_mask_with_holes_filled(filled) if ignore_holes else filled
    #     # Horizontal pass: all neighbors in row within ±line_width must be filled
    #     h_all = np.ones((h, w), dtype=bool)
    #     for dx in range(-line_width, line_width+1):
    #         shifted = np.zeros((h, w), dtype=bool)
    #         sx0 = max(0, -dx); sx1 = w - max(0, dx)
    #         dx0 = max(0, dx)
    #         # shifted[:, dx0:dx0 + sx1 - sx0] = filled[:, sx0:sx1]
    #         shifted[:, dx0:dx0 + sx1 - sx0] = filled_for_erosion[:, sx0:sx1] # The maybe-holes-filled mask
    #         h_all &= shifted
    #     # Vertical pass: all neighbors in column within ±line_width must be filled (after h-pass)
    #     all_filled = np.ones((h, w), dtype=bool)
    #     for dy in range(-line_width, line_width+1):
    #         shifted = np.zeros((h, w), dtype=bool)
    #         sy0 = max(0, -dy); sy1 = h - max(0, dy)
    #         dy0 = max(0, dy)
    #         shifted[dy0:dy0 + sy1 - sy0, :] = h_all[sy0:sy1, :]
    #         all_filled &= shifted
    #     # Edge pixels: filled AND NOT all neighbors filled
    #     edge_mask = filled & ~all_filled
    #     new_pixels[edge_mask] = color_digit
    #     d['_pixels'] = new_pixels # rebind, just like paint/outline/flip/rotate
    #     # The inset ring isn't "original" anymore: a later paint() leaves it alone, so inset-then-paint
    #     # gives the same result as paint-then-inset (just like the grow-outline below)
    #     d['_original_shadow'] = np.where(edge_mask, np.int8(0), d['_original_shadow'])
    #
    #     # inset doesn't grow or move the shape: shadow/area/dims all stay put
    #     d['_edits_cost'] = d.get('_edits_cost', 0) + 1  # Every action = 1 edit
    #     #self._enforce_static()
    #     return

    ########################################################################
    ## EXTENDING: grow the shape's leading pixels a few steps one way (the new pixels are part of the SAME shape)
    ########################################################################

    def _extend_by_step(self, step_x:int, step_y:int, steps:int, color:Optional[str]) -> None:
        """Grow every leading pixel (the outermost one on each line going that way) by `steps` pixels."""
        d = self.data
        # Nowhere to go, or a missing color? Do nothing, just like paint()
        if (step_x == 0 and step_y == 0) or steps <= 0 or color == 'MISSING' or color == 'c-1':
            return

        # Every filled pixel, in the shape's local coords
        filled_ys, filled_xs = np.nonzero(d['_shadow'] == 1)
        # Which line (parallel to the direction) each pixel sits on, and how far along the direction it is.
        # E.g. going right: the line is the row, and "how far" is x. Works for diagonals too
        line_ids:np.ndarray = filled_xs * step_y - filled_ys * step_x
        progress:np.ndarray = filled_xs * step_x + filled_ys * step_y
        # Sort by line, then by progress: the last pixel of each line is its leading pixel
        order:np.ndarray = np.lexsort((progress, line_ids))
        is_last_of_line:np.ndarray = np.append(line_ids[order][1:] != line_ids[order][:-1], True)
        leading:np.ndarray = order[is_last_of_line]
        leading_xs:np.ndarray = filled_xs[leading]
        leading_ys:np.ndarray = filled_ys[leading]

        # The new pixels: 1..steps past each leading pixel, in its own color unless a color was given
        step_counts:np.ndarray = np.arange(1, steps + 1)
        new_xs:np.ndarray = (leading_xs[:, None] + step_counts[None, :] * step_x).ravel()
        new_ys:np.ndarray = (leading_ys[:, None] + step_counts[None, :] * step_y).ravel()
        if color is None:
            new_colors:np.ndarray = np.repeat(d['_pixels'][leading_ys, leading_xs], steps)
        else:
            new_colors = np.full(len(new_xs), int(color[1:]), dtype=np.int8) # [1:] to convert c0 => 0, etc

        # How much the bounding box grows on each side (only the side we're heading towards grows)
        old_height, old_width = d['_pixels'].shape
        grow_left:int = max(0, -int(new_xs.min()))
        grow_top:int = max(0, -int(new_ys.min()))
        grow_right:int = max(0, int(new_xs.max()) - (old_width - 1))
        grow_bottom:int = max(0, int(new_ys.max()) - (old_height - 1))
        padding:Tuple[Tuple[int, int], Tuple[int, int]] = ((grow_top, grow_bottom), (grow_left, grow_right))

        # Pad the pixel grid (-1 = not in the shape), then drop the new pixels in
        new_pixels:np.ndarray = np.pad(d['_pixels'], padding, constant_values=-1)
        new_pixels[new_ys + grow_top, new_xs + grow_left] = new_colors
        d['_pixels'] = new_pixels
        d['_shadow'] = (new_pixels != -1).astype(np.int8)
        # Keep the local-coord masks lined up with the grown grid, like outline() does
        # (holes: -1 = not a hole; original shadow: 0 = not an original pixel, so a later paint() skips the new pixels)
        d['_hole_grids'] = [np.pad(hole_grid, padding, constant_values=-1) for hole_grid in d.get('_hole_grids', [])]
        d['_original_shadow'] = np.pad(d['_original_shadow'], padding, constant_values=0)

        # Refresh area, size & position (growing left/up moves the top-left corner)
        d['area'] = int(np.count_nonzero(d['_shadow']))
        d['height'], d['width'] = new_pixels.shape
        self._set_position(left=d['left'] - grow_left, top=d['top'] - grow_top)
        d['edited'] = True # it always grows by at least one pixel
        d['_edits_cost'] = d.get('_edits_cost', 0) + 1  # Every action = 1 edit

    def extend(self, direction:str = '', steps:int = 1, extend_color:Optional[str] = None) -> None:
        """
        shape.extend( direction:str, steps:int=1, extend_color:str=<each pixel's own color> )
        Directions: u, d, l, r (and the diagonals ul, ur, dl, dr)
        """
        # No shape (e.g. one() found nothing)? Nothing to grow
        if self.data['type'] == 'MISSING':
            return
        step_x, step_y = _COMPASS_DIRECTIONS[direction] # An unknown direction errors out, so the program fails
        self._extend_by_step(step_x, step_y, int(steps), extend_color)

    def _extend_relative_to_target(self, away:bool, extend_color:Optional[str], steps:int, target_selectors:Dict) -> None:
        """extend_towards / extend_away: grow towards (or away from) the closest selected shape, any of 8 directions."""
        if self.data['type'] == 'MISSING':
            return
        # Every selected shape but me (or I'd always be the closest, at distance 0)
        targets:List[Dict] = [target for target in _select_shape_datas(self.scene['shapes'], target_selectors) if target is not self.data]
        if not targets:
            return # nobody to grow towards (or away from)
        # Away uses fall_away's rule: away from the sides we touch targets on, else opposite the closest one's bounds.
        # Towards heads for the closest target by pixel gap, random among equally close ones (like move_towards)
        step_x, step_y = _direction_away(self.data, targets) if away else _direction_towards(self.data, _closest_target(self.data, targets))
        self._extend_by_step(step_x, step_y, int(steps), extend_color)

    def extend_towards(self, extend_color:Optional[str] = None, steps:int = 1, **target_selectors) -> None:
        """
        shape.extend_towards( extend_color:str=<each pixel's own color>, steps:int=1, **target selectors )
        Every kwarg except extend_color & steps is a target selector, like move_towards (so color= picks the target)
        """
        self._extend_relative_to_target(False, extend_color, steps, target_selectors)

    def extend_away(self, extend_color:Optional[str] = None, steps:int = 1, **target_selectors) -> None:
        """
        shape.extend_away( extend_color:str=<each pixel's own color>, steps:int=1, **target selectors )
        Every kwarg except extend_color & steps is a target selector, like fall_away (so color= picks the target)
        """
        self._extend_relative_to_target(True, extend_color, steps, target_selectors)

    ########################################################################
    ## RAYS: shoot a line out of a shape (makes NEW shapes, the origin stays as it is)
    ########################################################################

    # Rays shoot out of every leading-edge pixel (see rays.leading_edge_pixels): a dot's only pixel is its
    # leading edge every way. With from_center (and always for "all"), only from the one in line with the center

    def _shoot_rays(self, rays:List[Tuple[List[int], List[int], int]], ray_color:Optional[str]) -> None:
        """Turn a list of traced rays (xs, ys, the color digit of the pixel it shot from) into ray shapes."""
        for ray_xs, ray_ys, pixel_color_digit in rays:
            # Each ray is its start pixel's color, unless ray_color says otherwise
            this_ray_color:str = ray_color if ray_color is not None else f"c{pixel_color_digit}"
            # A missing color does nothing, just like paint()
            if this_ray_color == 'MISSING' or this_ray_color == 'c-1':
                continue
            self._add_ray_shape(ray_xs, ray_ys, this_ray_color)

    def _add_ray_shape(self, ray_xs:List[int], ray_ys:List[int], ray_color:str) -> None:
        """Turn traced ray pixels into a new type="ray" shape, and put it in the scene."""
        ray_shape = make_ray_shape(self.scene, ray_xs, ray_ys, int(ray_color[1:])) # [1:] to convert c0 => 0, etc
        if ray_shape is not None:
            self.scene['shapes'].append(ray_shape)

    def add_ray(self, direction:str = '', ray_color:Optional[str] = None, cut_corners:bool = False, from_center:bool = False) -> None:
        """
        shape.add_ray( direction:str, ray_color:str=<my color>, cut_corners:bool=False, from_center:bool=False )
        Directions: u, d, l, r, ul, ur, dl, dr, or the shortcuts "lr", "ud", "+", "x", "all" (one ray shape per direction)
        from_center=True: one ray per direction, from the edge pixel in line with my center. "all" always does this
        """
        # No shape (e.g. one() found nothing)? No rays
        if self.data['type'] == 'MISSING':
            return
        # (The ray color check lives in _shoot_rays(), since each ray can have its own color)
        # Where's everyone? (Built once per call, BEFORE adding any rays: rays don't block each other anyway)
        # ...minus me: the origin doesn't block its own rays
        blocker_grid:np.ndarray = blocker_index_grid(self.scene, origin_shape_data=self.data)
        # A starburst ("all") always shoots from the center (Nicky: there's no "all" that isn't)
        from_center = from_center or direction == 'all'
        # One ray shape per direction, and per leading-edge pixel in that direction (or just the centered one)
        for single_direction in ray_directions(direction):
            start_pixels:List[Tuple[int, int, int]] = leading_edge_pixels(self.data, single_direction)
            if from_center:
                start_pixels = center_edge_pixel(start_pixels, self.data, single_direction)
            rays:List[Tuple[List[int], List[int], int]] = []
            for start_x, start_y, pixel_color_digit in start_pixels:
                ray_xs, ray_ys, _ = trace_ray(blocker_grid, start_x, start_y, single_direction, cut_corners)
                rays.append((ray_xs, ray_ys, pixel_color_digit))
            self._shoot_rays(rays, ray_color)

    def add_ray_towards(self, cut_corners:bool = False, ray_color:Optional[str] = None, **target_selectors) -> None:
        """
        shape.add_ray_towards( cut_corners:bool=False, ray_color:str=<my color>, **target selectors )
        Shoot ONE ray in whichever direction hits a selected target first (straight directions get first dibs)
        """
        # Same "nothing to do" cases as add_ray
        if self.data['type'] == 'MISSING':
            return
        # (The color check lives in _shoot_rays(), like in add_ray)
        # Who are the targets? (Like move_towards: every other kwarg is a target selector.)
        # Stored as positions in scene['shapes'], since that's what the blocker grid holds
        scene_shapes:List[Dict] = self.scene['shapes']
        target_ids:set = set(id(target) for target in _select_shape_datas(scene_shapes, target_selectors))
        target_indices:set = set(index for index, shape_data in enumerate(scene_shapes) if id(shape_data) in target_ids)
        if not target_indices:
            return # nobody to shoot at
        # The origin doesn't block its own rays (see blocker_index_grid)
        blocker_grid:np.ndarray = blocker_index_grid(self.scene, origin_shape_data=self.data)
        # Straight directions first, and only if none of them hits a target, the diagonals.
        # (A diagonal can reach a target "first" even when the intended ray is a straight one)
        for direction_group in (STRAIGHT_DIRECTIONS, DIAGONAL_DIRECTIONS):
            # Every direction whose FIRST hit is a target: (distance, its rays)
            # A hit = at least ONE leading-edge pixel's ray hits a target first. Its distance is the
            # closest such pixel's, and we keep ALL that direction's rays (even the ones a non-target blocks: those
            # still get drawn, just shorter, like a bar's ray stopping at another bar on the way to the wall)
            hits:List[Tuple[int, List[Tuple[List[int], List[int], int]]]] = []
            for single_direction in direction_group:
                rays:List[Tuple[List[int], List[int], int]] = []
                target_hit_distances:List[int] = []
                for start_x, start_y, pixel_color_digit in leading_edge_pixels(self.data, single_direction):
                    ray_xs, ray_ys, blocking_shapes = trace_ray(blocker_grid, start_x, start_y, single_direction, cut_corners)
                    rays.append((ray_xs, ray_ys, pixel_color_digit))
                    if blocking_shapes & target_indices:
                        # Chebyshev distance from the origin pixel to the pixel where the target got hit = ray length + 1
                        target_hit_distances.append(len(ray_xs) + 1)
                if target_hit_distances:
                    hits.append((min(target_hit_distances), rays))
            if hits:
                # Closest one wins, random tie-break (random instead of first, like everywhere else)
                closest_distance:int = min(hit[0] for hit in hits)
                # The direction is picked once for the whole origin, then every leading-edge pixel shoots that way
                _, chosen_rays = random.choice([hit for hit in hits if hit[0] == closest_distance])
                self._shoot_rays(chosen_rays, ray_color)
                return

    ########################################################################
    ## COPYING: stamp NEW copies of a shape onto targets (the original stays as it is)
    ########################################################################

    def copy_to(self, **target_selectors) -> None:
        """
        shape.copy_to( **target selectors )
        Copy me wherever my pixels land on the most same-colored target pixels (one copy per match, greedily)
        """
        # No shape (e.g. one() found nothing)? No copies
        if self.data['type'] == 'MISSING':
            return
        # No selectors = no target, so do nothing. (An empty selector would match EVERY shape,
        # so other templates would get stamped onto each other)
        if not target_selectors:
            return
        copy_to_targets(self.data, self.scene, target_selectors)

    ########################################################################
    ## MARKING for physics, attraction, camera
    ########################################################################

    def fall(self, **kwargs):
        # A named direction ("d", "ul", "in"...) is just another way to say x and y
        if 'direction' in kwargs:
            kwargs['x'], kwargs['y'] = _direction_to_unit_step(kwargs['direction'], self.data, self.scene)
        # Don't do anything yet, just mark data in "_fall", for apply_gravity to deal with.
        self.data['_fall'] = {
            'x': int(kwargs.get('x',0)),
            'y': int(kwargs.get('y',0))
            #,
            #'passthru': kwargs.get('passthru',{})
        }

    def fall_towards(self, **kwargs):
        # Don't do anything yet, just mark data in "_fall_towards", for apply_gravity to deal with.
        # No selectors = no target, so silently do nothing (the shape doesn't move).
        # (An empty selector would match EVERY shape, so it'd fall towards the closest shape of any kind.)
        if not kwargs:
            return
        # Target selector is flat kwargs — just store them directly for apply_gravity
        self.data['_fall_towards'] = {**kwargs}

    def fall_away(self, **kwargs):
        # The opposite of fall_towards: mark data in "_fall_away", for apply_gravity to deal with.
        # No selectors = nothing to fall away from, so do nothing (same reason as fall_towards).
        if not kwargs:
            return
        self.data['_fall_away'] = {**kwargs}

    # Jigsaw time! Don't do anything yet, just flag the shape with "_moving_fit", for apply_fit to deal with
    # (same flag-now-act-later pattern as fall() & apply_gravity)
    def fit(self):
        # '_moving_fit': the flagged shapes are the MOVING ones (the rest are targets)
        self.data['_moving_fit'] = True

    # Magnets! Don't do anything yet, just flag the shape with "_moving_attract" (its target selector), for
    # apply_attract to deal with. Every kwarg except overlap is a target selector (none = every unmarked shape).
    def attract(self, **kwargs):
        # TODO: overlap=False (attract until touching, without overlapping). For now it does nothing at all
        if not kwargs.get('overlap', False):
            return
        self.data['_moving_attract'] = {key: value for key, value in kwargs.items() if key != 'overlap'}

    # Sorting! Don't do anything yet, just flag the shape with "_sort", for apply_sort to deal with
    # (same flag-now-act-later pattern as fall() & apply_gravity)
    def sort(self):
        self.data['_sort'] = True

    ########################################################################
    ## ENFORCING _STATIC (and recalculate other variables) TODO re-clean & comment & double-check
    ########################################################################

    def static(self, **kwargs):
        """(NOT ACTUALLY CURRENTLY USED; WIP)"""
        d = self.data
        static_props = d.setdefault('_static', set())
        saved = d.setdefault('_static_saved', {})

        if not kwargs:
            for prop in list(d.keys()):
                if not prop.startswith('_'):
                    static_props.add(prop)
                    saved[prop] = _copy_val(d[prop])
            return

        for prop, val in kwargs.items():
            if val:
                static_props.add(prop)
                if prop in d:
                    saved[prop] = _copy_val(d[prop])

    def static_clear(self):
        """(NOT ACTUALLY CURRENTLY USED; WIP)"""
        self.data.pop('_static', None)
        self.data.pop('_static_saved', None)

    def _enforce_static(self):
        d = self.data
        static = d.get('_static')
        saved = d.get('_static_saved')
        if not static or not saved:
            return

        # Restore all saved static values
        for prop in static:
            if prop in saved:
                d[prop] = _copy_val(saved[prop])

        # Color constraint: re-derive pixels to match restored color
        if 'color' in static and '_pixels' not in static and d.get('color', 'MISSING') != 'MISSING':
            color_digit = int(d['color'][1:])
            d['_pixels'] = np.where(d['_shadow'] == 0, np.int8(-1), np.int8(color_digit))

        # X-axis constraint: re-derive from pinned anchor + current dimensions
        if static & _X_PROPS:
            if 'center_x' in static:
                left = d['center_x'] - d['width'] / 2
                d['left'] = int(left) if left == int(left) else round(left)
                d['right'] = d['left'] + d['width']
            elif 'right' in static:
                d['left'] = d['right'] - d['width']
                d['center_x'] = d['left'] + d['width'] / 2
            elif 'left' in static:
                d['right'] = d['left'] + d['width']
                d['center_x'] = d['left'] + d['width'] / 2

        # Y-axis constraint: re-derive from pinned anchor + current dimensions
        if static & _Y_PROPS:
            if 'center_y' in static:
                top = d['center_y'] - d['height'] / 2
                d['top'] = int(top) if top == int(top) else round(top)
                d['bottom'] = d['top'] + d['height']
            elif 'bottom' in static:
                d['top'] = d['bottom'] - d['height']
                d['center_y'] = d['top'] + d['height'] / 2
            elif 'top' in static:
                d['bottom'] = d['top'] + d['height']
                d['center_y'] = d['top'] + d['height'] / 2


######################################################################
## HELPER METHODS
######################################################################

def _copy_val(val):
    """
    For safe copying by value. Do NOT modify originals.
    """
    if isinstance(val, np.ndarray):
        return val.copy()
    if isinstance(val, list):
        # for _hole_grids: list of numpy arrays
        return [item.copy() if isinstance(item, np.ndarray) else item for item in val]
    return val
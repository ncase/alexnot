# The AlexNot DSL
> Written by Claude, from talking with Nicky.

## What is this? (the ELI12)

Imagine an ARC puzzle grid as a table covered in paper cutouts: a red L here, a blue square there, a few
single dots. The solver's little language (its **DSL**, "domain-specific language") is a way of writing down
instructions like *"pick up every blue cutout and slide it down one square"* or *"paint the biggest cutout
yellow, then throw away the rest"*.

Every program is just two kinds of steps, over and over:

1. **Pick some shapes** ("all the blue ones", "the biggest one", "the ones with a hole in them").
2. **Do something to them** (move, paint, hide, rotate, drop them like they have gravity...).

The solver never writes these programs by hand. It *searches* for them: proposers look at how each demo input
differs from its output and suggest program lines (see [MONKEYS.md](MONKEYS.md) for who proposes what), and the
search keeps the **shortest program that gets every demo exactly right**. Because the programs are plain
Python, a human can read the answer and see *why* the solver thinks it's right.

## The technical picture

**A program is a Python string.** For example:

```python
for shape in all(color="c3"):
    shape.fill_holes(color="c4")
```

It gets run with `exec()` (in `arc/alexnot/transform/apply_transform.py`, `apply_transform_one`), once per grid.
A few things to know about that run:

- **No Python builtins.** The program runs with `{'__builtins__': {}}`, so `len`, `range`, `print`, `min` and
  friends don't exist. Loops, `if`, comparisons, arithmetic, `True`/`False`/`None`, and attribute access all work,
  since they're part of the language itself.
- **What's in scope:** `scene`, `all`, `one`, `replace_pixels`, `apply_gravity`, `apply_fit`, `apply_attract`,
  `apply_sort`, `change_background`, `camera_resize`, `focus`, `connect_with_lines`. Nothing else.
- **Crashes fail gracefully.** Any exception (an unknown name, a bad argument...) is caught, and the program's
  output becomes a fixed 5x5 "nothing" scene. That's guaranteed to score badly, so broken programs just lose
  the search instead of stopping it.
- **Each run gets a fresh copy** of the parsed scene, so a program can never scribble on the original.

**Scenes and shapes.** Before any program runs, the solver *parses* each grid into a **scene**: a background
color plus a list of **shapes** (connected blobs of pixels, found with whatever parser settings fit the demos
best). Then it *annotates* every shape with derived properties (size ranks, holes, symmetry, what it's
touching...). Under the hood a scene is a dict and each shape is a dict; the DSL wraps them so you can write
`scene.width` and `shape.color`.

- `all(...)` and `one(...)` hand you `Shape` wrapper objects (from `arc/alexnot/transform/dsl/shape.py`). Reading
  `shape.anything` returns that property, or `None` if the shape doesn't have it.
- `scene.anything` returns that scene property, or `0` if it doesn't exist (`scene["anything"]` works too).
- Actions change the shape dicts in place. Some (like `fall()`) only leave a note on the shape, and a separate
  top-level call (like `apply_gravity()`) acts on all the notes at once. That's the "physics" section below.

**How the output grid is made.** After the program finishes, the scene is drawn back into pixels
(`arc/alexnot/parse/reverse_parse.py`):

1. Start with a `scene.width` x `scene.height` grid filled with the background color (a scene parsed with no
   background, `"c-1"`, gets filled with color 0).
2. Draw every shape that isn't hidden, back to front: lowest `z` first, and among equal `z`, the most recently
   moved shape last (so it ends up on top). Pixels hanging off the edge are simply cut off.
3. Apply any `replace_pixels(...)` find-and-replaces, in program order.

**Colors** are always strings: `"c0"` to `"c9"`, where the digit is ARC's color index. `"c-1"` means "no color"
(for example `touching_color_most` on a shape that touches nothing). No color is special to the DSL.

**Coordinates:** `x` grows to the right, `y` grows **down**, `(0, 0)` is the top-left corner. `right` and `bottom`
are one *past* the last pixel, so `right = left + width`.

**What the search optimizes.** Every program is scored on accuracy (how close its outputs are to the demo
outputs) plus simplicity: each token costs a little, each new concept costs a bit more, and each action a shape
takes counts as an "edit" (more on edits below). So the DSL is designed so that the *simple* explanation of a
puzzle is also the *short* program.

---

## 1. Selecting shapes

The ELI12: a selector is a description like "red ones that are bigger than 3 pixels". You hand it to `all()`
and get back every shape that fits.

### `all(**selectors) -> list[Shape]`

Every shape in the scene that passes **all** the filters you give it (filters are ANDed). No filters = every shape.

```python
for shape in all(color="c2", area_LTE=3):
    shape.hide()
```

The list follows the scene's shape order, which the parser shuffles, so never rely on it. Hidden shapes are
still included (`hidden=True` is just another property you can filter on, once a shape has been hidden).

### `one(**selectors) -> Shape`

One shape that passes the filters.

```python
other = one(type="rect", holes=shape.holes)
```

- If **nothing** matches, you get a blank "MISSING" shape (`type == "MISSING"`, `color == "MISSING"`). Most actions
  quietly do nothing with it, e.g. `paint(color=other.color)` skips a `"MISSING"` color. (Actions that need its
  position, like `place()`, raise an error, and the program fails gracefully.)
- If **several** match, it picks one at random **and** adds a big edit penalty (`ONE_AMBIGUOUS_EDITS_COST`, 100,
  in `config/default.py`), once per call. That's on purpose: `one()` is for "the" shape, so an ambiguous selector
  is a dice roll the search shouldn't trust.

### Filters

Each keyword is a property name (from the list in section 2), with one of four forms:

- `prop=value`: exact match. The shape must have the property. Example: `all(type="dot")`.
- `prop_GTE=number`: the property is a number and `>= number`. Example: `all(width_GTE=5)`.
- `prop_LTE=number`: the property is a number and `<= number`. Example: `all(area_LTE=2)`.
- `prop_NOT=value`: the shape has the property and it's anything **but** this value. Example: `all(color_NOT="c5")`.

A filter on a property no shape has matches nothing. True/False properties are numbers in Python, but range
filters skip them on purpose, so use `prop=True` for those.

Values don't have to be literals. A filter can use another shape's properties or the scene's:

```python
for shape in all():
    other = one(color="c1")
    shape.place(bottom=other.bottom)        # line everything up with the c1 shape's bottom

for shape in all(width=scene.width):       # shapes as wide as the whole grid
    shape.hide()
```

---

## 2. Shape properties

The ELI12: every shape comes with a "fact sheet" (how big it is, where it is, how many holes it has, who it's
touching...). Selectors filter on these facts, and actions can read them (`shape.color`, `other.right`).

**Important:** the annotated facts are computed **once**, right after parsing. Position, size and color facts
(`left`, `top`, `right`, `bottom`, `center_x`, `center_y`, `width`, `height`, `color`, `area`, `holes`) are kept up
to date by the actions that change them. The rest (ranks, `touching_*`, `within_color`, twins, symmetry, halves...)
describe the shape **as it started**, even after it moves.

### From the parse (`arc/alexnot/parse/apply_parse.py`, `shape_coords_to_shape_dict`)

- `type`: `"dot"` (1x1), `"line"` (1 pixel thick, or a perfect diagonal), `"rect"` (at least 90% of its
  bounding-box edge filled), or `"blob"` (everything else). Shapes made by rays have `type="ray"`.
- `subtype`: lines only: `"horizontal"`, `"vertical"`, `"diag_up"` (the `/` diagonal) or `"diag_down"`. Other shapes
  don't have this property at all.
- `color`: the shape's main color (its most common pixel color; a tie is broken randomly). Changes when painted.
- `start_color`: the color it was parsed with. Never changes, so it's handy after a paint.
- `monochrome`: True if every pixel is one color.
- `left`, `top`, `right`, `bottom`: bounding box edges (`right`/`bottom` are exclusive).
- `center_x`, `center_y`: bounding box center (can be a half, like `4.5`).
- `width`, `height`: bounding box size.
- `area`: number of pixels.
- `holes`: empty pockets fully enclosed by the shape (4-connected).
- `gaps`: empty pockets that touch a side of the bounding box but none of its corners, like the opening of a "C".
- `edited`: False until an action **actually changes** the shape (see "Edits" below).
- `z`: stacking order. Every shape is `0` unless the parser stacked shapes on top of each other (then the
  backmost is `0`, the next `1`, and so on).

### Ranks (`arc/alexnot/annotate/apply_annotate.py`, `annotate_ranks`)

Ranks start at `0`, shapes with equal values share a rank, and every rank has a `_reversed` twin that counts from
the other end.

- `rank_area`, `rank_width`, `rank_height`: `0` = biggest. (`rank_area_reversed=0` = smallest.)
- `rank_right`, `rank_bottom`: `0` = right-most / bottom-most edge.
- `rank_left`, `rank_top`: `0` = left-most / top-most edge.
- `rank_x`, `rank_y`: by **center**, `0` = left-most / top-most.
- `rank_x_mod2`, `rank_x_mod3`, `rank_x_mod4` (and the same for `rank_x_reversed`, `rank_y`, `rank_y_reversed`):
  the rank modulo 2/3/4, for "every other row" rules, e.g. `all(rank_y_mod2=1)`.
- `rank_z`: `0` = frontmost. `rank_z_reversed`: `0` = backmost.
- `rank_shape_color_commonness`: `0` = the color shared by the most shapes.
- `rank_pixels_color_commonness`: `0` = the color with the most pixels (across all shapes).

Every one of these also has its `_reversed` version, e.g. `rank_top_reversed`, `rank_pixels_color_commonness_reversed`.

### Twins

- `color_twins`: how many **other** shapes have the same color.
- `shadow_twins`: how many other shapes have exactly the same silhouette (ignoring color).
- `pixels_twins`: how many other shapes are exact copies, colors included.
- `transformed_shadow_twins`, `transformed_pixels_twins`: the same, but also counting copies that are rotated or
  flipped.

```python
for shape in all(pixels_twins=0):   # the odd one out: no exact copy anywhere
    shape.paint(color="c2")
```

### Touching

"Touching" means a pixel directly left, right, above or below (not diagonal).

- `touching_border`: True if the bounding box touches the edge of the grid.
- `touching_count`: how many different shapes touch it.
- `touching_left_count`, `touching_right_count`, `touching_top_count`, `touching_bottom_count`: the same, per side.
- `touching_left_color`, `touching_right_color`, `touching_top_color`, `touching_bottom_color`: the most common
  color touching that side, weighted by area (`"c-1"` if none).
- `touching_color_most`, `touching_color_least`: the color it touches the most / least, weighted by the touched
  shapes' areas (ties broken randomly, `"c-1"` if touching nothing).
- `touching_color_other_most`, `touching_color_other_least`: the same, but only counting shapes of a **different**
  color than itself.

### Inside and outside

- `within_color`: the color of the shape whose hole this shape sits fully inside (innermost wins), or `"c-1"`.
- `contains_count`: how many shapes sit fully inside its holes.
- `contains_pixel_count`: for a one-color shape, how many pixels those inner shapes have. For a multicolor shape,
  how many of its own pixels aren't its main color.
- `over_color`: the color of the overlapping shape just **below** it in the stack (`"c-1"` if none).
- `under_color`: the color of the overlapping shape just **above** it (`"c-1"` if none).

### Dimensions and halves

- `dim_min`, `dim_max`: the smaller / bigger of width and height.
- `wider`: `1` if wider than tall, `-1` if taller than wide, `0` if square.
- `start_vhalf`: `-1` if the whole bounding box is in the top half of the grid, `1` if the bottom half, `0` if it
  straddles the middle.
- `start_hhalf`: the same for left (`-1`) and right (`1`).

### Symmetry (of the silhouette)

- `symm_h`: mirror-symmetric left-to-right (like the letter A).
- `symm_v`: mirror-symmetric top-to-bottom (like B).
- `symm_d_1`, `symm_d_2`: symmetric across one diagonal or the other. `symm_d`: either one.
- `symm_rot`: looks the same turned 180 degrees.
- `symm_flip`: any of the mirror symmetries. `symm_any`: any symmetry at all.

### Added by actions

- `hidden`: set to True by `hide()`. (Before that, shapes don't have it, so `all(hidden=False)` matches nothing.)

### Scene properties (`scene.X`)

- `scene.width`, `scene.height`, `scene.left`, `scene.top`, `scene.right`, `scene.bottom`, `scene.center_x`,
  `scene.center_y`: the grid's size and bounds.
- `scene.background`: the background color (`"c-1"` if the parse has none).
- `scene.shape_count`: number of shapes. `scene.color_count`: number of distinct shape colors.
- `scene.count_color_c0` ... `scene.count_color_c9`: how many shapes have that color.
- `scene.count_color_pixels_c0` ... `scene.count_color_pixels_c9`: how many shape pixels have that color.
- `scene.most_common_nonbg_color`, `scene.most_common_nonbg_color_2nd`, `scene.most_common_nonbg_color_3rd`,
  `scene.least_common_nonbg_color`, `scene.least_common_nonbg_color_2nd`, `scene.least_common_nonbg_color_3rd`:
  shape colors ranked by pixel count (only as many as there are colors; ties broken randomly).
- `scene.shapes`: the raw list of shape dicts (rarely useful in a program; use `all()`).

```python
for shape in all():
    shape.place(bottom=scene.height)   # everything sits on the floor
```

---

## 3. Shape actions

The ELI12: once you've picked a shape, these are the verbs. Each one is a method: `shape.verb(...)`.

**Edits.** Each action call adds 1 to the shape's edit count (the search's "how much did you have to change"
cost), and sets `edited=True` only if something **really** changed (moving to where it already is, or painting a
shape its own color, doesn't count). That's what makes `all(edited=True)` mean "whatever actually moved".

Most actions take keyword arguments only. Directions everywhere are `"u"`, `"d"`, `"l"`, `"r"` and the diagonals
`"ul"`, `"ur"`, `"dl"`, `"dr"`.

### Color and visibility

- **`shape.paint(color)`**: recolor the shape's originally parsed pixels (an outline ring or filled holes added
  later keep their color).
  `shape.paint(color="c4")`
- **`shape.hide()`**: remove the shape from the output. (It's still in the scene, so `all()` still sees it.)
  `shape.hide()`
- **`shape.fill_holes(color)`**: fill every enclosed hole with that color. Updates `area` and sets `holes` to 0.
  `shape.fill_holes(color="c4")`
- **`shape.outline(color, line_width=1, ignore_holes=False)`**: grow a ring of that color around the shape
  (square corners). The shape gets bigger by `line_width` on every side. `line_width` is capped at 5.
  `ignore_holes=True` draws it as if the holes were filled, so nothing gets painted inside them.
  `shape.outline(color="c7")`

### Moving

None of these check for collisions: shapes can slide on top of each other (the drawing order decides who's
visible). For "move until you bump into something", see the physics section.

- **`shape.place(left=, right=, top=, bottom=, center_x=, center_y=)`**: move to an absolute position, by
  whichever edge (or center) you give.
  `shape.place(left=shape.right)`  (jump right by its own width)
- **`shape.move(x=0, y=0)`**: move by a relative amount.
  `shape.move(y=1)`
- **`shape.move(direction, steps=1)`**: move `steps` pixels in a named direction. Besides the 8 compass directions,
  `"in"` heads towards the grid's center (per axis; an axis it's already centered on stays put) and `"out"` heads
  away from it.
  `shape.move(direction="dr", steps=2)`
- **`shape.move(ux=, uy=, steps=1)`**: the same, with the unit step spelled out as numbers.
  `shape.move(ux=1, uy=0, steps=3)`
- **`shape.move(z=)`, `shape.move(max_z=)`**: change the stacking order. `z=1` / `z=-1` steps forward / back,
  `max_z=1` jumps to the very front, `max_z=-1` to the very back.
  `shape.move(max_z=1)`
- **`shape.move_towards(steps=1, **target_selectors)`**: take `steps` steps towards the closest target (closest =
  smallest gap between pixels, ties random), straight or diagonal. Every keyword except `steps` is a target
  selector, so `color=` picks the **target's** color.
  `shape.move_towards(color="c5", steps=1)`
- **`shape.move_away(steps=1, **target_selectors)`**: one straight step (l/r/u/d) away from the targets. If it's
  touching targets, each touching (target, side) pair pushes it the other way, and the total push is rounded to a
  straight direction. If it's not touching any, it heads away from the closest one.
  `shape.move_away(color="c6", steps=1)`
- **`shape.pathfind(passthru=None, **target_selectors)`**: wind through the gaps between shapes (any of 8 directions,
  not just a straight line) to end up as close as possible to the nearest target, by walking distance. Other
  shapes' pixels are walls, except the `passthru` color, which it may walk onto.
  `shape.pathfind(color="c2", passthru="c2")`

### Turning

Both pivot around the shape's center.

- **`shape.rotate(quarters=1)`**: rotate clockwise by 90-degree steps (`2` = 180, `3` = 270).
  `shape.rotate(quarters=2)`
- **`shape.flip(mode="horizontal")`**: mirror the shape. Modes: `"horizontal"` (left-right), `"vertical"`
  (top-bottom), `"diagonal_1"` (across the top-right to bottom-left diagonal), `"diagonal_2"` (across the top-left
  to bottom-right diagonal, a transpose).
  `shape.flip(mode="vertical")`

### Growing

The new pixels become part of the **same** shape.

- **`shape.extend(direction, steps=1, extend_color=None)`**: grow the shape's leading pixels (the outermost pixel
  on each row/column going that way) `steps` pixels further. No `extend_color` = each leading pixel's own color.
  `shape.extend(direction="r", steps=2, extend_color="c3")`
- **`shape.extend_towards(extend_color=None, steps=1, **target_selectors)`**: the same, towards the closest target
  (any of 8 directions).
  `shape.extend_towards(color="c5")`
- **`shape.extend_away(extend_color=None, steps=1, **target_selectors)`**: the same, away from the targets (using
  `move_away`'s rule).
  `shape.extend_away(rank_area=0, extend_color="c4", steps=2)`

### Making new shapes

These add **brand-new** shapes to the scene. The original stays as it is.

- **`shape.add_ray(direction, ray_color=None, cut_corners=False, from_center=False)`**: shoot a 1-pixel-wide ray
  from every leading-edge pixel until it hits another (non-ray) shape or the edge of the grid. Each ray is a new
  shape with `type="ray"`, colored like the pixel it came from unless `ray_color` is given. Besides single
  directions, `direction` takes the shortcuts `"lr"`, `"ud"`, `"+"` (4 straight), `"x"` (4 diagonal) and `"all"` (8).
  Rays pass through each other and through their own origin. `from_center=True` fires just one ray per direction,
  from the edge pixel in line with the shape's center (`"all"` always does this). `cut_corners=True` lets a diagonal
  ray slip between two shapes that only touch at their corners.
  `shape.add_ray(direction="ud")`
- **`shape.add_ray_towards(cut_corners=False, ray_color=None, **target_selectors)`**: shoot rays in the **one**
  direction whose rays hit a target first (straight directions are tried before diagonals; ties random).
  `shape.add_ray_towards(color="c5", ray_color="c3")`
- **`shape.copy_to(**target_selectors)`**: stamp copies of the shape wherever its pixels land on the most
  same-colored target pixels, greedily, until no spot covers anything new. A spot where any pixel would land on a
  differently-colored target pixel is ruled out. No selectors = no copies. Each copy has `edited=True`.
  `shape.copy_to(area_LTE=2)`

---

## 4. Physics: mark now, act later

The ELI12: some actions only make sense for everybody **at once**. If a stack of blocks falls, the bottom one has
to land before the one above it knows where to stop. So you first put a sticky note on every shape that should
fall ("you're falling down"), then call `apply_gravity()` once, and everyone falls together.

The pattern is always:

```python
for shape in all(...):
    shape.<mark>(...)
apply_<physics>(...)
```

Each `apply_*()` uses up the marks, and counts the whole thing as **one** edit per shape that actually moved.

### Gravity

- **`shape.fall(x=0, y=0)`** or **`shape.fall(direction)`**: mark the shape to fall in that direction (each of
  `x`/`y` is clipped to -1..1; `direction` takes the 8 compass names plus `"in"`/`"out"`).
  `shape.fall(direction="d")`
- **`shape.fall_towards(**target_selectors)`**: mark it to fall towards the closest target that's in straight or
  diagonal line of sight. (No line of sight = it stays put.)
  `shape.fall_towards(color="c5")`
- **`shape.fall_away(**target_selectors)`**: mark it to fall away from the targets (`move_away`'s rule, no line of
  sight needed).
  `shape.fall_away(color="c6")`
- **`apply_gravity()`**: every marked shape slides one pixel at a time until the next step would leave the grid or
  hit another shape. Shapes it was already overlapping at the start don't block it, and two shapes both using
  plain `fall()` in different directions pass through each other.

```python
for shape in all():
    shape.fall(y=1)
apply_gravity()
```

### Jigsaw fitting

- **`shape.fit()`**: mark the shape as a moving piece.
- **`apply_fit()`**: every marked shape slides into the gaps of the unmarked (visible) shapes, like jigsaw pieces,
  scored by how snugly corners and edges pinch together. A search over placement orders picks the best overall
  arrangement, and each placed piece becomes something the next one can fit against. (Translation only, no turning.)

```python
for shape in all(rank_area_GTE=1):
    shape.fit()
apply_fit()
```

### Magnets

- **`shape.attract(overlap=True, **target_selectors)`**: mark the shape to snap onto its targets so their pixels
  overlap, colors matching (like a red dot landing on a red dot). No selectors = every unmarked shape is a target.
  `overlap=True` is required: without it, the call currently does nothing.
- **`apply_attract()`**: snaps the marked shapes onto the targets, one at a time, rare colors counting the most.
  A spot that would put a pixel on a different color isn't allowed. If no targets are left (everyone is marked),
  the biggest marked shape stays put and becomes the first target.

```python
for shape in all(width_GTE=5):
    shape.attract(overlap=True)
apply_attract()
```

### Sorting

- **`shape.sort()`**: mark the shape for sorting.
- **`apply_sort(key, direction, descending=False, spacing=0)`**: stack the marked shapes in order of the property
  `key` (e.g. `"area"`), in `direction` `"u"`, `"d"`, `"l"` or `"r"`, starting from the old group's edge on that
  side, with `spacing` empty pixels between them. Each shape keeps its position on the other axis. Ties keep their
  old order. `key` and `direction` are required (without them nothing happens).

```python
for shape in all():
    shape.sort()
apply_sort(key="area", direction="u", descending=True, spacing=1)
```

---

## 5. Scene-level functions

These are called on their own, not on a shape (and usually not inside a loop).

- **`focus(padding=0, hide_non_matches=False, one_match=False, hug=False, margin=0, **selectors)`**: crop the grid
  to the bounding box of every shape matching the selectors (hidden shapes don't count), plus `padding` on each
  side (negative padding trims). `hide_non_matches=True` hides everyone else. `one_match=True` frames just one
  random match. `hug=True` squishes the matches side by side along their longer direction with no gaps (and hides
  everyone else); `hug="2d"` packs them into a grid of cells instead, with `margin` empty pixels between cells.
  `focus(rank_area=0)`
- **`camera_resize(width=, height=)`**: set the output grid's size (clamped to 1..30, ARC's limits). Shapes keep
  their positions, measured from the top-left.
  `camera_resize(width=scene.width, height=3)`
- **`change_background(color)`**: change the background color.
  `change_background(color="c1")`
- **`connect_with_lines(directions, line_color=None, same_color=True, **selectors)`**: for every pair of selected
  shapes whose centers line up exactly (same row, same column, or exactly 45 degrees), draw a new 1-pixel line shape
  in the gap between them. `directions` is required: `"all"`, `"+"`, `"x"`, `"lr"`, `"ud"`, `"diag_up"` or
  `"diag_down"`. `same_color=True` only joins same-colored pairs. No `line_color` = the shapes' color. Lines go
  behind the shapes they connect and are never blocked.
  `connect_with_lines(directions="x", color="c1")` (here `color="c1"` is a selector: join the `c1` shapes, in their own
  color; `line_color="c2"` would draw the lines in `c2` instead)
- **`replace_pixels(old, new)`**: a find-and-replace on the final rendered pixels. `old` and `new` are same-sized 2D
  lists of color digits (`0`-`9`); `-1` in `old` means "match anything", and `-1` in `new` means "leave this pixel
  alone". It's recorded and applied after everything is drawn, wherever it sits in the program. Every matched spot
  counts as one edit.
  `replace_pixels([[1, 1]], [[-1, 2]])`

(`rotate_scene` exists in `scene_transforms.py` but is not in the program namespace, so programs can't call it.)

---

## 6. Complete examples

Each of these was checked on the training set: parsed, run, and compared pixel-for-pixel against every demo
output. The levels are ARC-AGI-1 *training* puzzles.

**Recolor by size** (`6e82a1ae`). Every shape is the same color; the biggest ones turn one color, the middle ones
another, the smallest a third. Ranks do the work, so the program doesn't care what the sizes actually are:

```python
for shape in all(rank_area=0):
    shape.paint(color="c1")
for shape in all(rank_area=1):
    shape.paint(color="c2")
for shape in all(rank_area=2):
    shape.paint(color="c3")
```

**Everything falls** (`1e0a9b12`). Scattered pixels drop to the bottom of the grid and pile up on each other.
This is exactly why gravity is "mark now, act later": the order the pixels land in matters.

```python
for shape in all():
    shape.fall(direction="d")
apply_gravity()
```

**Crop to the big one** (`1f85a75f`). The output is just the largest shape, cut out of the noise around it:

```python
focus(rank_area=0)
```

**Fill the enclosed spaces** (`00d62c1b`). Closed loops get their insides filled with a new color:

```python
for shape in all(color="c3"):
    shape.fill_holes(color="c4")
```

**Paint and nudge** (`a79310a0`). One shape changes color and moves down by one, two actions on the same shape:

```python
for shape in all():
    shape.paint(color="c2")
    shape.move(y=1)
```

For more (z-ordering, rays, copying, pathfinding, magnets, hugging...), see `DSL_EXAMPLES.md` at the repo root.

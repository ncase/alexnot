# The Monkeys
> Written by Claude, from talking with Nicky.

The "monkeys" are the idea generators inside the solver's program search. This page explains what they are, how
they plug into the search, and what every single one of them proposes. Grab a snack, there are 33 of them. 🐒

---

## The short version (ELI12)

Imagine you're handed a few before/after pictures and asked to write down the rule that turns each "before" into
its "after". Now imagine you have a room full of little helpers, and each one knows exactly **one trick**:

- One helper only ever notices "hey, that thing changed color!"
- Another only notices "that thing slid over!"
- Another only notices "a line got drawn from that dot!"
- Another only ever looks at the rules people *already* wrote, and says "what if we deleted that bit?"

Every round, each helper looks at the best rule so far, sees what it still gets wrong, and **shouts out
suggestions** for how to fix it. Most suggestions are bad. That's fine! A judge tries every suggestion on the
example pictures and keeps the rules that are **right** and **short**. Then the helpers look at the new best rules
and shout again. After a few rounds, the shortest rule that gets every example right wins.

The helpers are the monkeys (as in "a room full of monkeys at typewriters", except these monkeys are a lot less
random). The judge is the scorer.

---

## The technical picture

### Where monkeys sit in the solver

The solver works in three stages:

1. **Parse**: turn each grid into a *scene*, a list of shapes (each with pixels, position, color, and so on).
2. **Annotate**: add derived properties to every shape and scene (area, holes, symmetry, rank by size, what it's
   touching...).
3. **Transform search**: an evolutionary search for a short program in the DSL (a small Python-flavored language)
   that turns every demo input scene into its demo output. This is where the monkeys live.

A DSL program looks like this:

```python
for shape in all(color="c5"):
    shape.paint(color="c2")
for shape in all(area_LTE=2):
    shape.hide()
```

`all(...)` selects shapes by their properties, and the shape methods (`paint`, `move`, `hide`, `fall`, ...) act on
them. See [DSL.md](DSL.md) for every selector and action. Colors are written `"c0"` to `"c9"`, and no color is
special: the background is whatever the parse decided it was for that level.

### The evolutionary loop

The search lives in `arc/alexnot/transform/search_transform.py`. With the default config, it goes like this:

1. **Start** with the blank program `''` (it does nothing, so the predicted output is just the input). It's the only
   parent in round 0.
2. **Every round:**
   1. **Propose.** Every monkey in the `MONKEYS` list gets called with the demo pairs and the current parents,
      and returns a set of complete new program strings. Programs already tried (or already known to crash) are
      filtered out, and each monkey's list is randomly capped at `MONKEY_CAP_START` (1000).
   2. **Score.** Each monkey's proposals are shuffled and scored until that monkey's wall-clock budget runs out
      (`CLOCK_BUDGET_PER_MONKEY`, 0.3 s per round, or 0.4 s for the three busiest monkeys: `shapes_repaint`,
      `shapes_move` and `shapes_gravity`). So a monkey that proposes 5,000 programs doesn't get 5,000 programs
      scored, it gets a random slice. A program that's too long to possibly become a parent (even with zero
      wrong pixels) is skipped without being run. A program that crashes on any demo is dropped.
   3. **Pick parents** ("islands"), all de-duplicated:
      - the best program by **total score** (this is also where the final winner comes from),
      - the best program by **shape-diff score** (right shapes in roughly the right spots, even if the pixels
        aren't right yet), and
      - the best program on **each demo pair** separately (so a program that nails demo 2 but not demo 1 can
        still be built on).
   4. **Early stop** if the best total score hasn't improved for 2 rounds (`EVO_EARLY_STOP_COUNTDOWN`), or after
      10 rounds (`EVO_MAX_GENERATIONS`).
3. **Return** every scored program, best first. A program only counts as a solution if it gets 100% of the pixels
   right on every demo.

A monkey that throws an exception just proposes nothing that round (the traceback is logged), so one buggy monkey
can't sink a whole search.

### What a monkey receives

Every monkey has the same signature:

```python
def some_monkey(train_pairs: list[dict], parent_program_results: list[dict]) -> set[str]:
```

- **`train_pairs`**: the parsed, annotated demo pairs. Each has an `input` scene and an `output` scene. Each scene
  has its `shapes`, its size, its background color, its scene-level properties, and its raw pixel grid (`_raw`).
- **`parent_program_results`**: this round's parents (usually 2 to 5 of them). Each one has the `program` string,
  plus what running it produced: `predicted_scenes` (one per demo, each with its own raw `_raw` grid),
  `predicted_pixels`, and its scores (`total_score`, `pair_scores`, `total_wrong_pixels`, ...).

The key idea is **execution-guided** proposing: most monkeys compare the parent's **predicted** scene against the
real demo output, not the raw input against the output. So if the parent already recolored the right shapes,
the repaint monkey has nothing left to say about them, and the other monkeys focus on what's still wrong.

### What a monkey returns

A set of **complete programs**, as strings. There are three flavors of monkey:

- **Appenders** (most of them): the parent program, plus one new block at the end. E.g. the parent, then
  `for shape in all(color="c5"):\n\tshape.paint(color="c2")`.
- **Editors**: the parent program with something removed, reordered or rewritten (the compression monkeys and
  the selector re-roller).
- **Generalizers**: the parent program with a hard-coded value swapped for a reference to some property
  (the coincidence monkeys). These turn "move down by 3" into "move down by your own height".

### How proposals get scored (the judge)

The scorer is in `arc/alexnot/transform/score_transform.py`. Lower is better. Roughly:

- **Accuracy**: 25 points per wrong pixel, summed over every demo (a wrong-sized grid counts the missing/extra
  area as wrong too).
- **Simplicity**: the program's token cost. Every word costs about 1 (some words have hand-set prices). A
  hard-coded value costs more: `0`, `1`, `-1`, `True` and `False` are free, colors are cheap, but a "magic number"
  like `7` costs 5. On top of that, every unique concept and every unique line (with the literal values blanked
  out) costs 1.5 each.
- **Edits**: 2 points per actual edit a program makes to the shapes, so "undo and redo" programs lose.

This is basically MDL (minimum description length): the winner is the program that explains the demos best in
the fewest, most reusable words. It's also why the coincidence monkeys matter so much: swapping the magic number
`7` for `scene.height` is *cheaper*, and it generalizes to test inputs of a different size.

---

## Shared machinery

A few helpers show up in almost every monkey. Knowing them makes the monkey descriptions much shorter.

### The selector proposer ("the Bongard function")

**ELI12:** You point at a few shapes and say "these ones". The helper figures out a short description that picks
exactly those shapes and not the others, like "the grey ones" or "the ones with holes" or "the biggest one".

**Technical:** `_propose_selectors(ingroup, all_shapes, N)` in `arc/alexnot/utils/selector_helpers.py`. It picks
a random shape from the *ingroup* (the shapes we want) and builds candidate selectors from that shape's own
property values:

- the empty selector (`all()`, picks everyone),
- every single property (`color="c3"`, `holes=1`, `rank_area=0`...),
- every pair of properties (`color="c3",type="line"`),
- **range** selectors for numbers (`area_LTE=2`, `width_GTE=5`), using the ingroup's own min or max,
- **NOT** selectors (`color_NOT="c4"`) when one value describes exactly everyone *except* the ingroup.

Each candidate scores "ingroup shapes it picks, minus outgroup shapes it picks, minus 0.2 per key", with tiny
random tie-breaks, and the top `N` come back. On top of that, up to `N` extra single-property selectors that
grab the *whole* ingroup (even if they also grab some outgroup shapes) come along, because sometimes "all the
lines" is the true rule and the extra lines just didn't need to change.

Every value in a selector is a **literal** read off a real shape. Selectors never invent values.

### The ShapeActionBatcher

**ELI12:** Lots of shapes need the same fix ("paint red"). Instead of writing one rule per shape, the batcher
collects every shape that needs "paint red", then asks the Bongard function for one description that covers
them all.

**Technical:** `ShapeActionBatcher` in `arc/alexnot/transform/monkeys/monkey_helpers.py`. A monkey calls
`batcher.add(shape, 'paint(color="c2")')` for every (shape, action) it notices, across **all** demos (so the
selector has to work in every demo at once). Some actions also take a target shape:
`batcher.add(shape, 'move_towards(steps=2)', target)`. Then `batcher.flush()` returns code:

- It keeps the top 5 actions by how many pixels they'd affect (`BATCHER_TOP_N_PIXELS`).
- For a plain action, it proposes 10 actor selectors, each giving
  `for shape in all(<selector>):\n\tshape.<action>`.
- For an action with a target, it proposes 5 actor selectors × 5 target selectors, and splices the target
  selector into the action's own arguments: `shape.move_towards(steps=2, color="c5")`. If the action reads a
  property off the target (`place(top=other.top)`), the target gets its own line instead:
  `other = one(<target selector>)`.

Many monkeys use **more than one batcher**, so rarer but important ideas (like "paint with your own neighbor's
color") don't get crowded out of the top 5 by common ones.

### Moved-shape matching

`_get_moved_shape_pairs(before_scene, after_scene)` in `shape_change.py` pairs each predicted shape with output
shapes that have the **same silhouette** (and, for multicolor shapes, the same pixels) but a different position
or color. Shapes already sitting correctly are skipped, and if one shape could match several output shapes, only
the matches with the fewest changes are kept. The move, gravity, fit, attract, sort and pathfind monkeys all start
from these pairs.

### Line of sight

`_shapes_in_line_of_sight(scene, shape, dx, dy)` (in `arc/alexnot/grid/`) finds the shapes lying in a direction
from a shape. The move and gravity monkeys use it to guess *what* a shape moved towards (look ahead of it) or away
from (look behind it).

### Scenes before a block

The editing and generalizing monkeys need to know what the shapes looked like *just before* a given block ran
(earlier blocks may have moved or recolored them). `scenes_before_block(program, position)` in `monkey_helpers.py`
returns those scenes, reusing the search's own scored results when it can and re-running the program prefix
otherwise.

### Literals vs `scene.X` vs `shape.X` vs `other.X`

Where do the values in a proposal come from? This is the heart of how the solver generalizes, so here's the rule
of thumb:

- **Literals** (`"c3"`, `4`, `True`) come straight from the diff: the color a shape turned, how far it moved, a
  property of a real shape. Almost every appender monkey proposes literals.
- **`scene.X`** (a property of the whole scene, like `scene.width` or `scene.most_common_nonbg_color`) comes
  from either the move monkey (when a shape lands exactly on the scene's edge or center) or the
  `scene_prop_coincidence` monkey.
- **`shape.X`** (the acting shape's own property, like `shape.height`) comes from the coincidence monkeys
  (`self_coincidence`, `join_coincidence`), with a few deliberate exceptions inside the appender monkeys, all
  listed below: `paint(color=shape.<prop>)`, moves by `shape.<prop>`, and `color=shape.color` targets on
  `move_towards` / `fall_towards`.
- **`other.X`** (a property of some other shape picked with `one(...)`) comes from `any_coincidence`,
  `join_coincidence` and `center_coincidence`, plus the batcher when an action reads off a target.

---

## Every monkey

All 33 monkeys registered in the `MONKEYS` list in `search_transform.py`, grouped by the kind of change they
explain. File paths are relative to `arc/alexnot/transform/monkeys/`.

### Recoloring

**`shapes_repaint`** (`shape_change.py`)
- **Notices:** a predicted shape whose silhouette shows up in the same spot in the output, but in a different
  color. It also checks pixel by pixel, so a recolored shape that *merged* with a same-colored neighbor in the
  output still counts.
- **Proposes:** `paint` blocks via two batchers: a hard-coded new color, and (separately) the shape's own
  property that happens to hold the new color (like "paint yourself the color of whatever you touch most"). The
  second kind lets one rule give different shapes different colors.
- **Example:** `shape.paint(color="c2")` or `shape.paint(color=shape.touching_color_most)`

**`shapes_fill_holes`** (`shape_change_fancy.py`)
- **Notices:** a shape with holes whose silhouette still exists in the output, but the most common color inside
  its holes changed.
- **Proposes:** filling the holes with the new hole color.
- **Example:** `shape.fill_holes(color="c4")`

**`shapes_outline`** (`shape_change_fancy.py`)
- **Notices:** a shape that survived completely unchanged, but is now hugged by a brand-new output shape (a ring
  around it), or whose 1-pixel ring in the output is all one new color.
- **Proposes:** an outline in the ring's color, with `line_width` = the biggest gap between the shape's box and
  the ring's box (capped at 5), both with and without `ignore_holes=True`. Only the *tightest* ring around each
  shape is used.
- **Example:** `shape.outline(color="c8", line_width=1)`

**`change_background`** (`scene_change.py`)
- **Notices:** the predicted scene's background color differs from the output's.
- **Proposes:** one top-level line changing the background to the output's.
- **Example:** `change_background(color="c1")`

### Hiding and stacking

**`shapes_hide`** (`shape_change.py`)
- **Notices:** a visible, on-camera predicted shape whose exact silhouette is *not* in the same spot in the output.
- **Proposes:** hiding it.
- **Example:** `shape.hide()`

**`shapes_restack`** (`shape_change.py`)
- **Notices:** a shape that's identical in both (same spot, color and silhouette) but ranks differently in the
  stacking order (which shape is drawn on top).
- **Proposes:** three moves in whichever direction it went: one step, two steps, or all the way.
- **Example:** `shape.move(z=1)`, `shape.move(z=-2)`, `shape.move(max_z=1)`

**`shapes_bring_to_front`** (`shape_change.py`)
- **Notices:** looks at up to 3 random wrong pixels. If some shape underneath one of them already has the
  output's color at that exact spot, it's probably just covered up.
- **Proposes:** bringing that shape to the very front.
- **Example:** `shape.move(max_z=1)`

### Moving

**`shapes_move`** (`shape_change.py`)
- **Notices:** moved shapes (from the moved-shape matching above).
- **Proposes:** lots of ways to describe the same move, using four batchers so the rarer ones get a fair shot:
  - absolute placement by one edge: `place(left=4)`, and by the scene's size or center if the edge lands exactly
    there: `place(right=scene.width)`,
  - placement on both axes at once (one of left/right/center_x × one of top/bottom/center_y), or one center axis
    alone: `place(center_x=scene.center_x,bottom=7)`,
  - a relative move, naming only the axes that change: `move(x=2,y=-1)`,
  - the same move as a named direction, when it's a straight or 45° line (`"u"`, `"dr"`, and also `"in"` /
    `"out"`, towards or away from the scene's center): `move(direction="dr",steps=3)`,
  - a move by one of the shape's own whole-number properties, for steps of 2 or more:
    `move(x=0,y=shape.height)` (or `-shape.height` for the other way),
  - moving towards a shape in its line of sight: `move_towards(steps=3, color="c5")`, and towards something of
    its *own* color: `move_towards(steps=1,color=shape.color, touching_border=True)`,
  - moving away from the closest shape behind it: `move_away(steps=2, rank_area=0)`.

**`shapes_gravity`** (`shape_change.py`)
- **Notices:** the same moved shapes.
- **Proposes:** physics instead of exact placement. Each block is followed by `apply_gravity()`, which lets every
  flagged shape fall until it hits something:
  - fall in the direction it moved (clamped to one step per axis): `fall(x=0,y=1)`,
  - fall towards a shape in its line of sight: `fall_towards(color="c3")`, or towards its own color:
    `fall_towards(color=shape.color, type="line")`,
  - fall away from the closest shape behind it: `fall_away(rank_area=0)`.
- **Example:**
  ```python
  for shape in all(color="c4"):
      shape.fall(x=0,y=1)
  apply_gravity()
  ```

**`shapes_fit`** (`shape_change.py`)
- **Notices:** a moved shape that wasn't overlapping anyone's bounding box before, but is inside someone's box
  now: a jigsaw piece slotting into a gap. Only for demos where the grid is already the right size (crop first,
  fit second).
- **Proposes:** `fit()` on those shapes, then `apply_fit()`.
- **Example:** `shape.fit()` followed by `apply_fit()`

**`shapes_attract`** (`shape_change.py`)
- **Notices:** a moved shape that, in the output, sits on top of another shape where every overlapping pixel has
  the same color in both (like a red dot snapping onto a red dot).
- **Proposes:** magnets: `attract(overlap=True)` on those shapes, then `apply_attract()`.
- **Example:** `shape.attract(overlap=True)` followed by `apply_attract()`

**`shapes_pathfind`** (`pathfind_change.py`)
- **Notices:** a shape that moved along a *bent* path (not straight, not 45°) and ended up touching (or sitting
  on) a shape it wasn't touching before.
- **Proposes:** pathfinding towards that shape, with `passthru` set to any color it ended up standing on.
- **Example:** `shape.pathfind(color="c2", passthru="c2")`

**`shapes_sort`** (`sort_change.py`)
- **Notices:** 3 or more shapes moving in one demo, where their new order along an axis lines up with one of
  their number properties in every demo (area, height, their old position...), and one edge of the group stayed
  put.
- **Proposes:** marking the moved shapes with `sort()`, then one `apply_sort(...)` line per fitting key,
  direction and spacing (the most common gap between neighbors).
- **Example:**
  ```python
  for shape in all():
      shape.sort()
  apply_sort(key="area", direction="u", descending=True, spacing=1)
  ```

### Reshaping and copying

**`shapes_transform`** (`shape_change.py`)
- **Notices:** a predicted shape that isn't in the output as-is, but some output shape is exactly a rotation or
  flip of it (checked on both the silhouette and the colored pixels).
- **Proposes:** every rotation or flip that matches (a 90° turn is proposed both as `quarters=1` and
  `quarters=-3`).
- **Example:** `shape.rotate( quarters=1 )` or `shape.flip( mode="horizontal" )`

**`shapes_copy`** (`shape_change.py`)
- **Notices:** an output shape that's pixel-for-pixel identical to a predicted shape but somewhere else, sitting
  on top of another predicted shape with color-matching overlap (a stamp landing on its marker).
- **Proposes:** `copy_to` with a target selector for the markers.
- **Example:** `shape.copy_to(area_LTE=2)`

### Drawing new things

**`shapes_add_ray`** (`ray_change.py`)
- **Notices:** probes every direction from every shape's leading-edge pixels. A direction "fires" if the next few
  empty pixels that way (up to 5, stopping at anything in the way) are filled with that pixel's color in the
  output.
- **Proposes:** `add_ray` with the set of directions that fired (as a shortcut like `"+"` or `"x"` when it fits,
  borrowing directions that were blocked right away to complete a shortcut), diagonal variants with
  `cut_corners=True`, a `from_center=True` version, and `add_ray_towards` aimed at whatever stopped each ray.
- **Example:** `shape.add_ray(direction="ud")` or `shape.add_ray_towards(color="c5")`

**`shapes_connect_with_lines`** (`line_change.py`)
- **Notices:** pairs of shapes that line up exactly (same row, same column, or exactly 45°), where the empty gap
  between them is filled with one color in the output.
- **Proposes:** one top-level `connect_with_lines(...)` call, with a selector for the connected shapes,
  `directions` as both `"all"` and the narrowest setting that covers every line seen, `line_color` when the
  lines aren't the shapes' own color, and `same_color=False` when some connected pair had two colors.
- **Example:** `connect_with_lines(line_color="c1", directions="x", color="c3")`

**`replace_pixels`** (`pixels_change.py`)
- **Notices:** works on raw pixels instead of shapes. It finds connected patches of changed pixels, and looks for
  2 or more identical patches (across all demos). For those, it works out what those spots had in common in the
  predicted grid, cell by cell.
- **Proposes:** a pixel find-and-replace, with and without a 1-pixel ring of context around it (`-1` means "don't
  care").
- **Example:** `replace_pixels([[-1,5,-1],[5,5,5]], [[-1,2,-1],[2,2,2]])`

### Camera (grid size and cropping)

**`camera_resize`** (`camera_change.py`)
- **Notices:** the predicted grid's width and/or height differs from the output's.
- **Proposes:** resizing the camera to the output's width, height, or both.
- **Example:** `camera_resize(width=3,height=3)`

The four focus monkeys all propose the same kind of line, `focus(<selector>, padding=N, ...)`, which crops the
camera to the selected shapes. They differ in **how they decide which shapes** the camera should be on. Each
selector gets combined with each padding (the smallest and biggest gap between the output's shapes and its edges)
and each "spotlight" option: plain, `one_match=True`, `hide_non_matches=True`, both, `hug=True` (squish the
matches together), and `hug="2d"` with a margin (only if every demo agrees on that margin).

**`focus`** (`camera_change.py`)
- **Notices:** a size change. The ingroup is every predicted shape whose silhouette appears in the output. It
  also checks if the output is exactly the *inside* of some frame-shaped shape, and if so focuses on the frame with
  a negative padding to shave it off.
- **Example:** `focus(symm_v=True, padding=0)` or `focus(type="rect", padding=-1)`

**`focus_by_sliding_window`** (`camera_change.py`)
- **Notices:** slides the output's pixels over the predicted grid to find the best-matching crop window(s). The
  ingroup is the shapes fully inside the window.
- **Example:** `focus(rank_area=0, padding=1, hide_non_matches=True)`

**`focus_on_window_edge`** (`camera_change.py`)
- **Notices:** the same crop windows, but the ingroup is only the shapes inside the window that *touch its edge*
  (for "these markers show where to crop" puzzles, where other shapes inside the window would make "everything
  inside" impossible to select).
- **Example:** `focus(color="c8", padding=0)`

**`focus_by_output_size`** (`camera_change.py`)
- **Notices:** never looks at the output's pixels, only its size. For each padding from -3 to 3, the ingroup is the
  shapes whose box, padded that much, is exactly the output's size, and it needs a match in *every* demo. (For
  "crop to the region under this shape" puzzles, where the output matches nothing in the input.)
- **Example:** `focus(color="c3", padding=-1)`

### Compression and editing

These don't look at the diff at all. They tidy up programs that other monkeys built.

**`delete_lines`** (`compression.py`)
- **Proposes:** the parent with any one line removed.

**`delete_blocks`** (`compression.py`)
- **Proposes:** the parent with any one whole `for` block (the loop plus its body) removed.

**`delete_arguments`** (`compression.py`)
- **Proposes:** the parent with any one argument removed from any one call. This is how an overfit selector like
  `all(color="c5",area=3)` becomes `all(color="c5")`.

**`permute_code_blocks`** (`compression.py`)
- **Proposes:** the parent with its top-level blocks reordered. Order matters because later shapes are drawn on top
  of earlier ones. Programs with up to 4 blocks get every ordering, longer ones get 23 random orderings.

**`reroll_selectors`** (`synonym.py`)
- **Notices:** for each `all(...)`, `one(...)` and action target selector in the parent, which shapes it actually
  picks (in the scenes just before that block runs).
- **Proposes:** the same program with just that one selector swapped for another description of the same shapes
  (20 alternatives per selector, from the Bongard function). Joins like `holes=shape.holes` and non-selector
  arguments like `steps=2` or `padding=1` are kept as they were.
- **Example:** `all(height_GTE=3)` → `all(multicolor=True)`, if those pick the same shapes.

### Coincidence monkeys (generalizing hard-coded values)

**ELI12:** Say the best rule so far is "move every dot down 3". Why 3? Maybe it's a coincidence... or maybe 3 is
exactly the height of the dot, or the width of the grid, or the top of the blue box. These monkeys go looking for
that "why", and suggest the rule that reads the number off the thing that explains it. If the explanation is right,
it's cheaper (no magic number) and it works on new inputs too.

**Technical:** each of these looks at the parent's best-scoring demo pair, in the scene just before the block runs.
Most of them only rewrite blocks of the exact form `for shape in all(...):` followed directly by one `shape.action(...)`
line. A value only counts as "hard-coded" if it's a string, or a number of 2 or more (in absolute value), since a 0
or 1 matching some property is nearly always a fluke. Negatives count too: `move(y=-3)` can become
`move(y=-shape.height)`.

**`self_coincidence`** (`causality.py`)
- **Notices:** a hard-coded action value that equals one of the *acting shape's own* properties.
- **Proposes:** reading it off the shape. If that property was also in the loop's selector, the selector part is
  dropped, since it now just restates the value.
- **Example:** `for shape in all(area=5): shape.move(y=5)` → `for shape in all(): shape.move(y=shape.area)`

**`any_coincidence`** (`causality.py`)
- **Notices:** a hard-coded action value that equals some property of *any* shape in the scene.
- **Proposes:** an `other = one(...)` line selecting that shape, and the value read off it (for every property
  that holds the value), with both the loop's own selector and plain `all()`.
- **Example:**
  ```python
  for shape in all():
      other = one(color="c1")
      shape.place(top=other.top)
  ```

**`join_coincidence`** (`causality.py`)
- **Notices:** a hard-coded action value where *every* acting shape has its own "instruction" shape: an unselected
  shape holding the value, which shares some property with it (same row, same number of holes...).
- **Proposes:** a join: `other = one(...)` with that shared property tied to the acting shape, then the value read
  off `other`. Also proposed with the loop's selector generalized (minus the join property, and plain `all()`).
- **Example:**
  ```python
  for shape in all(color="c5"):
      other = one(left=0, rank_bottom=shape.rank_bottom)
      shape.paint(color=other.color)
  ```

**`center_coincidence`** (`causality.py`)
- **Notices:** a `place(center_x=X,center_y=Y)` with both centers hard-coded, where (X, Y) is exactly the center of
  the scene or of some shape. (`any_coincidence` only swaps one value at a time, so it can't do both axes.)
- **Proposes:** reading both off the scene, or off that shape via `other = one(...)`.
- **Example:** `shape.place(center_x=other.center_x,center_y=other.center_y)`

**`scene_prop_coincidence`** (`causality.py`)
- **Notices:** *any* hard-coded value anywhere in the program (selectors included, but not numbers inside a pixel
  grid literal) that equals a property of the whole scene.
- **Proposes:** the program with that value swapped for `scene.<property>`. A non-number (like a color) is only
  swapped if every demo's scene has that property as the same kind of value.
- **Example:** `place(bottom=10)` → `place(bottom=scene.height)`, or
  `paint(color="c3")` → `paint(color=scene.most_common_nonbg_color)`

---

## How to see what a monkey proposes

The quickest way to get a feel for a monkey is to run it on one level and look at its best proposals. From the
`arc/` folder:

```bash
cd arc
../.venv/bin/python -m alexnot.tests.test_code_monkey
```

Open `arc/alexnot/tests/test_code_monkey.py` first and edit its **CHOOSE** section at the top:

- **the level**: a path under `arc/levels/`, e.g. `LEVELS_DIR / "arc-1-train/00d62c1b.json"`,
- **the parse config**: background, diagonals, multicolor and so on (`parse/` has the options),
- **the monkey**: e.g. `monkey = shape_change_fancy.shapes_fill_holes`,
- **the parent program**: `PARENT_PROGRAM = ''` to act like round 0 of the search, or any DSL program to see what
  the monkey would add on top of it (handy for the coincidence and compression monkeys, which need a parent with
  something in it).

It parses and annotates the level, scores the parent, runs that one monkey, scores everything it proposed, and
prints the top 10 by total score (with wrong pixels and token cost), then the best program's predictions next to
the expected outputs. For example, `shapes_fill_holes` on the training level above, from the blank program, prints
proposals like:

```
  #0 | score=404.1 | wrong_px=15 | tokens=9.6
       for shape in all(holes_NOT=0):
       	shape.fill_holes(color="c4")
```

(Not perfect yet, and that's normal: in the real search, other monkeys and later rounds would build from there.)

Run with `ARC_DEBUG=1` for debug logging. To watch the *whole* search instead, with every monkey and every round,
try `../.venv/bin/python -m alexnot.tests.test_search_transform`, and look at the reasoning trace: each round lists
how many programs every monkey proposed and how many got scored, and which monkey suggested each parent.

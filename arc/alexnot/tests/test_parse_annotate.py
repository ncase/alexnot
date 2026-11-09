import json
import sys
import numpy as np
from pathlib import Path

# So it works in both pytest & VSCode Debug
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.parse.apply_parse import apply_parse_one_grid
from alexnot.parse.reverse_parse import reverse_parse_one
from alexnot.transform.apply_transform import apply_transform_one
from alexnot.annotate.apply_annotate import apply_annotate_one_scene

ROOT = Path(__file__).resolve().parent.parent.parent
LEVELS_DIR = ROOT / "levels"

####################################################################################

def _print_scene(scene):
    # Pretty-print one scene so we can eyeball the parse
    # (Plus the scene's shape_count & color_count annotations: shape_count should match the len() count,
    # and the most/least common non-background colors, by their shapes' total pixels)
    print(f"{scene['width']}x{scene['height']}, bg={scene['background']}, {len(scene['shapes'])} shapes, "
          f"shape_count={scene.get('shape_count','?')}, color_count={scene.get('color_count','?')}, "
          f"most_common_nonbg_color={scene.get('most_common_nonbg_color','?')}, least_common_nonbg_color={scene.get('least_common_nonbg_color','?')}")
    # Shape colors ranked by their shapes' total pixels: 1st/2nd/3rd most common, then 3rd/2nd/1st least common
    # ('?' = missing, when the scene has fewer than 3 distinct shape colors)
    print("    shape colors by pixels: most=" + "/".join(scene.get(name,'?') for name in
              ('most_common_nonbg_color', 'most_common_nonbg_color_2nd', 'most_common_nonbg_color_3rd')) +
          ", least=" + "/".join(scene.get(name,'?') for name in
              ('least_common_nonbg_color_3rd', 'least_common_nonbg_color_2nd', 'least_common_nonbg_color')))
    for i, shape in enumerate(scene["shapes"]):
        print(
            # (Only lines have a subtype: "horizontal", "vertical", "diag_up" or "diag_down")
            f"    [{i}] type={shape['type']:<5} subtype={shape.get('subtype',''):<10} color={shape['color']:<3} "
            # f"section={shape.get('section','?'):<2} " # Off: sections are disabled, so this would always print '?'
            f"bounds=({shape['left']},{shape['top']})-({shape['right']},{shape['bottom']}) "
            f"area={shape['area']} holes={shape['holes']} "
            f"color_twins={shape.get('color_twins','?')} "
            f"rank_area={shape.get('rank_area','?')} "
            # Position ranks: rank_top=0 / rank_left=0 should be the top-most / left-most shape,
            # rank_bottom=0 / rank_right=0 the bottom-most / right-most one
            f"rank_top={shape.get('rank_top','?')} rank_bottom={shape.get('rank_bottom','?')} "
            f"rank_left={shape.get('rank_left','?')} rank_right={shape.get('rank_right','?')} "
            # Color ranks: by shape count, then by pixel count
            f"rank_shape_color_commonness={shape.get('rank_shape_color_commonness','?')}/{shape.get('rank_shape_color_commonness_reversed','?')} "
            f"rank_pixels_color_commonness={shape.get('rank_pixels_color_commonness','?')}/{shape.get('rank_pixels_color_commonness_reversed','?')} "
            # Which color's hole is this shape inside? ("c-1" = not inside any shape's hole)
            f"within_color={shape.get('within_color','?')} "
            # How many shapes sit in my holes, and how many "inner" pixels do I have?
            f"contains_count={shape.get('contains_count','?')} contains_pixel_count={shape.get('contains_pixel_count','?')} "
            # Stacking: the color of the shape just under me / just over me ("c-1" = nothing there)
            f"z={shape.get('z','?')} over_color={shape.get('over_color','?')} under_color={shape.get('under_color','?')}"
        )

####################################################################################
####################################################################################

if __name__ == "__main__":

    #######################################################################
    ## CHOOSE THE LEVEL
    #######################################################################

    #path = LEVELS_DIR / "arc-2-train/5034a0b5.json"  # move 1 to own color
    #path = LEVELS_DIR / "arc-1-train/0b148d64.json"  # color_twins test
    #path = LEVELS_DIR / "arc-1-train/1f85a75f.json"  # crop to big 'un
    # path = LEVELS_DIR / "arc-3-test/m0r0.json"  # mirror maze (crashes in the touching-color annotation: it has colors above 9)
    # Rank check: shapes laid out top/bottom/left/right of a middle square, so the position ranks are easy
    # to eyeball (Use with "bg_mode": "c0" below, diagonals on, multicolor off)
    path = LEVELS_DIR / "concept-arc/HorizontalVertical8.json"

    #######################################################################
    ## CHOOSE THE PARSE PROGRAM
    #######################################################################

    # Parse config
    parse_config = {
        "bg_mode": "c0", # For the rank check above ("no_bg" below isn't read by the parser, bg_mode is)
        "no_bg": False,
        "diagonals": True,
        "columns": True,
        "rows": True,
        "multicolor": False,
    }

    #######################################################################
    ## NOW parse & annotate all scenes in that level
    #######################################################################

    # Load level
    with open(path, "r") as file:
        level = json.load(file)
    
    # Test parse against ALL train (input,output) and test (input) scenes
    all_grids = []
    for pair in level["train"]:
        all_grids.append( np.array( pair['input'], dtype=np.int8 ) ) # don't forget to numpify
        all_grids.append( np.array( pair['output'], dtype=np.int8 ) )
    for pair in level["test"]:
        all_grids.append( np.array( pair['input'], dtype=np.int8 ) )

    # Parse & Annotate all
    results = []
    for grid in all_grids:
        parsed_scene = apply_parse_one_grid(grid, parse_config)
        annotated_scene = apply_annotate_one_scene(parsed_scene) # DON'T FORGET TO ANNOTATE!
        results.append(annotated_scene)

    # Print stats
    print(f"\nParsed {path.name} with config {parse_config}\n")
    for i, parsed_scene in enumerate(results):
        print(f"Grid {i}, parsed:")
        _print_scene(parsed_scene)
        print()


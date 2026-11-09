import logging

from .dsl.top_level_methods import DSL_CONFIG, all, one, replace_pixels
from .dsl.lines import connect_with_lines
# Physics lives in one file per kind (gravity, fitting, attract, sort)
from .dsl.physics_gravity import apply_gravity
from .dsl.physics_fitting import apply_fit
from .dsl.physics_attract import apply_attract
from .dsl.physics_sort import apply_sort
from .dsl.scene_transforms import rotate_scene, change_background, camera_resize, focus

# Wrapper class to let DSL programs use "scene.prop_name" instead of scene["prop_name"].
# Returns 0 for missing properties so programs fail gracefully instead of crashing.
class SceneWrapper:
    def __init__(self, scene_dict:dict):
        # Stash the underlying dict where DSL_CONFIG and reverse_parse can still reach it
        object.__setattr__(self, '_dict', scene_dict)

    def __getattr__(self, prop_name:str):
        return self._dict.get(prop_name, 0)

    def __setattr__(self, prop_name:str, value):
        self._dict[prop_name] = value

    # Keep dict-style access working too, so existing scene["prop"] programs don't break
    def __getitem__(self, key:str):
        return self._dict.get(key, 0)

    def __setitem__(self, key:str, value):
        self._dict[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self._dict

    def items(self):
        return self._dict.items()

    def get(self, key:str, default=None):
        return self._dict.get(key, default)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

_NO_BUILTINS = {'__builtins__': {}}
_NOTHING_SCENE = {
    'width': 5,
    'height': 5,
    # Same bound names as a parsed scene, consistent with the 5x5 size
    'left': 0,
    'top': 0,
    'right': 5,
    'bottom': 5,
    'center_x': 2.5,
    'center_y': 2.5,
    'background': "c2",
    'shapes': []
}

def _copy_shape(shape:dict) -> dict:
    """
    Copy a shape so a running program can't scribble on the original parsed scene (which we reuse
    to score every candidate program). A shallow .copy() handles all the immutable scalar/string
    props; we just deep-copy the MUTABLE grids on top. Manual list-comps, NOT copy.deepcopy --
    this runs 1000s of times per level (once per candidate program, per pair), and deepcopy's
    memo + recursion is way too slow for that hot path.
    """
    copy:dict = shape.copy() # scalars & string bindings: cheap, and all we need for those
    copy['_pixels'] = shape['_pixels'].copy() # numpy 2D array: .copy() is a full deep copy
    copy['_shadow'] = shape['_shadow'].copy() # numpy 2D array: .copy() is a full deep copy
    copy['_hole_grids'] = [g.copy() for g in shape['_hole_grids']] # list of numpy 2D arrays (usually empty)
    # No copy needed for '_original_shadow': nothing ever writes into it in place (every op rebinds a fresh array)
    return copy

def apply_transform(program:str, scenes:list[dict]) -> list[dict]:
    return [apply_transform_one(program,scene) for scene in scenes]

# RUN PROGRAM: SCENE => TRANSFORMED SCENE
def apply_transform_one(program:str, scene:dict) -> dict:
    """
    Apply a program to Scenes in bulk

    Args:
        program (str): The program to run
        scene (Scene): The pre-parsed Scene

    Returns:
        Scene, after transformation (it modifies the scene in-place anyway)
    """ 

    # Make scene globally accessible (copy to NOT edit the original parsed scene).
    # Wrap in SceneWrapper so DSL programs can use scene.prop_name dot notation.
    scene_dict = {**scene, 'shapes': [_copy_shape(s) for s in scene['shapes']]}
    DSL_CONFIG['scene'] = SceneWrapper(scene_dict)

    # OKAY LET'S EXECUTE THE PROGRAM
    namespace = {

        'scene': DSL_CONFIG['scene'],

        'all': all,
        'one': one,
        'replace_pixels': replace_pixels, # Records pixel find-and-replaces, applied by reverse_parse at the end
        'apply_gravity': apply_gravity,
        'apply_fit': apply_fit, # Jigsaw: every moving (shape.fit()-flagged) shape slides into the target shapes' gaps
        'apply_attract': apply_attract, # Magnets: every shape.attract()-flagged shape snaps onto its targets, colors matching
        'apply_sort': apply_sort, # Sorting: every shape.sort()-flagged shape gets stacked in order of a property
        'change_background': change_background,
        'camera_resize': camera_resize,
        'focus': focus,
        'connect_with_lines': connect_with_lines, # Joins every exactly-lined-up pair of selected shapes with a new line shape

        #'rotate_scene': rotate_scene,

    }
    try:
        exec(program.strip(), _NO_BUILTINS, namespace)
    except Exception as e:
        logging.debug(f"Program exec failed: {type(e).__name__}: {e}")
        logging.debug(f"Failed program was:\n{program}")
        DSL_CONFIG['scene'] = _NOTHING_SCENE

    # Gimme
    return DSL_CONFIG['scene'] # pyright: ignore[reportReturnType] it's fine shut up
    
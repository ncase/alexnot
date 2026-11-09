# Physics is split into one file per kind: physics_gravity.py (apply_gravity), physics_fitting.py (apply_fit & its cache),
# physics_attract.py (apply_attract & its cache), physics_sort.py (apply_sort), plus physics_utils.py (small helpers).
# This file just re-exports some of them, so "from .physics import ..." lines keep working.
from .physics_gravity import apply_gravity
from .physics_fitting import apply_fit, clear_fit_cache
from .physics_attract import apply_attract, clear_attract_cache

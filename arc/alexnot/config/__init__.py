"""
The solver's config: picks ONE config file from this folder, once, at import time.
Set the env var ALEXNOT_CONFIG to the file's name (default: "default" = default.py) BEFORE alexnot gets imported,
e.g. `ALEXNOT_CONFIG=double_time python ...`, or BLAST_EM's `--config double_time`.
Then any file can just do `from ..config import TIME_LIMIT_PER_ATTEMPT` (no namespaces needed).
"""

import os
import importlib

# Import every knob from default.py *statically* first, so editors (Pylance) know all their names & types.
# (Every config starts from default anyway, and the chosen config overrides these values just below.)
from .default import *

# Which config file? (An unknown name crashes loudly right here, with a ModuleNotFoundError)
CONFIG_NAME:str = os.environ.get("ALEXNOT_CONFIG", "default")
_config_module = importlib.import_module(f".{CONFIG_NAME}", __name__)

# Re-export all its ALL_CAPS knobs, as if they were defined right here
globals().update({name: value for name, value in vars(_config_module).items() if name.isupper()})

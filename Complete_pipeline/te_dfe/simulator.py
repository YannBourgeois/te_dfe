"""
simulator.py — model dispatcher for the SLiM simulator.

Re-exports DFEBurstSimulator (and the module helpers) for the active model.
Both simulators expose the same call signature and now return a UNIFORM
4-tuple per window: (ts, covariate, recomb_map, gene_intervals). The simple
model's 4th element (gene_intervals) is used by background selection and
ignored by the plain simple processor; the context model uses it for the
dist-to-gene channel. This uniform interface lets the pipeline scripts unpack
windows the same way regardless of model.

NOTE: the realism batch (Q-scaling, background selection, coverage mask,
dead-pixel missingness, polarization, sharding hooks) currently lives in the
simple simulator. Porting it to simulator_context.py is a clean follow-up
(edit that one file); the dispatch keeps both models working in the meantime.
"""
from .config import MODEL_NAME as _MODEL

if _MODEL == "context":
    from .simulator_context import *      # noqa: F401,F403
    from . import simulator_context as _active
else:
    from .simulator_simple import *       # noqa: F401,F403
    from . import simulator_simple as _active

for _name in dir(_active):
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = getattr(_active, _name)

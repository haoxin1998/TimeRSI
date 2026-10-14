"""Complete candidate entry point. Copy and edit; this is NOT an auto-selected seed.

Inspect actual field semantics before populating BINDINGS. Both numerical changes
and changes to the whole algorithm are allowed; probe the resulting program.
Calendar priors and geographic assumptions are absent unless YOU justify them.
"""
from .event_structure import (
    FieldBindings, StructureConfig, predict_structure,
)

BINDINGS = FieldBindings(
    open_column=None,
    event_columns=(),
    categorical_columns=(),
    continuous_columns=(),
)
CONFIG = StructureConfig()


def predict(view, tools, memory):
    return predict_structure(view, tools, memory, bindings=BINDINGS, config=CONFIG)

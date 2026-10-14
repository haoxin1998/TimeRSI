"""Complete candidate, selected only after establishing cumulative semantics.

Unlike example_flow.py, this writes a new CPU statistical forecaster. The agent
may change its projector, uncertainty, period or blend; no dataset router chooses
this method automatically. No previously saved predictions are dependencies.
"""
from .cumulative_specialist import CumulativeConfig, predict_cumulative

CONFIG = CumulativeConfig()


def predict(view, tools, memory):
    return predict_cumulative(view, tools, memory, CONFIG, native_weight=0.0)

"""Complete editable candidate, not an automatically enabled task solution.

Set the explicit field bindings after reading their meaning, or author inputs()
to derive exposure and regressors yourself. None exposure means the explicit
unit-exposure hypothesis; it does NOT infer any field from its name. This is a
statistical expert by default. Optionally test role-separated native fusion.
"""
import numpy as np
from .short_history_exposure import ShortExposureConfig, predict_short_exposure
from .native_residual_fusion import ResidualFusionConfig

EXPOSURE_COLUMN = None
REGRESSOR_COLUMNS = ()
NORMAL_COLUMN = None
USE_NATIVE_FUSION = False
CONFIG = ShortExposureConfig()
FUSION = ResidualFusionConfig()


def _known(item, name):
    raw = item.get('known_raw', {})
    return np.asarray(raw[name] if name in raw else item['known'][name], float)


def inputs(item):
    length = item['y'].shape[1]
    exposure = (np.ones(len(item['timestamps'])) if EXPOSURE_COLUMN is None
                else _known(item, EXPOSURE_COLUMN))
    regressors = {name: _known(item, name) for name in REGRESSOR_COLUMNS}
    # For an explicitly supplied normal-state indicator, only known future values
    # are used. Otherwise this example proposes that all positive-exposure future
    # positions are ordinary. CHANGE this assumption when events affect them.
    normal = (exposure[length:] > 0 if NORMAL_COLUMN is None
              else _known(item, NORMAL_COLUMN)[length:].astype(bool))
    return {'exposure': exposure, 'regressors': regressors, 'normal_mask': normal}


def predict(view, tools, memory):
    return predict_short_exposure(view, tools, memory, inputs, CONFIG,
                                  native_fusion=FUSION if USE_NATIVE_FUSION else None)

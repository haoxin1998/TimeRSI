"""Editable representation hypotheses: rate/exposure and cumulative/flow.

Neither is selected by dataset name. The agent must establish target semantics
and test the hypothesis on R. `cumulative_flow` quantiles use a comonotonic sum,
NOT an exact distribution of a sum; compare against Raw and calibrate separately.
"""
from dataclasses import replace
import numpy as np
from timersi_u1.core import Forecast


def exposure_rate(view, tools, memory, exposure_column, minimum_exposure=1e-6):
    """Model level / known exposure, then restore the forecast's original units.

    Examples of semantics, not column-name rules: demand / operating capacity,
    energy / available capacity, or count / population. Future exposure must be a
    supplied known field. Zero exposure implies zero target only when the agent
    has established that domain constraint. This is NOT valid for net signed flow.
    """
    items = []; future_scales = []
    for item in view.items:
        length = item["y"].shape[1]
        values = np.asarray(item["known"][exposure_column], float)
        if values.shape != (length + view.horizon,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Exposure must be a nonnegative known context+horizon field")
        eligible = np.flatnonzero(values[:length] > minimum_exposure)
        if not len(eligible):
            raise ValueError("Exposure normalization requires positive historical exposure")
        rates = np.stack([np.interp(np.arange(length), eligible,
                                   target[eligible] / values[eligible]) for target in item["y"]])
        # Keep other covariates. Remove only the scale channel already factored out.
        known = {k: v for k, v in item["known"].items() if k != exposure_column}
        known_raw = {k: v for k, v in item.get("known_raw", {}).items() if k != exposure_column}
        items.append({**item, "y": rates, "known": known, "known_raw": known_raw})
        future_scales.append(values[length:])
    forecast = tools.timesfm(replace(view, items=items))
    scale = np.asarray(future_scales)[:, :, None]
    return Forecast(forecast.point * scale, forecast.quantiles * scale[..., None]).validate()


def cumulative_flow(view, tools, memory, nonnegative_increment=False, context=15360):
    """Forecast increments, then integrate from each target's final visible level.

    Use for an actual accumulated total, not an arbitrary trending time series.
    Negative corrections in cumulative records may be meaningful: the default
    permits them. Setting nonnegative_increment=True encodes an explicit monotone
    process hypothesis. Native covariates retain their matching final timestamps.
    """
    items = []
    last = []
    for item in view.items:
        increments = np.diff(item["y"], axis=1)
        if not increments.shape[1]:
            raise ValueError("Flow hypothesis requires at least two historical samples")
        items.append({**item, "y": increments, "timestamps": item["timestamps"][1:],
                      "known": {k: np.asarray(v)[1:] for k, v in item["known"].items()},
                      "past": {k: np.asarray(v)[1:] for k, v in item["past"].items()},
                      "known_raw": {k: np.asarray(v)[1:] for k, v in item.get("known_raw", {}).items()},
                      "past_raw": {k: np.asarray(v)[1:] for k, v in item.get("past_raw", {}).items()}})
        last.append(item["y"][:, -1])
    flow = tools.timesfm(replace(view, items=items), context=context, signed=not nonnegative_increment)
    point = np.maximum(flow.point, 0) if nonnegative_increment else flow.point
    quantiles = np.maximum(flow.quantiles, 0) if nonnegative_increment else flow.quantiles
    last = np.asarray(last)[:, None, :]
    return Forecast(last + np.cumsum(point, axis=1),
                    last[..., None] + np.cumsum(quantiles, axis=1)).validate()

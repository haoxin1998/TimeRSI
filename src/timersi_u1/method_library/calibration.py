"""Optional post-hoc repair from completed historical forecasts only.

This is editable and opt-in. It is not a reason to modify every dataset. Use
source='self' for candidate-specific memory or source='raw' for a Raw correction;
do not silently calibrate a transformed expert with another model's residuals.
"""
import numpy as np
from timersi_u1.core import Forecast


def residual_calibration(forecast, view, tools, memory, source="raw", weight=0.5, max_windows=6):
    """Mix predictive quantiles with the historical residual distribution.

    Horizon residuals are pooled within each entity/target. This is a simple
    stationary-residual hypothesis, not a universal coverage guarantee. The
    caller's memory contains only fully matured windows; repeated evaluations of
    one origin are deduplicated. No current/future labels enter this operation.
    """
    if not 0 <= weight <= 1:
        raise ValueError("Calibration weight must be in [0, 1]")
    if source == "self":
        source = getattr(tools, "active_program", None)
    point = forecast.point.copy(); quantiles = forecast.quantiles.copy()
    for i, item in enumerate(view.items):
        for d, target in enumerate(view.targets):
            records = {}
            for row in memory:
                if row["source"] != source or item["id"] not in row["ids"] or target not in row["targets"]:
                    continue
                j = row["ids"].index(item["id"]); k = row["targets"].index(target)
                records[np.datetime64(row["ends"][j])] = row["truth"][j, :, k] - row["point"][j, :, k]
            selected = sorted(records.items())[-max_windows:]
            if not selected:
                continue
            errors = np.concatenate([np.asarray(values) for _, values in selected])
            errors = errors[np.isfinite(errors)]
            if not len(errors):
                continue
            shift = float(np.median(errors))
            residual_q = np.quantile(errors, np.arange(1, 10) / 10)
            point[i, :, d] += weight * shift
            alternate_q = forecast.point[i, :, d, None] + residual_q[None, :]
            quantiles[i, :, d] = (1 - weight) * quantiles[i, :, d] + weight * alternate_q
    return Forecast(point, quantiles).validate()

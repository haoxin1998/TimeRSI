"""Editable damped-increment specialist for semantically cumulative totals.

This is intentionally more than a native input transform: the agent may replace
the foundation forecast with a new statistical model. A recent increment level
and a damped growth estimate are integrated from the last observed total. Its
uncertainty is fitted by re-running that same projector at earlier origins inside
the supplied item's past. No named dataset, stored winner or old run is consulted.

The default hypothesis is a nondecreasing accumulated quantity; negative revisions
in the input may occur but future increments are nonnegative. Do not select this
model merely because an ordinary level series trends upward.
"""
from dataclasses import dataclass
import numpy as np
from timersi_u1.core import Forecast


ZQ = np.array([-1.28155157, -.84162123, -.52440051, -.2533471,
                0., .2533471, .52440051, .84162123, 1.28155157])


@dataclass(frozen=True)
class CumulativeConfig:
    # None uses the view's seasonality; set explicitly when semantics suggest
    # another aggregation period. These units are observations, not always days.
    period: int | None = None
    level_window: int | None = None
    trend_periods: int = 3
    trend_recency: float = 0.6
    trim_fraction: float = 0.1
    growth_shrink: float = 0.8
    growth_bias: float = 0.0
    growth_lower: float = -0.08
    growth_upper: float = 0.08
    damping: float = 0.97
    backtest_origins: int = 6
    backtest_recency: float = 0.8
    sigma_prior_scale: float = 0.2
    sigma_prior_power: float = 0.5
    sigma_prior_weight: float = 4.0
    sigma_multiplier: float = 1.0
    sigma_min: float = 0.03
    sigma_max: float = 0.7
    pool_target_uncertainty: bool = True


def _mean(values, trim_fraction):
    values = np.sort(np.asarray(values)[np.isfinite(values)])
    if not len(values):
        return np.nan
    trim = int(len(values) * trim_fraction)
    if trim:
        values = values[trim:-trim]
    return float(np.mean(values))


def project_increments(increments, horizon, config, period):
    """Expose the proposed future flow, recent level and per-step growth rate."""
    increments = np.asarray(increments, float)
    window = min(config.level_window or period, len(increments))
    if window < 1:
        raise ValueError("Increment projector needs at least one historical increment")
    level = _mean(increments[-window:], config.trim_fraction)
    if not np.isfinite(level):
        level = _mean(increments[-3 * window:], config.trim_fraction)
    if not np.isfinite(level):
        level = 0.0
    means = []
    for j in range(config.trend_periods + 1):
        hi = len(increments) - j * period; lo = hi - period
        if lo < 0:
            break
        means.append(_mean(increments[lo:hi], config.trim_fraction))
    ratios = []; weights = []
    for j, (recent, previous) in enumerate(zip(means, means[1:])):
        if np.isfinite(recent) and np.isfinite(previous) and min(recent, previous) > 1e-12:
            ratios.append(np.log(recent / previous) / period)
            weights.append(config.trend_recency ** j)
    growth = float(np.average(ratios, weights=weights)) if ratios else 0.0
    growth = float(np.clip(config.growth_shrink * growth + config.growth_bias,
                           config.growth_lower, config.growth_upper))
    h = np.arange(1, horizon + 1, dtype=float)
    if not 0 <= config.damping <= 1:
        raise ValueError("Damping must lie in [0, 1]")
    effective_steps = (config.damping * (1 - config.damping ** h) / (1 - config.damping)
                       if config.damping < 1 else h)
    # Recent aggregate is centred before the origin; account for its age.
    offset = (window - 1) / 2 * config.damping
    level = max(level, 0.0)
    return level * np.exp(growth * (effective_steps + offset)), level, growth


def internal_backtests(history, horizon, config, period):
    """Causal projector residuals with explicit per-origin receipts.

    The forecast at origin o reads only history[:o]. Its outcomes come from
    history[o:], which is already visible at the real deployment origin. Partial
    historical horizons are allowed and only contribute their observed steps.
    """
    history = np.asarray(history, float)
    length = len(history)
    minimum = max(3 * period + 1, 8)
    origins = list(range(length - period, minimum - 1, -period))[:config.backtest_origins]
    residuals = {}; receipts = []
    for order, origin in enumerate(origins):
        past = history[:origin]
        if not np.isfinite(past[-1]):
            continue
        steps = min(horizon, length - origin)
        flow, level, growth = project_increments(np.diff(past), steps, config, period)
        rise = np.cumsum(flow)
        actual = history[origin:origin + steps] - past[-1]
        stabilizer = max(0.5 * level, 1.0)
        weight = config.backtest_recency ** order
        for step, (observed, predicted) in enumerate(zip(actual, rise), start=1):
            if np.isfinite(observed):
                error = np.log((max(observed, 0) + stabilizer) / (predicted + stabilizer))
                residuals.setdefault(step, []).append((float(error), weight))
        receipts.append({"origin": origin, "training_stop_exclusive": origin,
                         "validation_stop_exclusive": origin + steps,
                         "anchor": float(past[-1]), "level": level, "growth": growth,
                         "predicted_rise": rise.copy()})
    return residuals, receipts


def sigma_profile(residuals, horizon, config):
    """Fit a regularized horizon-dependent uncertainty profile on log total rise."""
    h = np.arange(1, horizon + 1, dtype=float)
    prior = config.sigma_prior_scale * h ** config.sigma_prior_power
    observed_h = []; observed_sigma = []; observed_weight = []
    for step, values in sorted(residuals.items()):
        errors = np.array([value for value, _ in values])
        weights = np.array([weight for _, weight in values])
        observed_h.append(step)
        observed_sigma.append(max(float(np.sqrt(np.average(errors ** 2, weights=weights))), 1e-3))
        observed_weight.append(float(weights.sum()))
    if len(observed_h) >= 3:
        x = np.log(observed_h); y = np.log(observed_sigma); weights = np.array(observed_weight)
        xc = x - np.average(x, weights=weights); yc = y - np.average(y, weights=weights)
        denominator = float(np.sum(weights * xc ** 2))
        slope = float(np.sum(weights * xc * yc) / denominator) if denominator > 1e-12 else config.sigma_prior_power
        slope = np.clip(slope, 0.0, 1.0)
        intercept = float(np.average(y - slope * x, weights=weights))
        fitted = np.exp(intercept) * h ** slope
        data_weight = weights.sum() / (weights.sum() + config.sigma_prior_weight * horizon)
        profile = np.exp(data_weight * np.log(fitted) + (1 - data_weight) * np.log(prior))
    else:
        profile = prior
    return np.clip(config.sigma_multiplier * profile, config.sigma_min, config.sigma_max)


def forecast_item(item, horizon, config, period):
    """Fit/project each target; optionally pool normalized uncertainty within item."""
    values = np.asarray(item["y"], float).copy()
    if "observed" in item:
        values[~np.asarray(item["observed"], bool)] = np.nan
    series = []; all_residuals = {}; receipts = []
    for target in values:
        if len(target) < 2 or not np.isfinite(target[-1]):
            raise ValueError("Cumulative specialist requires a visible final anchor")
        flow, level, growth = project_increments(np.diff(target), horizon, config, period)
        residuals, audit = internal_backtests(target, horizon, config, period)
        series.append((target[-1], flow, level, growth, residuals))
        receipts.append(audit)
        for step, rows in residuals.items():
            all_residuals.setdefault(step, []).extend(rows)
    points = []; quantiles = []; profiles = []
    for anchor, flow, level, growth, residuals in series:
        sigma = sigma_profile(all_residuals if config.pool_target_uncertainty else residuals, horizon, config)
        rise = np.cumsum(flow)
        stabilizer = max(0.5 * level, 1.0)
        point = anchor + rise
        quantile = anchor + np.maximum((rise[:, None] + stabilizer) * np.exp(sigma[:, None] * ZQ)
                                        - stabilizer, 0)
        points.append(point); quantiles.append(quantile); profiles.append(sigma)
    return np.stack(points, axis=1), np.stack(quantiles, axis=1), {
        "period": period, "sigma": np.stack(profiles), "backtests": receipts,
        "growth": [series[d][3] for d in range(len(series))],
    }


def predict_cumulative(view, tools, memory, config=CumulativeConfig(), native_weight=0.0):
    """Full candidate program: new expert alone or explicit convex native blend.

    A foundation-model call is not compulsory: native_weight=0 emits this newly
    written model directly. The runtime still compares the candidate against Raw
    through the same research/evaluation API.
    """
    if not 0 <= native_weight <= 1:
        raise ValueError("Native weight must lie in [0, 1]")
    period = config.period or max(1, int(view.seasonality))
    outputs = [forecast_item(item, view.horizon, config, period) for item in view.items]
    result = Forecast(np.stack([p for p, _, _ in outputs]),
                      np.stack([q for _, q, _ in outputs])).validate()
    if native_weight:
        return tools.blend([result, tools.timesfm(view)], [1 - native_weight, native_weight])
    return result

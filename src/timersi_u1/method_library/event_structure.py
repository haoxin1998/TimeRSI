"""A complete, editable multiplicative structure + foundation-model expert.

The agent binds fields by their *meaning* after inspecting the task. The mechanism
is generic: exposure/operating state, events interacting with day of week, closure
proximity, smooth monthly structure, robust fitting, and reversible normalization.
No dataset names, pretrained per-task coefficients, or old forecasts are used.

Use for nonnegative levels with a plausible multiplicative structure. Signed or
very sparse series need a different hypothesis, not an automatic log transform.
All fitting uses the target history of the supplied view; future inputs may only
come from supplied known covariates and calendars. Optional calendar priors are
explicit coefficients authored by the agent, EMPTY by default.
"""
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from timersi_u1.core import Forecast


@dataclass(frozen=True)
class FieldBindings:
    # Explicit semantics, e.g. set to the actual known operating-state column.
    open_column: str | None = None
    event_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    continuous_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructureConfig:
    recent: int = 730
    half_life_days: float = 130.0
    ridge: float = 0.15
    robust_iterations: int = 2
    huber_c: float = 1.75
    min_observations: int = 30
    month_harmonics: int = 2
    year_harmonics: int = 0
    hour_harmonics: int = 0
    trend_damping: float = 0.5
    expert_weight: float = 0.5
    spread_mix: float = 0.6
    empirical_quantile_mix: float = 0.0
    spread_scale: float = 1.0
    horizon_growth: float = 0.0
    normalize: bool = True
    point_log_shift: float = 0.0
    residual_median_shift: float = 0.0
    # Names refer to columns from extra_features(). NO built-in regional dates.
    calendar_priors: Mapping[str, float] = field(default_factory=dict)
    prior_strength: float = 4.0


def closure_proximity(open_values):
    """Known operating-path features, without assumptions about weekend rules.

    Looking one or two steps forward is valid only because `open_values` is an
    officially known future field. Unknown positions beyond this view are never
    manufactured as open/closed observations.
    """
    closed = np.asarray(open_values) <= 0.5
    before = np.zeros(len(closed), bool)
    after = before.copy(); before_two = before.copy(); after_two = before.copy()
    before[:-1] = closed[1:]
    after[1:] = closed[:-1]
    before_two[:-2] = closed[1:-1] & closed[2:]
    after_two[2:] = closed[:-2] & closed[1:-1]
    open_now = ~closed
    return {
        "before_closed": (open_now & before).astype(float),
        "after_closed": (open_now & after).astype(float),
        "before_long_closure": (open_now & before_two).astype(float),
        "after_long_closure": (open_now & after_two).astype(float),
    }


def _known(item, name, n):
    source = item.get("known_raw", {})
    values = np.asarray(source[name] if name in source else item["known"][name])
    if values.shape != (n,):
        raise ValueError(f"Known field {name!r} must contain context + horizon")
    return values


def design_matrix(item, bindings, config, extra_features=None):
    """Return interpretable feature columns and a train-origin-centred trend.

    extra_features(item) -> {name: vector[context + horizon]} is deliberately
    editable: add date-defined events, physical proxies or interactions here.
    Its corresponding optional `calendar_priors` must be stated explicitly.
    Categorical levels and numeric normalizers are learned ONLY from history.
    """
    dates = pd.DatetimeIndex(item["timestamps"])
    length = item["y"].shape[1]; n = len(dates)
    dow = np.asarray(dates.dayofweek)
    days = np.asarray(dates.day); month_length = np.asarray(dates.days_in_month)
    features = {"intercept": np.ones(n)}
    for weekday in range(7):
        features[f"weekday:{weekday}"] = (dow == weekday).astype(float)
    features["month_start"] = (days <= 3).astype(float)
    features["month_end"] = (days >= month_length - 2).astype(float)
    phases = {
        "month": (days - 1) / month_length,
        "year": (np.asarray(dates.dayofyear) - 1) / 365.25,
        "hour": (np.asarray(dates.hour) + np.asarray(dates.minute) / 60) / 24,
    }
    for cycle, count in (("month", config.month_harmonics),
                         ("year", config.year_harmonics),
                         ("hour", config.hour_harmonics)):
        for harmonic in range(1, count + 1):
            phase = 2 * np.pi * harmonic * phases[cycle]
            features[f"{cycle}_sin:{harmonic}"] = np.sin(phase)
            features[f"{cycle}_cos:{harmonic}"] = np.cos(phase)
    for name in bindings.event_columns:
        event = _known(item, name, n).astype(float)
        features[f"event:{name}"] = event
        for weekday in range(7):
            features[f"event:{name}:weekday:{weekday}"] = event * (dow == weekday)
    for name in bindings.categorical_columns:
        values = _known(item, name, n).astype(str)
        levels, counts = np.unique(values[:length], return_counts=True)
        baseline = levels[np.argmax(counts)]
        for level in levels:
            if level != baseline:
                features[f"category:{name}:{level}"] = (values == level).astype(float)
    for name in bindings.continuous_columns:
        values = _known(item, name, n).astype(float)
        centre = float(np.mean(values[:length])); scale = float(np.std(values[:length]))
        features[f"continuous:{name}"] = (values - centre) / max(scale, 1e-6)
    if bindings.open_column is not None:
        open_values = _known(item, bindings.open_column, n).astype(float)
        features.update(closure_proximity(open_values))
    else:
        open_values = np.ones(n)
    # Units are actual calendar days, so weekly data do not silently use daily age.
    elapsed_days = np.asarray((dates - dates[length - 1]).total_seconds()) / 86400
    features["trend"] = elapsed_days / 365.25
    if extra_features is not None:
        for name, values in extra_features(item).items():
            if name in features:
                raise ValueError(f"Custom calendar column collides with {name!r}")
            features[name] = np.asarray(values, float)
    for name, values in features.items():
        if np.asarray(values).shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"Feature {name!r} must be finite for context + horizon")
    unknown_priors = set(config.calendar_priors) - set(features)
    if unknown_priors:
        raise ValueError(f"Calendar priors have no corresponding features: {unknown_priors}")
    return np.column_stack(list(features.values())), list(features), open_values, elapsed_days


def weighted_quantiles(values, weights, levels=np.arange(1, 10) / 10):
    order = np.argsort(values)
    values = np.asarray(values)[order]; weights = np.asarray(weights)[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(levels, cumulative, values)


def fit_structure(item, bindings=FieldBindings(), config=StructureConfig(), extra_features=None):
    """Fit a full per-entity, per-target expert; return None if unsuitable.

    This function exposes coefficients, column names and residuals to the agent.
    Inspect them; do not treat the default hyperparameters as universal answers.
    """
    y = np.asarray(item["y"], float)
    dimensions, length = y.shape
    if length < config.min_observations or np.any(y < 0):
        return None
    X, names, open_values, days = design_matrix(item, bindings, config, extra_features)
    n = len(X)
    prior = np.array([config.calendar_priors.get(name, 0.0) for name in names])
    structure_indices = [j for j, name in enumerate(names) if name not in ("intercept", "trend")]
    observed = np.asarray(item.get("observed", np.ones_like(y, bool)), bool)
    target_fits = []
    for d in range(dimensions):
        train = np.arange(max(0, length - config.recent), length)
        train = train[(open_values[train] > 0.5) & np.isfinite(y[d, train])
                      & observed[d, train] & (y[d, train] > 0)]
        if len(train) < config.min_observations:
            return None
        log_y = np.log(y[d, train])
        weight = np.exp2(np.maximum(days[train] / config.half_life_days, -100))
        penalties = np.full(X.shape[1], config.ridge * weight.sum() / 50)
        penalties[names.index("intercept")] = 1e-8
        for name in config.calendar_priors:
            penalties[names.index(name)] = config.prior_strength
        current_weight = weight.copy()
        for _ in range(config.robust_iterations + 1):
            gram = (X[train].T * current_weight) @ X[train] + np.diag(penalties)
            rhs = X[train].T @ (current_weight * log_y) + penalties * prior
            beta = np.linalg.solve(gram, rhs)
            residual = log_y - X[train] @ beta
            sigma = np.sqrt(max(float(np.average(residual ** 2, weights=current_weight)), 1e-8))
            current_weight = weight * np.minimum(1.0, config.huber_c * sigma / np.maximum(abs(residual), 1e-8))
        log_fit = X @ beta
        # Preserve the fitted history and origin; damp only FUTURE extrapolation.
        trend = names.index("trend")
        log_fit[length:] += (config.trend_damping - 1) * X[length:, trend] * beta[trend]
        structural_log = X[:, structure_indices] @ beta[structure_indices]
        structural_log -= np.average(structural_log[train], weights=current_weight)
        factor = np.exp(np.clip(structural_log, -8, 8))
        normalized = np.interp(np.arange(length), train, y[d, train] / factor[train])
        expert = np.exp(np.clip(log_fit[length:], np.min(log_y) - 4, np.max(log_y) + 4))
        residual = log_y - log_fit[train]
        sigma = np.sqrt(max(float(np.average(residual ** 2, weights=current_weight)), 1e-8))
        target_fits.append({
            "beta": beta, "normalized": normalized, "factor": factor,
            "expert": expert, "sigma": sigma,
            "residual_quantiles": weighted_quantiles(residual, current_weight),
            "train": train, "residual": residual,
        })
    return {"names": names, "targets": target_fits,
            "closed": open_values[length:] <= 0.5, "length": length, "n": n}


def predict_structure(view, tools, memory, bindings=FieldBindings(), config=StructureConfig(),
                      extra_features: Callable | None = None):
    """Complete structural pipeline with a real native call and unit restoration.

    Set normalize=False to test direct expert/native fusion as a separate branch.
    `memory` is accepted for the standard program API, but never read here: every
    fit uses this origin's past. Optional mature-memory calibration is a separate
    editable operation (see calibration.py).
    """
    if not (0 <= config.expert_weight <= 1 and 0 <= config.spread_mix <= 1
            and 0 <= config.empirical_quantile_mix <= 1):
        raise ValueError("Mixture weights must lie in [0, 1]")
    fits = [fit_structure(item, bindings, config, extra_features) for item in view.items]
    native_items = []
    for item, fit in zip(view.items, fits):
        if fit is not None and config.normalize:
            native_items.append({**item, "y": np.stack([f["normalized"] for f in fit["targets"]]),
                                 "known": {}, "past": {}, "known_raw": {}, "past_raw": {}})
        else:
            native_items.append(item)
    native = tools.timesfm(replace(view, items=native_items))
    point = native.point.copy(); quantiles = native.quantiles.copy()
    gaussian = np.array([-1.28155157, -.84162123, -.52440051, -.2533471,
                         0., .2533471, .52440051, .84162123, 1.28155157])
    for i, fit in enumerate(fits):
        if fit is None:
            continue
        for d, target in enumerate(fit["targets"]):
            floor = max(1e-8, float(np.median(target["expert"])) * 1e-8)
            native_p = np.maximum(native.point[i, :, d], floor)
            native_q = np.maximum(native.quantiles[i, :, d], floor)
            restored = native_p * target["factor"][fit["length"]:] if config.normalize else native_p
            log_centre = ((1 - config.expert_weight) * np.log(restored)
                          + config.expert_weight * np.log(target["expert"]))
            log_centre += config.point_log_shift + config.residual_median_shift * target["residual_quantiles"][4]
            p = np.exp(log_centre)
            native_shape = np.log(native_q / native_p[:, None])
            expert_shape = ((1 - config.empirical_quantile_mix) * gaussian * target["sigma"]
                            + config.empirical_quantile_mix * target["residual_quantiles"])
            growth = np.sqrt(1 + config.horizon_growth * np.arange(view.horizon) / max(view.horizon - 1, 1))
            shape = ((1 - config.spread_mix) * native_shape
                     + config.spread_mix * expert_shape[None, :] * growth[:, None])
            q = p[:, None] * np.exp(np.clip(config.spread_scale * shape, -8, 8))
            p[fit["closed"]] = 0.; q[fit["closed"]] = 0.
            point[i, :, d] = p; quantiles[i, :, d] = np.sort(q, axis=-1)
    return Forecast(point, quantiles).validate()

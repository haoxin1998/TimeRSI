"""Editable role-separated fusion: structural expert first, native residual second.

This reference does NOT select an expert, infer a task/field name, fit on future
labels, or call a foundation model. The caller supplies two complete forecasts,
the positions where native/expert disagreement is interpretable as a level signal,
known structural zeros, historical sample counts, and past-fitted residual scales.

Native/expert log disagreement is decomposed into robust level, damped trend and
remaining shape. Their gains are independent. Forecast uncertainty has a separate
gain and retains the expert's nine-quantile shape. By default irregular/event
positions inherit only the smooth correction, not the native's event shape or
interval: expert semantics are not erased by an indiscriminate average.

Positive multiplicative levels are the hypothesis. For signed/additive targets,
the agent should edit the representation rather than force this log-space recipe.
All defaults below are generic reference choices, NOT recovered benchmark weights.
The caller should probe alternatives or reject this method altogether.
"""
from dataclasses import dataclass
import numpy as np
from timersi_u1.core import Forecast


@dataclass(frozen=True)
class ResidualFusionConfig:
    # Separate gains, not a scalar weight mixing two entire forecasts.
    level_weight: float = 0.5
    trend_weight: float = 0.5
    shape_weight: float = 0.5
    spread_weight: float = 0.5
    # L/(L+reference), in observations. The caller supplies the meaning of L.
    # 32 is an untuned reference scale, not a frequency or dataset rule.
    history_reference: float = 32.0
    minimum_level_reliability: float = 0.5
    # Native disagreement on NORMAL positions, measured in supplied log sigmas.
    # None disables this factor for an explicit mechanism ablation.
    disagreement_scale: float | None = 4.0
    # smooth discounts all disagreement; threshold discounts only excess beyond
    # the supplied scale. Both are hypotheses to probe, not confidence estimates.
    disagreement_mode: str = 'smooth'
    max_shape_sigma: float | None = 2.0
    max_correction_sigma: float | None = 2.0
    native_spread_min_ratio: float = 0.5
    native_spread_max_ratio: float = 2.0
    shape_on_events: bool = False
    spread_on_events: bool = False


def _theil_sen(x, y):
    """Median pair slope over explicitly normal forecast positions only."""
    if len(x) < 2:
        return 0.0
    slopes = [(y[j + 1:] - y[j]) / (x[j + 1:] - x[j]) for j in range(len(x) - 1)]
    return float(np.median(np.concatenate(slopes)))


def _horizon_scale(value, shape):
    array = np.asarray(value, float)
    if array.shape == (shape[0], shape[2]):
        array = array[:, None, :]
    return np.broadcast_to(array, shape)


def fuse_native_residual(expert, native, *, normal_mask, zero_mask,
                         history_length, residual_log_scale,
                         config=ResidualFusionConfig(), return_diagnostics=False):
    """Combine positive Forecast[N,H,D] objects while preserving expert structure.

    Required caller inputs:
      normal_mask: bool[N,H,D], or broadcastable scalar, known BEFORE forecasting.
          Mark positions where the expert's special event/exposure effects do not
          invalidate interpreting disagreement as a common level/trend signal.
      zero_mask: same shape; an explicit, justified structural-zero constraint.
      history_length: scalar or [N,D], in observations actually used by the caller.
      residual_log_scale: positive scalar, [N,D], or [N,H,D], estimated using PAST
          errors of this expert. It is not estimated from current forecast labels.

    No normal positions -> retain that expert unchanged, apart from explicit zeros.
    One normal position -> a level correction but no slope. Unknown/exceptional
    positions are never silently substituted for the caller's normal positions.
    Output point and quantiles follow the expert's convention (its point need not
    equal q50). Inputs are not modified. Optional diagnostics expose every accepted
    component/gain so the agent can inspect the mechanism, not just its final score.
    """
    expert.validate(); native.validate()
    if expert.point.shape != native.point.shape or expert.point.ndim != 3:
        raise ValueError("Expected expert/native with the same [N,H,D] shape")
    shape = expert.point.shape
    normal = np.broadcast_to(np.asarray(normal_mask, bool), shape)
    zeros = np.broadcast_to(np.asarray(zero_mask, bool), shape)
    lengths = np.broadcast_to(np.asarray(history_length, float), (shape[0], shape[2]))
    sigma = _horizon_scale(residual_log_scale, shape)
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("Supply positive finite past-fitted log residual scales")
    if np.any(lengths < 0) or not np.isfinite(lengths).all() or config.history_reference < 0:
        raise ValueError("History lengths and reference must be nonnegative")
    gains = (config.level_weight, config.trend_weight, config.shape_weight, config.spread_weight)
    if any(not 0 <= value <= 1 for value in gains) or not 0 <= config.minimum_level_reliability <= 1:
        raise ValueError("Component weights/reliability must lie in [0,1]")
    if not 0 < config.native_spread_min_ratio <= config.native_spread_max_ratio:
        raise ValueError("Spread-ratio bounds must be positive and ordered")
    if config.disagreement_scale is not None and config.disagreement_scale <= 0:
        raise ValueError("Disagreement scale must be positive or None")
    if config.disagreement_mode not in ('smooth','threshold'):
        raise ValueError("Unknown disagreement weighting mode")
    if np.any(expert.point < 0) or np.any(expert.quantiles < 0):
        raise ValueError("This multiplicative reference expects a nonnegative expert")
    point = expert.point.copy(); quantiles = expert.quantiles.copy()
    diagnostics = {name: np.zeros(shape) for name in
                   ("level", "trend", "remaining_shape", "correction", "output_log_scale")}
    diagnostics.update({name: np.zeros((shape[0], shape[2])) for name in
                        ("normal_count", "history_reliability", "agreement_reliability",
                         "level_gain", "trend_gain", "shape_gain", "spread_gain")})
    x = np.arange(shape[1], dtype=float)
    eps = 1e-12
    for i in range(shape[0]):
        for d in range(shape[2]):
            usable = (expert.point[i, :, d] > 0) & (native.point[i, :, d] > 0) & ~zeros[i, :, d]
            fit_mask = normal[i, :, d] & usable
            diagnostics["normal_count"][i, d] = fit_mask.sum()
            diagnostics["output_log_scale"][i, :, d] = sigma[i, :, d]
            if not fit_mask.any():
                continue
            ratio = np.zeros(shape[1])
            ratio[usable] = np.log(native.point[i, usable, d] / expert.point[i, usable, d])
            intercept = float(np.median(ratio[fit_mask]))
            slope = _theil_sen(x[fit_mask], ratio[fit_mask])
            trend = slope * (x - np.median(x[fit_mask]))
            remaining = ratio - intercept - trend
            if config.max_shape_sigma is not None:
                bound = config.max_shape_sigma * sigma[i, :, d]
                remaining = np.clip(remaining, -bound, bound)
            # Known events keep the expert's shape unless the caller opts in.
            shape_mask = usable if config.shape_on_events else fit_mask
            remaining = np.where(shape_mask, remaining, 0.0)
            length = lengths[i, d]
            evidence = length / max(length + config.history_reference, eps)
            standardized = ratio[fit_mask] / sigma[i, fit_mask, d]
            disagreement = float(np.sqrt(np.mean(standardized ** 2)))
            excess=(disagreement/config.disagreement_scale if config.disagreement_scale is not None else 0.)
            if config.disagreement_mode=='threshold':excess=max(excess-1.,0.)
            agreement=1/(1+excess**2)
            level_gain = config.level_weight * agreement * (
                config.minimum_level_reliability + (1 - config.minimum_level_reliability) * evidence)
            trend_gain = config.trend_weight * evidence * agreement
            shape_gain = config.shape_weight * evidence * agreement
            spread_gain = config.spread_weight * evidence * agreement
            correction = level_gain * intercept + trend_gain * trend + shape_gain * remaining
            if config.max_correction_sigma is not None:
                bound = config.max_correction_sigma * sigma[i, :, d]
                correction = np.clip(correction, -bound, bound)
            correction = np.where(usable, correction, 0)
            centre = expert.point[i, :, d] * np.exp(correction)

            # The expert supplies skew/shape. Native contributes only a separately
            # controlled relative scale, not a second unexamined quantile forecast.
            nq = np.maximum(native.quantiles[i, :, d], eps)
            native_sigma = (np.log(nq[:, 8]) - np.log(nq[:, 0])) / (2 * 1.2815515655446004)
            scale_ratio = np.clip(native_sigma / sigma[i, :, d],
                                  config.native_spread_min_ratio, config.native_spread_max_ratio)
            spread_mask = usable if config.spread_on_events else fit_mask
            scale_multiplier = np.exp(np.where(spread_mask, spread_gain * np.log(scale_ratio), 0))
            expert_offset = np.log(np.maximum(expert.quantiles[i, :, d], eps)
                                   / np.maximum(expert.point[i, :, d, None], eps))
            output_q = centre[:, None] * np.exp(expert_offset * scale_multiplier[:, None])
            point[i, usable, d] = centre[usable]
            quantiles[i, usable, d] = output_q[usable]
            for name, value in (("level", np.full(shape[1], intercept)), ("trend", trend),
                                ("remaining_shape", remaining), ("correction", correction),
                                ("output_log_scale", sigma[i, :, d] * scale_multiplier)):
                diagnostics[name][i, :, d] = value
            for name, value in (("history_reliability", evidence), ("agreement_reliability", agreement),
                                ("level_gain", level_gain), ("trend_gain", trend_gain),
                                ("shape_gain", shape_gain), ("spread_gain", spread_gain)):
                diagnostics[name][i, d] = value
    point[zeros] = 0.; quantiles[zeros] = 0.
    output = Forecast(point, np.sort(quantiles, axis=-1)).validate()
    return (output, diagnostics) if return_diagnostics else output

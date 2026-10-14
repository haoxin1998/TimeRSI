"""Editable small-sample exposure-rate expert, with optional native residual fusion.

The caller supplies exposure and numerical known regressors by their semantics.
There is no task/column-name dispatch and no stored fitted model. A short history
fits a small hypothesis: log rate intercept, then only as many varying independent
effects as observed samples support. Exposure is a numerical amount, NOT a binary
open flag: every strictly positive value is valid; exactly zero restores to zero.

All defaults are generic research starting points, not old benchmark parameters.
This positive-rate model does not model intermittent zeros or signed net flows.
"""
from dataclasses import dataclass
import numpy as np

from timersi_u1.core import Forecast
from .native_residual_fusion import fuse_native_residual


ZQ = np.array([-1.2815515655446004, -.8416212335729143, -.5244005127080407,
               -.2533471031357997, 0., .2533471031357997, .5244005127080407,
               .8416212335729143, 1.2815515655446004])


@dataclass(frozen=True)
class ShortExposureConfig:
    min_feature_observations: int = 3
    observations_per_extra_feature: int = 6
    max_features: int = 6
    ridge: float = 0.2
    half_life_observations: float | None = None
    prior_log_scale: float = 0.25
    prior_observations: float = 6.0
    minimum_log_scale: float = 0.03
    horizon_scale_growth: float = 0.25
    extrapolation_margin: float = 0.25
    max_log_rate_extrapolation: float = 3.0


def _fit_ridge(design, response, weight, ridge):
    penalty = np.full(design.shape[1], ridge)
    penalty[0] = 0.0
    return np.linalg.solve((design.T * weight) @ design + np.diag(penalty),
                           design.T @ (weight * response))


def fit_short_exposure(item, exposure, regressors=None, config=ShortExposureConfig()):
    """Fit one item and return point[H,D], quantiles[H,D,9] and fit evidence.

    exposure: explicit finite nonnegative [L+H] amounts, in consistent units.
    regressors: {label: numerical [L+H] vector}; interactions/calendars are authored
        by the caller. Future values must be known. Labels do not select methods.
    item['observed'], when supplied, identifies actually observed targets [D,L].

    Feature standardization, support and ranking use observed history only.
    Uncertainty combines in-sample residuals with an explicit short-sample prior;
    it is NOT claimed to be calibrated causal backtest uncertainty. A candidate
    can replace it with errors from replaying this exact expert on earlier origins.
    """
    y = np.asarray(item['y'], float)
    dimensions, length = y.shape
    exposure = np.asarray(exposure, float)
    if exposure.ndim != 1 or len(exposure) <= length:
        raise ValueError('Exposure must span historical and future positions')
    if not np.isfinite(exposure).all() or np.any(exposure < 0):
        raise ValueError('Exposure amounts must be finite and nonnegative')
    if np.any(y[np.isfinite(y)] < 0):
        raise ValueError('This reference assumes a nonnegative positive-rate target')
    if (config.min_feature_observations < 2 or config.observations_per_extra_feature < 1
            or config.max_features < 0 or config.ridge <= 0
            or config.prior_log_scale <= 0 or config.prior_observations <= 0
            or config.minimum_log_scale <= 0 or config.horizon_scale_growth < 0):
        raise ValueError('Use positive regularization/uncertainty and valid feature support')
    if config.half_life_observations is not None and config.half_life_observations <= 0:
        raise ValueError('Half-life must be positive, or None for equal weights')
    regressors = {str(name): np.asarray(values, float) for name, values in (regressors or {}).items()}
    for name, values in regressors.items():
        if values.shape != exposure.shape or not np.isfinite(values).all():
            raise ValueError(f'Known regressor {name!r} must be finite and span L+H')
    observed = np.asarray(item.get('observed', np.ones_like(y, bool)), bool)
    horizon = len(exposure) - length
    log_rates = np.empty((horizon, dimensions))
    scales = np.empty_like(log_rates)
    fits = []
    for d in range(dimensions):
        valid = observed[d] & np.isfinite(y[d]) & (y[d] > 0) & (exposure[:length] > 0)
        train = np.flatnonzero(valid)
        if not len(train):
            raise ValueError('Positive-rate expert needs a positive observed target/exposure pair')
        weight = np.ones(len(train))
        if config.half_life_observations is not None:
            weight = np.exp2((train - train[-1]) / config.half_life_observations)
        effective_n = float(weight.sum() ** 2 / np.sum(weight ** 2))
        weight *= len(train) / weight.sum()
        response = np.log(y[d, train]) - np.log(exposure[train])
        design = np.ones((len(train), 1))
        future = np.ones((horizon, 1))
        coefficient = _fit_ridge(design, response, weight, config.ridge)
        candidates = {}
        normalizers = {}
        for name, values in regressors.items():
            centre = float(np.average(values[train], weights=weight))
            scale = float(np.sqrt(np.average((values[train] - centre) ** 2, weights=weight)))
            if scale <= 1e-12:
                continue
            z = (values - centre) / scale
            lo, hi = float(z[train].min()), float(z[train].max())
            margin = config.extrapolation_margin * (hi - lo)
            candidates[name] = (z[train], np.clip(z[length:], lo - margin, hi + margin))
            normalizers[name] = (centre, scale)
        support = int(np.floor(effective_n + 1e-9))
        budget = (0 if support < config.min_feature_observations else
                  1 + (support - config.min_feature_observations) // config.observations_per_extra_feature)
        budget = min(budget, config.max_features, len(train) - 1)
        selected = []
        for _ in range(budget):
            residual = response - design @ coefficient
            best = None
            root_weight = np.sqrt(weight)
            for name, (past_z, future_z) in candidates.items():
                # Remove already-explained directions; duplicate columns cannot
                # consume the small history's feature budget.
                projection = np.linalg.lstsq(design * root_weight[:, None],
                                             past_z * root_weight, rcond=None)[0]
                independent = past_z - design @ projection
                energy = float(np.sum(weight * independent ** 2))
                if energy <= 1e-10:
                    continue
                score = abs(float(np.sum(weight * independent * residual))) / np.sqrt(energy)
                if best is None or score > best[0]:
                    best = (score, name, past_z, future_z)
            if best is None or best[0] <= 1e-10:
                break
            _, name, past_z, future_z = best
            selected.append(name)
            design = np.column_stack([design, past_z])
            future = np.column_stack([future, future_z])
            candidates.pop(name)
            coefficient = _fit_ridge(design, response, weight, config.ridge)
        residual = response - design @ coefficient
        degrees = max(effective_n - design.shape[1], 0.0)
        residual_variance = float(np.average(residual ** 2, weights=weight))
        if degrees > 0:
            residual_variance *= effective_n / degrees
        variance = (degrees * residual_variance + config.prior_observations * config.prior_log_scale ** 2)
        variance /= degrees + config.prior_observations
        scale0 = max(float(np.sqrt(variance)), config.minimum_log_scale)
        log_rates[:, d] = np.clip(future @ coefficient,
            response.min() - config.max_log_rate_extrapolation,
            response.max() + config.max_log_rate_extrapolation)
        scales[:, d] = scale0 * np.sqrt(1 + config.horizon_scale_growth
                                        * np.arange(1, horizon + 1) / max(effective_n, 1))
        fits.append({'features': selected, 'coefficients': coefficient,
                     'normalizers': {name: normalizers[name] for name in selected},
                     'train': train, 'effective_observations': effective_n,
                     'residual': residual, 'log_scale_at_origin': scale0})
    point = exposure[length:, None] * np.exp(log_rates)
    quantiles = point[..., None] * np.exp(scales[..., None] * ZQ)
    return {'point': point, 'quantiles': quantiles, 'log_scale': scales,
            'history_length': np.array([fit['effective_observations'] for fit in fits]),
            'zero_mask': np.broadcast_to(exposure[length:, None] == 0, point.shape), 'fits': fits}


def predict_short_exposure(view, tools, memory, input_builder,
                           config=ShortExposureConfig(), native_fusion=None,
                           return_details=False):
    """Complete forecast; native is optional, never a hidden required component.

    input_builder(item) -> {'exposure': [L+H], 'regressors': {name: [L+H]},
                            'normal_mask': bool[H] or bool[H,D]}.
    normal_mask is required only when native_fusion is supplied. It is an explicit
    caller choice of future positions suitable for estimating native level/trend.
    native_fusion is a ResidualFusionConfig; None returns this statistical expert.
    """
    inputs = [input_builder(item) for item in view.items]
    fits = [fit_short_exposure(item, values['exposure'], values.get('regressors'), config)
            for item, values in zip(view.items, inputs)]
    expert = Forecast(np.stack([fit['point'] for fit in fits]),
                      np.stack([fit['quantiles'] for fit in fits])).validate()
    if expert.point.shape != (len(view.items), view.horizon, len(view.targets)):
        raise ValueError('Caller exposure must match this view horizon and target count')
    details = {'fits': fits}
    output = expert
    if native_fusion is not None:
        normals = []
        for values, fit in zip(inputs, fits):
            normal = np.asarray(values['normal_mask'], bool)
            if normal.shape == (view.horizon,):
                normal = normal[:, None]
            normals.append(np.broadcast_to(normal, fit['point'].shape))
        output, details['fusion'] = fuse_native_residual(
            expert, tools.timesfm(view), normal_mask=np.stack(normals),
            zero_mask=np.stack([fit['zero_mask'] for fit in fits]),
            history_length=np.stack([fit['history_length'] for fit in fits]),
            residual_log_scale=np.stack([fit['log_scale'] for fit in fits]),
            config=native_fusion, return_diagnostics=True)
    return (output, details) if return_details else output

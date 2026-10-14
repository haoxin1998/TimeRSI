"""Small R-fitted nearest-scenario policy: no entity ID or S-label fitting."""
import numpy as np
import pandas as pd


BASE_FEATURES = ['log_level', 'relative_level_change', 'relative_trend',
                 'relative_volatility', 'relative_diff_volatility', 'zero_fraction',
                 'negative_fraction', 'seasonal_error', 'last_relative_level',
                 'horizon_over_season', 'year_sin', 'year_cos', 'weekend_fraction',
                 'month_start_fraction', 'month_end_fraction']


def scenario_features(view, known_columns):
    """Features at a forecast origin, using past y and known future only.

    Keys identify output cells but are deliberately absent from the features.
    Explicit covariate columns retain their task-local identity across origins.
    """
    result = {}
    for item in view.items:
        y = np.asarray(item['y'], float)
        length = y.shape[1]
        recent = min(length, max(8, 2 * view.horizon, 2 * view.seasonality))
        dates = pd.DatetimeIndex(item['timestamps'][length:length + view.horizon])
        year_phase = (np.asarray(dates.dayofyear, float) - 1) / 365.25
        calendar = [float(np.mean(np.sin(2 * np.pi * year_phase))),
                    float(np.mean(np.cos(2 * np.pi * year_phase))),
                    float(np.mean(dates.dayofweek >= 5)),
                    float(np.mean(dates.day <= 3)),
                    float(np.mean(dates.day >= dates.days_in_month - 2))]
        covariates = []
        for name in known_columns:
            values = item.get('known', {}).get(name)
            if values is None:
                covariates.extend([0., 0.])
                continue
            values = np.asarray(values, float)
            past = values[max(0, length - recent):length]
            future = values[length:length + view.horizon]
            scale = max(float(np.std(past)), float(np.mean(abs(past))) * .01, 1e-6)
            covariates.extend([float(np.clip((np.mean(future) - np.mean(past)) / scale, -10, 10)),
                               float(np.clip((np.std(future) - np.std(past)) / scale, -10, 10))])
        for d, target in enumerate(view.targets):
            tail = y[d, -recent:]
            before = y[d, max(0, length - 2 * recent):length - recent]
            level = float(np.mean(tail));scale = max(1e-6, float(np.mean(abs(tail))))
            half = max(1, len(tail) // 2)
            season = max(1, int(view.seasonality))
            seasonal = (float(np.mean(abs(y[d, season:] - y[d, :-season]))) / scale
                        if length > season else 0.)
            features = [np.log1p(scale), (level - np.mean(before)) / scale if len(before) else 0.,
                        (np.mean(tail[-half:]) - np.mean(tail[:half])) / scale,
                        np.std(tail) / scale, np.std(np.diff(tail)) / scale if len(tail) > 1 else 0.,
                        np.mean(tail == 0), np.mean(tail < 0), seasonal,
                        (tail[-1] - level) / scale,
                        np.log1p(view.horizon / max(1, view.seasonality)), *calendar, *covariates]
            result[(str(item['id']), str(target))] = np.nan_to_num(
                np.asarray(features, float), nan=0., posinf=10., neginf=-10.).tolist()
    return result


def fit_scenario_policy(views, folds, research_results, global_scores, neighbors=8):
    """Freeze R input features and candidate losses for later loss retrieval.

    Retrieval ranks expected log SQL ratio to Raw. Normalizing each historical
    cell by Raw prevents a high-loss entity from dominating all other entities.
    No entity identifier enters distances; new entities can use similar cases.
    """
    members = sorted(n for n, value in global_scores.items()
                     if np.isfinite(value) and n in research_results)
    fallback = min(members, key=lambda n: (global_scores[n], n))
    columns = sorted({name for fold in folds for item in views[fold].items
                      for name in item.get('known', {})})
    descriptors = {fold: scenario_features(views[fold], columns) for fold in folds}
    losses = {}
    for name in members:
        for row in research_results[name]['folds']:
            detail = row['error_details'];fold = row['offset']
            if fold not in descriptors:
                continue
            for item in detail['items']:
                for target, value in zip(detail['target_names'], item['SQL_by_target']):
                    if value is not None and np.isfinite(value):
                        losses.setdefault((fold, str(item['id']), str(target)), {})[name] = float(value)
    examples = []
    for (fold, item, target), values in sorted(losses.items()):
        if len(values) != len(members) or (item, target) not in descriptors[fold]:
            continue
        denominator = max(values.get('raw', values[fallback]), 1e-12)
        examples.append({'offset': int(fold), 'target': target,
            'features': descriptors[fold][(item, target)],
            'log_SQL_ratios': [float(np.clip(np.log(max(values[n], 1e-12) / denominator), -8, 8))
                               for n in members]})
    width = len(BASE_FEATURES) + 2 * len(columns)
    features = np.asarray([row['features'] for row in examples], float).reshape(-1, width)
    centre = np.mean(features, axis=0) if len(features) else np.zeros(width)
    scale = np.maximum(np.std(features, axis=0), .05) if len(features) else np.ones(width)
    return {'members': members, 'fallback': fallback, 'known_columns': columns,
            'feature_names': BASE_FEATURES + [f'{name}:{stat}' for name in columns
                                             for stat in ('mean_shift', 'spread_shift')],
            'centre': centre.tolist(), 'scale': scale.tolist(), 'examples': examples,
            'neighbors': int(neighbors), 'min_research_windows': 2,
            'fit_scope': 'R only; scenario features contain no future target or entity ID',
            'research_offsets': sorted(int(k) for k in folds)}


def scenario_routes(view, policy):
    """Retrieve similar historical origins for each current entity/target."""
    queries = scenario_features(view, policy['known_columns'])
    centre, scale = np.asarray(policy['centre']), np.asarray(policy['scale'])
    targets = {}
    for example in policy['examples']:
        targets.setdefault(example['target'], []).append(example)
    prepared = {}
    for target, examples in targets.items():
        if len({row['offset'] for row in examples}) < policy['min_research_windows']:
            continue
        prepared[target] = ((np.asarray([row['features'] for row in examples]) - centre) / scale,
                            np.asarray([row['log_SQL_ratios'] for row in examples]))
    routes = {}
    for (item, target), features in queries.items():
        if target not in prepared:
            continue
        contexts, losses = prepared[target]
        distances = np.mean((contexts - (np.asarray(features) - centre) / scale) ** 2, axis=1)
        count = min(policy['neighbors'], len(distances))
        nearest = np.argsort(distances, kind='stable')[:count]
        weights = 1. / (distances[nearest] + 1e-3)
        estimates = np.average(losses[nearest], axis=0, weights=weights)
        name = policy['members'][int(np.argmin(estimates))]
        routes.setdefault(item, {})[target] = {'members': [name], 'weights': [1.]}
    return routes


def apply_scenario(view, predict_member, policy):
    """Only run members actually selected now, each on the complete view."""
    from .selection import route_forecasts
    routes = scenario_routes(view, policy)
    needed = {policy['fallback']} | {n for row in routes.values() for route in row.values()
                                     for n in route['members']}
    forecasts = {name: predict_member(name) for name in sorted(needed)}
    return route_forecasts(view, forecasts, routes, policy['fallback'])

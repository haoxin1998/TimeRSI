"""Entity/target routes fitted on research SQL, never on selection labels."""
import numpy as np


def fit_routes(research_results, global_scores, top_k=1, min_windows=2):
    """Return a serializable route table and a global-R fallback.

    Each cell uses mean official, seasonally scaled quantile loss from the same
    completed R folds. A cell with fewer than min_windows falls back globally.
    top_k>1 blends the best local programs with inverse-loss weights; the complete
    resulting program still has to compete on the independent S windows.
    """
    valid = {n: float(v) for n, v in global_scores.items() if np.isfinite(v)}
    fallback = min(valid, key=lambda n: (valid[n], n))
    cells = {}
    for name, result in research_results.items():
        if name not in valid:
            continue
        for fold in result['folds']:
            detail = fold['error_details']
            for item in detail['items']:
                for target, loss in zip(detail['target_names'], item['SQL_by_target']):
                    if loss is not None and np.isfinite(loss):
                        cells.setdefault((str(item['id']), str(target)), {}).setdefault(name, {})[
                            int(fold['offset'])] = float(loss)
    routes = {}
    for (item_id, target), candidates in cells.items():
        # Compare the full set of available folds, not a lucky incomplete subset.
        folds = set().union(*(set(values) for values in candidates.values()))
        if len(folds) < min_windows:
            continue
        means = {name: float(np.mean(list(values.values()))) for name, values in candidates.items()
                 if set(values) == folds}
        ranked = sorted(means, key=lambda n: (means[n], n))[:max(1, top_k)]
        if not ranked:
            continue
        losses = np.asarray([means[n] for n in ranked])
        weights = 1. / np.maximum(losses, 1e-12)
        weights /= weights.sum()
        routes.setdefault(item_id, {})[target] = {
            'members': ranked, 'weights': weights.tolist(),
            'research_windows': len(folds), 'research_SQL': [means[n] for n in ranked]}
    return {'routes': routes, 'fallback': fallback,
            'fit_scope': 'R only; official per-entity-target SQL',
            'top_k': top_k, 'min_windows': min_windows}


def route_forecasts(view, forecasts, routes, fallback):
    """Apply frozen routes by semantic IDs, independent of entity/target order."""
    from .core import Forecast
    base = forecasts[fallback]
    point, quantiles = base.point.copy(), base.quantiles.copy()
    for i, item in enumerate(view.items):
        item_routes = routes.get(str(item['id']), {})
        for d, target in enumerate(view.targets):
            route = item_routes.get(str(target))
            if route is None:
                continue
            weights = np.asarray(route['weights'], float)
            weights = weights / weights.sum()
            point[i, :, d] = sum(w * forecasts[n].point[i, :, d]
                                 for w, n in zip(weights, route['members']))
            quantiles[i, :, d] = sum(w * forecasts[n].quantiles[i, :, d]
                                     for w, n in zip(weights, route['members']))
    return Forecast(point, quantiles).validate()

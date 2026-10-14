"""Compare complete programs on independent, already historical S windows."""
import numpy as np


def selection_values(selection, objective='mean-sql'):
    if objective == 'mean-sql':
        return {name: float(row['SQL']) for name, row in selection.items()
                if row['SQL'] is not None and np.isfinite(row['SQL'])}
    if objective != 'raw-ratio':
        raise ValueError(f'Unknown selection objective: {objective}')
    raw = {fold['offset']: float(fold['metrics']['SQL'])
           for fold in selection['raw']['folds']}
    result = {}
    for name, row in selection.items():
        losses = {fold['offset']: float(fold['metrics']['SQL']) for fold in row['folds']}
        if set(losses) != set(raw):
            raise ValueError('Compare complete programs on the same S windows')
        # Equal relative importance per window, rather than letting a high-loss
        # season dominate. This is not a confidence gate or an oracle fallback.
        result[name] = float(np.mean([losses[k]/max(raw[k], 1e-12) for k in raw]))
    return result


def choose_selection(selection, objective='mean-sql'):
    values = selection_values(selection, objective)
    return (min(values, key=values.get) if values else 'raw'), values

"""Complete visible inputs for a decision agent's own analysis, no future targets."""
from pathlib import Path
import numpy as np
from timersi_u1.core import write


def display_array(values):
    array = np.asarray(values)
    if array.dtype.kind == 'O':
        try:
            return np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return np.asarray(['<missing>' if x is None else str(x) for x in values], dtype=str)
    return array


def publish_history(view, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}; rows = []
    for i, item in enumerate(view.items):
        prefix = f'item_{i:05d}'
        y = np.asarray(item['y']); length = y.shape[1]
        dates = np.asarray(item['timestamps']).astype(str)
        arrays[prefix+'__past_targets'] = y
        arrays[prefix+'__observed'] = np.asarray(item.get('observed', np.isfinite(y)), dtype=bool)
        arrays[prefix+'__past_dates'] = dates[:length]
        arrays[prefix+'__forecast_dates'] = dates[length:length+view.horizon]
        row = dict(id=str(item['id']), prefix=prefix, past_length=length,
                   past_start=str(dates[0]), past_end=str(dates[length-1]),
                   forecast_start=str(dates[length]), static=item.get('static', {}),
                   known_covariates=[], past_covariates=[])
        for j, (name, encoded) in enumerate(item.get('known', {}).items()):
            key = prefix+f'__known_{j:03d}'
            arrays[key+'__encoded'] = np.asarray(encoded)
            arrays[key+'__raw'] = display_array(item.get('known_raw', {}).get(name, encoded))
            row['known_covariates'].append(dict(name=name, key=key, scope='Past plus explicitly known future'))
        for j, (name, encoded) in enumerate(item.get('past', {}).items()):
            key = prefix+f'__past_{j:03d}'
            arrays[key+'__encoded'] = np.asarray(encoded)
            arrays[key+'__raw'] = display_array(item.get('past_raw', {}).get(name, encoded))
            row['past_covariates'].append(dict(name=name, key=key, scope='Past only'))
        rows.append(row)
    np.savez_compressed(directory/'inputs.npz', **arrays)
    metadata = dict(items=rows, targets=list(view.targets), horizon=view.horizon,
        frequency=view.frequency, seasonality=view.seasonality, facts=view.facts,
        arrays='inputs.npz', past_target_orientation='[target, time]',
        observed_note='False marks a filled target value, not an observed measurement',
        scope='Complete available history and explicitly known future covariates; no future targets or scoring arrays')
    write(directory/'INDEX.json', metadata)
    return metadata

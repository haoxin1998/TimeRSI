"""Publish only R inputs, labels and forecasts for CPU analysis by the code agent."""
import numpy as np
import shutil
from pathlib import Path
from .core import write
from .data import predictions


def publish_native_calls(folder, calls, checkpoint):
    """Accumulate actual outputs of this public R fold, never copy the whole cache."""
    import json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder/'INDEX.json'
    rows = json.loads(index_path.read_text())['calls'] if index_path.exists() else []
    seen = {row['native_key'] for row in rows}
    for call in calls:
        key = call['native_key']
        if key in seen:
            continue
        source = Path(call['path'])
        shutil.copy2(source, folder/(key+'.npz'))
        rows.append({'native_key': key, 'arrays': key+'.npz'})
        seen.add(key)
    write(index_path, {'scope': 'Native outputs actually evaluated on this public R fold only',
        'checkpoint': checkpoint, 'calls': rows,
        'layout': 'point[N,H,D], quantiles[N,H,D,9]; native result in original target units'})
    return folder


def publish(folder, views, folds, warm, cache, task):
    folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for fold in folds:
        view=views[fold];raw=cache[fold]['raw']
        truth,_=warm[fold]._prepare_arrays(predictions(raw,task),task.quantile_levels)
        arrays={'truth':truth['y_true'],'raw_point':raw.point,'raw_quantiles':raw.quantiles,
                'metric_y_past':truth['y_past'],'metric_y_past_lengths':truth['y_past_lengths']}
        items=[]
        for i,item in enumerate(view.items):
            arrays[f'y_{i}']=item['y']
            if 'observed' in item:arrays[f'observed_{i}']=item['observed']
            arrays[f'timestamps_{i}']=np.asarray(item['timestamps']).astype(str)
            fields={}
            for kind in ('known','past','known_raw','past_raw'):
                fields[kind]={}
                for j,(name,value) in enumerate(item.get(kind,{}).items()):
                    key=f'{kind}_{i}_{j}';a=np.asarray(value)
                    arrays[key]=a.astype(str) if a.dtype.kind=='O' else a
                    fields[kind][name]=key
            items.append({'id':item['id'],'y':f'y_{i}','timestamps':f'timestamps_{i}',
                          'static':item['static'],**({'observed':f'observed_{i}'} if 'observed' in item else {}),**fields})
        filename=f'R-{fold}.npz';np.savez_compressed(folder/filename,**arrays)
        records.append({'offset':fold,'arrays':filename,'items':items,'targets':view.targets,
                        'horizon':view.horizon,'seasonality':view.seasonality,
                        'frequency':view.frequency,'facts':view.facts,
                        'quantile_levels':task.quantile_levels,
                        'metrics':[{'name':type(metric).__name__,'kwargs':vars(metric)} for metric in task.metrics]})
    write(folder/'INDEX.json',{'scope':'R only; no selection or official future labels',
                             'layout':'y[D,L], truth/raw_point[N,H,D], raw_quantiles[N,H,D,9]',
                             'folds':records})
    return folder

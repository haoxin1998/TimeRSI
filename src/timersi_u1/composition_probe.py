"""Score actual cached mixtures on completed history only; derived from experiment158."""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np


def mixed_arrays(arrays, manifest, decision):
    names=manifest['candidates']; ids=list(map(str,arrays['item_ids'].tolist()))
    def mixture(spec):
        weights=spec['weights']; values=np.asarray(list(weights.values()),float)
        if not weights or not set(weights)<=set(names):raise ValueError('Unknown or unavailable past candidates')
        if not np.isfinite(values).all() or np.any(values<0) or values.sum()<=0:raise ValueError('Invalid weights')
        values=values/values.sum()
        return tuple(sum(w*arrays[name+suffix] for name,w in zip(weights,values))
                     for suffix in ('__point','__quantiles'))
    point,quantiles=mixture(decision['default']);point=point.copy();quantiles=quantiles.copy()
    specs=decision.get('groups',{});groups={g['key']:g for g in manifest['groups']}
    if not set(specs)<=set(groups):raise ValueError('Unknown group')
    for key,spec in specs.items():
        group=groups[key];d=manifest['targets'].index(group['target'])
        rows=[i for i,item in enumerate(ids) if item in group['members']]
        p,q=mixture(spec);point[rows,:,d]=p[rows,:,d];quantiles[rows,:,d]=q[rows,:,d]
    return point,quantiles


def evaluate(directory,decision):
    from fev.metrics import SQL,MASE,WAPE,WQL
    directory=Path(directory);manifest=json.loads((directory/'MANIFEST.json').read_text());episodes=[]
    for episode in manifest['episodes']:
        with np.load(directory/episode['arrays'],allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files};point,quantiles=mixed_arrays(arrays,manifest,decision)
        scoring={k:v for k,v in arrays.items() if k.startswith('y_') or k=='q_pred'}
        scoring.update(y_pred=point,q_pred=quantiles)
        options=dict(seasonality=manifest['seasonality'],quantile_levels=manifest['quantile_levels'])
        metrics={name:float(np.mean(metric.compute(**scoring,**options))) for name,metric in
                 [('SQL',SQL()),('MASE',MASE()),('WAPE',WAPE(epsilon=1.)),('WQL',WQL(epsilon=1.))]}
        cells=SQL().compute(**scoring,**options,reduce_axes=(1,));groups=[]
        for group in manifest['groups']:
            d=manifest['targets'].index(group['target'])
            rows=[i for i,item in enumerate(arrays['item_ids'].tolist()) if str(item) in group['members']]
            values=cells[rows,d]
            groups.append(dict(key=group['key'],target=group['target'],SQL=float(np.nanmean(values)) if np.isfinite(values).any() else None,
                               visible_cells=int(np.isfinite(values).sum())))
        episodes.append(dict(context=episode['context'],metrics=metrics,groups=groups))
    return dict(decision=decision,episodes=episodes,current_labels_used=False,new_native_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--decision',type=Path);p.add_argument('--weights');p.add_argument('--group')
    p.add_argument('--input',type=Path,default=Path('INPUT.json'));a=p.parse_args()
    packet=json.loads(a.input.read_text());info=packet['composition_probe']
    if a.decision:decision=json.loads(a.decision.read_text())
    elif a.weights:
        decision=dict(default=dict(weights={packet['global_winner']:1.}),groups={});spec=dict(weights=json.loads(a.weights))
        if a.group:decision['groups'][a.group]=spec
        else:decision['default']=spec
    else:p.error('Provide --decision FILE or --weights JSON')
    result=evaluate(a.input.parent/info['directory'],decision);result['at']=time.time()
    directory=a.input.parent/'composition-probes';directory.mkdir(exist_ok=True);path=directory/(uuid.uuid4().hex+'.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({**result,'receipt':str(path)}))


if __name__=='__main__':main()

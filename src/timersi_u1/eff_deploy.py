"""Bound deployment comparisons without deleting the research program bank."""
import inspect
import json
import math
import os
from pathlib import Path
import time

def choose(names, selected, previous, scores, cap=5):
    # selected is required by the original planner's global-winner contract.
    chosen = list(dict.fromkeys(n for n in ['raw', previous, selected] if n and n in names))
    def score(n):
        v = scores.get(n, {}).get('SQL')
        return v if isinstance(v, (int, float)) and math.isfinite(v) else float('inf')
    for n in sorted(names, key=lambda n: (score(n), n)):
        if len(chosen) >= cap: break
        if n not in chosen: chosen.append(n)
    return chosen

def deployment_names(folder, bank, selected, epoch, requested):
    folder = Path(folder); index = int(folder.name.split('-')[-1])
    previous_path = folder.parent / f'window-{index-1:03d}' / 'DECISION.json'
    previous = json.loads(previous_path.read_text()).get('program') if previous_path.exists() else None
    names = list(dict.fromkeys(['raw', selected, *epoch.get('scores', {}), *requested]))
    if previous in bank.records and previous not in names: names.append(previous)
    chosen = choose(names, selected, previous, epoch.get('scores', {}))
    folder.mkdir(parents=True, exist_ok=True)
    from timersi_u1.core import write
    write(folder/'DEPLOYMENT-POOL182.json', dict(at=time.time(), policy='182-deployment5',
        window=index, before=len(names), after=len(chosen), chosen=chosen, previous=previous,
        selected=selected, scores_source='completed historical S', all_bank_records_retained=len(bank.records),
        scope='5 existing top-level candidates; recursive dependencies and new native requests are additional'))
    return chosen

def install():
    from timersi_u1 import window_planner
    source = inspect.getsource(window_planner.deploy)
    old = "names = list(dict.fromkeys(['raw',selected,*epoch.get('scores',{}),*requested]))"
    if source.count(old) != 1: raise RuntimeError('Unexpected deployment implementation')
    source = source.replace(old, 'names = deployment_names182(folder, bank, selected, epoch, requested)')
    window_planner.deployment_names182 = deployment_names
    exec(compile(source, str(Path(__file__)), 'exec'), window_planner.__dict__)
    root = Path(os.environ['TIMERSI_DEPLOY182_ROOT'])
    from timersi_u1.core import write
    write(root/'DEPLOY182-INSTALL.json', dict(at=time.time(), pid=os.getpid(), cap=5,
        scope='all subsequent deployment windows, research pool remains 181', source=str(Path(__file__).parent)))

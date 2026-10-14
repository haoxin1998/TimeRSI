"""Apply the frozen 189 rule at complete-program boundaries, before scoring."""
import json
import os
from pathlib import Path
import time
import numpy as np
from .numerical_guard import recover

POLICY = ('All complete candidate forecasts are checked before research/selection '
          'scoring and deployment. For an entity-target, any nonfinite value or '
          'prediction magnitude exceeding 100 times the maximum observed-history '
          'and current Raw magnitude causes its full point/quantile horizon to '
          'revert to Raw. Feedback scores include this fallback. Inspect '
          'numerical_guard feedback and fix unstable code; fallback is not evidence '
          'that the unguarded program is accurate. No current or future labels '
          'are used. Internal program exceptions remain ordinary failed proposals.')


def protect(result, raw, view, name, cache, spec):
    from .core import Forecast
    histories = [np.where(item['observed'], item['y'], np.nan) for item in view.items]
    p, q, mask, evidence = recover(result.point, result.quantiles,
                                  raw.point, raw.quantiles, histories)
    triggers = [dict(id=str(view.items[i]['id']), target=str(view.targets[d]),
                     nonfinite=bool(evidence['nonfinite'][i,d]),
                     excessive=bool(evidence['excessive'][i,d]),
                     scale=float(evidence['scale'][i,d]),
                     peak=float(evidence['peak'][i,d])) for i,d in np.argwhere(mask)]
    dependencies = {n:getattr(cache[n], 'numerical_guard', {}) for n in spec.get('members', [])
                    if n in cache and getattr(cache[n], 'numerical_guard', {}).get('affected')}
    note = dict(program=name, direct_fallbacks=int(mask.sum()), triggers=triggers,
                affected=bool(triggers or dependencies), guarded_dependencies=dependencies)
    output = Forecast(p, q)
    output.numerical_guard = note
    root = os.environ.get('TIMERSI_E185_ROOT')
    if root and note['affected']:
        with (Path(root)/'NUMERICAL-GUARD.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=time.time(), pid=os.getpid(),
                forecast_starts=[str(i['timestamps'][i['y'].shape[1]]) for i in view.items],
                **note), allow_nan=False)+'\n')
    return output

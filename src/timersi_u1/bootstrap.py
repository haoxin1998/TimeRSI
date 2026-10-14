"""Initialize TimeRSI's task-local runtime."""
import functools
import os
from pathlib import Path
import time
from .timing import event, phase


def install():
    from .compute import install as compute_install
    compute_install()
    root = os.environ['TIMERSI_E185_ROOT']
    for name in ['BALANCED_ROOT', 'POOL181_ROOT', 'DEPLOY182_ROOT']:
        os.environ['TIMERSI_'+name] = root
    from . import eff_budget, eff_pool, eff_deploy
    eff_budget.install()
    eff_pool.install()
    eff_deploy.install()
    from .core import ProgramBank, Forecast, write
    from .eff_forecast_cache import install as cache_install
    class Events:
        def append(self, row):
            row = dict(row)
            event(row.pop('event'), **row)
    cache_install(ProgramBank, Forecast, Events(), max_bytes=2*1024**3)
    from . import search, window_planner
    def timed(original, name):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            with phase(name):
                return original(*args, **kwargs)
        return wrapper
    search.search = timed(search.search, 'research_including_probes')
    window_planner.deploy = timed(window_planner.deploy, 'deployment_including_candidates_and_agent')
    write(Path(root)/'E185-INSTALL.json', dict(at=time.time(), pid=os.getpid(),
        version='TimeRSI-FEV', numerical_guard='entity-target bounded Raw fallback', balanced_budget=True, research_old_pool=12, deployment_old_pool=5,
        full_prediction_cache_bytes=2*1024**3, generic_lightgbm_fit_cache=True,
        initial_budget_unchanged=True))

"""One evaluator process per task: caches, hooks and mutable state never cross tasks."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import fev
import numpy as np
from .core import Forecast,write
from .data import predictions
from .fev_adapter import TimeRSIConfig,run_fev_task

def fit_predict(task,config):
    root=Path(config.run_root).resolve();root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='task-request-',dir=root) as temporary:
        request=Path(temporary)/'request.json'
        write(request,dict(task=task.to_dict(),config=asdict(config)))
        started=time.monotonic()
        subprocess.run([sys.executable,'-m','timersi_u1.task_process',str(request)],check=True)
        receipt=json.loads(request.with_suffix('.result.json').read_text())
        directory=Path(receipt['run_dir']);runtime=receipt['runtime']
        forecasts=[]
        for i in range(runtime['windows']):
            with np.load(directory/'composition/tasks/task/official'/f'window-{i:03d}'/'predictions.npz',allow_pickle=False) as saved:
                forecasts.append(predictions(Forecast(saved['point'],saved['quantiles']).validate(),task))
        total=time.monotonic()-started
        runtime['inference_time_s']+=max(0.,total-runtime['total_time_s'])
        runtime['total_time_s']=total
        return forecasts,runtime,directory

if __name__=='__main__':
    request=Path(sys.argv[1]);value=json.loads(request.read_text())
    _,runtime,directory=run_fev_task(fev.Task(**value['task']),TimeRSIConfig(**value['config']))
    write(request.with_suffix('.result.json'),dict(run_dir=str(directory),runtime=runtime))

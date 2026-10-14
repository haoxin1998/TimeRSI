"""FEV adapter: task-local research followed by causal composition feedback."""
from argparse import Namespace
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from .core import Forecast,write
from .data import predictions

@dataclass(frozen=True)
class TimeRSIConfig:
    checkpoint_path: str
    vertex_project: str
    run_root: str = 'timersi-runs'
    device: str = 'cuda:0'
    iterations: int = 12
    official_windows: int = 0
    online_updates: int = -1
    agent_timeout: int = 900

def run_fev_task(task,config):
    root=Path(config.run_root).resolve();root.mkdir(parents=True,exist_ok=True)
    run_dir=Path(tempfile.mkdtemp(prefix=task.task_name.replace('/','-')+'-',dir=root))
    source=run_dir/'research';source.mkdir()
    os.environ['TIMERSI_E185_ROOT']=str(source)
    os.environ['TIMERSI_E185_CONTROL']=str(root/'shared-compute')
    args=Namespace(task=task.task_name,tasks_yaml='',data_root=os.environ.get('FEV_DATASETS_PREFIX',''),
        checkpoint=config.checkpoint_path,output=str(source),device=config.device,max_items=64,
        iterations=config.iterations,official_windows=config.official_windows,research_windows=3,selection_windows=1,
        selection_objective='mean-sql',online_updates=config.online_updates,update_every=2,update_iterations=4,
        model='fable',effort='high',vertex_project=config.vertex_project,vertex_region='global',
        agent_timeout=config.agent_timeout,deployment_policy='global',research_design='old-style',parent_mixtures=[],
        resume=False,frozen_control=False,top3=False,interactive_research=True,current_window_planner=True,
        forecast_tool_requests=8,window_evidence='trajectory',reselect_every=0)
    from .bootstrap import install
    install()
    from .run import run
    started=time.monotonic()
    run(args,task_override=task)
    research_seconds=time.monotonic()-started
    count=config.official_windows or task.num_windows
    output=run_dir/'composition';(output/'source').mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name('composition_probe.py'),output/'source/composition_probe.py')
    job=dict(task=task.task_name,replay_label='task',expected_windows=count)
    from .composition import worker
    worker(dict(output=str(output),jobs=[job],vertex_project=config.vertex_project,
        scope='Current-origin candidates and historical feedback from this task-local research run.'),'task',source)
    deployment_seconds=time.monotonic()-started-research_seconds
    forecasts=[];num_forecasts=0
    for index in range(count):
        path=output/'tasks/task/official'/f'window-{index:03d}'/'predictions.npz'
        with np.load(path,allow_pickle=False) as saved:
            forecast=Forecast(saved['point'],saved['quantiles']).validate()
        num_forecasts+=forecast.point.shape[0]*forecast.point.shape[2]
        forecasts.append(predictions(forecast,task))
    # Research includes the first candidate-generation/deployment pass. Nothing is omitted.
    elapsed=time.monotonic()-started
    runtime=dict(training_time_s=research_seconds,inference_time_s=elapsed-research_seconds,
        num_forecasts=num_forecasts,total_time_s=elapsed,
        allocation='Pipeline research/adaptation including initial decisions; inference includes final chronological composition.',
        full_task=count==task.num_windows,windows=count)
    write(run_dir/'RUNTIME.json',runtime)
    write(run_dir/'COMPLETE.json',dict(task=task.task_name,**runtime))
    return forecasts,runtime,run_dir

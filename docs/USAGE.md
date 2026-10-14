# Running TimeRSI

Complete the environment setup in the [README](../README.md). Credentials remain on the host; code-agent sessions and generated candidates use Docker isolation.

## Model and provider

Use `google/timesfm-3.0-pytorch` or pass a local checkpoint directory. Set `TIMERSI_VERTEX_PROJECT` to your authorized Vertex AI project and run `gcloud auth application-default login`. Install Claude Code 2.1.245; set `CLAUDE_CODE_BINARY` if its executable is not on PATH.

## FEV evaluation

From the TimeRSI checkout, create a separate FEV checkout and copy the thin wrapper:

```bash
git clone https://github.com/autogluon/fev.git ../fev
cp -R fev_submission/models/timersi ../fev/models/timersi
cd ../fev
python models/evaluate.py -m timersi -n TimeRSI \
  -b benchmarks/fev_bench/tasks_mini.yaml \
  -k '{"checkpoint_path":"/path/to/timesfm-3.0-pytorch"}'
```

Use `benchmarks/fev_bench/tasks.yaml` for the full benchmark. The wrapper installs the released TimeRSI package through its requirements file. When using a local checkpoint, provide its absolute path.

## A single task

The installed package exposes the task runner directly:

```python
import os
from pathlib import Path
import fev
import yaml
from timersi_u1.fev_adapter import TimeRSIConfig
from timersi_u1.task_process import fit_predict

definitions = yaml.safe_load(Path('../fev/benchmarks/fev_bench/tasks_mini.yaml').read_text())['tasks']
task = next(fev.Task(**entry) for entry in definitions
            if fev.Task(**entry).task_name == 'hospital_admissions_1W')
forecasts, timing, run_dir = fit_predict(task, TimeRSIConfig(
    checkpoint_path='/path/to/timesfm-3.0-pytorch',
    vertex_project=os.environ['TIMERSI_VERTEX_PROJECT'],
))
print(run_dir)
print(timing)
```

This interface accepts a `fev.Task`. It loads that task's declared dataset and produces all its evaluation windows. Custom data must be described through the FEV task/dataset interface.

## Configuration and outputs

Defaults use Fable 5/high, up to 12 initial probes, adaptive follow-up budgets of 2–4 probes, three research folds and one selection fold when available, and 64 research entities. Formal forecasts use the complete task population.

Each task saves its research, chronological composition, forecasts and timing beneath `timersi-runs/`. Forecast-origin decisions use historical evidence and matured feedback. Fresh agent searches are stochastic. Use a service manager or detached session for long runs.

The package's runner and defaults are shared with [TimeRSI4FevBench](https://github.com/haoxin1998/TimeRSI4FevBench); benchmark results and inspection artifacts are maintained there.

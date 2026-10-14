# TimeRSI

**Time-Series Foundation Model Deployment Needs a Self-Improving Harness**

TimeRSI pairs frozen TimesFM 3.0 with a code agent that develops task-specific forecasting pipelines. The agent revises features, model inputs and forecast corrections using historical backtests; forecast-time selection and composition use available task context and feedback from completed windows. TimesFM weights remain frozen.

## Environment

Linux, Python 3.12, a CUDA-capable NVIDIA GPU, Docker, Claude Code 2.1.245, and access to Fable 5 through Vertex AI.

```bash
git clone https://github.com/haoxin1998/TimeRSI.git
cd TimeRSI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git@45e0a3bc7fc4acef17b7ba7910488be2159bae5f'
docker build -t timersi-isolation:1.1 -f docker/Dockerfile docker
export TIMERSI_VERTEX_PROJECT=YOUR_PROJECT
gcloud auth application-default login
```

Install Claude Code and make `claude` available on PATH. Use `google/timesfm-3.0-pytorch` or a local checkpoint. See the [usage guide](docs/USAGE.md) for running a task or benchmark.

## Code

- [Task runner](src/timersi_u1/fev_adapter.py): code research followed by chronological forecast composition.
- [Code research](src/timersi_u1/search.py) and [forecast composition](src/timersi_u1/composition.py).
- **Execution environment:** The public runner uses Docker to isolate code-agent sessions and generated candidates. [Implementation](src/timersi_u1/isolation.py).

## Benchmarks

[FEV-Bench results, evaluation instructions and mini20 inspection trajectories](https://github.com/haoxin1998/TimeRSI4FevBench) · [TIME result files](https://huggingface.co/datasets/Real-TSF/TIME-Output/tree/main/results/TimeRSI).

TimesFM weights have their own license; see the [checkpoint license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).

Updated at 2026-10-14 02:02 PDT.

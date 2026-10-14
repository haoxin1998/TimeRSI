"""Append-only phase evidence; task elapsed time is never a sum of nested spans."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import uuid

_lock = threading.Lock()
_diagnostics = []


def event(kind, **values):
    root = os.environ.get('TIMERSI_E185_ROOT')
    if not root:
        return
    folder = Path(root) / 'timing'
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(event=kind, at=time.time(), monotonic=time.monotonic(), pid=os.getpid(), **values)
    with _lock, (folder / f'events-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(record, default=str) + '\n')


@contextmanager
def phase(name, diagnostic=False, **values):
    token = uuid.uuid4().hex
    started = time.monotonic()
    event('phase_start', phase=name, span=token, diagnostic=diagnostic, **values)
    status = 'complete'
    try:
        yield
    except BaseException:
        status = 'failed'
        raise
    finally:
        ended = time.monotonic()
        if diagnostic:
            with _lock:
                _diagnostics.append((started, ended))
        event('phase_end', phase=name, span=token, diagnostic=diagnostic,
              seconds=ended-started, status=status, **values)


def union_seconds(intervals):
    end = None
    total = 0.
    for left, right in sorted(intervals):
        if end is None or left >= end:
            total += right-left
        elif right > end:
            total += right-end
        end = right if end is None else max(end, right)
    return total


def diagnostic_seconds():
    return union_seconds(_diagnostics)


def summarize_attempts(attempts, num_forecasts=None):
    ordered = sorted(attempts, key=lambda a:a['started_unix'])
    active = sum(a['active_seconds'] for a in ordered)
    diagnostic = sum(a.get('diagnostic_seconds', 0.) for a in ordered)
    gaps = sum(max(0., b['started_unix']-a['ended_unix']) for a,b in zip(ordered, ordered[1:]))
    result = dict(training_time_s=0., inference_time_s=active-diagnostic,
                  active_attempt_seconds=active, diagnostic_seconds=diagnostic,
                  interruption_gap_s=gaps,
                  failed_attempt_s=sum(a['active_seconds'] for a in ordered if a['status'] != 'complete'),
                  attempts=len(ordered), num_forecasts=num_forecasts,
                  model_service_setup_s=0., model_service_mode='pre-existing warm service; idle excluded',
                  allocation='Complete deployment request, including CPU fitting and LLM search, in inference')
    if num_forecasts:
        result['runtime_s_per100'] = 100*result['inference_time_s']/num_forecasts
    return result

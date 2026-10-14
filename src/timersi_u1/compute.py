"""Generic exact-fit reuse for new LightGBM programs, plus shared CPU admission."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import threading
import time

from .timing import event

_local = threading.local()


@contextmanager
def cpu_slot():
    if getattr(_local, 'admitted', False):
        yield
        return
    folder = Path(os.environ['TIMERSI_E185_CONTROL']) / 'cpu-slots'
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = int(os.environ.get('TIMERSI_E185_CPU_SLOTS', '10'))
    handle = None
    while handle is None:
        for index in range(count):
            candidate = (folder / f'{index:02d}.lock').open('a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = candidate
                break
            except BlockingIOError:
                candidate.close()
        if handle is None:
            time.sleep(.04)
    _local.admitted = True
    event('cpu_admitted', wait_seconds=time.monotonic()-started)
    try:
        yield
    finally:
        _local.admitted = False
        handle.close()


def safe_parameters(value):
    if callable(value):
        return False
    if isinstance(value, dict):
        return all(safe_parameters(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(safe_parameters(v) for v in value)
    return True


def install():
    import lightgbm as lgb
    import numpy as np
    if getattr(lgb, '_e185_installed', False):
        return
    lgb._e185_installed = True
    original_fit = lgb.LGBMModel.fit
    original_train = lgb.train
    root = Path(os.environ['TIMERSI_E185_ROOT'])
    cache = root / 'fit-cache'
    cache.mkdir(parents=True, exist_ok=True)
    minimum = float(os.environ.get('TIMERSI_E185_MIN_FIT_SECONDS', '1'))
    cap_bytes = int(os.environ.get('TIMERSI_E185_FIT_CACHE_BYTES', str(8*1024**3)))

    def fit(self, *args, **kwargs):
        old_threads = self.n_jobs
        if old_threads is None or old_threads <= 0 or old_threads > 8:
            self.set_params(n_jobs=8)
        params = self.get_params(deep=True)
        # Callbacks/custom objectives have observable state; let them execute.
        eligible = (safe_parameters(params) and safe_parameters(args) and not kwargs.get('callbacks')
                    and kwargs.get('init_model') is None and safe_parameters(kwargs))
        started = time.monotonic()
        key = None
        if eligible:
            try:
                payload = (lgb.__version__, np.__version__, type(self).__module__,
                           type(self).__qualname__, params, args, kwargs)
                key = hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()
            except (TypeError, pickle.PicklingError):
                eligible = False
        key_seconds = time.monotonic()-started
        lock = (cache / f'{key}.lock').open('a') if eligible else None
        try:
            if lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            path = cache / f'{key}.pkl' if eligible else None
            if path and path.exists():
                restored = pickle.loads(path.read_bytes())
                self.__dict__.clear()
                self.__dict__.update(restored.__dict__)
                event('lightgbm_fit', hit=True, key=key, key_seconds=key_seconds,
                      seconds=time.monotonic()-started, actual_threads=self.n_jobs)
                return self
            with cpu_slot():
                compute_started = time.monotonic()
                result = original_fit(self, *args, **kwargs)
                compute_seconds = time.monotonic()-compute_started
            saved = False
            if path and compute_seconds >= minimum:
                # Admission limit; keep existing successful cache entries intact.
                used = sum(p.stat().st_size for p in cache.glob('*.pkl'))
                if used < cap_bytes:
                    content = pickle.dumps(self, protocol=5)
                    if used + len(content) <= cap_bytes:
                        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
                        temporary.write_bytes(content)
                        temporary.replace(path)
                        saved = True
            event('lightgbm_fit', hit=False, key=key, key_seconds=key_seconds,
                  seconds=time.monotonic()-started, compute_seconds=compute_seconds,
                  cached=saved, requested_threads=old_threads, actual_threads=self.n_jobs)
            return result
        finally:
            if lock:
                lock.close()

    def train(params, *args, **kwargs):
        params = dict(params)
        aliases = ('num_threads','num_thread','nthread','n_jobs')
        requested = next((params[k] for k in aliases if k in params), 8)
        threads = 8 if requested is None or requested <= 0 else min(8, int(requested))
        for name in aliases:
            params.pop(name, None)
        params['num_threads'] = threads
        with cpu_slot():
            begin = time.monotonic()
            output = original_train(params, *args, **kwargs)
        event('lightgbm_native_train', seconds=time.monotonic()-begin, actual_threads=threads)
        return output

    lgb.LGBMModel.fit = fit
    lgb.train = train
    event('compute_installed', thread_cap=8, slots=int(os.environ.get('TIMERSI_E185_CPU_SLOTS','10')),
          fit_cache_bytes=cap_bytes, minimum_fit_seconds=minimum)

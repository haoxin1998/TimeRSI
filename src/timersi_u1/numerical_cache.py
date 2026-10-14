"""Opt-in cache for a pure numerical step, scoped to the complete program."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def cache_key(program, operation, inputs):
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('Use typed, non-object arrays for numerical cache inputs')
            digest.update(json.dumps(['array',value.dtype.str,value.shape]).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            digest.update(json.dumps(['dict',len(value)]).encode())
            for key in sorted(value):
                add(key);add(value[key])
        elif isinstance(value, (list,tuple)):
            digest.update(json.dumps([type(value).__name__,len(value)]).encode())
            for item in value:add(item)
        else:
            if isinstance(value,np.generic):
                digest.update(json.dumps(['np-scalar',value.dtype.str]).encode())
                add(value.item());return
            digest.update(json.dumps(['scalar',value],sort_keys=True).encode())
    add([program,operation,inputs])
    return digest.hexdigest()


def cached_array(tools, operation, inputs, compute):
    """Caller includes every input/parameter; stochastic/stateful work is not pure."""
    program=getattr(tools,'active_program',None) or 'standalone'
    key=cache_key(program,operation,inputs)
    output=getattr(tools,'output',None)
    started=time.monotonic()
    def calculate():
        result=np.asarray(compute()).copy()
        if result.dtype.hasobject:raise TypeError('Numerical cache outputs must be non-object arrays')
        return result
    if output is None:
        cache=tools.__dict__.setdefault('_numerical_arrays',{})
        hit=key in cache
        if not hit:cache[key]=calculate()
        result=cache[key].copy()
    else:
        folder=Path(output)/'numerical';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(key+'.npz');hit=path.exists()
        if hit:
            with np.load(path,allow_pickle=False) as data:result=data['array']
        else:
            result=calculate()
            np.savez_compressed(path,array=result)
        with (folder/'EVENTS.jsonl').open('a') as stream:
            stream.write(json.dumps({'key':key,'program':program,'operation':operation,'hit':hit,
                'seconds':time.monotonic()-started,'shape':result.shape})+'\n')
    name='numerical_cache_hits' if hit else 'numerical_cache_misses'
    setattr(tools,name,getattr(tools,name,0)+1)
    return result

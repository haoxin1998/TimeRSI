"""JSON-only RPC encoding. Never deserialize untrusted Python objects on the host."""
import base64
import json
import socket
import numpy as np

def pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'O': return [pack(x) for x in value.tolist()]
        return {'__array__': base64.b64encode(value.tobytes()).decode(), 'dtype': value.dtype.str, 'shape': value.shape}
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):pack(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [pack(v) for v in value]
    return value

def unpack(value):
    if isinstance(value,dict):
        if '__array__' in value:
            dtype=np.dtype(value['dtype'])
            if dtype.hasobject: raise ValueError('Object arrays are not accepted')
            return np.frombuffer(base64.b64decode(value['__array__'],validate=True),dtype=dtype).reshape(value['shape']).copy()
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list): return [unpack(v) for v in value]
    return value

def call(path, payload):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(1800);sock.connect(str(path))
        with sock.makefile('rwb') as stream:
            stream.write(json.dumps(pack(payload)).encode()+b'\n');stream.flush()
            result=json.loads(stream.readline())
    if 'error' in result: raise RuntimeError(result['error'])
    return unpack(result['result'])

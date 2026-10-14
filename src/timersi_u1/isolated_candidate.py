"""Candidate process entrypoint; only supplied context and numeric native RPC exist."""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .isolation_wire import call
from .tools import Tools
from .program import load_program

class NumericBackend:
    def predict_batch(self, **kwargs):
        outputs=call('/broker/native.sock',kwargs)
        return [SimpleNamespace(forecast=p,quantiles=q) for p,q in outputs]

def main():
    root=Path(sys.argv[1])
    # This input is written by the trusted evaluator; nothing pickled returns to it.
    with (root/'context.pkl').open('rb') as stream: context=pickle.load(stream)
    program=json.loads((root/'program.json').read_text())
    tools=Tools('resident-foundation-model','cpu',root/'native',backend=NumericBackend())
    with tools.record_native_calls() as records:
        result=load_program(program['code'],program.get('files'))['predict'](context['view'],tools,context['memory'])
    np.savez_compressed(root/'forecast.npz',point=result.point,quantiles=result.quantiles)
    (root/'native-records.json').write_text(json.dumps(dict(records=records,calls=tools.calls,hits=tools.hits)))

if __name__=='__main__':main()

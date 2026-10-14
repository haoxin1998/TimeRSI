"""Bounded complete-forecast reuse for the existing immutable ProgramBank API."""
from collections import OrderedDict
from copy import deepcopy
import contextvars
import hashlib
import pickle
import time
import weakref

CONTEXT=contextvars.ContextVar('forecast183_context',default=None)

def install(bank_class,forecast_class,events,max_bytes=256*1024*1024):
    original=bank_class.predict
    states=weakref.WeakKeyDictionary()
    def predict(self,name,view,tools,memory,cache):
        # Actual native-call receipts must still be recorded by the original path.
        if name in cache or getattr(tools,'native_records',None) is not None:
            return original(self,name,view,tools,memory,cache)
        context=CONTEXT.get();identity=(id(self),id(view),id(tools),id(memory))
        token=None;key_seconds=0.
        if context is None or context[0]!=identity:
            started=time.monotonic()
            payload=(view,memory,id(tools),getattr(tools,'checkpoint',None),getattr(tools,'raw',None))
            digest=hashlib.sha256(pickle.dumps(payload,protocol=5)).digest()
            key_seconds=time.monotonic()-started
            context=(identity,digest);token=CONTEXT.set(context)
        state=states.setdefault(self,dict(values=OrderedDict(),bytes=0))
        key=(name,context[1]);values=state['values'];started=time.monotonic()
        try:
            hit=key in values
            if hit:
                point,quantiles,guard=values.pop(key);values[key]=(point,quantiles,guard)
                result=forecast_class(point.copy(),quantiles.copy()).validate();cache[name]=result
                result.numerical_guard=deepcopy(guard)
            else:
                result=original(self,name,view,tools,memory,cache)
                size=result.point.nbytes+result.quantiles.nbytes
                if size<=max_bytes:
                    while values and state['bytes']+size>max_bytes:
                        _,(point,quantiles,_)=values.popitem(last=False);state['bytes']-=point.nbytes+quantiles.nbytes
                    values[key]=(result.point.copy(),result.quantiles.copy(),deepcopy(getattr(result,'numerical_guard',{})));state['bytes']+=size
            events.append(dict(event='complete_forecast',program=name,hit=hit,
                key_seconds=key_seconds,seconds=time.monotonic()-started,
                cache_bytes=state['bytes'],entries=len(values)))
            return result
        finally:
            if token is not None:CONTEXT.reset(token)
    bank_class.predict=predict
    return original

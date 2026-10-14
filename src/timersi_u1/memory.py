"""Prediction-error memory, released by real per-entity observation timestamps."""
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
from .core import write


class MaturedMemory:
    def __init__(self, folder, decoded_cache_bytes=256*1024**2):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.records={}
        self.decoded_cache_bytes=decoded_cache_bytes
        self._decoded=OrderedDict();self._decoded_bytes=0
        self.decode_hits=0;self.decode_misses=0
        for p in sorted(self.folder.glob('*.json')):
            row=json.loads(p.read_text());self.records[row['key']]=row

    def _arrays(self, name):
        # observe() never rewrites a committed key. Only private immutable arrays
        # are reused; NumPy's integer indexing below gives callers fresh copies.
        if name in self._decoded:
            self.decode_hits+=1;self._decoded.move_to_end(name)
            return self._decoded[name]
        self.decode_misses+=1
        with np.load(self.folder/name) as z:
            arrays={key:z[key] for key in ('point','quantiles','truth')}
        size=sum(a.nbytes for a in arrays.values())
        if size<=self.decoded_cache_bytes:
            while self._decoded and self._decoded_bytes+size>self.decoded_cache_bytes:
                _,old=self._decoded.popitem(last=False)
                self._decoded_bytes-=sum(a.nbytes for a in old.values())
            for a in arrays.values():a.flags.writeable=False
            self._decoded[name]=arrays;self._decoded_bytes+=size
        return arrays

    def observe(self, key, source, view, forecast, truth, official_sql=None):
        """Only called after forecasts are saved; visibility is separately enforced."""
        if key in self.records:return
        truth=np.asarray(truth,float)
        if truth.shape!=forecast.point.shape:raise ValueError('Memory truth/forecast shapes differ')
        path=self.folder/(key+'.npz')
        np.savez_compressed(path,point=forecast.point,quantiles=forecast.quantiles,truth=truth)
        row={'key':key,'source':source,'arrays':path.name,'targets':list(view.targets),
             'ids':[it['id'] for it in view.items],
             'starts':[str(it['timestamps'][it['y'].shape[1]]) for it in view.items],
             'ends':[str(it['timestamps'][-1]) for it in view.items]}
        if official_sql is not None:
            row['official_SQL_by_target']=[official_sql[str(it['id'])] for it in view.items]
        write(self.folder/(key+'.json'),row);self.records[key]=row

    def visible(self, view):
        cutoffs={it['id']:np.datetime64(it['timestamps'][it['y'].shape[1]-1]) for it in view.items}
        result=[]
        for row in self.records.values():
            keep=[i for i,item in enumerate(row['ids']) if item in cutoffs and
                  np.datetime64(row['ends'][i])<=cutoffs[item]]
            if not keep:continue
            z=self._arrays(row['arrays'])
            result.append({'source':row['source'],'key':row['key'],'targets':row['targets'],
                    'ids':[row['ids'][i] for i in keep],'ends':[row['ends'][i] for i in keep],
                    **({'starts':[row['starts'][i] for i in keep]} if 'starts' in row else {}),
                    **({'official_SQL_by_target':[row['official_SQL_by_target'][i] for i in keep]}
                       if 'official_SQL_by_target' in row else {}),
                    'point':z['point'][keep],'quantiles':z['quantiles'][keep],
                    'truth':z['truth'][keep]})
        return result

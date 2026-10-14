"""Exact CPU scoring and typed R fixtures for agent-side numerical analysis."""
import json
import tempfile
from pathlib import Path
import numpy as np
from .core import Forecast
from .data import View


def load_fold(directory, offset):
    directory=Path(directory)
    index=json.loads((directory/'INDEX.json').read_text())
    spec=next(row for row in index['folds'] if row['offset']==offset)
    with np.load(directory/spec['arrays'],allow_pickle=False) as arrays:
        data={key:arrays[key].copy() for key in arrays.files}
    items=[]
    for row in spec['items']:
        item={'id':row['id'],'static':row['static'],'y':data[row['y']],
              'timestamps':data[row['timestamps']].astype('datetime64[ns]')}
        if 'observed' in row:item['observed']=data[row['observed']].astype(bool)
        for kind in ('known','past','known_raw','past_raw'):
            item[kind]={name:data[key] for name,key in row[kind].items()}
        items.append(item)
    view=View(items,spec['horizon'],spec['seasonality'],spec['targets'],spec['frequency'],spec['facts'])
    raw=Forecast(data['raw_point'],data['raw_quantiles'])
    return view,raw,{'spec':spec,'arrays':data}


def score_forecast(forecast, evidence):
    """Same FEV metric classes, scaling histories, masks and epsilon as executor."""
    import fev.metrics
    spec=evidence['spec'];data=evidence['arrays']
    forecast.validate()
    kwargs={'y_true':data['truth'],'y_pred':forecast.point,'q_pred':forecast.quantiles,
            'y_past':data['metric_y_past'],'y_past_lengths':data['metric_y_past_lengths'],
            'seasonality':spec['seasonality'],'quantile_levels':spec['quantile_levels']}
    return {row['name']:float(np.nanmean(getattr(fev.metrics,row['name'])(**row['kwargs']).compute(**kwargs)))
            for row in spec['metrics']}


class CachedRaw:
    """Only an unchanged Raw call is available in this CPU-only fixture."""
    def __init__(self,view,raw,output=None):
        self.view,self.raw=view,raw
        self.output=output

    def cached_array(self, operation, inputs, compute):
        from .numerical_cache import cached_array
        return cached_array(self,operation,inputs,compute)

    def timesfm(self,view,**kwargs):
        if view is not self.view or kwargs:
            raise ValueError('Only the original Raw view is cached; native variants require executor evaluation')
        return Forecast(self.raw.point.copy(),self.raw.quantiles.copy())

    def shift(self,forecast,delta):
        from .tools import Tools
        return Tools.shift(self,forecast,delta)

    def spread(self,forecast,factor=1.):
        from .tools import Tools
        return Tools.spread(self,forecast,factor)

    def blend(self,predictions,weights):
        from .core import blend
        return blend(predictions,weights)


def load_native_tools(directory, output):
    """Replay exact, previously evaluated R inputs without loading a foundation model."""
    from .tools import Tools
    directory = Path(directory)
    index = json.loads((directory/'INDEX.json').read_text())
    return Tools(index['checkpoint'], 'cpu', output, cache_only=True, native_cache=directory)


def run_local(directory, offset, program_directory, output=None, native_directory=None, memory=()):
    """Run a CPU program with Raw or an exact per-fold native archive, never GPU."""
    from .program import load_program
    from .core import write
    directory=Path(directory);program_directory=Path(program_directory)
    view,raw,evidence=load_fold(directory,offset)
    files={str(p.relative_to(program_directory)):p.read_text()
           for p in program_directory.rglob('*.py') if '__pycache__' not in p.parts}
    program=load_program(files['candidate.py'],files)
    with tempfile.TemporaryDirectory(prefix='timersi-r-replay-') as scratch:
        tools = CachedRaw(view,raw,scratch) if native_directory is None else load_native_tools(native_directory,scratch)
        forecast=program['predict'](view,tools,list(memory))
    metrics=score_forecast(forecast,evidence)
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'predictions.npz',point=forecast.point,quantiles=forecast.quantiles)
        write(output/'METRICS.json',metrics)
    return forecast,metrics

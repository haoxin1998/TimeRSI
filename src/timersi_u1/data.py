"""FEV views: candidates receive inputs, never the scoring window object."""
from dataclasses import dataclass,field
import dataclasses
import hashlib
import warnings
import numpy as np
import datasets


def fill(values):
    a = np.asarray(values)
    if a.dtype.kind in 'OUS':
        a = np.asarray([float(abs(hash(x)) % 100) if isinstance(x,str) else 0. for x in a])
    a = np.asarray(a, np.float32).copy()
    good = np.isfinite(a)
    if not good.any(): return np.zeros_like(a)
    indices = np.maximum.accumulate(np.where(good, np.arange(len(a)), 0))
    a = a[indices]
    a[:np.flatnonzero(good)[0]] = a[np.flatnonzero(good)[0]]
    return a


@dataclass
class View:
    items: list
    horizon: int
    seasonality: int
    targets: list
    frequency: str
    facts: dict = field(default_factory=dict)


def make_view(window, task):
    past, future = window.get_input_data()
    items = []
    for p,f in zip(past, future):
        items.append({'id': str(p[task.id_column]),
            'y': np.stack([fill(p[n]) for n in task.target_columns]),
            'observed':np.stack([np.isfinite(np.asarray(p[n],float)) for n in task.target_columns]),
            'timestamps': np.concatenate([p[task.timestamp_column],f[task.timestamp_column]]),
            'known': {n: np.concatenate([fill(p[n]),fill(f[n])]) for n in task.known_dynamic_columns},
            'past': {n: fill(p[n]) for n in task.past_dynamic_columns},
            'known_raw': {n:list(p[n])+list(f[n]) for n in task.known_dynamic_columns},
            'past_raw': {n:list(p[n]) for n in task.past_dynamic_columns},
            'static': {n: p[n] for n in task.static_columns}})
    facts={'task':task.task_name,'description':getattr(task,'task_description',None),
        'column_descriptions':getattr(task,'column_descriptions',None)}
    return View(items, task.horizon, task.seasonality, list(task.target_columns), task.freq,facts)


def predictions(forecast, task):
    return datasets.DatasetDict({n: datasets.Dataset.from_dict({
        'predictions': forecast.point[:,:,j].tolist(),
        **{str(q): forecast.quantiles[:,:,j,k].tolist() for k,q in enumerate(task.quantile_levels)}
    }) for j,n in enumerate(task.target_columns)})


class EvaluationContext:
    """Judge-private immutable window data, prepared once across candidates."""
    def __init__(self, window, task):
        self.window=window;self.task=task;self.base=None;self.ids=None
        self.prepare_calls=0

    def prepare(self, forecast):
        if self.base is None:
            arrays,self.ids=self.window._prepare_arrays(predictions(forecast,self.task),self.task.quantile_levels)
            self.base={k:arrays[k] for k in ('y_true','y_past','y_past_lengths')}
            self.prepare_calls+=1
        else:
            shape=self.base['y_true'].shape
            if forecast.point.shape!=shape or forecast.quantiles.shape[:3]!=shape:
                raise ValueError('Forecast shape differs from official evaluation window')
            arrays={**self.base,'y_pred':np.array(forecast.point,dtype=np.float64,copy=True),
                'q_pred':np.array(forecast.quantiles[...,:len(self.task.quantile_levels)],dtype=np.float64,copy=True)}
        return PreparedEvaluation(arrays,self.ids,self.task)


class PreparedEvaluation:
    """Reuse the exact official arrays for scoring, error cells and memory."""
    def __init__(self, arrays, ids, task):
        self.arrays=arrays;self.ids=ids;self.task=task

    def score(self):
        scores={}
        # Same metric methods/reductions as FEV EvaluationWindow.compute_metrics.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=RuntimeWarning)
            for metric in self.task.metrics:
                values=metric.compute_scores(**self.arrays,seasonality=self.task.seasonality,
                    quantile_levels=self.task.quantile_levels,reduce_axes=(0,1),per_quantile_scores=False)
                for name,per_dim in values.items():scores[name]=float(np.mean(per_dim))
        return scores

    def official_sql_by_target(self):
        from fev.metrics import SQL
        loss=SQL().compute(**self.arrays,seasonality=self.task.seasonality,
            quantile_levels=self.task.quantile_levels,reduce_axes=(1,))
        return {str(item):loss[i] for i,item in enumerate(self.ids)}

    def error_details(self):
        arrays=self.arrays;task=self.task
        loss=task.metrics[0].compute(**arrays,seasonality=task.seasonality,
            quantile_levels=task.quantile_levels,reduce_axes=(1,))
        bias=np.nanmean(arrays['y_pred']-arrays['y_true'],axis=1)
        coverage=np.nanmean(arrays['y_true'][...,None]<=arrays['q_pred'],axis=1)
        width=np.nanmean(arrays['q_pred'][...,-1]-arrays['q_pred'][...,0],axis=1)
        return {'target_names':list(task.target_columns),'items':[
            {'id':str(item),'SQL_by_target':loss[i],'mean_point_bias_by_target':bias[i],
             'quantile_coverage_by_target':coverage[i],'mean_q90_minus_q10_by_target':width[i]}
            for i,item in enumerate(self.ids)]}


def score(forecast, window, task):
    return window.compute_metrics(predictions(forecast,task), metrics=task.metrics,
        seasonality=task.seasonality, quantile_levels=task.quantile_levels)


def official_sql_by_target(forecast, window, task):
    """Exact FEV SQL cells, retaining IDs and each origin's scaling history."""
    from fev.metrics import SQL
    arrays,ids=window._prepare_arrays(predictions(forecast,task),task.quantile_levels)
    loss=SQL().compute(**arrays,seasonality=task.seasonality,
        quantile_levels=task.quantile_levels,reduce_axes=(1,))
    return {str(item):loss[i] for i,item in enumerate(ids)}


def error_details(forecast, window, task):
    arrays,ids=window._prepare_arrays(predictions(forecast,task),task.quantile_levels)
    loss=task.metrics[0].compute(**arrays,seasonality=task.seasonality,
        quantile_levels=task.quantile_levels,reduce_axes=(1,))
    bias=np.nanmean(arrays['y_pred']-arrays['y_true'],axis=1)
    coverage=np.nanmean(arrays['y_true'][...,None]<=arrays['q_pred'],axis=1)
    width=np.nanmean(arrays['q_pred'][...,-1]-arrays['q_pred'][...,0],axis=1)
    return {'target_names':list(task.target_columns),'items':[
        {'id':str(item),'SQL_by_target':loss[i], 'mean_point_bias_by_target':bias[i],
         'quantile_coverage_by_target':coverage[i],'mean_q90_minus_q10_by_target':width[i]}
        for i,item in enumerate(ids)]}


def historical_windows(task, max_items, total_windows=3, anchor=None):
    first = task.get_window(0, num_proc=1) if anchor is None else anchor
    past,_ = first.get_input_data()
    ids = past[task.id_column]
    # Preserve the original open-ended research population and ordering. This is
    # a deterministic sampling seed, not a task-to-method dispatch rule.
    indexes = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f'{task.task_name}|{ids[i]}|20261005'.encode()).hexdigest())[:max_items]
    subset = past.select(indexes)
    result = {}
    for offset in range(total_windows,0,-1):
        eligible = [i for i,r in enumerate(subset) if len(r[task.timestamp_column])-offset*task.horizon >= task.min_context_length]
        if eligible:
            result[offset] = dataclasses.replace(first, full_dataset=subset.select(eligible),
                                                 cutoff=-offset*task.horizon)
    return result

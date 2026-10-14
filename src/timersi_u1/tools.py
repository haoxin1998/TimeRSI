"""Composable native inputs, reversible transforms, experts and postprocessing."""
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
from .core import Forecast, blend, write


def calendar(timestamps, known, context_length):
    d = pd.DatetimeIndex(timestamps)
    hour = np.asarray(d.hour+d.minute/60., float)
    phases = {'day':hour/24., 'week':(np.asarray(d.dayofweek)+hour/24)/7.,
              'year':(np.asarray(d.dayofyear)-1+hour/24)/365.25}
    return {name+suffix: fn(2*np.pi*phase) for name,phase in phases.items()
            for suffix,fn in (('_sin',np.sin),('_cos',np.cos))}


class Tools:
    def __init__(self, checkpoint, device, output, backend=None, cache_only=False, native_cache=None):
        self.checkpoint, self.device = checkpoint, device
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.backend = backend
        self.calls, self.hits = 0, 0
        self.cache_only = cache_only
        self.native_cache = Path(native_cache) if native_cache is not None else self.output
        self.native_records = None

    @contextmanager
    def record_native_calls(self):
        """Track this evaluation's actual outputs, including exact cache hits."""
        previous = self.native_records
        records = []
        self.native_records = records
        try:
            yield records
        finally:
            self.native_records = previous

    def _record_native(self, key, path):
        if self.native_records is not None:
            self.native_records.append({'native_key': key, 'path': str(path)})

    def _model(self):
        if self.backend is None:
            if os.environ.get('TIMERSI_U1_GPU_POOL'):
                from .gpu_pool import PoolBackend
                self.backend=PoolBackend(os.environ['TIMERSI_U1_GPU_POOL'],self.checkpoint,str(self.output))
                return self.backend
            import torch
            from timesfm3 import TimesFM3Evaluator, ModelConfig
            torch.set_num_threads(2)
            self.backend = TimesFM3Evaluator(ModelConfig(checkpoint_path=self.checkpoint,
                per_core_batch_size=16, use_variate_attention=True,use_sdpa=True,device=self.device))
        return self.backend

    blend = staticmethod(blend)
    calendar = staticmethod(calendar)

    def cached_array(self, operation, inputs, compute):
        from .numerical_cache import cached_array
        return cached_array(self,operation,inputs,compute)

    def calendar_expert(self, view, weight=.5, spread_weight=.6, recent=730):
        from .structure import calendar_expert
        return calendar_expert(self, view, weight, spread_weight, recent)

    def asinh_center(self, view, weight=1.):
        """Transform the centre while retaining Raw's quantile offsets, a candidate."""
        raw=self.timesfm(view)
        transformed=self.timesfm(view,transform='asinh')
        return self.shift(raw,weight*(transformed.point-raw.point))

    def native_flow(self, view, uncertainty='sum', signed=False, context=15360,
                    features=None, covariate_differences=()):
        """Actual increment inference and stock restoration, not a task dispatch.

        SUM and RSS are alternative uncertainty propagation hypotheses, not exact
        sum-distribution quantiles or calibration guarantees. Requested numeric
        past-only differences are appended without removing original channels.
        """
        from dataclasses import replace
        items=[]
        for item in view.items:
            past=dict(item['past']);past_raw=dict(item.get('past_raw',{}))
            for name in covariate_differences:
                values=np.asarray(item.get('past_raw',{}).get(name,past[name]),float)
                differences=np.diff(values,prepend=values[0])
                key='flow_difference__'+name
                past[key]=differences;past_raw[key]=differences
            items.append({**item,'past':past,'past_raw':past_raw})
        transformed=replace(view,items=items)
        forecast=self.timesfm(transformed,transform='flow',signed=signed,context=context,features=features)
        if uncertainty=='sum':return forecast
        if uncertainty!='rss':raise ValueError('Flow uncertainty must be sum or rss')
        last=np.asarray([item['y'][:,-1] for item in view.items])[:,None,:,None]
        increments=np.diff(np.concatenate([np.repeat(last,9,axis=-1),forecast.quantiles],axis=1),axis=1)
        deviations=increments-increments[...,4:5]
        spread=np.sign(np.arange(1,10)/10-.5)*np.sqrt(np.cumsum(deviations**2,axis=1))
        return Forecast(forecast.point.copy(),forecast.quantiles[...,4:5]+spread).validate()

    def scene(self,view,forecast=None,max_items=4,max_targets=8):
        from .scene import describe
        return describe(view,max_items=max_items,max_targets=max_targets,forecast=forecast)

    def timesfm(self, view, features=None, context=15360, transform='identity', signed=False):
        contexts, knowns, pasts, inverses = [],[],[],[]
        digest = hashlib.sha256(json.dumps({'checkpoint':self.checkpoint,'horizon':view.horizon,
            'signed':signed,'transform':transform,'context':context,'symmetric':True}).encode())
        for item in view.items:
            y = item['y']; L = y.shape[1]; C = min(L,int(context))
            known = dict(item['known']); past = dict(item['past'])
            if features is not None:
                extra = features(item['timestamps'], known, L)
                for name,value in extra.items():
                    a = np.asarray(value, np.float32)
                    if a.shape != (L+view.horizon,) or not np.isfinite(a).all(): raise ValueError('Invalid generated feature')
                    known['agent_'+name] = a
            scale = np.maximum(1., np.median(abs(y),axis=1))
            if transform == 'asinh': values = np.arcsinh(y/scale[:,None])
            elif transform == 'flow':
                values = np.diff(y,axis=1); C = min(C,values.shape[1])
            elif transform == 'identity': values = y
            else: raise ValueError('Supported transforms: identity/asinh/flow')
            arrays = [values[:,-C:].astype(np.float32),
                np.stack([v[-(C+view.horizon):] for v in known.values()]).astype(np.float32) if known else None,
                np.stack([v[-C:] for v in past.values()]).astype(np.float32) if past else None]
            for a in arrays:
                digest.update(str(None if a is None else (a.shape,a.dtype.str)).encode())
                if a is not None: digest.update(np.ascontiguousarray(a).tobytes())
            contexts.append(arrays[0]);knowns.append(arrays[1]);pasts.append(arrays[2])
            inverses.append((scale,y[:,-1]))
            # Inverse transforms depend on past even if native contexts coincide.
            digest.update(np.ascontiguousarray(scale).tobytes());digest.update(np.ascontiguousarray(y[:,-1]).tobytes())
        key = digest.hexdigest(); path = self.native_cache/(key+'.npz')
        if path.exists():
            self.hits += 1
            with np.load(path) as z: result = Forecast(z['point'],z['quantiles']).validate()
            self._record_native(key, path)
            return result
        if self.cache_only:
            raise ValueError('Native input is not in this R archive; submit a real probe for changed inputs')
        input_dir=self.output/'inputs';input_dir.mkdir(exist_ok=True)
        input_arrays={f'context_{i}':v for i,v in enumerate(contexts)}
        input_arrays.update({f'known_{i}':v for i,v in enumerate(knowns) if v is not None})
        input_arrays.update({f'past_{i}':v for i,v in enumerate(pasts) if v is not None})
        np.savez_compressed(input_dir/(key+'.npz'),**input_arrays)
        outputs=[];started=time.monotonic()
        batches=[]
        batch_size=int(os.environ.get('TIMERSI_U1_BATCH_SIZE','4'))
        for start in range(0,len(contexts),batch_size):
            batches.append(dict(contexts=contexts[start:start+batch_size],horizon=view.horizon,
                past_only_covariates=pasts[start:start+batch_size],past_future_covariates=knowns[start:start+batch_size],
                return_quantiles=True,use_symmetric_averaging=True,make_positive=not signed,sort_quantiles=True))
        backend=self._model()
        results=backend.predict_many(batches) if hasattr(backend,'predict_many') else (backend.predict_batch(**kw) for kw in batches)
        for raw in results:
            outputs.extend(raw);self.calls+=1
        points=[];quantiles=[]
        for output,(scale,last) in zip(outputs,inverses):
            p = np.asarray(output.forecast).reshape(len(view.targets),view.horizon).T
            q = np.asarray(output.quantiles).reshape(len(view.targets),view.horizon,9).transpose(1,0,2)
            if transform == 'asinh': p=scale[None,:]*np.sinh(p);q=scale[None,:,None]*np.sinh(q)
            if transform == 'flow': p=last[None,:]+np.cumsum(p,axis=0);q=last[None,:,None]+np.cumsum(q,axis=0)
            points.append(p);quantiles.append(q)
        result=Forecast(np.stack(points),np.stack(quantiles)).validate()
        np.savez_compressed(path,point=result.point,quantiles=result.quantiles)
        write(path.with_suffix('.json'),{'native_key':key,'items':len(contexts),'seconds':time.monotonic()-started,
            'transform':transform,'device':self.device,'context_lengths':[x.shape[1] for x in contexts],
            'batch_size':batch_size,'gpu_pool':os.environ.get('TIMERSI_U1_GPU_POOL'),
            'known_channels':[0 if x is None else len(x) for x in knowns],
            'past_channels':[0 if x is None else len(x) for x in pasts],
            'item_ids':[x['id'] for x in view.items],'actual_inputs':str(input_dir/(key+'.npz'))})
        self._record_native(key, path)
        return result

    def seasonal(self, view):
        points=[];quantiles=[]
        for item in view.items:
            y=item['y'];s=min(max(1,view.seasonality),y.shape[1]);D,L=y.shape
            p=y[:,L-s+np.arange(view.horizon)%s].T
            residual=y[:,s:]-y[:,:-s] if L>s else np.zeros((D,1))
            q=p[...,None]+np.quantile(residual,np.arange(1,10)/10,axis=1).T[None,:,:]
            points.append(p);quantiles.append(q)
        return Forecast(np.stack(points),np.stack(quantiles)).validate()

    def shift(self, forecast, delta):
        delta=np.broadcast_to(np.asarray(delta),forecast.point.shape)
        return Forecast(forecast.point+delta,forecast.quantiles+delta[...,None]).validate()

    def spread(self, forecast, factor=1.):
        centre=forecast.quantiles[...,4:5]
        return Forecast(forecast.point.copy(),centre+factor*(forecast.quantiles-centre)).validate()

    def residual(self, forecast, view, memory, weight=.25, source='raw', max_windows=3):
        """Historical residual correction; memory already excludes unmatured labels."""
        if source=='self':source=getattr(self,'active_program',None)
        delta=np.zeros_like(forecast.point)
        for i,item in enumerate(view.items):
            for d,target in enumerate(view.targets):
                records={}
                for row in memory:
                    if row['source']!=source or item['id'] not in row['ids'] or target not in row['targets']:continue
                    j=row['ids'].index(item['id']);k=row['targets'].index(target)
                    # Rolling research can revisit an already recorded origin.
                    # Count that forecast horizon once, not once per search epoch.
                    end=np.datetime64(row['ends'][j])
                    records[end]=row['truth'][j,:,k]-row['point'][j,:,k]
                if records:
                    recent=sorted(records.items(),key=lambda r:r[0])[-max_windows:]
                    errors=np.concatenate([r[1] for r in recent]);errors=errors[np.isfinite(errors)]
                    if len(errors):delta[i,:,d]=weight*np.median(errors)
        return self.shift(forecast,delta)

    def expert(self, view, kind='ridge', lags=None, recent=2048, fit_scope='item'):
        """Cache fitted predictions within this run, keyed by actual inputs/settings."""
        if kind not in ('ridge','lgbm') or fit_scope not in ('item','task'):
            raise ValueError('Unknown expert kind or fit_scope')
        lags=sorted(set(lags or [1,max(1,view.seasonality)]))
        digest=hashlib.sha256(json.dumps({'kind':kind,'scope':fit_scope,'lags':lags,
            'recent':recent,'horizon':view.horizon,'targets':view.targets}).encode())
        for item in view.items:
            digest.update(item['id'].encode())
            arrays=[item['y'],item['timestamps']]
            if 'observed' in item:arrays.append(item['observed'])
            for name in sorted(item['known']):
                digest.update(name.encode());arrays.append(item['known'][name])
            for value in arrays:
                a=np.asarray(value)
                digest.update(str((a.shape,a.dtype.str)).encode())
                digest.update(a.astype(str).tobytes() if a.dtype.kind=='O' else np.ascontiguousarray(a).tobytes())
        cache=self.output/'experts';cache.mkdir(exist_ok=True)
        path=cache/(digest.hexdigest()+'.npz')
        if path.exists():
            with np.load(path) as z:return Forecast(z['point'],z['quantiles']).validate()
        if fit_scope=='task':
            from .experts import pooled_expert
            result=pooled_expert(view,kind,lags,recent,self.output,calendar)
        else:result=self._expert_item(view,kind,lags,recent)
        np.savez_compressed(path,point=result.point,quantiles=result.quantiles)
        return result

    def _expert_item(self, view, kind='ridge', lags=None, recent=2048):
        """Causal item-level recursive expert; quantiles from own training residuals."""
        from sklearn.linear_model import Ridge
        from lightgbm import LGBMRegressor
        if kind not in ('ridge','lgbm'): raise ValueError('Unknown expert type')
        lags=sorted(set(lags or [1,max(1,view.seasonality)]))
        points=[];quantiles=[]
        for item in view.items:
            y=item['y'];D,L=y.shape
            if L <= max(lags)+8:
                raise ValueError('Expert needs sufficient lagged training rows')
            cal=np.stack(list(calendar(item['timestamps'],item['known'],L).values()),axis=1)
            known=np.stack(list(item['known'].values()),axis=1) if item['known'] else np.empty((L+view.horizon,0))
            exog=np.concatenate([cal,known],axis=1)
            rows=np.arange(max(max(lags),L-recent),L)
            p=np.empty((view.horizon,D));q=np.empty((view.horizon,D,9))
            for d in range(D):
                observed=np.asarray(item.get('observed',np.isfinite(y)),bool)[d]
                fit_rows=rows[observed[rows]]
                if not len(fit_rows):raise ValueError('No observed expert labels')
                scale=max(1.,float(np.median(abs(y[d,observed]))));history=list(y[d]/scale)
                X=np.column_stack([*[np.asarray(history)[fit_rows-lag] for lag in lags],exog[fit_rows]])
                model=Ridge(alpha=10.) if kind=='ridge' else LGBMRegressor(n_estimators=60,num_leaves=7,n_jobs=1,verbosity=-1,random_state=0)
                fit_start=time.monotonic()
                model.fit(X,np.asarray(history)[fit_rows])
                with (self.output/'expert-fits.jsonl').open('a') as log:
                    log.write(json.dumps({'at':time.time(),'kind':kind,'item':item['id'],
                        'target':view.targets[d],'rows':len(fit_rows),'features':X.shape[1],
                        'last_training_time':str(item['timestamps'][L-1]),
                        'seconds':time.monotonic()-fit_start,'lags':lags,
                        'model_type':type(model).__name__,'fit_scope':'item'})+'\n')
                residual=np.asarray(history)[fit_rows]-model.predict(X)
                spread=np.quantile(residual,np.arange(1,10)/10)*scale
                for h in range(view.horizon):
                    row=np.r_[[history[-lag] for lag in lags],exog[L+h]]
                    pred=float(model.predict(row[None,:])[0]);history.append(pred)
                    p[h,d]=pred*scale;q[h,d]=p[h,d]+spread
            points.append(p);quantiles.append(q)
        return Forecast(np.stack(points),np.stack(quantiles)).validate()

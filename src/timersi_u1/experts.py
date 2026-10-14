"""Task-level causal specialists; one fit per target, not per inference batch."""
import json
import time
import numpy as np
from .core import Forecast


def pooled_expert(view,kind,lags,recent,output,calendar):
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    names=sorted({n for it in view.items for n in it['known']})
    # A later entity must not contribute labels beyond another entity's origin.
    cutoff=min(np.datetime64(it['timestamps'][it['y'].shape[1]-1]) for it in view.items)
    prepared=[]
    for item in view.items:
        y=item['y'];D,L=y.shape
        if L<=max(lags)+8:raise ValueError('Expert needs sufficient lagged training rows')
        cal=np.stack(list(calendar(item['timestamps'],item['known'],L).values()),axis=1)
        known=np.column_stack([item['known'].get(n,np.zeros(L+view.horizon)) for n in names]) if names else np.empty((L+view.horizon,0))
        exog=np.concatenate([cal,known],axis=1)
        rows=np.arange(max(max(lags),L-recent),L)
        rows=rows[np.asarray(item['timestamps'])[rows]<=cutoff]
        visible=np.asarray(item['timestamps'][:L])<=cutoff
        observed=np.asarray(item.get('observed',np.isfinite(y)),bool)
        scale=np.array([max(1.,float(np.median(abs(y[d,visible&observed[d]]))))
            if (visible&observed[d]).any() else 1. for d in range(D)])
        prepared.append((y/scale[:,None],scale,exog,rows,observed))
    points=np.empty((len(prepared),view.horizon,len(view.targets)))
    quantiles=np.empty(points.shape+(9,))
    for d,target in enumerate(view.targets):
        features=[];labels=[]
        for y,scale,exog,rows,observed in prepared:
            rows=rows[observed[d,rows]]
            if not len(rows):continue
            features.append(np.column_stack([*[y[d,rows-lag] for lag in lags],exog[rows]]))
            labels.append(y[d,rows])
        if not features:raise ValueError('No shared-origin training rows')
        X=np.concatenate(features);truth=np.concatenate(labels)
        model=Ridge(alpha=10.) if kind=='ridge' else LGBMRegressor(n_estimators=60,num_leaves=7,n_jobs=1,verbosity=-1,random_state=0)
        start=time.monotonic();model.fit(X,truth)
        residual=truth-model.predict(X)
        spread=np.quantile(residual,np.arange(1,10)/10)
        with (output/'expert-fits.jsonl').open('a') as log:
            log.write(json.dumps({'at':time.time(),'kind':kind,'target':target,
                'rows':len(truth),'features':X.shape[1],'last_training_time':str(cutoff),
                'seconds':time.monotonic()-start,'lags':lags,'fit_scope':'task',
                'items':[it['id'] for it in view.items],'model_type':type(model).__name__})+'\n')
        for i,(y,scale,exog,rows,observed) in enumerate(prepared):
            history=list(y[d]);L=len(history)
            for h in range(view.horizon):
                row=np.r_[[history[-lag] for lag in lags],exog[L+h]]
                prediction=float(model.predict(row[None,:])[0]);history.append(prediction)
                points[i,h,d]=prediction*scale[d]
                quantiles[i,h,d]=points[i,h,d]+spread*scale[d]
    return Forecast(points,quantiles).validate()

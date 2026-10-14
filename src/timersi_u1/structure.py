"""Past-fitted multiplicative calendar/covariate decomposition, not task policies."""
from dataclasses import replace
import numpy as np
import pandas as pd
from .core import Forecast


def design(item, length):
    dates = pd.DatetimeIndex(item['timestamps'])
    n = len(dates)
    # Trend is separate from the calendar factor removed before native inference.
    columns = [np.arange(n) / max(length, 1)]
    columns += [(np.asarray(dates.dayofweek) == day).astype(float) for day in range(6)]
    year = np.asarray(dates.dayofyear) / 365.25
    columns += [np.sin(2*np.pi*year), np.cos(2*np.pi*year)]
    for name, values in item.get('known_raw', item['known']).items():
        values = np.asarray(values)
        if values.dtype.kind in 'OUS':
            values = values.astype(str)
            levels = sorted(set(values[:length]))
            if len(levels) <= 12:
                columns += [(values == level).astype(float) for level in levels[1:]]
            continue
        values = values.astype(float)
        if not np.isfinite(values).all():
            continue
        std = np.std(values[:length])
        if std > 1e-8:
            columns.append((values-np.mean(values[:length]))/std)
    return np.column_stack(columns)


def fit_item(item, horizon, recent=730):
    y = np.asarray(item['y'], float)
    length = y.shape[1]
    if length < 28 or np.any(y < 0):
        return None
    X = design(item, length)
    # Only an actual supplied open indicator licenses closure handling.
    open_values = next((np.asarray(v, float) for n,v in item['known'].items()
                        if n.lower() in ('open', 'is_open')), np.ones(length+horizon))
    train = np.arange(max(0, length-recent), length)
    train = train[open_values[train] > .5]
    if len(train) < max(24, X.shape[1]+3):
        return None
    A = np.column_stack([np.ones(len(X)), X])
    weights = np.exp((train-length+1)/max(60., recent/2))
    gram = (A[train].T*weights)@A[train]
    penalty = np.eye(A.shape[1])*2.; penalty[0,0] = 1e-6
    factors=[]; experts=[]; spreads=[]; normalized=[]
    for target in y:
        floor = max(1e-6, np.median(target[train])*1e-3)
        log_y = np.log(np.maximum(target, floor))
        beta = np.linalg.solve(gram+penalty, A[train].T@(weights*log_y[train]))
        # Calendar/covariates only: do not remove the level or trend from TimesFM.
        structural = A[:,2:]@beta[2:]
        structural -= np.average(structural[train], weights=weights)
        factor = np.exp(np.clip(structural, -2.5, 2.5))
        norm = target/factor[:length]
        closed = open_values[:length] <= .5
        observed = np.flatnonzero(~closed)
        if closed.any():
            norm[closed] = np.interp(np.flatnonzero(closed), observed, norm[observed])
        log_fit = A@beta
        # Extrapolation bound is relative to this entity's observed log range.
        future_log = np.clip(log_fit[length:], np.min(log_y[train])-1., np.max(log_y[train])+1.)
        residual = log_y[train]-log_fit[train]
        factors.append(factor[length:]);experts.append(np.exp(future_log))
        spreads.append(np.quantile(residual, np.arange(1,10)/10))
        normalized.append(norm)
    return {'normalized':np.asarray(normalized), 'factor':np.asarray(factors).T,
            'expert':np.asarray(experts).T, 'spread':np.asarray(spreads),
            'closed':open_values[length:]<=.5, 'training_rows':len(train)}


def calendar_expert(tools, view, weight=.5, spread_weight=.6, recent=730):
    if not 0 <= weight <= 1 or not 0 <= spread_weight <= 1:
        raise ValueError('Mixture weights must be in [0,1]')
    fits=[fit_item(item,view.horizon,recent) for item in view.items]
    items=[]
    for item,fit in zip(view.items,fits):
        if fit is None:
            items.append(item)
        else:
            items.append({**item, 'y':fit['normalized'], 'known':{}, 'past':{},
                          'known_raw':{}, 'past_raw':{}})
    native=tools.timesfm(replace(view,items=items))
    point=native.point.copy();quantiles=native.quantiles.copy()
    for i,fit in enumerate(fits):
        if fit is None:
            continue
        restored=np.maximum(native.point[i]*fit['factor'],1e-8)
        p=np.exp((1-weight)*np.log(restored)+weight*np.log(np.maximum(fit['expert'],1e-8)))
        native_shape=np.log(np.maximum(native.quantiles[i],1e-8)/np.maximum(native.point[i,...,None],1e-8))
        shape=(1-spread_weight)*native_shape+spread_weight*fit['spread'][None,:,:]
        q=p[...,None]*np.exp(np.clip(shape,-8.,8.))
        p[fit['closed']]=0.;q[fit['closed']]=0.
        point[i]=p;quantiles[i]=np.sort(q,axis=-1)
    return Forecast(point,quantiles).validate()

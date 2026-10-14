"""Compact forecast-origin facts, computed only from visible past and known future."""
import numpy as np
import pandas as pd


def describe(view,max_items=4,max_targets=8,forecast=None):
    result={'horizon':view.horizon,'frequency':view.frequency,'seasonality':view.seasonality,
        'facts':view.facts,'total_items':len(view.items),'total_targets':len(view.targets),'items':[]}
    for i,item in enumerate(view.items[:max_items]):
        y=np.asarray(item['y']);L=y.shape[1];dates=pd.DatetimeIndex(item['timestamps'][L:])
        recent=min(L,max(view.horizon,view.seasonality,8));tail=y[:,-recent:]
        previous=y[:,max(0,L-2*recent):L-recent]
        targets=[]
        for d,name in enumerate(view.targets[:max_targets]):
            scale=max(1e-6,float(np.mean(abs(tail[d]))));mean=float(np.mean(tail[d]))
            info={'name':name,'last':float(y[d,-1]),'recent_mean':mean,
                'recent_std':float(np.std(tail[d])),'recent_negative_fraction':float(np.mean(tail[d]<0)),
                'recent_zero_fraction':float(np.mean(tail[d]==0)),
                'normalized_level_change':float((mean-np.mean(previous[d]))/scale) if previous.shape[1] else None}
            if forecast is not None:
                info['forecast_mean']=float(np.mean(forecast.point[i,:,d]))
                info['forecast_interval_mean']=float(np.mean(forecast.quantiles[i,:,d,-1]-forecast.quantiles[i,:,d,0]))
            targets.append(info)
        covariates={}
        for name,values in item['known'].items():
            original=item.get('known_raw',{}).get(name,values)
            future_original=list(original[L:])
            if any(isinstance(v,str) for v in future_original):
                counts={}
                for value in future_original:counts[str(value)]=counts.get(str(value),0)+1
                covariates[name]={'original_future_categories':counts}
            else:
                values=np.asarray(values,float);past=values[max(0,L-recent):L];future=values[L:]
                scale=max(1e-6,float(np.std(past)),float(np.mean(abs(past)))*.01)
                covariates[name]={'future_min':float(np.min(future)),'future_max':float(np.max(future)),
                    'future_mean':float(np.mean(future)),'recent_past_mean':float(np.mean(past)),
                    'standardized_mean_change':float((np.mean(future)-np.mean(past))/scale)}
        result['items'].append({'id':item['id'],'past_end':str(item['timestamps'][L-1]),
            'forecast_start':str(dates[0]),'forecast_end':str(dates[-1]),
            'forecast_weekday_fractions':{str(day):float(np.mean(dates.dayofweek==day)) for day in range(7)},
            'recent_points':recent,'targets':targets,'known_future':covariates,'static':item['static']})
    return result

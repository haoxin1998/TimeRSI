"""Enrich the same current-origin packet with visible numerical trajectories."""
from copy import deepcopy
import numpy as np

FEATURE_NAMES=['recent_std_over_level_scale','recent_increment_over_level_scale',
    'increment_acceleration_over_level_scale','recent_zero_fraction',
    'raw_terminal_change_over_level_scale','raw_interval_width_over_level_scale']


def change_unit(y, seasonality):
    lag=min(max(1,int(seasonality)),max(1,len(y)-1))
    changes=y[lag:]-y[:-lag]
    unit=float(np.mean(abs(changes))) if len(changes) else 0.
    if unit<=1e-9:unit=max(1e-6,float(np.std(y)),float(np.mean(abs(y)))*.01)
    return unit


def enrich(view, forecasts, packet):
    result=deepcopy(packet)
    result['evidence_representation']='Visible history and actual forecast trajectories in original and past-change units'
    result['normalization_note']='Per entity/target: (value-last_observed)/mean_abs_past_seasonal_change; constant-history fallback from past only. Display units, not a recomputed official metric.'
    result['group_feature_names']=FEATURE_NAMES
    lookup={str(item['id']):i for i,item in enumerate(view.items)}
    leads=np.asarray(result['lead_indices'],int)
    for group in result['groups']:
        d=view.targets.index(group['target']);indices=[lookup[i] for i in group['members']]
        normalized={name:[] for name in forecasts};details=[]
        representative_ids=[r['id'] for r in group['representatives']['items']]
        for i in indices:
            item=view.items[i];y=np.asarray(item['y'][d],float);last=float(y[-1]);unit=change_unit(y,view.seasonality)
            width=min(len(y),min(256,max(64,2*view.seasonality,4*view.horizon)))
            tail=y[-width:];dates=item['timestamps'];L=len(y)
            for name,f in forecasts.items():normalized[name].append((f.point[i,:,d]-last)/unit)
            if str(item['id']) not in representative_ids:continue
            increments=np.diff(tail);span=min(7,max(1,len(increments)//2))
            recent=float(np.mean(increments[-span:])) if len(increments) else None
            previous=float(np.mean(increments[-2*span:-span])) if len(increments)>span else None
            details.append(dict(id=str(item['id']),target=group['target'],last_observed=last,
                past_change_unit=unit,past_dates=[str(t) for t in dates[L-width:L]],
                past_values=tail.tolist(),past_relative_change_units=((tail-last)/unit).tolist(),
                observed_increments=increments.tolist(),recent_mean_increment=recent,previous_mean_increment=previous,
                forecast_dates=[str(dates[L+h]) for h in leads],
                known_future={name:list(item.get('known_raw',{}).get(name,values)[L:]) for name,values in item['known'].items()},
                past_covariates={name:list(item.get('past_raw',{}).get(name,values)[-width:]) for name,values in item['past'].items()},
                forecasts={name:dict(point_relative_change_units=((f.point[i,leads,d]-last)/unit).tolist(),
                    q10_relative_change_units=((f.quantiles[i,leads,d,0]-last)/unit).tolist(),
                    q50_relative_change_units=((f.quantiles[i,leads,d,4]-last)/unit).tolist(),
                    q90_relative_change_units=((f.quantiles[i,leads,d,8]-last)/unit).tolist(),
                    terminal_change_original=float(f.point[i,-1,d]-last),
                    average_implied_increment=float((f.point[i,-1,d]-last)/view.horizon)) for name,f in forecasts.items()}))
        group['trajectory_evidence']=dict(representatives=details,
            all_member_point_change_units={name:dict(mean=np.mean(values,axis=0)[leads].tolist(),
                terminal_p10_p50_p90=np.quantile(np.asarray(values)[:,-1],[.1,.5,.9]).tolist()) for name,values in normalized.items()})
    return result

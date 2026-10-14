"""Paired outcomes of this run, released before the held-out selection period."""
import numpy as np


def _errors(record):
    truth=np.asarray(record['truth']);residual=truth-record['point']
    delta=truth[...,None]-record['quantiles']
    levels=np.arange(1,10)/10
    pinball=2*np.maximum(levels*delta,(levels-1)*delta)
    result={'MAE':float(np.mean(np.abs(residual))),
            'mean_truth_minus_prediction':float(np.mean(residual)),
            'mean_quantile_loss':float(np.mean(pinball))}
    if 'official_SQL_by_target' in record:
        values=np.asarray(record['official_SQL_by_target'],float)
        result['official_SQL_by_target']={}
        for j,target in enumerate(record['targets']):
            cells=values[:,j];cells=cells[np.isfinite(cells)]
            result['official_SQL_by_target'][target]=float(np.mean(cells)) if len(cells) else None
        target_means=[value for value in result['official_SQL_by_target'].values() if value is not None]
        result['official_SQL_mean_visible_cells']=float(np.mean(target_means)) if target_means else None
    return result


def deployment_feedback(memory,selection_boundary,bank,limit=12):
    """Use the existing per-entity visibility rule at the OLDEST S origin.

    Unscaled diagnostics and optional origin-exact SQL cells are separate.
    Only visible entities contribute; partial-cell means are not full-task scores.
    Current S target intervals cannot appear in this packet.
    """
    visible={r['key']:r for r in memory.visible(selection_boundary)
             if r['key'].startswith('official-')}
    identities={}
    for name in bank.records:
        identities.setdefault(bank.memory_source(name),[]).append(name)
    pairs=[]
    for key in sorted(visible):
        if not key.endswith('-raw'):continue
        raw=visible[key];selected=visible.get(key[:-3]+'selected')
        if selected is None:continue
        if raw['ids']!=selected['ids'] or raw['targets']!=selected['targets']:
            raise ValueError('Paired deployment feedback has different entities or targets')
        a=_errors(raw);b=_errors(selected)
        raw_sql=a.get('official_SQL_mean_visible_cells');deployed_sql=b.get('official_SQL_mean_visible_cells')
        pairs.append({'window':key[:-4],'programs':identities.get(selected['source'],[]),
            'program_identity':selected['source'],'visible_entities':len(raw['ids']),
            'forecast_starts':sorted(set(raw.get('starts',[]))),
            'forecast_ends':sorted(set(raw['ends'])),
            'raw':a,'deployed':b,
            'relative_official_SQL_change':deployed_sql/raw_sql-1
                if raw_sql is not None and deployed_sql is not None and raw_sql>0 else None,
            'relative_MAE_change':b['MAE']/a['MAE']-1 if a['MAE']>0 else None,
            'relative_quantile_loss_change':b['mean_quantile_loss']/a['mean_quantile_loss']-1
                if a['mean_quantile_loss']>0 else None})
    cutoffs=[str(it['timestamps'][it['y'].shape[1]-1]) for it in selection_boundary.items]
    return {'scope':'This run only; official deployments mature before the oldest S origin',
        'metric_note':'MAE/quantile loss are unscaled; optional official_SQL cells use FEV SQL and original origin histories. '
            'Visible-cell means are not full-task or benchmark scores. Negative relative change is better; inspect SQL separately.',
        'visibility_boundary_min':min(cutoffs),'visibility_boundary_max':max(cutoffs),
        'available_pairs':len(pairs),'recent_pairs':pairs[-limit:]}

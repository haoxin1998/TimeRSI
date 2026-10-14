"""Current-origin decisions over actual own-run forecasts, not future scores."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .core import write
from .scene import describe
from .agent import propose


GUIDE = '''You are the deployment decision agent in an adaptive forecasting loop.
Read INPUT.json. Every candidate is an ACTUAL forecast from this run's frozen
model/programs at THIS forecast origin, not a suggested model or an oracle answer.
Read candidate-code/ only if useful. Do not write or retrain prediction programs
here: code research is a separate responsibility within this same loop.
Compare current observed levels, recent changes and acceleration, known future
calendar/covariates, candidate forecast trajectories and interval widths. Use
mature per-candidate errors and historical selection evidence, but do not assume
past average winners remain best under today's conditions. Do not overfit historical
data. Semantic names are context, not evidence for a fixed task-name recipe.
Each group is one target across the listed member entities; representative samples
are not the whole group. Weights apply to ALL group members and every horizon point,
including all nine quantiles. Check the group summaries as well as representative
curves. Select or convex-combine useful complementary candidates, normally 1-3,
or retain the supplied global winner when there is no useful reason to change.
Candidate disagreement alone is neither proof of improvement nor reason to abstain.
Errors in INPUT are from COMPLETED past windows only; current outcomes are unavailable.
Write decision.json with {"default":{"weights":{"EXACT_CANDIDATE_ID":1}},
"groups":{"g000":{"weights":{"EXACT_CANDIDATE_ID":0.7,
"OTHER_EXACT_ID":0.3},"reason":"current evidence"}}}.
Use only candidate IDs actually listed in INPUT. Supply nonnegative finite weights
with positive sum. Omitted groups use default. Include your explanation in each
modified group's reason. Do not run git commands, inspect credentials, fetch external
data or read outside this workspace. Leave the decision file, not merely prose.
'''


def groups_for(view, raw):
    """Target-specific clusters of currently visible behaviour; small N stays exact."""
    from sklearn.cluster import KMeans
    groups = []
    for d, target in enumerate(view.targets):
        features = []
        for i, item in enumerate(view.items):
            y = np.asarray(item['y'][d], float); width = min(len(y), max(8, view.horizon))
            tail = y[-width:]; scale = max(1e-6, float(np.mean(abs(tail))), float(np.std(tail)))
            inc = np.diff(tail); half = max(1, len(inc)//2)
            recent = float(np.mean(inc[-half:])) if len(inc) else 0.
            earlier = float(np.mean(inc[:half])) if len(inc) else 0.
            features.append([np.std(tail)/scale, recent/scale, (recent-earlier)/scale,
                float(np.mean(tail == 0)), (raw.point[i,-1,d]-y[-1])/scale,
                float(np.mean(raw.quantiles[i,:,d,-1]-raw.quantiles[i,:,d,0]))/scale])
        X = np.asarray(features)
        if len(X) <= 16: labels = np.arange(len(X))
        else:
            normalized = (X-np.median(X, axis=0))/np.maximum(np.std(X, axis=0), 1e-6)
            labels = KMeans(n_clusters=min(16,len(np.unique(X,axis=0))), n_init=3, random_state=0).fit_predict(normalized)
        for label in sorted(set(labels)):
            members = np.flatnonzero(labels == label).tolist()
            centre = np.mean(X[members], axis=0)
            reps = sorted(members, key=lambda i: (float(np.sum((X[i]-centre)**2)), str(view.items[i]['id'])))[:3]
            groups.append(dict(key=f'g{len(groups):03d}', target=target, target_index=d,
                members=[str(view.items[i]['id']) for i in members], indices=members,
                representative_indices=reps, mean_features=centre.tolist()))
    return groups


def mature_errors(view, group, bank, forecasts, visible):
    result = {}
    member_ids=set(group['members'])
    for name in forecasts:
        source = bank.memory_source(name); records = {}
        for row in visible:
            if row['source'] != source or not row['key'].startswith('official-') or 'official_SQL_by_target' not in row: continue
            if group['target'] not in row['targets']: continue
            d = row['targets'].index(group['target'])
            for j, item in enumerate(row['ids']):
                if str(item) not in member_ids: continue
                loss = row['official_SQL_by_target'][j][d]
                if loss is not None and np.isfinite(loss): records[(str(item),row['ends'][j])] = float(loss)
        ends = sorted({end for _,end in records})[-3:]
        values = [loss for (_,end),loss in records.items() if end in ends]
        result[name] = dict(SQL=float(np.mean(values)) if values else None,
            visible_cells=len(values), forecast_ends=ends)
    return result


def packet_for(view, bank, forecasts, selected, selection, visible, groups=None):
    groups = groups_for(view, forecasts['raw']) if groups is None else groups
    public = []
    leads = np.unique(np.linspace(0,view.horizon-1,min(24,view.horizon)).astype(int))
    for group in groups:
        d = group['target_index']; members = group['indices']; reps = group['representative_indices']
        candidate_curves = {}
        for name, forecast in forecasts.items():
            candidate_curves[name] = dict(
                group_mean_point=forecast.point[members,:,d].mean(axis=0)[leads],
                group_mean_q10=forecast.quantiles[members,:,d,0].mean(axis=0)[leads],
                group_mean_q50=forecast.quantiles[members,:,d,4].mean(axis=0)[leads],
                group_mean_q90=forecast.quantiles[members,:,d,8].mean(axis=0)[leads],
                representatives=[dict(id=str(view.items[i]['id']), point=forecast.point[i,leads,d],
                    q10=forecast.quantiles[i,leads,d,0], q50=forecast.quantiles[i,leads,d,4],
                    q90=forecast.quantiles[i,leads,d,8]) for i in reps])
        public.append({**{k:v for k,v in group.items() if k not in ('indices','representative_indices')},
            'representatives':describe(replace(view,items=[view.items[i] for i in reps]),max_items=3,max_targets=len(view.targets)),
            'candidate_forecasts':candidate_curves,
            'mature_candidate_errors':mature_errors(view,group,bank,forecasts,visible)})
    packet = dict(task_facts=view.facts, horizon=view.horizon, frequency=view.frequency,
        global_winner=selected, lead_indices=leads.tolist(), groups=public,
        candidates={name:dict(kind=bank.records[name]['kind'], identity=bank.memory_source(name),
            historical_S_SQL=selection.get(name,{}).get('SQL')) for name in forecasts},
        evidence_scope='Visible past/known future/current forecasts and completed history only; no current truth',
        composition='Convex point/quantile averaging; not a mixture-distribution quantile calculation')
    return packet, groups


def route_from_decision(view, groups, forecasts, decision):
    def route(spec):
        weights = spec['weights']
        if not weights or any(n not in forecasts for n in weights): raise ValueError('Unknown or empty candidates')
        values = np.asarray(list(weights.values()),float)
        if not np.isfinite(values).all() or np.any(values<0) or values.sum()<=0: raise ValueError('Invalid weights')
        return dict(members=list(weights),weights=(values/values.sum()).tolist())
    default = route(decision['default']); supplied = decision.get('groups',{})
    if set(supplied)-{g['key'] for g in groups}: raise ValueError('Unknown group')
    routes = {}
    for group in groups:
        local = route(supplied[group['key']]) if group['key'] in supplied else default
        for item in group['members']: routes.setdefault(item,{})[group['target']] = local
    return routes


def deterministic_decision(packet):
    groups = {}
    for group in packet['groups']:
        scores = {name:value['SQL'] for name,value in group['mature_candidate_errors'].items() if value['SQL'] is not None}
        # With no matched mature cells, use the same historical S evidence.
        if not scores: scores = {name:v['historical_S_SQL'] for name,v in packet['candidates'].items() if v['historical_S_SQL'] is not None}
        members = sorted(scores,key=lambda n:(scores[n],n))[:3]
        if members:
            values = 1/np.maximum([scores[n] for n in members],1e-12); values /= values.sum()
            groups[group['key']] = dict(weights=dict(zip(members,values.tolist())),reason='Inverse historical SQL, top3; no LLM')
    return dict(default=dict(weights={packet['global_winner']:1.}),groups=groups)


def deploy(folder, view, bank, tools, visible, cache, selected, epoch, args, task=None, window=None, memory=None):
    folder = Path(folder); forecasts = {}; failures = {}
    requested=[name for name,node in bank.records.items() if node.get('origin')=='deployment_agent_native_request']
    names = list(dict.fromkeys(['raw',selected,*epoch.get('scores',{}),*requested]))
    for name in names:
        try: forecasts[name] = bank.predict(name,view,tools,visible,cache)
        except Exception as error: failures[name] = repr(error)
    selection=dict(epoch.get('scores',{}))
    packet, groups = packet_for(view,bank,forecasts,selected,selection,visible)
    trajectory=getattr(args,'window_evidence','summary')=='trajectory'
    if trajectory:
        from .trajectory_evidence import enrich
        packet=enrich(view,forecasts,packet)
    packet['current_candidate_failures'] = failures
    workspace = folder/'agent-session/workspace'; workspace.mkdir(parents=True,exist_ok=True)
    for name in forecasts:
        node = bank.records[name]
        if node['kind']=='code':
            code_path = workspace/'candidate-code'/name/'candidate.py'
            code_path.parent.mkdir(parents=True,exist_ok=True); code_path.write_text(node['code'])
            for path, text in node.get('files',{}).items():
                target=code_path.parent/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    initial_packet=packet
    refreshed_history={}
    def publish(workspace=None,requests=None):
        nonlocal packet
        packet,_=packet_for(view,bank,forecasts,selected,selection,visible,groups=groups)
        from .numerical_boundary import POLICY
        packet['numerical_guard_policy']=POLICY
        packet['numerical_guard']={n:getattr(f,'numerical_guard',{}) for n,f in forecasts.items()}
        if trajectory:packet=enrich(view,forecasts,packet)
        packet['current_candidate_failures']=failures
        if refreshed_history:packet['latest_past_selection_evidence']=refreshed_history
        if requests:packet['native_request_results']=requests
        write(folder/'PUBLIC-INPUT.json',packet)
        np.savez_compressed(folder/'CANDIDATES.npz',item_ids=[it['id'] for it in view.items],
            **{name+'__point':f.point for name,f in forecasts.items()},
            **{name+'__quantiles':f.quantiles for name,f in forecasts.items()})
        if workspace is not None:
            from .agent import readable_json
            target=Path(workspace)/'INPUT.json'
            original=json.loads(target.read_text()) if target.exists() else {}
            write(target,{**original,**packet})
            target.write_text(readable_json(json.loads(target.read_text()))+'\n')
    publish()
    write(folder/'INITIAL-PUBLIC-INPUT.json',initial_packet)
    services=[];requests=None;service_factory=None;guide=GUIDE
    budget=getattr(args,'forecast_tool_requests',0)
    if budget:
        from .forecast_requests import NativeRequests,NativeRequestService
        from .data import historical_windows
        if task is None or window is None:raise ValueError('Native requests require the current task and origin')
        warm=historical_windows(task,args.max_items,args.selection_windows,anchor=window)
        requests=NativeRequests(folder,view,bank,tools,visible,cache,forecasts,selection,warm,task,memory,publish)
        # Compare requested variants against ALL existing candidates on identical
        # latest past windows, not against stale scalar scores from an old epoch.
        for name in list(forecasts):
            past_folder=folder/'latest-past-selection'/name;past_folder.mkdir(parents=True,exist_ok=True)
            try:
                history=requests.past_evidence(name,past_folder)
                refreshed_history[name]=history
                selection[name]={'SQL':float(np.mean([row['metrics']['SQL'] for row in history])) if history else None}
                write(past_folder/'RESULTS.json',history)
            except Exception as exc:
                refreshed_history[name]={'error':repr(exc)}
                selection[name]={'SQL':None}
        publish()
        def service_factory(workspace):
            service=NativeRequestService(workspace,budget,requests);services.append(service);return service
        guide+='''\nYou may request additional ACTUAL native forecasts by running
`<python_executable> native_forecast.py --transform flow --uncertainty rss`
using the Python executable from INPUT.json. See `--help` for identity/asinh/flow,
asinh-center/calendar-expert, optional --calendar, --signed, --context and named
--covariate-difference. These are tools, not required recipes. Flow is appropriate
for accumulated totals, not every positive or trending variable; SUM/RSS are
different interval hypotheses, not guaranteed calibrated quantiles. Requests use
the same native model and preserve original covariates. A request returns actual
current forecasts plus official per-entity/target SQL on COMPLETED past selection
windows, then updates INPUT.json and candidate-code/. Read the updated evidence
and continue this SAME session before choosing per-target weights. Current labels
are never returned. Do not hardcode task names or assume the tool always wins.\n'''
        import sys
        packet.update(python_executable=sys.executable,native_request_budget=budget)
        write(folder/'PUBLIC-INPUT.json',packet)
    error = None; receipt = None
    try:
        text, receipt = propose(folder/'agent-session',packet,[], '',[],model=args.model,
            effort=args.effort,timeout=900,guide=guide,max_turns=64,workspace_service=service_factory,
            vertex_project=args.vertex_project,vertex_region=args.vertex_region,output_artifact='decision.json',
            publish_session_history=getattr(args,'publish_decision_history',True))
        decision = json.loads(text)
        routes = route_from_decision(view,groups,forecasts,decision)
    except Exception as exc:
        error = repr(exc); decision = dict(default=dict(weights={selected:1.}),groups={})
        routes = route_from_decision(view,groups,forecasts,decision)
    name = 'current-window-'+folder.name
    bank.add_route(name,routes,selected)
    output = bank.predict(name,view,tools,visible,cache)
    deterministic = deterministic_decision(packet)
    deterministic_name = name+'-deterministic'
    bank.add_route(deterministic_name,route_from_decision(view,groups,forecasts,deterministic),selected)
    other = bank.predict(deterministic_name,view,tools,visible,cache)
    record = dict(program=name,global_program=selected,decision=decision,operational_fallback=error,
        candidates=list(forecasts),candidate_failures=failures,deterministic_program=deterministic_name,
        deterministic_decision=deterministic,real_agent_receipt=receipt,
        changed_groups=sum(v.get('weights')!={selected:1.} for v in decision.get('groups',{}).values()),
        native_requests=services[0].count if services else 0,
        completed_native_requests=requests.records if requests is not None else [],
        window_evidence='trajectory' if trajectory else 'summary')
    write(folder/'PLANNER-RESULT.json',record)
    return output, other, record, forecasts

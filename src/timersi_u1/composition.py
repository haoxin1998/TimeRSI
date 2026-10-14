"""Chronological composition of this run's own researched candidates."""
from copy import deepcopy
import hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
METRICS=('SQL','MASE','WAPE','WQL')
def cached_scouts(*args):return None,None

EXTRA_GUIDE = '''
This experiment reuses the code research and actual candidate forecasts of one
completed forecasting run. Code is fixed. Your responsibility is to improve its
deployment decisions using actual forecasts, visible context and completed errors.
incumbent153 is the original automated deployment recipe at this SAME origin,
including its target-specific combinations. It is an eligible complete candidate.
Its forecast is not a future answer. Retain, reduce, replace or combine it according
to evidence. Raw is also available. Do not discard useful complete recipes merely
because their individual components have different ranks on one historical window.

actual-candidates.npz contains current point and quantile arrays for all candidates.
visible-history/ contains all visible history and declared known-future covariates.
You can run the supplied Python executable with composition_probe.py --decision
draft.json or --weights '{"raw":0.3,"incumbent153":0.7}' [--group g000].
This scores the REAL candidate mixtures against REAL completed historical labels
using official SQL, MASE, WAPE and WQL. It returns group-level errors and saves
receipts. Make useful hypotheses, test them, inspect the actual errors and revise
the decision. Do not invent pseudo-labels. The recent S population may be a sample
of the full current population. A single historical success is limited evidence;
use current levels, increments, calendar and candidate mechanisms to judge transfer.
Do not overfit historical data. Changes with unstable evidence can retain Raw or
the incumbent; no candidate or blend is mandatory and no fixed weight is prescribed.

New native model calls and code research are unavailable in this cheap replay.
Some framework native variants may already be cached; use only actually listed IDs.
replay_feedback contains only outcomes already observable at this origin. Candidate
forecasts remain on the original research/memory trajectory; your new decisions
form their own chronological deployment feedback, not new code research.
Do not read outside this workspace. Current/future outcomes and benchmark scores
are private to the evaluator. Finish with decision.json in the required format.
'''

def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_task(source, cache):
    import datasets
    import fev
    args = read(source / 'RUN.json')
    os.environ['FEV_DATASETS_PREFIX'] = args['data_root']
    datasets.config.HF_DATASETS_CACHE = str(cache)
    task = fev.Task(**read(source / 'TASK.json'))
    task.load_full_dataset(num_proc=1)
    return task

def array_archive(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def score_arrays(point, quantiles, scoring, task):
    import numpy as np
    from fev.metrics import SQL, MASE, WAPE, WQL
    kw = dict(scoring, y_pred=point, q_pred=quantiles, seasonality=task.seasonality,
              quantile_levels=task.quantile_levels)
    result = {name:float(np.mean(metric.compute(**kw))) for name, metric in
              [('SQL', SQL()), ('MASE', MASE()), ('WAPE', WAPE(epsilon=1.)), ('WQL', WQL(epsilon=1.))]}
    cells = SQL().compute(**kw, reduce_axes=(1,))
    return result, cells

def past_files(original, packet):
    files = {n:original / 'latest-past-selection' / n / 'S-1.npz' for n in packet['candidates']}
    planner = read(original / 'PLANNER-RESULT.json')
    requested = planner.get('completed_native_requests', [])
    for name in files:
        if files[name].is_file(): continue
        for folder in sorted((original / 'native-requests').glob('request-*')):
            record = read(folder / 'RESULT.json', {})
            if record.get('candidate') == name: files[name] = folder / 'S-1.npz'
        if not files[name].is_file():
            for record in requested:
                if record.get('candidate') != name: continue
                # Actual request receipts may store their own folder.
                for key in ('directory', 'folder'):
                    if record.get(key): files[name] = Path(record[key]) / 'S-1.npz'
    return files

def prepare(config, job, source, task, window, index, destination, prior):
    import numpy as np
    from timersi_u1.core import Forecast, ProgramBank, write as typed_write
    from timersi_u1.data import make_view, historical_windows, predictions
    from timersi_u1.window_planner import groups_for, packet_for
    from timersi_u1.trajectory_evidence import enrich
    from .composition_probe import mixed_arrays, evaluate
    from .past_context import publish_history
    original = source / 'official' / f'window-{index:03d}'
    packet0 = read(original / 'PUBLIC-INPUT.json'); original_plan = read(original / 'PLANNER-RESULT.json')
    arrays = array_archive(original / 'CANDIDATES.npz'); ids = list(map(str, arrays['item_ids']))
    view = make_view(window, task)
    assert ids == [it['id'] for it in view.items]
    records = deepcopy(read(source / 'PROGRAM-BANK.json'))
    forecasts = {n:Forecast(arrays[n+'__point'], arrays[n+'__quantiles']).validate() for n in packet0['candidates']}
    files = past_files(original, packet0); selection = {n:dict(SQL=v['historical_S_SQL']) for n,v in packet0['candidates'].items()}
    histories = deepcopy(packet0.get('latest_past_selection_evidence', {}))
    scout_run, scout_folder = cached_scouts(source, index, task)
    scouts = []
    if scout_folder is not None:
        added = array_archive(scout_folder / 'CANDIDATES.npz'); added_ids = list(map(str, added['item_ids']))
        order = [added_ids.index(i) for i in ids]
        for suffix in ('__point', '__quantiles'):
            np.testing.assert_allclose(arrays['raw'+suffix], added['raw'+suffix][order], rtol=1e-6, atol=1e-5)
        scout_bank = read(scout_run / 'PROGRAM-BANK.json')
        for number, receipt in enumerate(read(scout_folder / 'PLANNER-RESULT.json')['automatic_native_scouts']):
            assert receipt['origin'] == 'automatic_native_scout' and not receipt['current_labels_used']
            old = receipt['candidate']; name = 'scout::'+old
            forecasts[name] = Forecast(added[old+'__point'][order], added[old+'__quantiles'][order]).validate()
            records[name] = scout_bank[old]; selection[name] = dict(SQL=receipt['historical_S_SQL'])
            histories[name] = receipt['past_selection']
            # Match the stored request receipt rather than relying on completed list order.
            folder = next(p for p in (scout_folder/'native-scouts').glob('request-*')
                          if read(p/'RESULT.json', {}).get('candidate') == old)
            files[name] = folder / 'S-1.npz'; scouts.append(dict(candidate=name, request=receipt['request']))
    raw_s = array_archive(files['raw'])
    warm = historical_windows(task, read(source/'RUN.json')['max_items'], 1, anchor=window)[1]
    scoring, sids = warm._prepare_arrays(predictions(Forecast(raw_s['point'], raw_s['quantiles']), task), task.quantile_levels)
    assert list(map(str,sids)) == list(map(str,raw_s['item_ids']))
    observed_ends = {str(item['id']):np.datetime64(item['timestamps'][len(item['y'][0])-1]) for item in view.items}
    past, future = warm.get_input_data()
    for item, dates in zip(past[task.id_column], future[task.timestamp_column]):
        assert np.datetime64(dates[-1]) <= observed_ends[str(item)]
    past_arrays = {'item_ids':np.asarray(sids)}; available=[]; missing=[]
    for name in forecasts:
        if not files[name].is_file(): missing.append(name); continue
        archive = array_archive(files[name]); order = [list(map(str,archive['item_ids'])).index(str(i)) for i in sids]
        p, q = archive['point'][order], archive['quantiles'][order]
        if not np.isfinite(p).all() or not np.isfinite(q).all(): missing.append(name); continue
        past_arrays[name+'__point'] = p; past_arrays[name+'__quantiles'] = q; available.append(name)
    group_defs = [dict(key=g['key'],target=g['target'],members=g['members']) for g in packet0['groups']]
    prior_recipe = original_plan['decision']
    recipe_ids = set(prior_recipe['default']['weights'])
    recipe_ids |= {n for g in prior_recipe.get('groups',{}).values() for n in g['weights']}
    if recipe_ids <= set(available):
        p,q = mixed_arrays(past_arrays, dict(candidates=available,targets=list(task.target_columns),groups=group_defs), prior_recipe)
        past_arrays['incumbent153__point']=p; past_arrays['incumbent153__quantiles']=q; available.append('incumbent153')
        metrics, _ = score_arrays(p,q,scoring,task); selection['incumbent153']=dict(SQL=metrics['SQL'])
    else: selection['incumbent153']=dict(SQL=None)
    current = array_archive(original / 'predictions.npz'); assert list(map(str,current['item_ids'])) == ids
    forecasts['incumbent153'] = Forecast(current['point'],current['quantiles']).validate()
    records['incumbent153'] = dict(kind='route', routes={}, members=list(recipe_ids), fallback=packet0['global_winner'],
                                    origin='original153_current_origin_deployment_recipe')
    bank = ProgramBank(records); groups = groups_for(view,forecasts['raw'])
    original_by_members = {(g['target'],tuple(sorted(g['members']))):g for g in packet0['groups']}
    fresh, _ = packet_for(view,bank,forecasts,'incumbent153',selection,[],groups=groups)
    for g in fresh['groups']:
        old = original_by_members.get((g['target'],tuple(sorted(g['members']))))
        if old: g['mature_candidate_errors'].update(old['mature_candidate_errors'])
    fresh = enrich(view,forecasts,fresh)
    workspace = destination / 'agent-session/workspace'; workspace.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(workspace/'actual-candidates.npz', item_ids=ids,
        **{n+'__point':f.point for n,f in forecasts.items()}, **{n+'__quantiles':f.quantiles for n,f in forecasts.items()})
    publish_history(view,workspace/'visible-history')
    composition = workspace/'composition-past'; composition.mkdir()
    manifest = dict(candidates=available,targets=list(task.target_columns),groups=[dict(key=g['key'],target=g['target'],members=g['members']) for g in fresh['groups']],
                    seasonality=task.seasonality,quantile_levels=task.quantile_levels,
                    episodes=[dict(arrays='recent.npz',context=dict(kind='recent',offset=1,
                        ids=list(map(str,sids)),forecast_ends=[str(d[-1]) for d in future[task.timestamp_column]]))])
    write(composition/'MANIFEST.json',manifest); np.savez_compressed(composition/'recent.npz',**past_arrays,**scoring)
    shutil.copy2(Path(config['output'])/'source/composition_probe.py',workspace/'composition_probe.py')
    max_difference = 0.
    for n in available:
        if n == 'incumbent153' or not isinstance(histories.get(n),list): continue
        evaluated = evaluate(composition,dict(default=dict(weights={n:1}),groups={}))['episodes'][0]['metrics']
        expected = histories[n][0]['metrics']
        max_difference = max(max_difference,max(abs(evaluated[m]-expected[m]) for m in METRICS))
    assert max_difference < 1e-8, max_difference
    # Publish chronological feedback only after its forecast end is observed.
    feedback=[]
    for row in prior[-3:]:
        visible=[]
        for item, ends, cells in zip(row['ids'],row['forecast_ends'],row['cells']):
            if item in observed_ends and np.datetime64(ends)<=observed_ends[item]:
                visible.append(dict(id=item,forecast_end=ends,SQL_by_target=cells))
        if visible: feedback.append(dict(window=row['window'],targets=row['targets'],decision=row['decision'],visible=visible))
    for name in packet0['candidates']:
        node=records[name]
        if node['kind']!='code': continue
        code = workspace/'candidate-code'/name; code.mkdir(parents=True,exist_ok=True)
        (code/'candidate.py').write_text(node['code'])
        for relative,text in node.get('files',{}).items():
            path=code/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    fresh.update(python_executable=sys.executable,native_request_budget=0,current_candidate_arrays='actual-candidates.npz',
        numerical_guard_policy=packet0.get('numerical_guard_policy'),numerical_guard=packet0.get('numerical_guard',{}),
        complete_visible_history=dict(index='visible-history/INDEX.json',arrays='visible-history/inputs.npz'),
        composition_probe=dict(directory='composition-past',script='composition_probe.py'),
        original_deployment_recipe=prior_recipe,latest_past_selection_evidence=histories,
        replay_feedback=feedback,available_past_candidates=available,missing_past_candidates=missing,
        automatic_cached_scouts=scouts,scope=config['scope'])
    typed_write(destination/'PUBLIC-INPUT.json',fresh)
    write(destination/'PREPARATION.json',dict(source=str(source),window=index,source_candidate_sha256=digest(original/'CANDIDATES.npz'),
        source_packet_sha256=digest(original/'PUBLIC-INPUT.json'),candidates=len(forecasts),past_candidates=len(available),
        scouts=len(scouts),past_control_metric_max_difference=max_difference,current_labels_published=False,new_native_calls=0))
    return view, forecasts, fresh, workspace

def worker(config, label, source):
    import numpy as np
    from timersi_u1.agent import propose,long_context_model
    from timersi_u1.data import predictions
    from timersi_u1.window_planner import GUIDE,route_from_decision
    from timersi_u1.selection import route_forecasts
    from timersi_u1.core import write as typed_write
    from .composition_probe import mixed_arrays
    job = next(j for j in config['jobs'] if j['replay_label']==label)
    out = Path(config['output'])/'tasks'/label; out.mkdir(parents=True,exist_ok=True)
    task=load_task(source,out/'dataset-cache'); rows=[]; prior=[];started=time.time()
    assert job['expected_windows']<=task.num_windows
    for index in range(job['expected_windows']):
        directory=out/'official'/f'window-{index:03d}'
        write(out/'STATUS.json',dict(phase='prepare',window=index,pid=os.getpid(),at=time.time()))
        if (directory/'RESULTS.json').is_file():
            row=read(directory/'RESULTS.json'); rows.append(row); prior.append(row['feedback']); continue
        directory.mkdir(parents=True,exist_ok=True)
        window=task.get_window(index,num_proc=1)
        view,forecasts,packet,workspace=prepare(config,job,source,task,window,index,directory,prior)
        write(out/'STATUS.json',dict(phase='LLM_decision',window=index,pid=os.getpid(),at=time.time()))
        failure=None; receipt=None; decision=None
        try:
            text,receipt=propose(directory/'agent-session',packet,[],'',[],model='fable',effort='high',timeout=1800,
                max_turns=64,guide=GUIDE+EXTRA_GUIDE,vertex_project=config['vertex_project'],vertex_region='global',
                output_artifact='decision.json',publish_session_history=False)
            assert receipt['configured_model']==long_context_model('fable')
            assert receipt['effort']=='high' and 1000000 in receipt['reported_context_windows']
            assert 'claude-fable-5' in receipt['observed_models'] and receipt['exit_code']==0 and not receipt['timeout']
            decision=json.loads(text)
            routes=route_from_decision(view,packet['groups'],forecasts,decision)
            forecast=route_forecasts(view,forecasts,routes,'incumbent153')
        except Exception:
            failure=traceback.format_exc(); decision=dict(default=dict(weights={'incumbent153':1.}),groups={})
            forecast=forecasts['incumbent153']; receipt=read(directory/'agent-session/AGENT-RESULT.json',receipt)
        # Current truth is prepared ONLY AFTER a decision has been fixed.
        scoring,ids=window._prepare_arrays(predictions(forecasts['raw'],task),task.quantile_levels)
        assert list(map(str,ids))==[it['id'] for it in view.items]
        original=read(source/'official'/f'window-{index:03d}'/'METRICS.json')
        raw,raw_cells=score_arrays(forecasts['raw'].point,forecasts['raw'].quantiles,scoring,task)
        incumbent,inc_cells=score_arrays(forecasts['incumbent153'].point,forecasts['incumbent153'].quantiles,scoring,task)
        actual,cells=score_arrays(forecast.point,forecast.quantiles,scoring,task)
        delta=max(abs(v[m]-original[arm][m]) for arm,v in [('raw',raw),('loop',incumbent)] for m in METRICS)
        assert delta<1e-8,delta
        # Independent application of the saved JSON must reproduce the scored output.
        arrays=array_archive(workspace/'actual-candidates.npz')
        replay_p,replay_q=mixed_arrays(arrays,dict(candidates=list(forecasts),targets=list(task.target_columns),groups=packet['groups']),decision)
        np.testing.assert_allclose(replay_p,forecast.point,rtol=0,atol=1e-8)
        np.testing.assert_allclose(replay_q,forecast.quantiles,rtol=0,atol=1e-8)
        candidate_errors={n:score_arrays(f.point,f.quantiles,scoring,task)[0]['SQL'] for n,f in forecasts.items()}
        ends=[str(it['timestamps'][len(it['y'][0])+task.horizon-1]) for it in view.items]
        feedback=dict(window=index,ids=list(map(str,ids)),targets=list(task.target_columns),forecast_ends=ends,
                      cells=[dict(new=c.tolist(),raw=r.tolist(),incumbent153=b.tolist()) for c,r,b in zip(cells,raw_cells,inc_cells)],decision=decision)
        np.savez_compressed(directory/'predictions.npz',item_ids=ids,point=forecast.point,quantiles=forecast.quantiles)
        row=dict(window=index,metrics=actual,original153=incumbent,raw=raw,decision=decision,
                 operational_fallback=failure,receipt=receipt,control_metric_max_difference=delta,feedback=feedback,
                 candidate_SQL=candidate_errors,oracle_SQL=min(candidate_errors.values()),
                 actual_probe_receipts=len(list((workspace/'composition-probes').glob('*.json'))),new_native_calls=0)
        typed_write(directory/'RESULTS.json',row); rows.append(read(directory/'RESULTS.json'));prior.append(rows[-1]['feedback'])
        write(out/'STATUS.json',dict(phase='window_complete',window=index,complete_windows=len(rows),pid=os.getpid(),at=time.time()))
    result=dict(task=task.task_name,full_task=len(rows)==task.num_windows,windows=len(rows),scores={arm:{m:float(np.mean([r[arm][m] for r in rows])) for m in METRICS}
        for arm in ('metrics','original153','raw')},new_LLM_sessions=len(rows),
        failed_decisions=sum(bool(r['operational_fallback']) for r in rows),new_native_calls=0,
        actual_probe_receipts=sum(r['actual_probe_receipts'] for r in rows),seconds=time.time()-started,source=str(source),
        scope=config['scope'],source_result_sha256=digest(source/'RESULTS.json'))
    write(out/'RESULTS.json',result);write(out/'STATUS.json',dict(phase='complete',at=time.time(),pid=os.getpid(),complete_windows=len(rows)))
    print(json.dumps(dict(task=task.task_name,SQL=result['scores']['metrics']['SQL'],failed=result['failed_decisions'])),flush=True)

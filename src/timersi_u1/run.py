"""One end-to-end search loop, reused when new forecast errors become observable."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import yaml
import fev
from .core import ProgramBank, write
from .data import make_view, score, predictions as fev_predictions, official_sql_by_target
from .memory import MaturedMemory
from .tools import Tools
from .search import search, export_program


def run(args,task_override=None):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    import datasets
    datasets.config.HF_DATASETS_CACHE=root/'dataset-cache'
    started=time.monotonic()
    resume=getattr(args,'resume',False)
    if (root/'RUN.json').exists():
        if not resume:raise ValueError('Existing run: use --resume with the original frozen source')
        original=json.loads((root/'RUN.json').read_text())
        for key,value in vars(args).items():
            if key not in ('resume','device') and original.get(key)!=value:
                raise ValueError(f'Resume configuration changed: {key}')
        if (root/'RESULTS.json').exists():return json.loads((root/'RESULTS.json').read_text())
    else:
        original={**vars(args),'pid':os.getpid(),'started_unix':time.time(),
            'old_predictions_used':False,'protocol':'development pilot; exact official windows, possibly limited scope'}
        write(root/'RUN.json',original)
    with (root/'ATTEMPTS.jsonl').open('a') as stream:
        stream.write(json.dumps({'pid':os.getpid(),'started_unix':time.time(),'resume':resume})+'\n')
    if task_override is None:
        if args.data_root:os.environ['FEV_DATASETS_PREFIX']=args.data_root
        definitions=yaml.safe_load(Path(args.tasks_yaml).read_text())['tasks']
        choices=[d for d in definitions if fev.Task(**d).task_name==args.task]
        if len(choices)!=1:raise ValueError(f'Expected one task definition, got {len(choices)}')
        task=fev.Task(**choices[0])
    else:task=task_override
    task.load_full_dataset(num_proc=1)
    write(root/'TASK.json',task.to_dict())
    # A fixed research budget must span the deployment, not all expire in its first six windows.
    import math
    research_every=args.update_every or (2 if args.online_updates<0 else
        max(1,math.ceil(task.num_windows/(args.online_updates+1))))
    reselect_every=getattr(args,'reselect_every',0)
    write(root/'SEARCH-SCHEDULE.json',{'research_every_mature_windows':research_every,
        'reselect_every_mature_windows':reselect_every,'max_new_code_updates':args.online_updates,
        'research_spacing_uses':'known task window count only, never future targets'})
    tools=Tools(args.checkpoint,args.device,root/'native-cache')
    memory=MaturedMemory(root/'memory-private')
    saved=json.loads((root/'CHECKPOINT.json').read_text()) if resume and (root/'CHECKPOINT.json').exists() else None
    bank=ProgramBank(saved['bank'] if saved else None)
    history=saved['history'] if saved else []
    previous=[root/p for p in saved['previous']] if saved else []
    def status(phase,**values):
        write(root/'STATUS.json',{'phase':phase,'pid':os.getpid(),'at':time.time(),**values})
    if saved:
        selected=saved['selected'];epochs=saved['epochs'];updates=saved['updates']
        last_refresh_count=saved['last_refresh_count'];update_errors=saved['update_errors'];rows=saved['rows']
        last_selection_count=saved.get('last_selection_count',last_refresh_count)
        champions=saved.get('deployment_champions',[selected])
    else:
        selected,initial=search(args,root,task,task.get_window(0,num_proc=1),
            bank,tools,memory,history,previous)
        epochs=[initial];updates=0;last_refresh_count=0;update_errors=[];rows=[]
        last_selection_count=0
        champions=[]
    def checkpoint():
        write(root/'CHECKPOINT.json',{'bank':bank.records,'history':history,
            'previous':[str(p.relative_to(root)) for p in previous],
            'selected':selected,'epochs':epochs,'updates':updates,
            'last_refresh_count':last_refresh_count,'last_selection_count':last_selection_count,
            'deployment_champions':champions,'update_errors':update_errors,'rows':rows})
    checkpoint()
    first_pending=len(rows)
    for index,window in enumerate(task.iter_windows(num_proc=1)):
        if args.official_windows and index>=args.official_windows:break
        if index<first_pending:continue
        view=make_view(window,task);visible=memory.visible(view)
        complete={r['key'] for r in visible if r['source']=='raw' and r['key'].startswith('official-')
            and r['key'].endswith('-raw')
            and set(r['ids'])==set(memory.records[r['key']]['ids'])}
        if len(complete)-last_refresh_count>=research_every and (args.online_updates<0 or updates<args.online_updates):
            updates+=1;last_refresh_count=len(complete)
            write(root/'epochs'/f'e{updates:03d}'/'TRIGGER.json',
                {'official_window':index,'fully_matured_official_records':sorted(complete)})
            try:
                selected,epoch=search(args,root,task,window,bank,tools,memory,history,previous,
                    epoch=updates,incumbent=selected,iterations=args.update_iterations,retained_programs=champions)
                epochs.append(epoch)
            except Exception:
                error={'epoch':updates,'window':index,'retained_program':selected,
                       'traceback':traceback.format_exc()}
                update_errors.append(error)
                write(root/'epochs'/f'e{updates:03d}'/'ERROR.json',error)
            last_selection_count=len(complete)
            checkpoint()
            visible=memory.visible(view)
        elif reselect_every and len(complete)-last_selection_count>=reselect_every:
            tag=f'e{updates:03d}-refresh-w{index:03d}'
            write(root/'epochs'/tag/'TRIGGER.json',{'official_window':index,
                'fully_matured_official_records':sorted(complete),'new_agent_calls':0})
            try:
                selected,selection=search(args,root,task,window,bank,tools,memory,history,previous,
                    epoch=updates,incumbent=selected,iterations=0,record_tag=tag,retained_programs=champions)
                epochs.append(selection)
            except Exception:
                error={'epoch':updates,'window':index,'record_tag':tag,'retained_program':selected,
                    'traceback':traceback.format_exc()}
                update_errors.append(error);write(root/'epochs'/tag/'ERROR.json',error)
            last_selection_count=len(complete);checkpoint();visible=memory.visible(view)
        status('official',window=index,selected=selected,online_updates=updates)
        cache={}
        output=bank.predict(selected,view,tools,visible,cache)
        global_output=output
        deployment_program=selected
        planner_record=None
        candidate_forecasts={}
        deterministic_output=None
        if getattr(args,'current_window_planner',False):
            from .window_planner import deploy
            status('current_window_planner',window=index,selected=selected)
            output,deterministic_output,planner_record,candidate_forecasts=deploy(
                root/'official'/f'window-{index:03d}',view,bank,tools,visible,cache,selected,epochs[-1],args,
                task=task,window=window,memory=memory)
            deployment_program=planner_record['program']
        raw=bank.predict('raw',view,tools,visible,cache)
        metrics={'loop':score(output,window,task)}
        metrics['raw']=metrics['loop'] if deployment_program=='raw' else score(raw,window,task)
        control_arrays={}
        if planner_record is not None:
            from .timing import phase
            with phase('comparison_metrics', diagnostic=True):
                metrics['historical_program']=score(global_output,window,task)
                metrics['deterministic_combination']=score(deterministic_output,window,task)
            control_arrays.update(historical_point=global_output.point,historical_quantiles=global_output.quantiles,
                deterministic_point=deterministic_output.point,deterministic_quantiles=deterministic_output.quantiles)
        if getattr(args,'deployment_policy','global')=='adaptive':
            global_selected=epochs[-1].get('global_selected',selected)
            global_output=bank.predict(global_selected,view,tools,visible,cache)
            metrics['global_selector']=metrics['loop'] if global_selected==selected else score(global_output,window,task)
            control_arrays.update(global_point=global_output.point,global_quantiles=global_output.quantiles)
        if getattr(args,'frozen_control',False):
            frozen=bank.predict(epochs[0]['selected'],view,tools,visible,cache)
            if epochs[0]['selected']==deployment_program:metrics['frozen_code']=metrics['loop']
            elif epochs[0]['selected']=='raw':metrics['frozen_code']=metrics['raw']
            else:metrics['frozen_code']=score(frozen,window,task)
            control_arrays.update(frozen_point=frozen.point,frozen_quantiles=frozen.quantiles)
        folder=root/'official'/f'window-{index:03d}';folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'predictions.npz',point=output.point,quantiles=output.quantiles,
                            raw_point=raw.point,raw_quantiles=raw.quantiles,item_ids=[it['id'] for it in view.items],**control_arrays)
        write(folder/'METRICS.json',metrics);rows.append(metrics)
        write(folder/'DECISION.json',{'program':deployment_program,'research_program':selected,'epoch':updates,
            'global_selector_program':epochs[-1].get('global_selected',selected),
            'selection_record':epochs[-1].get('record_tag',f'e{updates:03d}')})
        write(folder/'MEMORY-VISIBILITY.json',{'records':[{'key':r['key'],'source':r['source'],
            'ids':r['ids'],'ends':r['ends']} for r in visible]})
        arrays,_=window._prepare_arrays(fev_predictions(output,task),task.quantile_levels)
        raw_sql=official_sql_by_target(raw,window,task)
        selected_sql=raw_sql if deployment_program=='raw' else official_sql_by_target(output,window,task)
        memory.observe(f'official-{index:03d}-raw','raw',view,raw,arrays['y_true'],official_sql=raw_sql)
        memory.observe(f'official-{index:03d}-selected',bank.memory_source(deployment_program),view,output,arrays['y_true'],
                       official_sql=selected_sql)
        for name,forecast in candidate_forecasts.items():
            if name=='raw' or name==deployment_program:continue
            cell_sql=official_sql_by_target(forecast,window,task)
            memory.observe(f'official-{index:03d}-candidate-{name}',bank.memory_source(name),view,forecast,
                arrays['y_true'],official_sql=cell_sql)
        if planner_record is not None:
            memory.observe(f'official-{index:03d}-deterministic',bank.memory_source(planner_record['deterministic_program']),
                view,deterministic_output,arrays['y_true'],official_sql=official_sql_by_target(deterministic_output,window,task))
        champions=[n for n in champions if n!=selected]+[selected]
        checkpoint()
    (root/'selected_program.py').write_text(export_program(bank,deployment_program))
    write(root/'PROGRAM-BANK.json',bank.records)
    result={'task':args.task,'selected':selected,'official_windows':len(rows),
        'full_task':len(rows)==task.num_windows,'iterations':history,
        'successful_real_proposals':sum(e['real_calls'] for e in epochs),
        'native_batches_this_attempt':tools.calls,'native_cache_hits_this_attempt':tools.hits,
        'seconds':time.time()-original['started_unix'],'attempt_seconds':time.monotonic()-started,
        'runtime_includes_interruption_time':resume,
        'scores':{arm:{m:float(np.mean([r[arm][m] for r in rows])) for m in ('SQL','MASE','WAPE','WQL')} for arm in ('raw','loop')},
        'old_predictions_used':False,'matured_prediction_feedback_implemented':True,
        'online_code_updates_implemented':True,'online_searches':updates,
        'selection_refreshes':sum('refresh' in e.get('record_tag','') for e in epochs),
        'epoch_selections':[{'epoch':e['epoch'],'selected':e['selected']} for e in epochs],
        'update_errors':update_errors}
    if getattr(args,'frozen_control',False):
        result['frozen_code_scores']={m:float(np.mean([r['frozen_code'][m] for r in rows])) for m in ('SQL','MASE','WAPE','WQL')}
        result['frozen_control_note']='Initial code held fixed, same matured memory. Diagnostic compute included in wall time.'
    if getattr(args,'deployment_policy','global')=='adaptive':
        result['global_selector_scores']={m:float(np.mean([r['global_selector'][m] for r in rows])) for m in ('SQL','MASE','WAPE','WQL')}
        result['global_selector_note']='Same adaptive-run candidate bank, global S winner; diagnostic, not an independent search trajectory.'
    if getattr(args,'interactive_research',False):
        result['observed_agent_sessions']=sum(e['real_calls'] for e in epochs)
        result['successful_real_proposals']=sum(r['status']=='complete' for r in history)
        result['research_transport']='interactive_probe'
        result['proposal_count_note']='Executor probes, not LLM sessions; see observed_agent_sessions and session receipts.'
    if getattr(args,'current_window_planner',False):
        result['current_window_planner_implemented']=True
        result['deployment_program']=deployment_program
        result['comparison_scores']={arm:{m:float(np.mean([r[arm][m] for r in rows]))
            for m in ('SQL','MASE','WAPE','WQL')} for arm in ('historical_program','deterministic_combination')}
        decisions=[json.loads(p.read_text()) for p in sorted((root/'official').glob('*/PLANNER-RESULT.json'))]
        result['current_window_decision_calls']=len(decisions)
        result['current_window_operational_fallbacks']=[{'window':i,'error':d['operational_fallback']}
            for i,d in enumerate(decisions) if d['operational_fallback']]
        result['same_pool_control_scope']='Same research trajectory/current forecasts/mature memory; not independently researched loops'
    write(root/'RESULTS.json',result);status('complete');return result


def main():
    p=argparse.ArgumentParser()
    for name in ('task','tasks-yaml','data-root','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--max-items',type=int,default=16)
    p.add_argument('--iterations',type=int,default=2);p.add_argument('--official-windows',type=int,default=2)
    p.add_argument('--research-windows',type=int,default=2);p.add_argument('--selection-windows',type=int,default=1)
    p.add_argument('--selection-objective',choices=['mean-sql','raw-ratio'],default='mean-sql')
    p.add_argument('--online-updates',type=int,default=1);p.add_argument('--update-every',type=int,default=2)
    p.add_argument('--update-iterations',type=int,default=2)
    p.add_argument('--model',default='fable');p.add_argument('--effort',default='high')
    p.add_argument('--vertex-project');p.add_argument('--vertex-region',default='global')
    p.add_argument('--agent-timeout',type=int,default=900)
    p.add_argument('--deployment-policy',choices=['global','adaptive'],default='adaptive')
    p.add_argument('--research-design',choices=['control','growing','old-style','native-growth','placement'],default='control')
    p.add_argument('--parent-mixtures',type=float,nargs='*',default=[.25,.5])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--frozen-control',action='store_true')
    p.add_argument('--top3',action='store_true')
    p.add_argument('--interactive-research',action='store_true')
    p.add_argument('--current-window-planner',action='store_true')
    p.add_argument('--forecast-tool-requests',type=int,default=0)
    p.add_argument('--window-evidence',choices=['summary','trajectory'],default='summary')
    p.add_argument('--reselect-every',type=int,default=0)
    args=p.parse_args()
    try:run(args)
    except Exception:
        write(Path(args.output)/'ERROR.json',{'traceback':traceback.format_exc(),'at':time.time()})
        raise


if __name__=='__main__':main()

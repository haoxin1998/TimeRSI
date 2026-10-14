"""The same forecast-program search at initial and later prediction origins."""
import os
import hashlib
import json
import time
import traceback
import sys
import numpy as np
from .core import write, select
from .data import historical_windows,make_view,score,error_details,predictions as fev_predictions,EvaluationContext
from .agent import propose
from .research_packet import publish
from .research_packet import publish_native_calls
from contextlib import nullcontext
from .deployment_feedback import deployment_feedback
from .research_design import context_comparison
from .temporal_selection import choose_selection
from .scoreability import eligible_research_offsets,historical_mean,recovery_shortlist


def diverse_shortlist(scores, initial_incumbent, limit=7):
    """Reserve room for distinct code families before their weight variants."""
    ordered=sorted(scores,key=lambda n:(scores[n],n))
    chosen=list(dict.fromkeys(['raw',initial_incumbent]));families={n.split('_mix')[0] for n in chosen}
    for name in ordered:
        family=name.split('_mix')[0]
        if family not in families and len(chosen)<limit:
            chosen.append(name);families.add(family)
    for name in ordered:
        if name not in chosen and len(chosen)<limit:chosen.append(name)
    return chosen


def archive_candidates(bank,scores,retained_programs=()):
    """All this-run code and deployed compositions remain eligible, without age caps."""
    names=list(retained_programs)+[n for n,s in bank.records.items() if s['kind']=='code']
    chosen=[]
    for name in names:
        if name not in bank.records or name in scores or name in chosen:continue
        chosen.append(name)
    return chosen


def top3_members(bank,scores,caches,folds,incumbent='raw'):
    """Rank original programs on R; avoid copies of identical R predictions."""
    members=[];seen=set()
    for name in sorted(scores,key=lambda n:(scores[n],n)):
        if not np.isfinite(scores[name]):continue
        if bank.records[name]['kind']!='code' and name!=incumbent:continue
        signature=hashlib.sha256()
        for fold in folds:
            prediction=caches[fold][name]
            for values in (prediction.point,prediction.quantiles):
                signature.update(np.asarray(values,dtype=np.float64).tobytes())
        key=signature.hexdigest()
        if key in seen:continue
        seen.add(key);members.append(name)
        if len(members)==3:break
    return members


def split_offsets(warm,selection_windows):
    ordered=sorted(warm)
    # Reserve at least one research window when short history cannot fit defaults.
    count=min(selection_windows,max(0,len(ordered)-1))
    return sorted(ordered[count:],reverse=True),sorted(ordered[:count],reverse=True)


def export_program(bank, name):
    emitted={};blocks=['import numpy as np\nfrom timersi_u1.core import Forecast\nfrom timersi_u1.program import load_program\nfrom timersi_u1.selection import route_forecasts\nfrom timersi_u1.scenario_selection import apply_scenario\n']
    def emit(key):
        if key in emitted:return emitted[key]
        node=bank.records[key];fn='program_'+str(len(emitted));emitted[key]=fn
        if node['kind']=='code':
            blocks.append(f'_scope_{fn} = load_program({node["code"]!r}, {node.get("files",{})!r})\n{fn} = _scope_{fn}["predict"]\n')
        elif node['kind']=='route':
            children={n:emit(n) for n in node['members']}
            calls=', '.join(f'{n!r}: {fn}(view, tools, memory)' for n,fn in children.items())
            blocks.append(f'def {fn}(view, tools, memory):\n    return route_forecasts(view, {{{calls}}}, {node["routes"]!r}, {node["fallback"]!r})\n')
        elif node['kind']=='scenario':
            children={n:emit(n) for n in node['members']}
            callbacks=', '.join(f'{n!r}: {child}' for n,child in children.items())
            blocks.append(f'def {fn}(view, tools, memory):\n    callbacks = {{{callbacks}}}\n    return apply_scenario(view, lambda name: callbacks[name](view, tools, memory), {node["policy"]!r})\n')
        else:
            children=[emit(n) for n in node['members']]
            calls=', '.join(f'{n}(view, tools, memory)' for n in children)
            blocks.append(f'def {fn}(view, tools, memory):\n    return tools.blend([{calls}], {node["weights"]!r})\n')
        return fn
    entry=emit(name);blocks.append(f'predict = {entry}\n')
    return '\n'.join(blocks)


def search(args,root,task,anchor,bank,tools,memory,history,previous,epoch=0,incumbent='raw',iterations=None,record_tag=None,retained_programs=()):
    tag=record_tag or f'e{epoch:03d}';record=root/'epochs'/tag;record.mkdir(parents=True,exist_ok=True)
    budget=args.iterations if iterations is None else iterations
    warm=historical_windows(task,args.max_items,args.research_windows+args.selection_windows,anchor)
    R,S=split_offsets(warm,args.selection_windows)
    write(record/'SPLITS.json',{'research_offsets':R,'selection_offsets':S,'anchor_cutoff':str(anchor.cutoff)})
    if not S or not R:
        result={'epoch':epoch,'selected':incumbent,'scores':{},'errors':[],
            'global_selected':incumbent,'global_selection_SQL':None,
            'real_calls':0,'reason':'Insufficient history for separate research and selection; retain current program'}
        write(record/'SELECTION.json',result);write(root/'SELECTION.json',result)
        return incumbent,result
    views={k:make_view(w,task) for k,w in warm.items()}
    evaluations={k:EvaluationContext(w,task) for k,w in warm.items()}
    caches={k:{} for k in warm};scores={};research_results={};real_calls=0;initial_incumbent=incumbent
    scoreable_R=None
    def status(phase,**values):write(root/'STATUS.json',{'phase':phase,'epoch':epoch,'pid':os.getpid(),'at':time.time(),**values})
    def evaluate(name,folds):
        rows=[]
        for k in folds:
            status('evaluate',program=name,fold=k)
            visible=memory.visible(views[k])
            capture = tools.record_native_calls() if folds == R else nullcontext([])
            with capture as native_calls:
                output=bank.predict(name,views[k],tools,visible,caches[k])
            evaluation=evaluations[k].prepare(output)
            metrics=evaluation.score()
            folder=record/'judge'/f'fold-{k}'/name;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'predictions.npz',point=output.point,quantiles=output.quantiles)
            detail=evaluation.error_details()
            write(folder/'METRICS.json',metrics);write(folder/'ERROR-DETAILS.json',detail)
            if folds == R:
                publish_native_calls(record/'R-native'/f'fold-{k}', native_calls, tools.checkpoint)
            guard = getattr(output, 'numerical_guard', {})
            write(folder/'NUMERICAL-GUARD.json', guard)
            rows.append({'offset':k,'metrics':metrics,'error_details':detail,'numerical_guard':guard})
            memory.observe(f'{tag}-fold-{k}-{name}',bank.memory_source(name),views[k],output,evaluation.arrays['y_true'])
        result={'SQL':historical_mean(rows,scoreable_R if folds==R else None),'folds':rows}
        if folds==R:research_results[name]=result
        return result
    baseline=evaluate('raw',R)
    scoreable_R=eligible_research_offsets(baseline)
    baseline['SQL']=historical_mean(baseline['folds'],scoreable_R);scores['raw']=baseline['SQL']
    scoreability=dict(original_research_offsets=R,scoreable_research_offsets=scoreable_R,
        undefined_research_offsets=[k for k in R if k not in scoreable_R],selection_offsets=S,
        scope='Shared Raw-data eligibility for all candidates; official test and S windows unchanged',
        all_R_undefined_policy='Retain submitted executable candidates for independent historical S selection')
    write(record/'HISTORICAL-SCOREABILITY.json',scoreability)
    if incumbent!='raw':scores[incumbent]=evaluate(incumbent,R)['SQL']
    # Re-evaluate actual prior branches in this run, not only their old scores.
    archived=archive_candidates(bank,scores,retained_programs)
    write(record/'ARCHIVE-RECHECK.json',{'programs':archived,
        'prior_deployment_winners':list(retained_programs),'source':'this run only'})
    for name in archived:
        try:scores[name]=evaluate(name,R)['SQL']
        except Exception:write(record/(name+'-RECHECK-ERROR.json'),{'traceback':traceback.format_exc()})
    research_directory=publish(record/'public-research',views,R,warm,caches,task) if budget else None
    deployed=deployment_feedback(memory,views[S[0]],bank) if budget else None
    if deployed is not None:write(record/'DEPLOYMENT-FEEDBACK.json',deployed)
    earliest=views[R[0]]
    from .numerical_boundary import POLICY
    packet={'numerical_guard_policy':POLICY,'task':args.task,'epoch':epoch,'targets':earliest.targets,'frequency':earliest.frequency,
        'research_design':getattr(args,'research_design','control'),
        'context_comparison':context_comparison({k:views[k] for k in R},make_view(anchor,task)),
        'task_facts':earliest.facts,
        'current_forecast_inputs':tools.scene(make_view(anchor,task)),
        'matured_deployment_feedback':deployed,
        'research_contexts':{str(k):tools.scene(views[k]) for k in R},
        'horizon':earliest.horizon,'seasonality':earliest.seasonality,'raw_R':baseline,
        'python_executable':sys.executable,'research_arrays':'research-data/INDEX.json',
        'incumbent_R_SQL':scores[incumbent],'selection_scores_not_shared':True,
        'research_scoreability':scoreability,
        'examples':[{'id':it['id'],'tail_targets':earliest.targets[:8],'past_target_tail':it['y'][:8,-32:],
            'past_dates_tail':[str(x) for x in it['timestamps'][max(0,it['y'].shape[1]-32):it['y'].shape[1]]],
            'known_columns':list(it['known']),'past_columns':list(it['past']),
            'known_future':{n:v[it['y'].shape[1]:] for n,v in it['known'].items()},
            'known_future_original':{n:v[it['y'].shape[1]:] for n,v in it.get('known_raw',{}).items()},
            'static':{n:str(v) for n,v in it['static'].items()}} for it in earliest.items[:4]]}
    interactive=getattr(args,'interactive_research',False) and budget>0
    session_receipt=None
    if interactive:
        from .interactive import interactive_session,freeze_program
        import shutil
        def register_probe(workspace,i,request,restore=False):
            nonlocal incumbent
            folder=root/'iterations'/f'{tag}-iter-{i:03d}';folder.mkdir(parents=True,exist_ok=True)
            parent=request.get('parent') or incumbent
            try:
                if parent not in scores:raise ValueError('Parent must be Raw or a program evaluated in this epoch')
                if restore:
                    code=(folder/'candidate.py').read_text()
                    files={str(p.relative_to(folder/'code')):p.read_text() for p in (folder/'code').rglob('*.py')}
                else:code,files=freeze_program(workspace,folder)
                write(folder/'PROBE-REQUEST.json',{'iteration':i,'parent':parent,'epoch':epoch})
                name=f'{tag}_candidate_{i:03d}';bank.add(name,code,parent,files);names=[name]
                for weight in getattr(args,'parent_mixtures',(.25,.5)):
                    key=f'{name}_mix{int(weight*100)}';bank.add_blend(key,[parent,name],[1-weight,weight]);names.append(key)
                results={}
                for key in names:
                    result=evaluate(key,R);scores[key]=result['SQL'];results[key]=result
                incumbent=min(scores,key=scores.get)
                feedback={'epoch':epoch,'iteration':i,'parent':parent,'results':results,
                    'research_incumbent':incumbent,'incumbent_SQL':scores[incumbent],
                    'status':'complete','transport':'interactive_probe'}
            except Exception:
                feedback={'epoch':epoch,'iteration':i,'parent':parent,'status':'failed',
                    'transport':'interactive_probe','traceback':traceback.format_exc()}
            write(folder/'R-FEEDBACK.json',feedback);history.append(feedback);previous.append(folder)
            write(root/'PROGRAM-BANK.json',bank.records)
            if workspace is not None:
                archive=workspace/'archive'/f'probe-{i:03d}'
                shutil.copytree(folder,archive,dirs_exist_ok=True)
                for name in feedback.get('results',{}):
                    for k in R:
                        shutil.copytree(record/'judge'/f'fold-{k}'/name,
                            archive/'R-predictions'/f'fold-{k}'/name,dirs_exist_ok=True)
                        shutil.copytree(record/'R-native'/f'fold-{k}',
                            archive/'R-native'/f'fold-{k}',dirs_exist_ok=True)
                (workspace/'parent.py').write_text(export_program(bank,incumbent))
            status('interactive_agent',probe=i,incumbent=incumbent)
            return {'status':feedback['status'],'probe':i,'parent':parent,
                'results':{k:{'SQL':v['SQL'],'folds':[{'offset':r['offset'],'metrics':r['metrics'],'numerical_guard':r.get('numerical_guard',{})} for r in v['folds']]}
                    for k,v in feedback.get('results',{}).items()},
                'incumbent':incumbent,'incumbent_SQL':scores[incumbent],
                'archive':f'archive/probe-{i:03d}', 'error':feedback.get('traceback')}
        # Rebuild only this run's immutable submitted programs when resuming mid-search.
        for i in range(budget):
            folder=root/'iterations'/f'{tag}-iter-{i:03d}'
            feedback=folder/'R-FEEDBACK.json'
            if feedback.exists():
                old=__import__('json').loads(feedback.read_text())
                if old['status']=='complete':register_probe(None,i,{'parent':old['parent']},restore=True)
                else:history.append(old);previous.append(folder)
        session=record/'agent-session'
        status('interactive_agent',probe_budget=budget,incumbent=incumbent)
        session_receipt=interactive_session(session,packet,history,export_program(bank,incumbent),
            previous,research_directory,budget,register_probe,args.model,args.effort,
            timeout=budget*args.agent_timeout,vertex_project=getattr(args,'vertex_project',None),
            vertex_region=getattr(args,'vertex_region','global'))
        receipt=session/'AGENT-RESULT.json'
        if receipt.exists():
            real_calls=int(bool(__import__('json').loads(receipt.read_text()).get('observed_models')))
    for i in range(0 if interactive else budget):
        folder=root/'iterations'/f'{tag}-iter-{i:03d}';folder.mkdir(parents=True,exist_ok=True)
        independent=(i%3==1)
        parent='raw' if independent else incumbent
        packet['search_direction']=('Independent branch from Raw: investigate a different mechanism using R data.'
            if independent else 'Improve the parent mechanism using actual R diagnostics, or remove an ineffective component.')
        status('agent',iteration=i,parent=parent)
        try:
            receipt_path=folder/'AGENT-RESULT.json'
            if receipt_path.exists():
                receipt=json.loads(receipt_path.read_text())
                if (receipt.get('exit_code') and not receipt.get('usable_budget_limited_code')) or receipt.get('timeout') or not receipt.get('observed_models'):
                    raise RuntimeError('Prior failed proposal retained; no duplicate API request on resume')
                candidate=folder/'candidate.py'
                if not candidate.exists():
                    candidate=folder/'workspace/candidate.py'
                code=candidate.read_text()
                # A crash may have happened between receiving the reply and archiving it.
                if not (folder/'code').exists() and (folder/'workspace').exists():
                    import shutil
                    for source in (folder/'workspace').rglob('*.py'):
                        relative=source.relative_to(folder/'workspace')
                        if relative.parts[0]=='history' or '__pycache__' in relative.parts:continue
                        target=folder/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(source,target)
            else:
                code,receipt=propose(folder,packet,history,export_program(bank,parent),previous,
                    model=args.model,effort=args.effort,timeout=args.agent_timeout,
                    research_directory=research_directory,vertex_project=getattr(args,'vertex_project',None),
                    vertex_region=getattr(args,'vertex_region','global'))
            bundle=folder/'code'
            files={str(p.relative_to(bundle)):p.read_text() for p in bundle.rglob('*.py')} if bundle.exists() else {}
            real_calls+=1;name=f'{tag}_candidate_{i:03d}';bank.add(name,code,parent,files);names=[name]
            for weight in getattr(args,'parent_mixtures',(.25,.5)):
                blend_name=f'{name}_mix{int(weight*100)}'
                bank.add_blend(blend_name,[parent,name],[1-weight,weight]);names.append(blend_name)
            results={}
            for key in names:
                result=evaluate(key,R);scores[key]=result['SQL'];results[key]=result
            incumbent=min(scores,key=scores.get)
            feedback={'epoch':epoch,'iteration':i,'parent':parent,'results':results,
                'research_incumbent':incumbent,'incumbent_SQL':scores[incumbent],'status':'complete'}
        except Exception:
            feedback={'epoch':epoch,'iteration':i,'parent':parent,'status':'failed','traceback':traceback.format_exc()}
        write(folder/'R-FEEDBACK.json',feedback);history.append(feedback);previous.append(folder)
        write(root/'PROGRAM-BANK.json',bank.records)
    if getattr(args,'top3',False):
        members=top3_members(bank,scores,caches,R,initial_incumbent)
        if len(members)==3:
            name=f'{tag}_top3';bank.add_blend(name,members,[1.,1.,1.])
            scores[name]=evaluate(name,R)['SQL']
            write(record/'TOP3.json',{'members':members,'weights':[1/3]*3,
                'R_SQL':scores[name],'selection_inputs':'R only; distinct exact predictions'})
            write(root/'PROGRAM-BANK.json',bank.records)
    policy=getattr(args,'deployment_policy','global')
    if policy=='adaptive':
        from .selection import fit_routes
        # Freeze R-trained routes before any S evaluation. S judges whole programs.
        route_inputs=dict(research_results);route_scores=dict(scores)
        for top_k in (1,3):
            fitted=fit_routes(route_inputs,route_scores,top_k=top_k)
            name=f'{tag}_route_top{top_k}'
            bank.add_route(name,fitted['routes'],fitted['fallback'])
            write(record/(name+'-FIT.json'),fitted)
            scores[name]=evaluate(name,R)['SQL']
        from .scenario_selection import fit_scenario_policy
        fitted=fit_scenario_policy(views,R,route_inputs,route_scores)
        name=f'{tag}_scenario'
        bank.add_scenario(name,fitted)
        write(record/(name+'-FIT.json'),fitted)
        scores[name]=evaluate(name,R)['SQL']
        write(root/'PROGRAM-BANK.json',bank.records)
    # All evaluated programs may compete: more probes must not silently discard
    # earlier useful mechanisms at a fixed seven-candidate bottleneck.
    shortlist=recovery_shortlist(scores,scoreable_R)
    write(record/'FROZEN-CANDIDATES.json',{'shortlist':shortlist,'R_scores':scores,
        'deployment_policy':policy,'routing_fit_inputs':'R only' if policy=='adaptive' else None})
    selection={};errors=[]
    for name in shortlist:
        try:selection[name]=evaluate(name,S)
        except Exception:errors.append({'name':name,'traceback':traceback.format_exc()})
    objective=getattr(args,'selection_objective','mean-sql')
    selected,selection_objectives=choose_selection(selection,objective)
    result={'epoch':epoch,'selected':selected,'scores':selection,'errors':errors,'real_calls':real_calls}
    result.update(selection_objective=objective,selection_objectives=selection_objectives,
        selection_offsets=S,selection_evidence='Complete programs evaluated on the same historical S windows')
    global_candidates={n:r for n,r in selection.items() if bank.records[n]['kind'] not in ('route','scenario')}
    global_selected=choose_selection(global_candidates,objective)[0] if global_candidates else initial_incumbent
    result.update(global_selected=global_selected,
        global_selection_SQL=global_candidates[global_selected]['SQL'] if global_selected in global_candidates else None,
        global_selection_note='Same candidates and S evidence; exclude explicit route/scenario nodes. No official labels.')
    result['record_tag']=tag
    if interactive:
        result.update(research_transport='interactive_probe',agent_session=session_receipt,
            real_calls_unit='agent sessions, not probe requests',
            probe_requests=sum(r.get('epoch')==epoch for r in history))
    write(record/'SELECTION.json',result);write(root/'SELECTION.json',result)
    (record/'selected_program.py').write_text(export_program(bank,selected))
    return selected,result

"""Opt-in balanced follow-up research; no task-ID forecasting rules."""
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time

POLICY='178-balanced-v1'
def read(p, default=None):
    try:return json.loads(Path(p).read_text())
    except FileNotFoundError:return default
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)
def birth(name):
    m=re.fullmatch(r'e(\d+)_candidate_(\d+)',name)
    return (int(m[1]),int(m[2])) if m else None
def valid(v):return isinstance(v,(int,float)) and math.isfinite(v) and v>=0
def code_scores(scores):
    return {n:v for n,v in scores.items() if valid(v) and (birth(n) is not None or n=='raw' or n.startswith('native_'))}
def prior_gains(root,epoch):
    out=[]
    for path in sorted((Path(root)/'epochs').glob('e[0-9][0-9][0-9]/SELECTION.json')):
        e=int(path.parent.name[1:])
        if not 0<e<epoch:continue
        row=read(path,{})
        scores=code_scores({n:v.get('SQL') for n,v in row.get('scores',{}).items()})
        old=[v for n,v in scores.items() if birth(n) is None or birth(n)[0]<e]
        if scores and old and min(old)>0:
            out.append(dict(epoch=e,gain_pct=100*(1-min(scores.values())/min(old))))
    return out

class Gate:
    def __init__(self,flat,ceiling=4):
        self.flat=flat;self.cap=min(2,ceiling) if flat else ceiling
        self.ceiling=ceiling;self.marginal_R_gain_pct=None;self.missing_R=False
    def after_second(self,before,second):
        if self.flat:return self.cap
        if not valid(before) or before<=0 or not valid(second):
            self.missing_R=True;self.cap=self.ceiling
        else:
            self.marginal_R_gain_pct=100*(1-min(before,second)/before)
            self.cap=self.ceiling if self.marginal_R_gain_pct>=.1 else min(2,self.ceiling)
        return self.cap

def process_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

def install():
    # Imported only by the main worker; EDA Python processes must not mutate agents.
    from timersi_u1 import interactive
    if getattr(interactive,'_balanced_installed',False):return
    interactive._balanced_installed=True
    original=interactive.interactive_session
    base_service=interactive.ProbeService
    root=Path(os.environ['TIMERSI_BALANCED_ROOT'])
    write(root/'BALANCED-INSTALL.json',dict(policy=POLICY,pid=os.getpid(),at=time.time(),hook=__file__))
    def session(folder,packet,history,parent_code,previous,research_directory,budget,callback,
                model,effort,timeout,vertex_project=None,vertex_region='global'):
        folder=Path(folder);epoch=packet.get('epoch',0)
        # Existing sessions are restored through their original frozen probe receipts.
        if epoch==0 or (folder/'AGENT-PID.json').exists() or budget<=2:
            return original(folder,packet,history,parent_code,previous,research_directory,budget,callback,
                model,effort,timeout,vertex_project,vertex_region)
        gains=prior_gains(root,epoch)
        flat=len(gains)>=2 and all(x['gain_pct']<.1 for x in gains[-2:])
        gate=Gate(flat,budget)
        # This exact frozen executor exposes its current R scores in the callback.
        r_scores=inspect.getclosurevars(callback).nonlocals.get('scores')
        if not isinstance(r_scores,dict):raise RuntimeError('Balanced hook requires frozen R-score callback')
        state=dict(policy=POLICY,epoch=epoch,at=time.time(),flat2=flat,prior_executed_gains=gains[-2:],
            maximum=budget,cap=gate.cap,probes_completed=0,grace_seconds=30)
        write(folder/'BALANCED-BUDGET.json',state)
        note=('Adaptive budget: initial research is unchanged. This follow-up starts with two probes. '
              'After the second probe, the executor decides whether probes 3 and 4 are allowed. '
              'Use the budget_policy response; when remaining_probes is zero, stop research and '
              'finish your summary within 30 seconds. Completed probes are already archived. '
              'Do not continue EDA or issue more probes after budget closure.')
        packet={**packet,'adaptive_budget_policy':dict(name=POLICY,instruction=note,
                  initially_allowed_probes=2,maximum_probes=gate.cap)}
        def wrap_callback(workspace,i,request):
            before_scores=code_scores(r_scores)
            before=min(before_scores.values()) if before_scores else None
            response=callback(workspace,i,request)
            if i==1:
                values=[v.get('SQL') for n,v in response.get('results',{}).items() if birth(n)==(epoch,1)]
                second=next((v for v in values if valid(v)),None)
                gate.after_second(before,second)
            state.update(cap=gate.cap,probes_completed=i+1,marginal_R_gain_pct=gate.marginal_R_gain_pct,
                missing_R=gate.missing_R,at=time.time())
            write(folder/'BALANCED-BUDGET.json',state)
            return {**response,'budget_policy':dict(name=POLICY,cap=gate.cap,
                remaining_probes=max(0,gate.cap-i-1),instruction=note)}
        class Service(base_service):
            def __enter__(self):
                self.close_after=None;self.agent_identity=None
                obj=super().__enter__()
                self.guard=threading.Thread(target=self.watch,name='balanced-budget-stop',daemon=True)
                self.guard.start();return obj
            def __exit__(self,*args):
                super().__exit__(*args);self.guard.join(timeout=2)
            def serve(self):
                while not self.stop.is_set():
                    for request in sorted(self.queue.glob('*.request.json'),key=lambda p:p.stat().st_mtime_ns):
                        answer=request.with_name(request.name.replace('.request.json','.response.json'))
                        if answer.exists():continue
                        if self.count>=gate.cap:
                            result=dict(status='budget_exhausted',completed_probes=self.count,instruction=note)
                        else:
                            i=self.count;self.count+=1
                            try:result=self.callback(self.workspace,i,read(request))
                            except Exception:
                                import traceback
                                result=dict(status='failed',traceback=traceback.format_exc())
                            # On unknown R after probe failure retain the full budget.
                            if i==1 and state['probes_completed']<2:gate.after_second(None,None)
                        write(answer,result)
                        if self.count>=gate.cap and self.close_after is None:
                            self.close_after=time.monotonic()+30
                            write(folder/'BUDGET-CLOSED.json',dict(policy=POLICY,at=time.time(),cap=gate.cap,
                                completed_probes=self.count,grace_seconds=30,candidate_callback_finished=True))
                    self.stop.wait(.1)
            def watch(self):
                while not self.stop.wait(.25):
                    if self.agent_identity is None:
                        pidinfo=read(folder/'AGENT-PID.json',{})
                        if pidinfo:
                            pid=pidinfo['pid'];start=process_start(pid)
                            if start:self.agent_identity=(pid,start)
                    if self.close_after is None or time.monotonic()<self.close_after or not self.agent_identity:continue
                    pid,start=self.agent_identity
                    if process_start(pid)!=start:return
                    # Agent subprocess was explicitly launched with start_new_session=True.
                    if os.getpgid(pid)!=pid:return
                    write(folder/'BUDGET-STOP.json',dict(policy=POLICY,at=time.time(),pid=pid,
                        process_start=start,reason='completed probe budget, 30-second finalization elapsed'))
                    try:os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError:return
                    if not self.stop.wait(10) and process_start(pid)==start:
                        try:os.killpg(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    return
        interactive.ProbeService=Service
        try:
            result=original(folder,packet,history,parent_code,previous,research_directory,
                gate.cap,wrap_callback,model,effort,timeout*(gate.cap/budget),vertex_project,vertex_region)
        finally:interactive.ProbeService=base_service
        result['adaptive_budget_policy']=POLICY
        result['effective_probe_cap']=gate.cap
        result['controlled_budget_stop']=(folder/'BUDGET-STOP.json').exists()
        write(folder/'SESSION.json',result)
        return result
    interactive.interactive_session=session

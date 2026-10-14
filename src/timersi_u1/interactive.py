"""A persistent code-agent session with real, archived executor feedback."""
import json
from pathlib import Path
import shutil
import threading
import time
import traceback
from .agent import GUIDE,propose
from .core import write
from .research_design import executable_root,archive_analysis


CLIENT = '''import argparse,json,time,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--parent',default=None)
a=p.parse_args();root=Path(__file__).resolve().parent/'.probe-queue'
key=uuid.uuid4().hex;request=root/(key+'.request.json');tmp=request.with_suffix('.tmp')
tmp.write_text(json.dumps({'parent':a.parent}));tmp.replace(request)
answer=root/(key+'.response.json')
while not answer.exists():time.sleep(.25)
print(answer.read_text())
'''


def session_guide(budget):
    text=GUIDE.replace('The executor runs your proposal after submission and returns R feedback to the\nnext iteration; model probes are not available directly inside this session.',
        'You have a live executor: run the provided Python executable with probe.py after\nwriting candidate.py. It returns real native R metrics in THIS conversation. Read\nthe feedback, inspect the public archive, revise the program and probe again.')
    text=text.replace('Submit native changes for real executor evaluation.',
        'Use probe.py to evaluate native changes immediately through the real executor.')
    return text+f'''\n## Persistent research protocol
You have at most {budget} real probes within the wall limit in INPUT.json (failures
count). Use the live executor to validate useful hypotheses before ending the
session. Run `<python_executable> probe.py` and wait for each response. Investigate
data and errors as deeply as needed; there is no first-candidate tool-call deadline.
The default parent is the current R incumbent; --parent raw starts a fresh branch.
You choose the research path: deeper work on a supported mechanism, a distinct
branch, a restored program or a data-dependent combination. There is no mandatory
sequence of feature engineering, transformation, model fitting or correction.
Read matured_deployment_feedback before repeating an earlier program: a historical
research winner may have failed when deployed. Explain what new evidence or change
addresses that failure. Never fit rules to task names or a single anomalous fold.
Each probe freezes all your Python helpers, tests the full program and 25%/50% parent
mixtures, then publishes archive/probe-NNN/R-FEEDBACK.json and code/. Those copies
are read-only research history. All earlier public history/ and every archive/
probe remain available; restore by copying their code into your workspace and
probe again. Prior artifacts do not count as new evaluated results without a probe.
Feedback is R only; S is evaluated after this session has ended. Do not read the
private executor or other task directories. No direct GPU calls are needed.
Use per-fold errors to improve mechanisms, not to fit arbitrary task-name rules.
Leave the best code on disk, but final selection considers ALL completed probes.
The API.md text about unavailable native CPU fixtures still applies to CachedRaw;
probe.py, unlike CachedRaw, invokes real native inference via the outer executor.
'''


class ProbeService:
    """Serial file RPC; the outer executor owns native models and hidden S data."""
    def __init__(self,workspace,budget,callback):
        self.workspace=Path(workspace);self.budget=budget;self.callback=callback
        self.queue=self.workspace/'.probe-queue';self.stop=threading.Event();self.count=0

    def __enter__(self):
        self.queue.mkdir(exist_ok=True)
        (self.workspace/'probe.py').write_text(CLIENT)
        self.thread=threading.Thread(target=self.serve,name='research-probe',daemon=True)
        self.thread.start();return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join()

    def serve(self):
        while not self.stop.is_set():
            for request in sorted(self.queue.glob('*.request.json'),key=lambda p:p.stat().st_mtime_ns):
                answer=request.with_name(request.name.replace('.request.json','.response.json'))
                if answer.exists():continue
                from .isolation import frozen_workspace
                with frozen_workspace(self.workspace):
                    if self.count>=self.budget:
                        result={'status':'budget_exhausted','completed_probes':self.count}
                    else:
                        number=self.count;self.count+=1
                        try:result=self.callback(self.workspace,number,json.loads(request.read_text()))
                        except Exception:result={'status':'failed','traceback':traceback.format_exc()}
                    write(answer,result)
            self.stop.wait(.1)


def freeze_program(workspace,folder):
    """Freeze executable research code, excluding prior archives and the RPC client."""
    workspace=Path(workspace);folder=Path(folder);files={}
    runtime=executable_root(workspace)
    for source in runtime.rglob('*.py'):
        relative=source.relative_to(runtime)
        if relative.parts[0] in ('archive','history','.probe-queue','research-data','research-history','eda','notes') or '__pycache__' in relative.parts:continue
        if str(relative)=='probe.py':continue
        target=folder/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);files[str(relative)]=source.read_text()
    code=files['candidate.py'];compile(code,'candidate.py','exec')
    (folder/'candidate.py').write_text(code)
    for name in ('HYPOTHESIS.md','REFLECTION.md'):
        if (workspace/name).exists():shutil.copy2(workspace/name,folder/name)
    archive_analysis(workspace,folder)
    return code,files


def interactive_session(folder,packet,history,parent_code,previous,research_directory,
                        budget,callback,model,effort,timeout,vertex_project=None,vertex_region='global'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # A stopped session is not silently replayed. search() reconstructs its frozen probes.
    if (folder/'SESSION.json').exists():return json.loads((folder/'SESSION.json').read_text())
    if (folder/'AGENT-PID.json').exists():
        receipt={'resumed_without_new_session':True,'reason':'prior session attempt retained'}
        write(folder/'SESSION.json',receipt);return receipt
    services=[]
    def service(workspace):
        obj=ProbeService(workspace,budget,callback);services.append(obj);return obj
    start=time.monotonic();error=None
    try:
        propose(folder,{**packet,'probe_budget':budget,'session_wall_limit_seconds':timeout},
            history,parent_code,previous,model=model,effort=effort,timeout=timeout,
            research_directory=research_directory,guide=session_guide(budget),
            max_turns=256,workspace_service=service,
            vertex_project=vertex_project,vertex_region=vertex_region)
    except Exception:error=traceback.format_exc()
    receipt={'seconds':time.monotonic()-start,'probe_requests':services[0].count if services else 0,
        'agent_error':error,'agent_receipt':'AGENT-RESULT.json',
        'completed_probes_retained_on_session_error':True}
    write(folder/'SESSION.json',receipt);return receipt

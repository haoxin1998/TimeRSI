"""Docker isolation with narrow, credential-free Unix RPC interfaces."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import uuid
import numpy as np
from .core import Forecast,write
from .isolation_wire import pack,unpack

class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads=True

class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            request=unpack(json.loads(self.rfile.readline()))
            answer={'result':pack(self.server.callback(request))}
        except Exception as exc:answer={'error':f'{type(exc).__name__}: {exc}'}
        self.wfile.write(json.dumps(answer).encode()+b'\n')

@contextmanager
def broker(callback, name):
    with tempfile.TemporaryDirectory(prefix='timersi-broker-') as directory:
        folder=Path(directory);server=Server(str(folder/name),Handler);server.callback=callback
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:yield folder
        finally:server.shutdown();server.server_close();thread.join()

def docker_command(workspace, broker_dir=None, name=None):
    workspace=Path(workspace).resolve()
    package=Path(__file__).resolve().parents[1]
    command=['docker','run','--rm','--init','--name',name or 'timersi-'+uuid.uuid4().hex,
        '--network','none','--read-only','--cap-drop=ALL','--security-opt','no-new-privileges',
        '--user',f'{os.getuid()}:{os.getgid()}','--pids-limit','256','--cpus','8','--memory','16g',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=2g','--env','HOME=/tmp/home','--env','PYTHONDONTWRITEBYTECODE=1',
        '--mount',f'type=bind,src={workspace},dst={workspace}',
        '--mount',f'type=bind,src={package},dst=/opt/timersi/src,readonly',
        '--workdir',str(workspace)]
    if broker_dir:command+=['--mount',f'type=bind,src={broker_dir},dst=/broker,readonly']
    return command

def image():return os.environ.get('TIMERSI_ISOLATION_IMAGE','timersi-isolation:1.1')

def validate_workspace(workspace):
    """Host-side RPC must not follow links created by container code."""
    root=Path(workspace)
    for directory,folders,files in os.walk(root,followlinks=False):
        for name in folders+files:
            if (Path(directory)/name).is_symlink():
                raise ValueError('Workspace links are not accepted by the evaluator')

@contextmanager
def frozen_workspace(workspace):
    """Stop writers while the host snapshots a probe and publishes its response."""
    receipt=Path(workspace).parent/'ISOLATION.json'
    name=json.loads(receipt.read_text())['container_name'] if receipt.exists() else None
    paused=False
    if name:
        state=subprocess.run(['docker','pause',name],capture_output=True)
        paused=state.returncode==0
        if not paused:
            # The session may already have exited. Never read while it is running.
            alive=subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True)
            if alive.returncode==0 and alive.stdout.strip()=='true':raise RuntimeError('Unable to freeze probe workspace')
    try:
        validate_workspace(workspace)
        yield
    finally:
        if paused:subprocess.run(['docker','unpause',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def vertex_callback(model, project, region):
    # Credentials stay on the host. There is no general URL, file or shell RPC.
    import requests
    token=[None,0.]
    base=model.removesuffix('[1m]')
    hostname='aiplatform.googleapis.com' if region=='global' else f'{region}-aiplatform.googleapis.com'
    url=f'https://{hostname}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{base}:streamRawPredict'
    def invoke(request):
        if time.monotonic()-token[1]>2400 or not token[0]:
            token[0]=subprocess.check_output(['gcloud','auth','application-default','print-access-token'],text=True,stderr=subprocess.DEVNULL).strip()
            token[1]=time.monotonic()
        payload=dict(request['body']);requested=payload.pop('model','').removesuffix('[1m]')
        if requested!=base:raise ValueError('The broker only serves the configured model')
        payload['anthropic_version']='vertex-2023-10-16'
        headers={'Authorization':'Bearer '+token[0],'Content-Type':'application/json'}
        headers.update({k:v for k,v in request.get('headers',{}).items() if k.lower()=='anthropic-beta'})
        response=requests.post(url,headers=headers,json=payload,timeout=1500)
        return {'status':response.status_code,'content_type':response.headers.get('Content-Type','application/json'),'body':response.text}
    return invoke

@contextmanager
def agent_launch(command, workspace, model, effort, project, region):
    if not project:raise ValueError('Set TIMERSI_VERTEX_PROJECT for the isolated provider broker')
    executable=Path(os.environ.get('CLAUDE_CODE_BINARY') or shutil.which('claude')).resolve()
    name='timersi-agent-'+uuid.uuid4().hex
    with broker(vertex_callback(model,project,region),'messages.sock') as endpoint:
        cli=['/opt/claude','--bare','--print','--model',model,'--effort',effort,
             '--output-format','stream-json','--verbose','--max-turns',command[command.index('--max-turns')+1],
             '--permission-mode','acceptEdits','--tools','Read,Write,Edit,Bash',
             '--allowedTools','Read','Write','Edit','Bash','--setting-sources','','--',command[-1]]
        docker=docker_command(workspace,endpoint,name)+['--mount',f'type=bind,src={executable},dst=/opt/claude,readonly',
            image(),'python','-m','timersi_u1.isolated_agent',*cli]
        write(Path(workspace).parent/'ISOLATION.json',dict(filesystem='task workspace only plus read-only runtime',
            network='none; messages-only Unix broker',generated_code='separate networkless containers',
            current_future_targets_mounted=False,container_name=name))
        try:yield docker
        finally:
            subprocess.run(['docker','stop','--time','2',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            validate_workspace(workspace)

def predict(spec,view,tools,memory):
    root=Path(os.environ['TIMERSI_E185_ROOT'])/'isolated-candidates';root.mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='candidate-',dir=root)).resolve()
    write(folder/'program.json',spec)
    with (folder/'context.pkl').open('wb') as stream:pickle.dump(dict(view=view,memory=memory),stream,protocol=5)
    allowed={'contexts','horizon','past_only_covariates','past_future_covariates','return_quantiles',
             'use_symmetric_averaging','make_positive','sort_quantiles'}
    def forecast(request):
        if set(request)-allowed:raise ValueError('Unexpected native forecast parameter')
        outputs=tools._model().predict_batch(**request)
        return [(r.forecast,r.quantiles) for r in outputs]
    with broker(forecast,'native.sock') as endpoint:
        command=docker_command(folder,endpoint)+[image(),'python','-m','timersi_u1.isolated_candidate',str(folder)]
        with (folder/'execution.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=log,check=True,timeout=1800)
    validate_workspace(folder)
    with np.load(folder/'forecast.npz',allow_pickle=False) as saved:
        result=Forecast(saved['point'],saved['quantiles'])
    record=json.loads((folder/'native-records.json').read_text())
    tools.calls+=record['calls'];tools.hits+=record['hits']
    if tools.native_records is not None:tools.native_records.extend(record['records'])
    return result

"""Private local GPU service: share resident models, never share prediction caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
from multiprocessing.connection import Client,Listener
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace


def serve(pool,gpu,checkpoint):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    from timesfm3 import TimesFM3Evaluator,ModelConfig
    torch.set_num_threads(2)
    model=TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint,per_core_batch_size=16,
        use_variate_attention=True,use_sdpa=True,device='cuda:0'))
    config=json.loads((pool/'CONFIG.json').read_text())
    endpoint=config['endpoints'][str(gpu)]
    with Listener(endpoint,family='AF_UNIX') as listener:
        (pool/f'gpu-{gpu}.ready.json').write_text(json.dumps({'pid':os.getpid(),'gpu':gpu,'at':time.time()}))
        while True:
            with listener.accept() as connection:
                request=connection.recv()
                if request.get('shutdown'):
                    connection.send({'stopped':gpu});return
                started=time.monotonic()
                try:
                    outputs=[(out.forecast,out.quantiles) for out in model.predict_batch(**request['kwargs'])]
                    torch.cuda.synchronize()
                    result={'outputs':outputs,'gpu':gpu,'seconds':time.monotonic()-started}
                except Exception:
                    result={'error':traceback.format_exc(),'gpu':gpu,'seconds':time.monotonic()-started}
                with (pool/f'gpu-{gpu}.jsonl').open('a') as log:
                    log.write(json.dumps({k:v for k,v in result.items() if k!='outputs'}|{
                        'at':time.time(),'caller':request.get('caller'),'batch':request.get('batch'),
                        'items':len(request['kwargs']['contexts'])})+'\n')
                try:connection.send(result)
                except (BrokenPipeError,EOFError,ConnectionResetError):pass


class PoolBackend:
    def __init__(self,pool,checkpoint,caller):
        self.pool=Path(pool);self.config=json.loads((self.pool/'CONFIG.json').read_text())
        if str(Path(checkpoint).resolve())!=self.config['checkpoint']:raise ValueError('Pool checkpoint mismatch')
        self.caller=caller;self.request_count=0

    def request(self,batch,kwargs):
        gpus=self.config['gpus'];start=(os.getpid()+batch)%len(gpus)
        order=gpus[start:]+gpus[:start];deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            for gpu in order:
                with (self.pool/f'gpu-{gpu}.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    ready=self.pool/f'gpu-{gpu}.ready.json'
                    if not ready.exists():continue
                    pid=json.loads(ready.read_text())['pid'];proc=Path('/proc')/str(pid)
                    try:state=(proc/'stat').read_text().split(')')[-1].split()[0]
                    except FileNotFoundError:raise RuntimeError(f'Pool GPU {gpu} process exited')
                    if state=='Z':raise RuntimeError(f'Pool GPU {gpu} process is terminal')
                    with Client(self.config['endpoints'][str(gpu)],family='AF_UNIX') as connection:
                        connection.send({'kwargs':kwargs,'caller':self.caller,'batch':batch})
                        if not connection.poll(1800):raise TimeoutError(f'GPU {gpu} batch timeout')
                        result=connection.recv()
                    if 'error' in result:raise RuntimeError(result['error'])
                    return [SimpleNamespace(forecast=p,quantiles=q) for p,q in result['outputs']]
            time.sleep(.025)
        raise TimeoutError('No ready GPU service available')

    def predict_batch(self,**kwargs):
        self.request_count+=1
        return self.request(self.request_count,kwargs)

    def predict_many(self,batches):
        # At most one in-flight request per pool GPU from this task. Inputs are
        # complete multivariate entities; channel attention is never split apart.
        with ThreadPoolExecutor(max_workers=len(self.config['gpus'])) as workers:
            limit=len(self.config['gpus']);inputs=iter(enumerate(batches))
            pending={};ready={};next_output=0;exhausted=False
            def fill_slots():
                nonlocal exhausted
                while not exhausted and len(pending)<limit and len(pending)+len(ready)<2*limit:
                    try:index,kwargs=next(inputs)
                    except StopIteration:exhausted=True;break
                    pending[workers.submit(self.request,index,kwargs)]=index
            fill_slots()
            while pending or ready:
                if pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for job in done:ready[pending.pop(job)]=job.result()
                fill_slots()
                while next_output in ready:
                    yield ready.pop(next_output);next_output+=1
                    fill_slots()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--checkpoint',required=True)
    a=p.parse_args();serve(a.pool,a.gpu,a.checkpoint)

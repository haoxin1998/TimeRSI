"""A deployment agent can request real native forecasts inside its own session."""
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .core import write
from .data import make_view, score, official_sql_by_target,EvaluationContext
from .interactive import ProbeService


CLIENT = '''import argparse,json,time,uuid
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--transform',choices=['identity','asinh','flow','asinh-center','calendar-expert'],default='identity')
p.add_argument('--calendar',action='store_true');p.add_argument('--signed',action='store_true')
p.add_argument('--uncertainty',choices=['sum','rss'],default='sum')
p.add_argument('--context',type=int,default=15360)
p.add_argument('--covariate-difference',action='append',default=[])
a=p.parse_args();root=Path(__file__).resolve().parent/'.probe-queue'
key=uuid.uuid4().hex;request=root/(key+'.request.json');tmp=request.with_suffix('.tmp')
tmp.write_text(json.dumps(vars(a)));tmp.replace(request);answer=root/(key+'.response.json')
while not answer.exists():time.sleep(.25)
print(answer.read_text())
'''


def requested_program(request):
    transform=request.get('transform','identity')
    if transform not in ('identity','asinh','flow','asinh-center','calendar-expert'):
        raise ValueError('Unknown native representation')
    context=int(request.get('context',15360))
    if context<2:raise ValueError('Native context needs at least two observations')
    signed=bool(request.get('signed',False));calendar=bool(request.get('calendar',False))
    feature='tools.calendar' if calendar else 'None'
    if transform=='flow':
        uncertainty=request.get('uncertainty','sum')
        if uncertainty not in ('sum','rss'):raise ValueError('Unknown uncertainty propagation')
        differences=list(request.get('covariate_difference',[]))
        if not all(isinstance(name,str) for name in differences):raise ValueError('Named numeric covariates required')
        expression=f'tools.native_flow(view, uncertainty={uncertainty!r}, signed={signed!r}, context={context}, features={feature}, covariate_differences={differences!r})'
    elif transform=='asinh-center':expression='tools.asinh_center(view)'
    elif transform=='calendar-expert':expression='tools.calendar_expert(view)'
    else:expression=f'tools.timesfm(view, transform={transform!r}, signed={signed!r}, context={context}, features={feature})'
    return 'def predict(view, tools, memory):\n    return '+expression+'\n'


class NativeRequestService(ProbeService):
    def __enter__(self):
        super().__enter__()
        (self.workspace/'native_forecast.py').write_text(CLIENT)
        return self


class NativeRequests:
    """Only past warm windows are scored; the current view has no outcomes."""
    def __init__(self,folder,view,bank,tools,visible,cache,forecasts,selection,warm,task,memory,publish):
        self.folder=Path(folder);self.view=view;self.bank=bank;self.tools=tools
        self.visible=visible;self.cache=cache;self.forecasts=forecasts;self.selection=selection
        self.warm=warm;self.task=task;self.memory=memory;self.publish=publish;self.records=[]
        self._past_templates={};self._evaluations={}

    def past_evidence(self,name,output):
        history=[]
        for offset,window in self.warm.items():
            if offset not in self._past_templates:
                self._past_templates[offset]=make_view(window,self.task)
                self._evaluations[offset]=EvaluationContext(window,self.task)
            past=deepcopy(self._past_templates[offset])
            visible=self.memory.visible(past) if self.memory is not None else []
            result=self.bank.predict(name,past,self.tools,visible,{})
            evaluation=self._evaluations[offset].prepare(result)
            history.append(dict(offset=offset,metrics=evaluation.score(),
                official_SQL_by_target=evaluation.official_sql_by_target(),
                targets=list(self.task.target_columns),
                forecast_ends=[str(item['timestamps'][-1]) for item in past.items]))
            np.savez_compressed(output/f'S-{offset}.npz',item_ids=[i['id'] for i in past.items],
                point=result.point,quantiles=result.quantiles)
        return history

    def __call__(self,workspace,number,request):
        code=requested_program(request)
        name='native_'+hashlib.sha256(code.encode()).hexdigest()[:16]
        if name not in self.bank.records:
            self.bank.add(name,code)
            self.bank.records[name]['origin']='deployment_agent_native_request'
        output=self.folder/'native-requests'/f'request-{number:03d}'
        output.mkdir(parents=True,exist_ok=True);write(output/'REQUEST.json',request)
        (output/'candidate.py').write_text(code)
        with self.tools.record_native_calls() as calls:
            forecast=self.bank.predict(name,self.view,self.tools,self.visible,self.cache)
            history=self.past_evidence(name,output)
        self.forecasts[name]=forecast;self.cache[name]=forecast
        sql=float(np.mean([row['metrics']['SQL'] for row in history])) if history else None
        self.selection[name]={'SQL':sql}
        np.savez_compressed(output/'CURRENT-PREDICTIONS.npz',item_ids=[i['id'] for i in self.view.items],
            point=forecast.point,quantiles=forecast.quantiles)
        record=dict(status='completed',candidate=name,request=request,past_selection=history,
            historical_S_SQL=sql,native_inputs=calls,current_labels_used=False)
        self.records.append(record);write(output/'RESULT.json',record)
        path=Path(workspace)/'candidate-code'/name/'candidate.py';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(code)
        self.publish(workspace,self.records)
        return {**record,'updated_input':'INPUT.json',
            'archive':str(output.resolve()),'note':'Current outcomes unavailable. Continue comparing actual curves and past errors, then write the final decision.'}

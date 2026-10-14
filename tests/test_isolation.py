import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import numpy as np
from types import SimpleNamespace
from timersi_u1.data import View
from timersi_u1.tools import Tools
from timersi_u1.isolation import docker_command,image,predict,frozen_workspace

def test_filesystem_network_and_candidate():
    with tempfile.TemporaryDirectory(prefix='timersi-isolation-test-') as tmp:
        root=Path(tmp);private=root/'evaluator-secret';private.write_text('unreleased labels')
        workspace=root/'workspace';workspace.mkdir()
        command=docker_command(workspace)+[image(),'python','-c',
            'from pathlib import Path; import socket,json; '
            f'assert not Path({str(private)!r}).exists(); '
            's=socket.socket();s.settimeout(.5);code=s.connect_ex(("1.1.1.1",443)); '
            'assert code!=0; print(json.dumps({"private_file_hidden":True,"external_network_blocked":True}))']
        receipt=json.loads(subprocess.check_output(command,text=True))
        os.environ['TIMERSI_E185_ROOT']=str(workspace)
        class Backend:
            def predict_batch(self,**kw):
                return [SimpleNamespace(forecast=np.array([[5.,6.]]),quantiles=np.tile(np.array([5.,6.])[:,None],(1,9))[None])]
        tools=Tools('fake','cpu',workspace/'native',backend=Backend())
        dates=np.arange('2020-01-01','2020-01-06',dtype='datetime64[D]')
        view=View([dict(id='a',y=np.array([[1.,2.,3.]]),timestamps=dates,known={},past={},observed=np.ones((1,3),bool))],2,1,['y'],'D')
        code=f'''def predict(view,tools,memory):
    from pathlib import Path
    assert not Path({str(private)!r}).exists()
    raw=tools.timesfm(view)
    return tools.shift(raw,1.)
'''
        forecast=predict(dict(code=code),view,tools,[])
        np.testing.assert_array_equal(forecast.point.ravel(),[6.,7.])
        # Exercise the real host probe boundary, including writer suspension.
        name='timersi-probe-test-'+uuid.uuid4().hex
        (root/'ISOLATION.json').write_text(json.dumps(dict(container_name=name)))
        process=subprocess.Popen(docker_command(workspace,name=name)+[image(),'python','-c',
            'from pathlib import Path;import time;Path("ready").touch();time.sleep(30)'],stdout=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+20
            while not (workspace/'ready').exists():
                assert time.monotonic()<deadline
                time.sleep(.1)
            with frozen_workspace(workspace):
                assert subprocess.check_output(['docker','inspect','--format','{{.State.Paused}}',name],text=True).strip()=='true'
            assert subprocess.check_output(['docker','inspect','--format','{{.State.Paused}}',name],text=True).strip()=='false'
            (workspace/'outside-link').symlink_to(private)
            try:
                with frozen_workspace(workspace):raise AssertionError('Link was accepted')
            except ValueError:pass
            receipt['host_probe_links_rejected']=True
            receipt['host_probe_writer_suspension']=True
        finally:
            subprocess.run(['docker','stop','--time','1',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process.wait()
        return receipt

if __name__=='__main__':print(json.dumps(test_filesystem_network_and_candidate()))

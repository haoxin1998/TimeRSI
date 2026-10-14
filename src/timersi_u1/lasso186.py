"""Exact deterministic Lasso reuse and audited independent-hour execution."""
from concurrent.futures import ThreadPoolExecutor
import copy,fcntl,functools,hashlib,inspect,json,os,pickle,time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.linear_model import Lasso
from .timing import event

_installed=False

def cached(kind,payload,compute):
    start=time.monotonic()
    key=hashlib.sha256(pickle.dumps((sklearn.__version__,np.__version__,kind,payload),protocol=5)).hexdigest()
    key_seconds=time.monotonic()-start
    root=Path(os.environ['TIMERSI_E185_ROOT'])/'cache186'/kind/key[:2]
    root.mkdir(parents=True,exist_ok=True);path=root/(key+'.pkl')
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            result=pickle.loads(path.read_bytes());hit=True;compute_seconds=0.
        else:
            t=time.monotonic();result=compute();compute_seconds=time.monotonic()-t;hit=False
            # 16MiB/shard => <=4GiB/type/task; no eviction of an in-flight entry.
            if compute_seconds>=.02:
                content=pickle.dumps(result,protocol=5)
                if sum(p.stat().st_size for p in root.glob('*.pkl'))+len(content)<=16*1024**2:
                    temp=path.with_suffix(f'.{os.getpid()}.tmp');temp.write_bytes(content);temp.replace(path)
        event('cache186',cache_kind=kind,hit=hit,seconds=time.monotonic()-start,
              compute_seconds=compute_seconds,key_seconds=key_seconds,key=key)
        return result

def install():
    global _installed
    if _installed:return
    _installed=True
    original=Lasso.fit
    @functools.wraps(original)
    def fit(self,*args,**kwargs):
        params=self.get_params(deep=True)
        # copy_X=False can mutate the caller. Stateful/random fits must execute.
        if (type(self) is not Lasso or params.get('warm_start') or not params.get('copy_X',True)
                or params.get('selection','cyclic')!='cyclic'):
            return original(self,*args,**kwargs)
        state=copy.deepcopy(cached('lasso-fit',(params,args,kwargs),lambda:original(self,*args,**kwargs).__dict__))
        self.__dict__.clear();self.__dict__.update(state);return self
    Lasso.fit=fit
    event('lasso186_installed',hour_workers=4,solver_parameters_unchanged=True)

def accelerate_module(module,source):
    """Only full-source reviewed pure experts qualify; unknown source unchanged."""
    if os.environ.get('TIMERSI186_ENABLE','1')=='0':return
    registry=json.loads((Path(__file__).parent/'registry186.json').read_text())
    digest=hashlib.sha256(source.encode()).hexdigest();kind=registry.get(digest)
    if kind is None:return
    if kind=='pjm':
        def block(Xtr,Ytr,Xte,alpha=1e-3,max_iter=3000):
            mu=Xtr.mean(0);sd=Xtr.std(0)+1e-8
            Xtr=(Xtr-mu)/sd;Xte=(Xte-mu)/sd
            def one(h):
                m=Lasso(alpha=alpha,max_iter=max_iter);m.fit(Xtr,Ytr[:,h]);return m.predict(Xte)
            with ThreadPoolExecutor(max_workers=4) as pool:
                return np.column_stack(list(pool.map(one,range(24))))
        module._fit_block=block;name='lear_forecast'
    else:
        def fit(self,P_raw,X1,X2,dow,anchor):
            z,self.med,self.mad=module.asinh_transform(P_raw[:anchor].ravel());P_t=z.reshape(anchor,24)
            lo=max(7,anchor-self.cal_days);days=list(range(lo,anchor))
            F=np.asarray([module._features_for_day(P_t,X1,X2,dow,d) for d in days])
            self.mu=F.mean(axis=0);self.sd=F.std(axis=0);self.sd[self.sd<1e-8]=1.
            F=np.asfortranarray((F-self.mu)/self.sd);gram=np.dot(F.T,F)
            def one(h):
                m=Lasso(alpha=self.alpha,max_iter=2500,precompute=gram,tol=1e-3)
                m.fit(F,P_t[days,h]);return m
            with ThreadPoolExecutor(max_workers=4) as pool:self.models=list(pool.map(one,range(24)))
            return self
        module.LearModel.fit=fit;name='lear_run'
    original=getattr(module,name);signature=inspect.signature(original)
    @functools.wraps(original)
    def expert(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        return cached('lear-expert',(digest,name,bound.arguments),lambda:original(*args,**kwargs))
    setattr(module,name,expert)
    event('expert186_loaded',source_sha256=digest,expert_kind=kind,hour_workers=4)

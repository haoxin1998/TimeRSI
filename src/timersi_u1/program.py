"""Load a frozen Python source bundle without cross-candidate module collisions."""
import builtins
import importlib.util
import itertools
import sys
from types import ModuleType
import numpy as np
from .core import Forecast,program_identity

_LOAD_IDS=itertools.count()


def load_program(code, files=None):
    sources=dict(files or {});sources['candidate.py']=code
    digest=program_identity(code,files)
    # Real Python facilities (e.g. dataclasses with postponed annotations) look
    # up cls.__module__ in sys.modules. Register under a source-content namespace,
    # never shared names such as 'candidate' or 'helpers'. The instance suffix also
    # keeps repeated loads of identical code from replacing a live module's state.
    namespace=f'_timersi_bundle_{digest}_{next(_LOAD_IDS)}'
    root=ModuleType(namespace);root.__path__=[];root.__package__=namespace
    root.__spec__=importlib.util.spec_from_loader(namespace,loader=None,is_package=True)
    sys.modules[namespace]=root
    modules={'':root}

    def local_name(name):
        if name==namespace:return ''
        return name[len(namespace)+1:] if name.startswith(namespace+'.') else name

    def location(name):
        path=name.replace('.', '/')
        if path+'.py' in sources:return path+'.py',False
        if path+'/__init__.py' in sources:return path+'/__init__.py',True
        return None,False

    def load(name):
        if name in modules:return modules[name]
        path,package=location(name)
        if path is None:raise ImportError(f'No bundled module {name}')
        qualified=namespace+'.'+name
        module=ModuleType(qualified);modules[name]=module
        sys.modules[qualified]=module
        module.__file__=path
        module.__package__=qualified if package else qualified.rpartition('.')[0]
        module.__spec__=importlib.util.spec_from_loader(qualified,loader=None,is_package=package)
        if package:module.__path__=[]
        module.__dict__.update(np=np,Forecast=Forecast,
            __builtins__={**vars(builtins),'__import__':bundle_import})
        parent,_,child=name.rpartition('.')
        setattr(load(parent),child,module)
        exec(compile(sources[path],path,'exec'),module.__dict__)
        from .lasso186 import accelerate_module
        accelerate_module(module,sources[path])
        return module

    def bundle_import(name,globals=None,locals=None,fromlist=(),level=0):
        resolved=importlib.util.resolve_name('.'*level+name,globals.get('__package__','')) if level else name
        absolute=local_name(resolved)
        path,_=location(absolute)
        if path is None and absolute not in modules:
            return builtins.__import__(name,globals,locals,fromlist,level)
        module=load(absolute)
        if fromlist:
            for child in fromlist:
                child_name=absolute+'.'+child if absolute else child
                if child!='*' and not hasattr(module,child) and location(child_name)[0]:
                    load(child_name)
            return module
        return load(absolute.split('.')[0])

    try:
        entry=load('candidate').__dict__
        if not callable(entry.get('predict')):raise ValueError('Missing predict(view,tools,memory)')
    except Exception:
        for module in modules.values():sys.modules.pop(module.__name__,None)
        raise
    predict=entry['predict'];identity='program:'+digest
    def scoped_predict(view,tools,memory):
        previous=getattr(tools,'active_program',None)
        tools.active_program=identity
        try:return predict(view,tools,memory)
        finally:tools.active_program=previous
    entry['predict']=scoped_predict
    return entry

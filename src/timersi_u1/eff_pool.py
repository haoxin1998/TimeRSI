import contextvars
import functools
import inspect
import json
import os
from pathlib import Path
import time
from .eff_policy import choose,past_scores
CTX=contextvars.ContextVar('working_pool181',default=None)
def install():
    from timersi_u1 import search
    original=search.search;archive=search.archive_candidates;signature=inspect.signature(original)
    @functools.wraps(original)
    def bounded_search(*args,**kwargs):
        b=signature.bind(*args,**kwargs);b.apply_defaults()
        token=CTX.set(b.arguments)
        try:return original(*args,**kwargs)
        finally:CTX.reset(token)
    def bounded_archive(bank,scores,retained_programs=()):
        old=archive(bank,scores,retained_programs);context=CTX.get()
        if not context or context['epoch']==0:return old
        root=Path(context['root']);epoch=context['epoch'];tag=context['record_tag'] or f'e{epoch:03d}'
        protected=list(scores);universe=list(dict.fromkeys(protected+old))
        known,seen=past_scores(root,epoch);chosen=choose(universe,protected,known,seen,12)
        result=[n for n in chosen if n not in scores]
        receipt=dict(policy='181-old-pool12-rotation1',at=time.time(),epoch=epoch,cap=12,
            old_before=len(universe),old_after=len(chosen),protected=protected,working_pool=chosen,
            archived_not_evaluated=[n for n in universe if n not in chosen],ranking_uses='previous completed S only',
            new_candidates_separate_budget=True,complete_bank_retained=len(bank.records))
        folder=root/'epochs'/tag;folder.mkdir(parents=True,exist_ok=True)
        (folder/'WORKING-POOL181.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return result
    search.search=bounded_search;search.archive_candidates=bounded_archive
    root=Path(os.environ['TIMERSI_POOL181_ROOT'])
    (root/'POOL181-INSTALL.json').write_text(json.dumps(dict(at=time.time(),pid=os.getpid(),cap=12,
        policy='181-old-pool12-rotation1',source=str(Path(__file__).parent)),indent=2)+'\n')

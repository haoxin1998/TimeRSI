"""Prediction-only numerical sanity check. No task IDs or outcome labels."""
import numpy as np

MULTIPLIER = 100.0


def recover(point, quantiles, raw_point, raw_quantiles, histories):
    p,q,rp,rq=map(np.asarray,(point,quantiles,raw_point,raw_quantiles))
    assert p.shape==rp.shape and q.shape==rq.shape and q.shape[:3]==p.shape
    assert len(histories)==len(p)
    if not np.isfinite(rp).all() or not np.isfinite(rq).all():
        raise ValueError('Raw reference contains nonfinite values')
    history_peak=[]
    for y in histories:
        y=np.asarray(y,dtype=np.float64)
        history_peak.append(np.max(np.where(np.isfinite(y),np.abs(y),0.),axis=1))
    scale=np.maximum.reduce([np.asarray(history_peak),np.max(np.abs(rp),axis=1),
        np.max(np.abs(rq),axis=(1,3)),np.full((p.shape[0],p.shape[2]),1e-6)])
    nonfinite=~(np.isfinite(p).all(axis=1)&np.isfinite(q).all(axis=(1,3)))
    peak=np.maximum(np.max(np.abs(np.where(np.isfinite(p),p,0.)),axis=1),
                    np.max(np.abs(np.where(np.isfinite(q),q,0.)),axis=(1,3)))
    with np.errstate(over='ignore',invalid='ignore'):
        ratio=peak/scale
    excessive=ratio>MULTIPLIER
    mask=nonfinite|excessive
    fixed_p=np.where(mask[:,None,:],rp,p)
    fixed_q=np.where(mask[:,None,:,None],rq,q)
    return fixed_p,fixed_q,mask,dict(scale=scale,peak=peak,ratio=ratio,
                                    nonfinite=nonfinite,excessive=excessive)

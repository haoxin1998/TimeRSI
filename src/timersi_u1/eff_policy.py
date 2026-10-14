"""Past-only bounded working set; retain the complete program archive."""
import math
def choose(old,protected,known,last_seen,cap=12):
    old=list(dict.fromkeys(old));chosen=[n for n in dict.fromkeys(protected) if n in old]
    if len(chosen)>cap:raise ValueError('Protected set exceeds working budget')
    for n in sorted(old,key=lambda n:(known.get(n,float('inf')),n)):
        if len(chosen)>=cap-1:break
        if n not in chosen:chosen.append(n)
    remaining=[n for n in old if n not in chosen]
    if remaining and len(chosen)<cap:chosen.append(min(remaining,key=lambda n:(last_seen.get(n,-1),n)))
    return chosen

def past_scores(root,epoch):
    import json
    known={};seen={}
    for p in sorted((root/'epochs').glob('e[0-9][0-9][0-9]/SELECTION.json')):
        e=int(p.parent.name[1:])
        if e>=epoch:continue
        for n,row in json.loads(p.read_text()).get('scores',{}).items():
            v=row.get('SQL')
            if isinstance(v,(int,float)) and math.isfinite(v) and v>=0:known[n]=v;seen[n]=e
    return known,seen

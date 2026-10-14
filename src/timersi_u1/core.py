"""Complete programs and their compositions, without historical-run dependencies."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return convert(x.tolist())
        if isinstance(x, np.generic): return convert(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, dict): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)): return [convert(v) for v in x]
        return x
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(convert(value), indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


@dataclass
class Forecast:
    point: np.ndarray
    quantiles: np.ndarray

    def validate(self):
        self.point = np.asarray(self.point, float)
        self.quantiles = np.asarray(self.quantiles, float)
        if self.quantiles.shape != self.point.shape + (9,):
            raise ValueError('Expected point[N,H,D] and quantiles[N,H,D,9]')
        if not np.isfinite(self.point).all() or not np.isfinite(self.quantiles).all():
            raise ValueError('Nonfinite forecast')
        if np.any(np.diff(self.quantiles, axis=-1) < -1e-8):
            raise ValueError('Quantile crossing')
        return self


def blend(predictions, weights):
    weights = np.asarray(weights, float)
    if len(predictions) != len(weights) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid convex combination')
    weights = weights / weights.sum()
    if any(p.point.shape != predictions[0].point.shape for p in predictions):
        raise ValueError('Different forecast shapes')
    return Forecast(sum(w*p.point for w,p in zip(weights,predictions)),
                    sum(w*p.quantiles for w,p in zip(weights,predictions))).validate()


RAW = 'def predict(view, tools, memory):\n    return tools.timesfm(view)\n'


def program_identity(code,files=None):
    payload=json.dumps({'code':code,'files':files or {}},sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ProgramBank:
    def __init__(self, records=None):
        self.records = records or {'raw': {'kind': 'code', 'code': RAW, 'parent': None}}
        self.modules = {}

    def add(self, name, code, parent='raw', files=None):
        compile(code, name, 'exec')
        for path,source in (files or {}).items():compile(source,path,'exec')
        if name in self.records: raise ValueError('Program IDs are immutable')
        self.records[name] = {'kind': 'code', 'code': code, 'files':files or {}, 'parent': parent,
                              'sha256': program_identity(code,files)}

    def memory_source(self,name):
        node=self.records[name]
        if name=='raw':return 'raw'
        if node['kind']=='code':return 'program:'+program_identity(node['code'],node.get('files'))
        return node['kind']+':'+hashlib.sha256(json.dumps(node,sort_keys=True).encode()).hexdigest()

    def add_blend(self, name, members, weights):
        if name in self.records or any(n not in self.records for n in members):
            raise ValueError('Invalid immutable dependency')
        self.records[name] = {'kind': 'blend', 'members': members, 'weights': weights}

    def add_route(self, name, routes, fallback):
        members = sorted({fallback} | {n for row in routes.values() for route in row.values()
                                       for n in route['members']})
        if name in self.records or any(n not in self.records for n in members):
            raise ValueError('Invalid immutable route dependency')
        self.records[name] = {'kind': 'route', 'members': members,
                              'routes': routes, 'fallback': fallback}

    def add_scenario(self, name, policy):
        if name in self.records or any(n not in self.records for n in policy['members']):
            raise ValueError('Invalid immutable scenario dependency')
        self.records[name] = {'kind': 'scenario', 'members': list(policy['members']), 'policy': policy}

    def predict(self, name, view, tools, memory, cache):
        if name in cache: return cache[name]
        spec = self.records[name]
        if spec['kind'] == 'blend':
            result = blend([self.predict(n, view, tools, memory, cache) for n in spec['members']], spec['weights'])
        elif spec['kind'] == 'route':
            from .selection import route_forecasts
            result = route_forecasts(view,
                {n: self.predict(n, view, tools, memory, cache) for n in spec['members']},
                spec['routes'], spec['fallback'])
        elif spec['kind'] == 'scenario':
            from .scenario_selection import apply_scenario
            result = apply_scenario(view, lambda n: self.predict(n, view, tools, memory, cache),
                                    spec['policy'])
        else:
            if name == 'raw' and spec['code'] == RAW and not spec.get('files'):
                result = tools.timesfm(view)
            else:
                from .isolation import predict
                result = predict(spec,view,tools,memory)
        if name != 'raw':
            from .numerical_boundary import protect
            raw = self.predict('raw', view, tools, memory, cache)
            result = protect(result, raw, view, name, cache, spec)
        result.validate()
        cache[name] = result
        return result


def select(scores, limit=7):
    valid = sorted((float(v), k) for k,v in scores.items() if np.isfinite(v))
    names = ['raw'] + [k for _,k in valid if k != 'raw']
    return names[:limit]

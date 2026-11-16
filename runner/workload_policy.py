"""Versioned workload ceilings, distinct from permissions and decoder framing.

The specification is shared with native settings. Policy is immutable and scoped
per operation; changing a preference cannot mutate a running operation. Larger
presets are admitted only by callers/readers implementing this policy version.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

_SPEC=json.loads(Path(__file__).with_name('execution-workload-policy.json').read_bytes())
_IDS=frozenset(item['id'] for item in _SPEC['settings'])
_PRESETS=frozenset(_SPEC['presets'])
_scope=ContextVar('kwip_workload_policy',default=None)
_admitted=None


def _require(condition):
    if not condition:raise ValueError('Invalid captured workload policy')


@dataclass(frozen=True,init=False)
class WorkloadPolicy:
    preset:str
    limits:object

    def __init__(self,preset='legacy',limits=None):
        _require(type(preset) is str and preset in _PRESETS|{'custom'})
        if limits is None:
            _require(preset!='custom')
            limits=_SPEC['presets'][preset]
        _require(type(limits) is dict and set(limits)==_IDS)
        _require(all(type(n) is int and 1<=n<=_SPEC['integerMaximum'] for n in limits.values()))
        _require(preset=='custom' or limits==_SPEC['presets'][preset])
        object.__setattr__(self,'preset',preset)
        object.__setattr__(self,'limits',MappingProxyType(dict(limits)))

    @classmethod
    def from_document(cls,value):
        _require(type(value) is dict and set(value)=={'schemaVersion','preset','limits'})
        _require(type(value['schemaVersion']) is int and value['schemaVersion']==1)
        return cls(value['preset'],value['limits'])

    def to_document(self):
        return {'schemaVersion':1,'preset':self.preset,'limits':dict(self.limits)}

    def overriding(self,values):
        _require(type(values) is dict and set(values)<=_IDS)
        merged=dict(self.limits);merged.update(values)
        return WorkloadPolicy('custom',merged)

    def limit(self,name):
        return self.limits[name]


LEGACY=WorkloadPolicy()


def install_verified_policy(policy):
    """Bootstrap only, after launch-envelope validation and before numerical imports."""
    if not isinstance(policy,WorkloadPolicy):raise ValueError('A verified workload policy is required')
    global _admitted
    _admitted=policy
    _scope.set(policy)


def captured_policy():
    """None preserves historical reader limits when no versioned policy was captured."""
    return _scope.get()


def current_policy():
    return _scope.get() or LEGACY


@contextmanager
def workload_scope(policy):
    if not isinstance(policy,WorkloadPolicy):raise ValueError('A verified workload policy is required')
    if _admitted is not None and any(policy.limit(key)>_admitted.limit(key) for key in _IDS):
        raise ValueError('Workload scope exceeds its captured admission; choose an adequate run allowance before launch.')
    token=_scope.set(policy)
    try:yield policy
    finally:_scope.reset(token)

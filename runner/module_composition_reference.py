"""Independent bounded scalar composition oracle; no native module execution."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class AffineParameters:
    a: float
    b: float
    c: float
    d: float

@dataclass(frozen=True)
class Fixture:
    id: str
    parameters: AffineParameters
    rows: tuple[float, ...]
    upstream: tuple[float, ...]

@dataclass(frozen=True)
class Evaluation:
    output: tuple[float, ...]
    inputInfluence: tuple[float, ...]
    parameterInfluence: tuple[tuple[str, float], ...]

@dataclass(frozen=True)
class Update:
    left: AffineParameters
    right: AffineParameters
    loss: float
    gradients: tuple[tuple[str, float], ...]

FIXTURES = (
    Fixture('composition-1', AffineParameters(2, 1, 3, -2), (-2, .5, 3), (1, -2, .5)),
    Fixture('composition-2', AffineParameters(-1, .5, 2, 1), (-1.5, .25, 2), (-.5, 1, 2)),
    Fixture('composition-3', AffineParameters(.5, -2, -2, .25), (-3, 1, 2.5), (2, -.25, -1)),
    Fixture('composition-4', AffineParameters(3, -.5, .25, 2), (-.75, .5, 1.75), (-1, .5, 1.5)),
)
MAX_ROWS = 32
MAX_ABS_VALUE = 1_000_000

def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('Use finite real scalar values, not flags.')
    if not math.isfinite(value) or abs(value) > MAX_ABS_VALUE:
        raise ValueError('Scalar value exceeds the finite reference budget.')
    return float(value)

def _parameters(value):
    if not isinstance(value, AffineParameters):
        raise TypeError('Supply an immutable affine parameter record.')
    return tuple(_number(getattr(value, name)) for name in ('a','b','c','d'))

def _rows(rows, values):
    if not isinstance(rows, (tuple, list)) or not isinstance(values, (tuple, list)):
        raise TypeError('Supply bounded scalar rows and matching values.')
    if not 1 <= len(rows) <= MAX_ROWS or len(rows) != len(values):
        raise ValueError('Every bounded observation needs its matching scalar value.')
    return tuple(_number(x) for x in rows), tuple(_number(x) for x in values)

def evaluate(kind, parameters, rows, upstream):
    """Differentiate the explicitly selected scalar expression with weighted outputs."""
    if kind not in ('sequenceAB', 'sequenceBA', 'branch'):
        raise ValueError('Choose the admitted scalar expression.')
    a,b,c,d = _parameters(parameters)
    rows,upstream = _rows(rows,upstream)
    outputs=[]; inputs=[]; gradient=[0.,0.,0.,0.]
    for x,u in zip(rows,upstream):
        if kind == 'sequenceBA':
            intermediate=c*x+d
            y=a*intermediate+b
            dx=a*c
            local=(intermediate,1.,a*x,a)
        else:
            intermediate=a*x+b
            multiplier=c+1 if kind == 'branch' else c
            y=multiplier*intermediate+d
            dx=multiplier*a
            local=(multiplier*x,multiplier,intermediate,1.)
        outputs.append(y);inputs.append(u*dx)
        for index in range(4):gradient[index] += u*local[index]
    return Evaluation(tuple(outputs),tuple(inputs),tuple(zip(('a','b','c','d'),gradient)))

def evaluate_outer(left, right, rows, upstream):
    """Private left(x) minus private right(-x), with distinct owner contributions."""
    rows,upstream = _rows(rows,upstream)
    l=evaluate('branch',left,rows,upstream)
    # Differentiate minus-right with respect to its own argument first.
    r=evaluate('branch',right,tuple(-x for x in rows),tuple(-u for u in upstream))
    return Evaluation(tuple(a-b for a,b in zip(l.output,r.output)),
        tuple(a-b for a,b in zip(l.inputInfluence,r.inputInfluence)),
        tuple(('left.'+name,value) for name,value in l.parameterInfluence)+
        tuple(('right.'+name,value) for name,value in r.parameterInfluence))

def left_sgd_step(left, right, rows, targets, learning_rate=.05):
    """One simultaneous mean-squared-error adjustment of left only; right is immutable."""
    _parameters(left);_parameters(right)
    rows,targets = _rows(rows,targets)
    rate=_number(learning_rate)
    if not 0 < rate <= 1:
        raise ValueError('Learning rate must be positive and at most one.')
    outputs=evaluate('branch',left,rows,tuple(1. for _ in rows)).output
    n=len(rows);loss=0.;upstream=[]
    for y,t in zip(outputs,targets):
        difference=y-t;loss += difference*difference/n;upstream.append(2*difference/n)
    # Upstream here is computed arithmetic, not an externally supplied bounded input.
    if any(abs(u)>MAX_ABS_VALUE for u in upstream):
        raise ValueError('Adjustment exceeds the bounded scalar reference budget.')
    measured=evaluate('branch',left,rows,upstream)
    gradients=tuple(('left.'+name,value) for name,value in measured.parameterInfluence)
    values=tuple(getattr(left,name)-rate*value for name,value in measured.parameterInfluence)
    updated=AffineParameters(*values);_parameters(updated)
    return Update(updated,right,loss,gradients)

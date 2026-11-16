"""Explicit exact decimal text operations; never pass through binary floating point.

Text is the declared transport type. Ordinary Number arithmetic stays unchanged.
Oversized inputs/results fail explicitly, never round or become invalid-number flags.
"""
from decimal import Decimal, InvalidOperation, DecimalException, localcontext, Inexact, Rounded
import re

MAX_TEXT=16384
FUNCTIONS=('decimal_parse','decimal_is_finite','decimal_is_integer','decimal_abs',
           'decimal_add','decimal_subtract','decimal_multiply','decimal_compare')
BINARY={'decimal_add','decimal_subtract','decimal_multiply','decimal_compare'}
FINITE=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE]([+-]?\d+))?\Z')


def call_type(name,types):
    count=2 if name in BINARY else 1
    if len(types)!=count or any(t not in ('string','null') for t in types):
        raise ValueError(name+' requires '+str(count)+' explicit decimal text argument(s); Number/Boolean columns are not exact decimal sources.')
    return 'bool' if name in ('decimal_is_finite','decimal_is_integer') else 'int64' if name=='decimal_compare' else 'string'


def parse(value):
    if value is None:return None
    if type(value) is not str:raise ValueError('Exact decimal operations require text.')
    if len(value)>MAX_TEXT:raise ValueError('Decimal source exceeds its text budget.')
    # Decimal accepts Unicode digits and underscores. Preserve those semantics while
    # checking exponent size before construction, including exponents too large for Decimal.
    text=value.strip().replace('_','')
    match=FINITE.fullmatch(text)
    if match and match.group(1):
        exponent=match.group(1).lstrip('+-').lstrip('0') or '0'
        if len(exponent)>5 or int(exponent)>MAX_TEXT:
            raise ValueError('Decimal exponent exceeds its exact-processing bound.')
    try:result=Decimal(text)
    except InvalidOperation:return None
    return result if result.is_finite() else None


def canonical(value):
    if not value.is_finite():raise ValueError('Decimal arithmetic requires finite values.')
    if value.is_zero():return '0'
    # Avoid allocating exponent-sized strings even when a caller bypasses parse.
    exponent=value.as_tuple().exponent
    if abs(exponent)>MAX_TEXT or abs(value.adjusted())>MAX_TEXT:
        raise ValueError('Decimal result exceeds its text budget.')
    text=format(value,'f')
    if '.' in text:text=text.rstrip('0').rstrip('.')
    if len(text)>MAX_TEXT:raise ValueError('Decimal result exceeds its text budget.')
    return text


def exact(operation):
    try:
        with localcontext() as context:
            context.prec=MAX_TEXT+4
            context.Emax=MAX_TEXT*2
            context.Emin=-MAX_TEXT*2
            context.traps[Inexact]=True
            context.traps[Rounded]=True
            return operation()
    except DecimalException as error:
        raise ValueError('Decimal arithmetic exceeded its exact-processing budget; no rounded result was returned.') from error


def call_value(name,values):
    parsed=[parse(v) for v in values]
    if name=='decimal_is_finite':return None if values[0] is None else parsed[0] is not None
    if name=='decimal_is_integer':
        value=parsed[0]
        return None if values[0] is None else value is not None and value==value.to_integral_value()
    if name=='decimal_parse':return canonical(parsed[0]) if parsed[0] is not None else None
    if any(v is None for v in values):return None
    if any(v is None for v in parsed):raise ValueError('Exact decimal arithmetic received invalid or nonfinite text; validate or filter it first.')
    if name=='decimal_compare':return (parsed[0]>parsed[1])-(parsed[0]<parsed[1])
    if name=='decimal_abs':return canonical(parsed[0].copy_abs())
    operation={'decimal_add':lambda:parsed[0]+parsed[1],
               'decimal_subtract':lambda:parsed[0]-parsed[1],
               'decimal_multiply':lambda:parsed[0]*parsed[1]}[name]
    return canonical(exact(operation))


def sum_text(values):
    def total():
        result=Decimal(0)
        for text in values:
            value=parse(text)
            if value is None:raise ValueError('Exact decimal sum received invalid or nonfinite text.')
            result+=value
        return result
    return canonical(exact(total))

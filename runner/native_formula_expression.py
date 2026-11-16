"""Qualified native Formula expressions; exact reference fallback is explicit.

Decimal operations prove sufficient Decimal128 precision before evaluation. No
binary floating point or rounded decimal result is admitted. Regex support is a
bounded ASCII/prefix subset with no lookaround, backreferences or end anchors.
"""
import re
import polars as pl

DTYPES={'string':pl.String,'bool':pl.Boolean,'int64':pl.Int64,'float64':pl.Float64,'null':pl.Null}
SPACE=' \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0\u1680'+''.join(chr(i) for i in range(0x2000,0x200b))+'\u2028\u2029\u202f\u205f\u3000'
FUNCTIONS={'coalesce','trim','length','require','regex_match','regex_extract','decimal_parse','decimal_is_finite','decimal_is_integer','decimal_abs','decimal_add','decimal_subtract','decimal_multiply','decimal_compare'}


def supported(node):
    if node[0] in ('column','literal'):return True
    if node[0]=='call':return node[1] in FUNCTIONS and all(supported(n) for n in node[2])
    if node[0]=='binary':return node[1] in ('=','==','!=','<','<=','>','>=','AND','OR') and supported(node[2]) and supported(node[3])
    if node[0]=='isNull':return supported(node[2])
    if node[0]=='unary':return node[1]=='NOT' and supported(node[2])
    return False


def _decimal(value,frame):
    # Qualify ASCII fixed-point text. Exponents/Unicode retain Python Decimal's
    # broader exact contract; malformed ASCII fixed-point text becomes null.
    raw=value.str.strip_chars(SPACE).str.replace_all('_','')
    values=frame.select(raw.alias('_v')).get_column('_v')
    if (values.str.len_chars().max() or 0)>16384:return None
    if values.str.contains(r'[^\x00-\x7f]').fill_null(False).any():return None
    if values.str.contains(r'^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+$').fill_null(False).any():return None
    pattern=r'^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$'
    valid=raw.str.contains(pattern)
    cleaned=pl.when(valid).then(raw).otherwise(pl.lit(None,dtype=pl.String))
    metrics=frame.select(
        cleaned.str.split_exact('.',1).struct.field('field_1').str.len_chars().max().alias('scale'),
        cleaned.str.replace(r'^[+-]?0*','').str.split_exact('.',1).struct.field('field_0').str.len_chars().max().alias('integer'))
    scale=metrics.item(0,0) or 0;integer=metrics.item(0,1) or 0
    if scale+integer>38:return None
    return cleaned.cast(pl.Decimal(38,scale),strict=True),scale,integer


def _canonical(value):
    return value.cast(pl.String).str.replace(r'(\.[0-9]*?)0+$','$1').str.strip_suffix('.').str.replace(r'^-0$','0')


def compile_native(expression,node,frame):
    kind=node[0]
    if kind=='column':return pl.col(f'c{expression.columns[node[1]][0]}')
    if kind=='literal':
        dtype=expression.type_of(node)
        return pl.lit(node[1],dtype=DTYPES[dtype]) if dtype in DTYPES else None
    if kind=='isNull':
        value=compile_native(expression,node[2],frame)
        return None if value is None else value.is_not_null() if node[1] else value.is_null()
    if kind=='unary':
        value=compile_native(expression,node[2],frame)
        return ~value if value is not None and node[1]=='NOT' else None
    if kind=='binary':
        op=node[1];a=compile_native(expression,node[2],frame);b=compile_native(expression,node[3],frame)
        if a is None or b is None:return None
        types={expression.type_of(n) for n in node[2:]}-{'null'}
        if len(types)>1:return None
        functions={'=':lambda:a==b,'==':lambda:a==b,'!=':lambda:a!=b,'<':lambda:a<b,'<=':lambda:a<=b,'>':lambda:a>b,'>=':lambda:a>=b,'AND':lambda:a&b,'OR':lambda:a|b}
        return functions[op]() if op in functions else None
    if kind!='call':return None
    name,args=node[1:]
    values=[compile_native(expression,n,frame) for n in args]
    if any(v is None for v in values):return None
    value=values[0]
    if name=='coalesce':
        if len({expression.type_of(n) for n in args}-{'null'})>1:return None
        return pl.coalesce(values)
    if name=='trim':return value.str.strip_chars(SPACE)
    if name=='length':return value.str.len_chars().cast(pl.Int64)
    if name=='require':
        if not frame.select(value.fill_null(False).all()).item():return None
        return values[1]
    if name.startswith('decimal_'):
        a=_decimal(value,frame)
        if a is None:return None
        number,scale,integer=a
        if name=='decimal_parse':return _canonical(number)
        if name=='decimal_is_finite':return pl.when(value.is_null()).then(None).otherwise(number.is_not_null())
        if name=='decimal_is_integer':return pl.when(value.is_null()).then(None).otherwise(number.is_not_null() & (number==number.floor()).fill_null(False))
        if name=='decimal_abs':return _canonical(number.abs())
        b=_decimal(values[1],frame)
        if b is None:return None
        # Invalid non-null arithmetic must retain the reference error, not turn null.
        for original,parsed in ((value,number),(values[1],b[0])):
            if frame.select((original.is_not_null() & parsed.is_null()).any()).item():return None
        result_scale=scale+b[1] if name=='decimal_multiply' else max(scale,b[1])
        digits=integer+b[2] if name=='decimal_multiply' else max(integer,b[2])+(name!='decimal_compare')
        if result_scale+digits>38:return None
        left=number.cast(pl.Decimal(38,result_scale));right=b[0].cast(pl.Decimal(38,result_scale))
        if name=='decimal_compare':return pl.when(left.is_null()|right.is_null()).then(None).otherwise(pl.when(left>right).then(1).when(left<right).then(-1).otherwise(0)).cast(pl.Int64)
        result={'decimal_add':lambda:left+right,'decimal_subtract':lambda:left-right,'decimal_multiply':lambda:left*right}[name]()
        return _canonical(result)
    if name in ('regex_match','regex_extract'):
        if any(n[0]!='literal' for n in args[1:]):return None
        pattern=args[1][1];mode=args[2][1];flags=args[3 if name=='regex_match' else 4][1]
        if len(pattern)>256 or not re.fullmatch(r'\^?[A-Za-z0-9_ .,:\[\]{}()\-]+',pattern) or '^' in pattern[1:]:return None
        if flags not in ('','ignoreCase'):return None
        series=frame.select(value.alias('_v')).get_column('_v')
        if (series.str.len_chars().max() or 0)>16384 or series.str.contains(r'[^\x00-\x7f]').fill_null(False).any():return None
        if flags=='ignoreCase':pattern='(?i)'+pattern
        if mode=='full':pattern='^(?:'+pattern+')\\z'
        if name=='regex_match':return value.str.contains(pattern)
        if len(args)!=6 or not str(args[5][1]).isdigit():return None
        result=value.str.extract(pattern,int(args[5][1]))
        if args[3][1]=='error' and frame.select((value.is_not_null() & result.is_null()).any()).item():return None
        return result
    return None

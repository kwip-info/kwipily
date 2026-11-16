"""Allowlisted guided Formula temporal signatures, shared with native editor.

Options are string literals. Schema metadata is inferred from calls, never from
sample values. Conversion of timezone keeps physical UTC microseconds unchanged.
"""
if __package__: from . import dataflow_temporal as t
else: import dataflow_temporal as t

FUNCTIONS = ('temporal_parse','temporal_format','temporal_timezone','temporal_part','temporal_difference')


def call_contract(name, types, options):
    if name == 'temporal_parse':
        if len(types) != 9 or types[0] not in ('string','null') or len(options)!=8: raise ValueError('Parse needs text and eight explicit options')
        dtype,pattern,source,result,invalid,ambiguous,nonexistent,version=options
        if version != 'v1': raise ValueError('Use temporal contract v1')
        meta=t.metadata(dtype,result);t.checked_pattern(pattern)
        if source:t.zone(source)
        if invalid not in ('stop','missing') or ambiguous not in ('stop','earliest','latest','missing') or nonexistent not in ('stop','missing'):raise ValueError('Invalid Parse policy')
        return dtype,meta
    if name == 'temporal_format':
        if len(types)!=3 or types[0] not in t.DTYPES or len(options)!=2:raise ValueError('Format needs a temporal source, pattern and timezone')
        t.checked_pattern(options[0]);t.zone(options[1]);return 'string',None
    if name == 'temporal_timezone':
        if len(types)!=2 or types[0]!='datetime' or len(options)!=1:raise ValueError('Convert timezone needs DateTime and an explicit timezone')
        return 'datetime',t.metadata('datetime',options[0])
    if name == 'temporal_part':
        if len(types)!=3 or types[0] not in t.DTYPES or len(options)!=2 or options[0] not in t.PARTS[types[0]]:raise ValueError('Extract part needs a supported temporal part and timezone')
        t.zone(options[1]);return 'int64',None
    if name == 'temporal_difference':
        if len(types)!=4 or types[0] not in t.DTYPES or types[1]!=types[0] or len(options)!=2:raise ValueError('Difference needs two compatible temporal columns, units and representation')
        t.difference(0,0,types[0],unit=options[0],representation=options[1])
        return 'int64' if options[1]=='integer' else 'float64',None
    raise ValueError('Unknown temporal function')


def call_value(name, values, types):
    value=values[0]
    if name=='temporal_parse':
        dtype,pattern,source,result,invalid,ambiguous,nonexistent,_=values[1:]
        return t.parse(value,dtype,pattern=pattern,source_timezone=source,result_timezone=result,invalid=invalid,ambiguous=ambiguous,nonexistent=nonexistent)
    if name=='temporal_format':return t.format_value(value,types[0],pattern=values[1],timezone=values[2])
    if name=='temporal_timezone':return value
    if name=='temporal_part':return t.extract(value,types[0],values[1],timezone=values[2])
    if name=='temporal_difference':return t.difference(value,values[1],types[0],unit=values[2],representation=values[3])
    raise ValueError('Unknown temporal function')


def expression_metadata(expression, node=None):
    node=expression.tree if node is None else node
    if node[0]=='column':
        return expression.column_contracts[node[1]].get('temporal')
    if node[0]=='call':
        if node[1] in FUNCTIONS:
            args=node[2];types=[expression.type_of(arg) for arg in args]
            start=2 if node[1]=='temporal_difference' else 1
            return call_contract(node[1],types,[expression.literal(arg) for arg in args[start:]])[1]
        if node[1]=='coalesce':
            metas=[expression_metadata(expression,arg) for arg in node[2] if expression.type_of(arg) in t.DTYPES]
            if metas and any(meta!=metas[0] for meta in metas):raise ValueError('Temporal coalesce requires matching timezone/precision metadata')
            return metas[0] if metas else None
    return None

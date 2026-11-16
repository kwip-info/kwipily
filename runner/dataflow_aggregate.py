"""Bounded pre-partition aggregation with exact numeric accumulation."""
from dataclasses import dataclass
from fractions import Fraction
import json
import math

if __package__:
    from . import dataflow as df
    from . import formula_decimal
else:
    import dataflow as df
    import formula_decimal

MAX_ARTIFACT_BYTES=50_000_000
MAX_DIAGNOSTICS=16_384
REDUCERS={'countRows','countNonNull','sum','sumDecimal','mean','min','max','joinText'}


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def validate_config(config):
    if not isinstance(config,dict) or set(config)!={'groupBy','nullKeys','aggregations','maxRows'}:
        raise ValueError('Aggregate requires explicit grouping, null policy, reducers and row budget.')
    keys=config['groupBy']
    if not isinstance(keys,list) or (keys and not df.names(keys)):
        raise ValueError('Aggregate grouping columns must be an ordered unique column list.')
    if config['nullKeys'] not in ('group','drop','error'):
        raise ValueError('Invalid aggregate missing-key policy.')
    if type(config['maxRows']) is not int or not 1<=config['maxRows']<=df.MAX_ROWS:
        raise ValueError('Aggregate row limit must be between 1 and 100000.')
    fields=config['aggregations']
    if not isinstance(fields,list) or not 1<=len(fields)<=128:
        raise ValueError('Aggregate requires between 1 and 128 named reducers.')
    for field in fields:
        if not isinstance(field,dict) or not isinstance(field.get('op'),str) or field['op'] not in REDUCERS:
            raise ValueError('Invalid aggregate reducer.')
        reducer=field['op']
        expected={'output','op'} | ({'column'} if reducer!='countRows' else set()) | ({'nullPolicy'} if reducer not in ('countRows','countNonNull') else set())
        if reducer=='joinText':expected |= {'separator','distinct','skipEmpty'}
        if set(field)!=expected or not df.names([field.get('output')]):
            raise ValueError('Aggregate reducer fields and output name must be explicit.')
        if reducer=='joinText' and (not isinstance(field['separator'],str) or len(field['separator'])>1024 or type(field['distinct']) is not bool or type(field['skipEmpty']) is not bool):
            raise ValueError('Join text needs a separator of at most 1024 characters and explicit duplicate/empty choices.')
        if 'column' in field and not df.names([field['column']]):raise ValueError('Invalid reducer column.')
        if 'nullPolicy' in field and field['nullPolicy'] not in ('skip','propagate','error'):
            raise ValueError('Invalid reducer null policy.')
    if not df.names(keys+[f['output'] for f in fields]):
        raise ValueError('Aggregate output column names collide or exceed 128 columns.')


def _validate_table(table,*,validate_data=True,maximum_rows=None):
    if table.role!='unassigned' or table.partition_digest or table.fitted_states:
        raise ValueError('Aggregate before splitting or fitting transforms; existing isolation evidence cannot be discarded.')
    df._fitted_module().schema_valid(list(table.columns))
    maximum_rows=df.MAX_ROWS if maximum_rows is None else maximum_rows
    if not 1<=len(table.rows)<=maximum_rows or len(table.rows)!=len(table.row_ids):raise ValueError('Invalid aggregate input row bounds.')
    if not validate_data:return
    if any(not isinstance(x,str) or not 1<=len(x)<=200 for x in table.row_ids) or len(set(table.row_ids))!=len(table.row_ids):raise ValueError('Aggregate requires unique bounded observation identities.')
    for row in table.rows:
        if len(row)!=len(table.columns):raise ValueError('Aggregate input row width differs from schema.')
        for value,column in zip(row,table.columns):
            if value is None:
                if not column['nullable']:raise ValueError('Aggregate input null violates schema.')
            elif not df._fitted_module().typed(value,column['dtype']):raise ValueError('Aggregate input violates finite column dtype.')


def _key_value(value,dtype):
    if value is None:return None
    if dtype=='float64':
        # Preserve exact admitted integer values, including beyond 2**53.
        # Integral floats normalize to the same integer; signed zero to zero.
        return int(value) if isinstance(value,int) or value.is_integer() else value
    return value


def _reduce(table,members,field):
    reducer=field['op']
    if reducer=='countRows':return len(members)
    index=table.index(field['column'])
    # Numeric reductions retain canonical identity ordering; text concatenation
    # follows actual incoming row order, including after an explicit Sort.
    ordered=sorted(members) if reducer=='joinText' else members
    values=[table.rows[i][index] for i in ordered]
    present=[v for v in values if v is not None]
    if reducer=='countNonNull':return len(present)
    if len(present)!=len(values):
        if field['nullPolicy']=='error':raise ValueError('Resolve missing reducer values or choose an explicit null policy.')
        if field['nullPolicy']=='propagate':return None
    if not present:return None
    if reducer=='sumDecimal':return formula_decimal.sum_text(present)
    if reducer=='joinText':
        text=[v for v in present if not field['skipEmpty'] or v!='']
        if field['distinct']:text=list(dict.fromkeys(text))
        # Bound escaped output bytes before allocating the combined string.
        size=2+sum(len(encoded(v))-2 for v in text)+max(0,len(text)-1)*(len(encoded(field['separator']))-2)
        if size>MAX_ARTIFACT_BYTES:raise ValueError('Join text exceeds the serialized byte budget; reduce the group or selected text.')
        return field['separator'].join(text)
    if reducer=='min':return min(present)
    if reducer=='max':return max(present)
    # Exact binary rational accumulation; only the final result is rounded.
    # Division before float conversion avoids overflowing a finite mean.
    total=sum((Fraction(v) for v in present),Fraction())
    if reducer=='sum' and table.columns[index]['dtype']=='int64':
        value=int(total)
        if not -(2**63)<=value<2**63:raise ValueError('Aggregate int64 sum overflow; explicitly cast or revise data.')
        return value
    try:value=float(total/len(present) if reducer=='mean' else total)
    except OverflowError as error:raise ValueError('Aggregate numeric overflow; result is outside finite float64.') from error
    if not math.isfinite(value):raise ValueError('Aggregate numeric overflow; result is outside finite float64.')
    return value


@dataclass(frozen=True)
class AggregatePlan:
    table:object
    config:dict
    groups:tuple
    columns:tuple
    source_digest:str
    diagnostics:tuple
    artifact_bytes:int=0

    def iter_rows(self):
        for key,members,identity in self.groups:
            yield tuple(key)+tuple(_reduce(self.table,members,f) for f in self.config['aggregations']),identity


def aggregate_schema(table,config):
    indices=[table.index(name) for name in config['groupBy']]
    if any(table.columns[i]['dtype'] in ('image','audio','tensor') for i in indices) or any('column' in field and table.columns[table.index(field['column'])]['dtype'] in ('image','audio','tensor') for field in config['aggregations']):raise ValueError('Image columns cannot be grouping or scalar aggregation values; use an explicit image adapter.')
    columns=[{**table.columns[i],'nullable':table.columns[i]['nullable'] and config['nullKeys']=='group'} for i in indices]
    for field in config['aggregations']:
        reducer=field['op'];column=table.columns[table.index(field['column'])] if 'column' in field else None
        if reducer in ('sum','mean') and column['dtype'] not in ('int64','float64'):raise ValueError('Aggregate sum and mean require numeric columns, excluding bool.')
        if reducer=='joinText' and column['dtype']!='string':raise ValueError('Join text requires a text column; use Columns for explicit casts.')
        if reducer=='sumDecimal' and column['dtype']!='string':raise ValueError('Exact decimal sum requires original decimal text, not Number columns.')
        dtype='int64' if reducer.startswith('count') else 'float64' if reducer=='mean' else column['dtype']
        nullable=False if reducer.startswith('count') else column['nullable'] and field['nullPolicy']!='error'
        output={'name':field['output'],'dtype':dtype,'nullable':nullable}
        if dtype in ('date','datetime','time'):output['temporal']=dict(column['temporal'])
        columns.append(output)
    return indices,columns


def plan_aggregate(table,config):
    validate_config(config);_validate_table(table)
    indices,columns=aggregate_schema(table,config)
    groups={};dropped=0
    for index,row in enumerate(table.rows):
        key=tuple(_key_value(row[i],table.columns[i]['dtype']) for i in indices)
        if None in key:
            if config['nullKeys']=='error':raise ValueError('Resolve missing group key values or choose an explicit missing-key policy.')
            if config['nullKeys']=='drop':dropped+=1;continue
        if key not in groups:
            if len(groups)>=config['maxRows']:raise ValueError('Aggregate exceeds its output row limit; revise grouping or budget.')
            groups[key]=[]
        groups[key].append(index)
    if not groups:raise ValueError('Aggregate produced no rows after dropping missing group keys.')
    if len(groups)*len(columns)>5_000_000:raise ValueError('Aggregate exceeds five million output cells.')
    planned=[]
    for key,members in groups.items():
        members=tuple(sorted(members,key=lambda i:table.row_ids[i]))
        membership=df.membership_digest([table.row_ids[i] for i in members])
        identity='aggregate:'+df.digest(['kwip.aggregate.row.v1',table.source_digest,
            [[table.columns[i]['name'],table.columns[i]['dtype'],value] for i,value in zip(indices,key)],membership])
        planned.append((key,members,identity))
    diagnostic={'kind':'aggregate','inputRows':len(table.rows),'outputRows':len(groups),'droppedKeyRows':dropped,
                'sourceDigest':table.source_digest,'reducers':[
                    {'output':field['output'],'op':field['op'],'nullInputRows':
                     0 if field['op']=='countRows' else sum(table.rows[i][table.index(field['column'])] is None for members in groups.values() for i in members)}
                    for field in config['aggregations']]}
    diagnostics=[];seen=set();diagnostic_bytes=2
    for item in (*table.diagnostics,diagnostic):
        content=encoded(item)
        if content in seen:continue
        diagnostic_bytes+=len(content)+1
        if len(diagnostics)>=MAX_DIAGNOSTICS or diagnostic_bytes>MAX_ARTIFACT_BYTES:raise ValueError('Aggregate diagnostic count or byte budget exceeded.')
        seen.add(content);diagnostics.append(item)
    source=df.digest(['kwip.aggregate.sources.v1',table.source_digest,config['groupBy'],config['nullKeys']])
    plan=AggregatePlan(table,config,tuple(planned),tuple(columns),source,tuple(diagnostics))
    envelope={'schemaVersion':1,'kind':'table','columns':columns,'rows':[],
        'lineage':{'role':'unassigned','sourceDigest':source,'membershipDigest':'0'*64,'observationIDs':[],
                   'dataVersion':'0'*64,'partitionDigest':'','fittedStateDigests':[],'diagnostics':diagnostics}}
    size=len(encoded(envelope))
    # Stream preflight: no output table/row collection or derive/hash until all
    # escaped bytes fit. Each temporary row is discarded before the next group.
    for index,(row,identity) in enumerate(plan.iter_rows()):
        size+=len(encoded(row))+len(encoded(identity))+(2 if index else 0)
        if size>MAX_ARTIFACT_BYTES:raise ValueError('Aggregate exceeds its 50 MB serialized byte budget.')
    return AggregatePlan(table,config,plan.groups,plan.columns,source,plan.diagnostics,size)


def aggregate_table(table,config):
    plan=plan_aggregate(table,config)
    rows=[];ids=[]
    for row,identity in plan.iter_rows():rows.append(row);ids.append(identity)
    return df.derive(df.Table(plan.columns,tuple(rows),tuple(ids),plan.source_digest,diagnostics=plan.diagnostics))


def verify_aggregate_output(table,config,output):
    plan=plan_aggregate(table,config)
    if (output.columns!=plan.columns or output.source_digest!=plan.source_digest or output.role!='unassigned' or
        output.partition_digest or output.fitted_states or output.diagnostics!=plan.diagnostics or
        len(output.rows)!=len(plan.groups) or len(output.row_ids)!=len(plan.groups)):
        raise ValueError('Cached aggregate schema or lineage mismatch')
    for (row,identity),actual,actual_id in zip(plan.iter_rows(),output.rows,output.row_ids):
        if row!=actual or identity!=actual_id:raise ValueError('Cached aggregate values or membership mismatch')

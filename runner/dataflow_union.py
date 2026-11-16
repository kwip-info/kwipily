"""Ordered, streaming table append with explicit alignment and checked types."""
import re
from itertools import zip_longest
if __package__:
    from . import dataflow as df
    from .dataflow_join import _diagnostics
    from .execution_checkpoints import RowCheckpoints
else:
    import dataflow as df
    from dataflow_join import _diagnostics
    from execution_checkpoints import RowCheckpoints

def validate_config(c):
    if not isinstance(c,dict) or set(c)!={'inputs','match','coverage','typePolicy','sourceColumn'}:
        raise ValueError('Union requires input order, matching, coverage, type policy and identifier settings.')
    ports=c['inputs']
    if not isinstance(ports,list) or not 2<=len(ports)<=32 or any(not isinstance(p,str) or not re.fullmatch(r'table[1-9][0-9]{0,5}',p) for p in ports) or len(set(ports))!=len(ports):
        raise ValueError('Union needs 2–32 unique stable table input ports.')
    if c['match'] not in ('name','position') or c['coverage'] not in ('all','shared','exact') or c['typePolicy'] not in ('strict','losslessNumeric'):
        raise ValueError('Invalid Union matching, coverage or type policy.')
    if not isinstance(c['sourceColumn'],str) or len(c['sourceColumn'])>200 or (c['sourceColumn'] and not c['sourceColumn'].strip()):
        raise ValueError('Union input identifier must be blank or a nonblank column name of at most 200 characters.')

def plan(args,c):
    validate_config(c)
    if set(args)!=set(c['inputs']):raise ValueError('Connect exactly one table to each configured Union input.')
    tables=[args[p] for p in c['inputs']]
    for t in tables:
        if t.role!='unassigned' or t.partition_digest or t.fitted_states:
            raise ValueError('Union tables before partitioning or fitting transforms; partition isolation must not be discarded.')
        df._fitted_module().schema_valid(list(t.columns))
        if len(t.rows)!=len(t.row_ids):raise ValueError('Union row identities do not match input rows.')
    keys=[[col['name'] if c['match']=='name' else i for i,col in enumerate(t.columns)] for t in tables]
    if c['coverage']=='exact' and any(set(k)!=set(keys[0]) for k in keys):
        raise ValueError('Union requires identical column names or position counts.')
    ordered=list(dict.fromkeys(k for group in keys for k in group))
    if c['coverage']=='shared':ordered=[k for k in ordered if all(k in group for group in keys)]
    columns=[];mappings=[]
    for key in ordered:
        found=[t.columns[group.index(key)] for t,group in zip(tables,keys) if key in group]
        types={col['dtype'] for col in found}
        if len(types)>1 and not (c['typePolicy']=='losslessNumeric' and types<= {'int64','float64'}):
            raise ValueError(f'Union column {found[0]["name"]!r} has incompatible types; choose checked numeric widening or use Columns first.')
        columns.append(dict(name=found[0]['name'],dtype='float64' if len(types)>1 else found[0]['dtype'],nullable=len(found)<len(tables) or any(col['nullable'] for col in found)))
    for group in keys:mappings.append([group.index(k) if k in group else None for k in ordered])
    if c['sourceColumn']:
        if c['sourceColumn'] in [col['name'] for col in columns]:raise ValueError('Union input identifier conflicts with an output column.')
        columns.append(dict(name=c['sourceColumn'],dtype='string',nullable=False))
    if not df.names([col['name'] for col in columns]):raise ValueError('Union requires 1–128 unique output columns; inspect alignment and coverage.')
    total=sum(len(t.rows) for t in tables)
    if total>df.MAX_TABLE_CELLS//len(columns):raise ValueError('Union exceeds the five million cell storage budget.')
    source=df.digest(['kwip.union.sources.v1',[(p,t.source_digest) for p,t in zip(c['inputs'],tables)]])
    diagnostic=dict(kind='union',inputs=[dict(port=p,rows=len(t.rows),sourceDigest=t.source_digest) for p,t in zip(c['inputs'],tables)],match=c['match'],coverage=c['coverage'],typePolicy=c['typePolicy'],outputRows=total)
    histories=_diagnostics(*(t.diagnostics for t in tables),(diagnostic,))
    return tuple(columns),mappings,source,histories,total

def pairs(args,c,columns,mappings,check):
    typed=df._fitted_module().typed
    checkpoints=RowCheckpoints(check)
    for port,mapping in zip(c['inputs'],mappings):
        t=args[port]
        for row,rid in df.row_pairs(t):
            checkpoints.admit(row)
            if len(row)!=len(t.columns) or not isinstance(rid,str) or not 1<=len(rid)<=200:raise ValueError('Union input row or identity is invalid.')
            for value,col in zip(row,t.columns):
                if (value is None and not col['nullable']) or (value is not None and not typed(value,col['dtype'])):raise ValueError('Union input violates its declared type or nullability.')
            result=[]
            for i,index in enumerate(mapping):
                value=None if index is None else row[index]
                if value is not None and columns[i]['dtype']=='float64' and type(value) is int:
                    converted=float(value)
                    if int(converted)!=value:raise ValueError('Union integer cannot be represented exactly as a number; use an explicit conversion in Columns.')
                    value=converted
                result.append(value)
            if c['sourceColumn']:result.append(port)
            yield tuple(result),'union:'+df.digest(['kwip.union.row.v1',port,rid])
    checkpoints.finish()

def union_tables(args,c,check_cancelled=lambda:None,*,use_native=True):
    columns,mappings,source,history,_=plan(args,c)
    if __package__:from .dataflow_native_relational import union_table
    else:from dataflow_native_relational import union_table
    native=union_table(args,c,columns,mappings,source,history,check_cancelled) if use_native else None
    if native is not None:return native
    base=df.Table(columns,(),(),source,diagnostics=history)
    return df.streaming_rows(base,pairs(args,c,columns,mappings,check_cancelled),check_cancelled=check_cancelled)

def verify_output(args,c,output,check_cancelled=lambda:None):
    columns,mappings,source,history,total=plan(args,c)
    if output.columns!=columns or output.source_digest!=source or output.diagnostics!=history or output.role!='unassigned' or output.partition_digest or output.fitted_states or len(output.rows)!=total:
        raise ValueError('Cached Union schema, lineage or count mismatch.')
    for expected,actual in zip_longest(pairs(args,c,columns,mappings,check_cancelled),df.row_pairs(output)):
        if expected!=actual:raise ValueError('Cached Union ordered rows or observation identities mismatch.')

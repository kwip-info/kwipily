"""Streaming melt and disk-grouped pivot; explicit observation-population changes."""
from contextlib import contextmanager
from fractions import Fraction
from itertools import zip_longest
import hashlib
import json
import math
import sqlite3
import tempfile
if __package__:
    from . import dataflow as df
    from .dataflow_aggregate import _key_value
else:
    import dataflow as df
    from dataflow_aggregate import _key_value

REDUCERS={'first','last','countRows','countNonNull','sum','mean','min','max'}
MAX_SCRATCH_BYTES=100_000_000
MAX_ARTIFACT_BYTES=50_000_000

def defaults():
    return dict(mode='long',idColumns=[],valueColumns=[],fieldName='field',valueName='value',typePolicy='strict',missingValues='keep',rowKeys=[],columnKey='',values=[],duplicates='stop',nullColumnNames='stop',nullLabel='missing',missingCombinations='missing',fillValue=None,columnOrder='firstSeen',maxColumns=128)

def validate_config(c):
    if not isinstance(c,dict) or set(c)!=set(defaults()):raise ValueError('Reshape Table requires explicit Long and Wide settings.')
    for key in ('idColumns','valueColumns','rowKeys'):
        if not isinstance(c[key],list) or (c[key] and not df.names(c[key])):raise ValueError('Reshape column lists must contain unique ordered names.')
    for key in ('fieldName','valueName','nullLabel'):
        if not df.names([c[key]]):raise ValueError('Reshape output names must be nonblank and bounded.')
    for key,allowed in [('mode',('long','wide')),('typePolicy',('strict','losslessNumeric')),('missingValues',('keep','drop')),('duplicates',('stop','aggregate')),('nullColumnNames',('stop','drop','label')),('missingCombinations',('missing','fill')),('columnOrder',('firstSeen','sorted'))]:
        if c[key] not in allowed:raise ValueError('Invalid Reshape Table '+key+' policy.')
    if not isinstance(c['columnKey'],str) or len(c['columnKey'])>200 or type(c['maxColumns']) is not int or not 1<=c['maxColumns']<=128:raise ValueError('Reshape Wide column limit must be 1–128.')
    if not isinstance(c['values'],list) or len(c['values'])>128:raise ValueError('Reshape Wide needs an ordered values grid.')
    if c['fillValue'] is not None and (type(c['fillValue']) not in (str,int,float,bool) or (type(c['fillValue']) is float and not math.isfinite(c['fillValue']))):raise ValueError('Wide fixed fill must be a finite scalar value.')
    for v in c['values']:
        if not isinstance(v,dict) or set(v)!={'column','aggregation','prefix','nullPolicy'} or not df.names([v['column']]) or v['aggregation'] not in REDUCERS or v['nullPolicy'] not in ('skip','propagate','error') or not isinstance(v['prefix'],str) or len(v['prefix'])>200:raise ValueError('Invalid Wide value, reducer, prefix or missing-value policy.')

def source_digest(t,c):return df.digest(['kwip.reshape.sources.v1',t.source_digest,c])

def _admit(t,c):
    validate_config(c)
    if t.role!='unassigned' or t.partition_digest or t.fitted_states:raise ValueError('Reshape Table belongs before Partition and fitted preprocessing; existing isolation cannot be discarded.')
    df._fitted_module().schema_valid(list(t.columns))
    if len(t.rows)!=len(t.row_ids):raise ValueError('Reshape Table input row identities mismatch.')

def _input(t,check):
    for row,rid in df.row_pairs(t):
        check()
        if len(row)!=len(t.columns) or not isinstance(rid,str) or not 1<=len(rid)<=200:raise ValueError('Reshape input row or identity is invalid.')
        for value,col in zip(row,t.columns):
            if (value is None and not col['nullable']) or (value is not None and not df._fitted_module().typed(value,col['dtype'])):raise ValueError('Reshape input violates declared types or nullability.')
        yield row,rid

def _long(t,c,check):
    if not c['valueColumns']:raise ValueError('Long needs at least one value column.')
    if set(c['idColumns']) & set(c['valueColumns']):raise ValueError('Long identifier and value columns must be disjoint.')
    ids=[t.index(n) for n in c['idColumns']];values=[t.index(n) for n in c['valueColumns']]
    types={t.columns[i]['dtype'] for i in values}
    if len(types)>1 and not (c['typePolicy']=='losslessNumeric' and types<= {'int64','float64'}):raise ValueError('Long value types differ; use checked lossless numeric widening or Columns.')
    dtype='float64' if len(types)>1 else next(iter(types))
    metadata=[t.columns[i].get('temporal') for i in values]
    if any(item!=metadata[0] for item in metadata):raise ValueError('Long temporal metadata must match exactly.')
    value_column=dict(name=c['valueName'],dtype=dtype,nullable=c['missingValues']=='keep' and any(t.columns[i]['nullable'] for i in values))
    if metadata[0] is not None:value_column['temporal']=metadata[0]
    columns=tuple([t.columns[i] for i in ids]+[dict(name=c['fieldName'],dtype='string',nullable=False),value_column])
    if not df.names([col['name'] for col in columns]):raise ValueError('Long output names collide or exceed the shared schema limit.')
    # Count expansion before output allocation, including the actual Drop policy.
    count=0
    for row,_ in _input(t,check):
        count+=sum(row[i] is not None or c['missingValues']=='keep' for i in values)
        if count>df.MAX_TABLE_CELLS//len(columns):raise ValueError('Long expansion exceeds the five million cell budget.')
    def pairs():
        for row,rid in _input(t,check):
            for i in values:
                check();v=row[i]
                if v is None and c['missingValues']=='drop':continue
                if v is not None and dtype=='float64' and type(v) is int:
                    widened=float(v)
                    if int(widened)!=v:raise ValueError('Long integer cannot be represented exactly as float64.')
                    v=widened
                yield tuple(row[j] for j in ids)+(t.columns[i]['name'],v),'reshape:'+df.digest(['kwip.reshape.long.row.v1',t.source_digest,rid,t.columns[i]['name']])
    return columns,pairs

def _encoded(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

def _dtype(col,v):
    op=v['aggregation'];dtype=col['dtype']
    if op in ('sum','mean') and dtype not in ('int64','float64'):raise ValueError('Wide Sum and Mean require numeric values.')
    if dtype in ('image','audio','tensor') and op not in ('first','last','countRows','countNonNull'):raise ValueError('Wide images support First, Last and counts only.')
    return 'int64' if op.startswith('count') else 'float64' if op=='mean' else dtype

def _reduce(db,group,key,index,v,dtype,check):
    op=v['aggregation'];count=0;present=0;total=Fraction();result=None;missing=False
    for (payload,) in db.execute('SELECT payload FROM records WHERE g=? AND k=? ORDER BY ordinal',(group,key)):
        check();value=json.loads(payload)[index];count+=1
        if value is None:
            missing=True
            if v['nullPolicy']=='error' and not op.startswith('count'):raise ValueError('Wide reducer encountered missing values.')
            continue
        present+=1
        if op=='first' and present==1:result=value
        elif op=='last':result=value
        elif op=='min':result=value if present==1 else min(result,value)
        elif op=='max':result=value if present==1 else max(result,value)
        elif op in ('sum','mean'):total+=Fraction(value)
    if op=='countRows':return count
    if op=='countNonNull':return present
    if (missing and v['nullPolicy']=='propagate') or not present:return None
    if op in ('sum','mean'):
        if op=='sum' and dtype=='int64':
            result=int(total)
            if not -(2**63)<=result<2**63:raise ValueError('Wide int64 sum overflow.')
        else:
            try:result=float(total/present if op=='mean' else total)
            except OverflowError as e:raise ValueError('Wide numeric overflow.') from e
            if not math.isfinite(result):raise ValueError('Wide numeric overflow.')
    return result

@contextmanager
def _wide(t,c,check):
    if not c['values'] or not c['columnKey']:raise ValueError('Wide needs a column-name field and at least one value.')
    rowidx=[t.index(n) for n in c['rowKeys']];keyidx=t.index(c['columnKey']);valueidx=[t.index(v['column']) for v in c['values']]
    if any(t.columns[i]['dtype'] in ('image','audio','tensor') for i in rowidx+[keyidx]):raise ValueError('Wide row and column-name keys must be scalar fields.')
    dtypes=[_dtype(t.columns[i],v) for i,v in zip(valueidx,c['values'])]
    if c['missingCombinations']=='fill':
        for dtype in dtypes:
            if c['fillValue'] is None or dtype in ('image','audio','tensor') or not df._fitted_module().typed(c['fillValue'],dtype):raise ValueError('Wide fixed fill must match every output value type; images need missing fill.')
    if __package__:from .dataflow_disk import current_scope, ScratchBudget
    else:from dataflow_disk import current_scope, ScratchBudget
    scope=current_scope();budget=scope[1] if scope else ScratchBudget();reserved=0
    with tempfile.TemporaryDirectory(prefix='kwip-reshape-',dir=scope[0] if scope else None) as directory:
        db=sqlite3.connect(directory+'/pivot.sqlite')
        def account():
            nonlocal reserved
            size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
            budget.resize(reserved,size);reserved=size
        try:
            db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA max_page_count='+str(min(MAX_SCRATCH_BYTES,budget.limit-budget.used)//4096))
            db.executescript('CREATE TABLE groups(g TEXT PRIMARY KEY, ordinal INTEGER); CREATE TABLE records(g TEXT,k TEXT,ordinal INTEGER,payload TEXT,rid TEXT UNIQUE); CREATE INDEX pairs ON records(g,k,ordinal); CREATE INDEX memberships ON records(g,rid COLLATE BINARY); CREATE INDEX group_order ON groups(ordinal);')
            account()
            keys={};labels={}
            for ordinal,(row,rid) in enumerate(_input(t,check)):
                raw=row[keyidx]
                if raw is None:
                    if c['nullColumnNames']=='stop':raise ValueError('Missing column-name key; choose Drop row or an explicit label.')
                    if c['nullColumnNames']=='drop':continue
                    label=c['nullLabel'];key='null'
                else:
                    key=_encoded(_key_value(raw,t.columns[keyidx]['dtype']));label=raw if type(raw) is str else key
                    if t.columns[keyidx]['dtype'] in ('date','datetime','time'):
                        if __package__:from .dataflow_temporal import format_value
                        else:from dataflow_temporal import format_value
                        label=format_value(raw,t.columns[keyidx]['dtype'],timezone=t.columns[keyidx]['temporal'].get('timezone','UTC'))
                if label in labels and labels[label]!=key:raise ValueError('Wide output names collide between column keys or missing labels.')
                labels[label]=key
                if key not in keys:
                    keys[key]=label
                    if len(rowidx)+len(keys)*len(valueidx)>c['maxColumns']:raise ValueError('Wide output column limit exceeded; reduce vocabulary or values.')
                g=_encoded([_key_value(row[i],t.columns[i]['dtype']) for i in rowidx])
                db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)',(g,ordinal))
                if c['duplicates']=='stop' and db.execute('SELECT 1 FROM records WHERE g=? AND k=? LIMIT 1',(g,key)).fetchone():raise ValueError('Repeated row-key / column-key pair; select Aggregate and explicit reducers.')
                try:db.execute('INSERT INTO records VALUES (?,?,?,?,?)',(g,key,ordinal,_encoded(row),rid))
                except sqlite3.IntegrityError as e:raise ValueError('Reshape requires unique input observation identities.') from e
                account()
            ordered=list(keys)
            if c['columnOrder']=='sorted':ordered.sort(key=lambda key: (key=='null',json.loads(key) if key!='null' else ''))
            columns=[t.columns[i] for i in rowidx]
            for v,dtype,i in zip(c['values'],dtypes,valueidx):
                for key in ordered:
                    col=dict(name=v['prefix']+keys[key],dtype=dtype,nullable=c['missingCombinations']=='missing' or (t.columns[i]['nullable'] and v['nullPolicy']!='error' and not v['aggregation'].startswith('count')))
                    if dtype in ('date','datetime','time'):col['temporal']=t.columns[i]['temporal']
                    columns.append(col)
            if not df.names([col['name'] for col in columns]):raise ValueError('Wide output names collide, are blank or exceed the shared schema limit.')
            count=db.execute('SELECT COUNT(*) FROM groups').fetchone()[0]
            if count*len(columns)>df.MAX_TABLE_CELLS:raise ValueError('Wide output exceeds the five million cell budget.')
            def pairs():
                for g,ordinal in db.execute('SELECT g,ordinal FROM groups ORDER BY ordinal'):
                    check();result=json.loads(g)
                    membership=hashlib.sha256(b'kwip.membership.v1\n')
                    for (rid,) in db.execute('SELECT rid FROM records WHERE g=? ORDER BY rid COLLATE BINARY',(g,)):
                        check();data=rid.encode();membership.update(str(len(data)).encode()+b':'+data)
                    identity='reshape:'+df.digest(['kwip.reshape.wide.row.v1',t.source_digest,[[t.columns[i]['name'],t.columns[i]['dtype'],value] for i,value in zip(rowidx,result)],membership.hexdigest()])
                    for v,i,dtype in zip(c['values'],valueidx,dtypes):
                        for key in ordered:
                            if db.execute('SELECT 1 FROM records WHERE g=? AND k=? LIMIT 1',(g,key)).fetchone():result.append(_reduce(db,g,key,i,v,dtype,check))
                            else:result.append(c['fillValue'] if c['missingCombinations']=='fill' else None)
                    yield tuple(result),identity
            yield tuple(columns),pairs
        except sqlite3.OperationalError as e:
            raise ValueError('Wide scratch storage budget exceeded or unavailable.') from e
        finally:
            db.close();budget.resize(reserved,0)

def _preflight(t,c,columns,factory,check):
    envelope=dict(schemaVersion=3,kind='table',columns=columns,rows=[],lineage=dict(role='unassigned',sourceDigest=source_digest(t,c),membershipDigest='0'*64,observationIDs=[],dataVersion='0'*64,partitionDigest='',fittedStateDigests=[],diagnostics=t.diagnostics))
    size=sum(map(len,df.iter_json(envelope)))
    if size>MAX_ARTIFACT_BYTES:raise ValueError('Reshape metadata exceeds the 50 MB serialized byte budget.')
    for index,(row,rid) in enumerate(factory()):
        check()
        row_bytes=sum(map(len,df.iter_json(row)))
        if row_bytes>1_048_576:raise ValueError('Reshape output row exceeds the one MiB record byte budget.')
        size+=row_bytes+sum(map(len,df.iter_json(rid)))+(2 if index else 0)
        if size>MAX_ARTIFACT_BYTES:raise ValueError('Reshape output exceeds the 50 MB serialized byte budget.')
    return columns,factory()

@contextmanager
def _plan(t,c,check):
    _admit(t,c);check()
    if c['mode']=='long':
        columns,factory=_long(t,c,check)
        yield _preflight(t,c,columns,factory,check)
    else:
        with _wide(t,c,check) as (columns,factory):yield _preflight(t,c,columns,factory,check)

def reshape_table(table,c,check_cancelled=lambda:None):
    if __package__:from .dataflow_disk import disk_scope
    else:from dataflow_disk import disk_scope
    with disk_scope(check_cancelled=check_cancelled), _plan(table,c,check_cancelled) as (columns,pairs):
        base=df.Table(columns,(),(),source_digest(table,c),diagnostics=table.diagnostics)
        return df.streaming_rows(base,pairs,check_cancelled=check_cancelled)

def verify_output(table,c,output,check_cancelled=lambda:None):
    with _plan(table,c,check_cancelled) as (columns,pairs):
        if output.columns!=columns or output.source_digest!=source_digest(table,c) or output.diagnostics!=table.diagnostics or output.role!='unassigned' or output.partition_digest or output.fitted_states:raise ValueError('Cached Reshape schema or lineage mismatch.')
        for expected,actual in zip_longest(pairs,df.row_pairs(output)):
            check_cancelled()
            if expected!=actual:raise ValueError('Cached Reshape values or observation identities mismatch.')


def config_schema():
    """Authoritative portable shape; runtime supplies cross-field/type validation."""
    name={'type':'string','minLength':1,'maxLength':200}
    names={'type':'array','maxItems':128,'uniqueItems':True,'items':name}
    enum=lambda *values:dict(type='string',enum=list(values))
    properties=dict(mode=enum('long','wide'),idColumns=names,valueColumns=names,fieldName=name,valueName=name,typePolicy=enum('strict','losslessNumeric'),missingValues=enum('keep','drop'),rowKeys=names,columnKey={'type':'string','maxLength':200},values={'type':'array','maxItems':128,'items':{'type':'object','additionalProperties':False,'required':['column','aggregation','prefix','nullPolicy'],'properties':{'column':name,'aggregation':enum(*sorted(REDUCERS)),'prefix':{'type':'string','maxLength':200},'nullPolicy':enum('skip','propagate','error')}}},duplicates=enum('stop','aggregate'),nullColumnNames=enum('stop','drop','label'),nullLabel=name,missingCombinations=enum('missing','fill'),fillValue={'type':['null','string','integer','number','boolean']},columnOrder=enum('firstSeen','sorted'),maxColumns={'type':'integer','minimum':1,'maximum':128})
    return dict(type='object',additionalProperties=False,required=list(properties),properties=properties)

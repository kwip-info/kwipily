"""Bounded deterministic next-token windows. Never fits vocabulary or re-splits data."""
import json
import sqlite3
from collections import deque
if __package__:from . import dataflow as df
else:import dataflow as df

DEFAULTS={'mode':'table','tokenColumn':'tokens','scope':'row','documentColumn':'','orderColumn':'','direction':'ascending','contextLength':128,'stride':128,'incomplete':'drop','paddingID':1,'inputColumn':'input_ids','targetColumn':'target_ids','maskColumn':'loss_mask','lengthColumn':'valid_length'}
METADATA=('window','document_id','start','parent_id')


def validate_config(c):
    if type(c) is not dict or set(c)!=set(DEFAULTS):raise ValueError('Sequence windows requires its exact versioned fields.')
    for key,choices in [('mode',('table','partitions')),('scope',('row','document')),('direction',('ascending','descending')),('incomplete',('drop','pad'))]:
        if c[key] not in choices or type(c[key]) is not str:raise ValueError('Invalid Sequence windows '+key+'.')
    for key,lo,hi in [('contextLength',1,4096),('stride',1,65536),('paddingID',0,65535)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError('Sequence windows '+key+' is outside its declared bounds.')
    for key in ('tokenColumn','documentColumn','orderColumn','inputColumn','targetColumn','maskColumn','lengthColumn'):
        value=c[key]
        if type(value) is not str or '\0' in value or len(value)>200 or (key not in ('documentColumn','orderColumn') and not value.strip()):raise ValueError('Choose bounded nonblank Sequence windows column names.')
    if c['scope']=='document' and (not c['documentColumn'].strip() or not c['orderColumn'].strip()):raise ValueError('Grouped sequences require document and order columns.')
    names=[c[k] for k in ('inputColumn','targetColumn','maskColumn','lengthColumn')]
    if len(set(names))!=4 or set(names)&set(METADATA):raise ValueError('Sequence output names must be distinct from each other and metadata columns.')
    return c


def ports(c,connected=None):
    if c['mode']=='table':return {'table'},{'table'},{'table'}
    return {'training','validation','test'},{'training'}, {'training'}|({'validation','test'} if connected is None else set(connected)&{'validation','test'})


def output_columns(c):
    columns=[{'name':name,'dtype':dtype,'nullable':False} for name,dtype in [('window','int64'),('document_id','string'),('start','int64'),('parent_id','string'),(c['inputColumn'],'int64Sequence'),(c['targetColumn'],'int64Sequence'),(c['lengthColumn'],'int64')]]
    if c['incomplete']=='pad':columns.append({'name':c['maskColumn'],'dtype':'int64Sequence','nullable':False})
    return tuple(columns)


def _tokens(value):
    if type(value) not in (list,tuple) or len(value)>65536 or any(type(v) is not int or not 0<=v<=65535 for v in value):raise ValueError('Select bounded exact nonnegative integer token sequences (0–65535); nulls and coercion are not supported.')
    return value


def _documents(table,c,check):
    index=table.index(c['tokenColumn'])
    if table.columns[index]['dtype']!='int64Sequence':raise ValueError('Select a typed integer token-sequence column.')
    if c['scope']=='row':
        for n,(row,rid) in enumerate(df.row_pairs(table)):
            check();yield ('Record '+str(n+1),rid,iter(_tokens(row[index])))
        return
    doc=table.index(c['documentColumn']);order=table.index(c['orderColumn'])
    if table.columns[doc]['dtype'] not in ('string','int64') or table.columns[order]['dtype'] not in ('string','int64'):raise ValueError('Document and order columns require exact text or integer values.')
    store=df.DiskStore(check)
    try:
        db=store.connection
        db.execute('CREATE TABLE fragments (document TEXT NOT NULL, ordering_key TEXT NOT NULL, ordinal_int INTEGER, ordinal_text TEXT COLLATE BINARY, tokens TEXT NOT NULL, UNIQUE(document,ordering_key))')
        for n,(row,rid) in enumerate(df.row_pairs(table)):
            check();document=row[doc];ordinal=row[order]
            for value,col in [(document,table.columns[doc]),(ordinal,table.columns[order])]:
                if col['dtype']=='string' and (type(value) is not str or not value or len(value.encode())>65536) or col['dtype']=='int64' and (type(value) is not int or not -2**63<=value<2**63):raise ValueError('Document and order values must be nonnull, exact and bounded.')
            values=_tokens(row[index])
            try:db.execute('INSERT INTO fragments VALUES (?,?,?,?,?)',(json.dumps(document),json.dumps(ordinal),ordinal if type(ordinal) is int else None,ordinal if type(ordinal) is str else None,json.dumps(values)))
            except sqlite3.IntegrityError as error:raise ValueError('Each document needs unique order values; repair the ambiguous order.') from error
            if n%128==0:store._allocation_limit();store._account()
        db.execute('CREATE INDEX fragment_order ON fragments(document,ordinal_int,ordinal_text)');db.commit();store._account()
        direction='ASC' if c['direction']=='ascending' else 'DESC'
        for (key,) in db.execute('SELECT DISTINCT document FROM fragments ORDER BY document COLLATE BINARY'):
            check()
            parent='group:'+df.digest({'document':json.loads(key),'source':table.source_digest,'partition':table.partition_digest})
            def stream(key=key):
                for (text,) in db.execute('SELECT tokens FROM fragments WHERE document=? ORDER BY ordinal_int '+direction+', ordinal_text '+direction,(key,)):
                    check();yield from json.loads(text)
            yield str(json.loads(key)),parent,stream()
    finally:store.close()


def _one(table,c,graph,check):
    columns=output_columns(c);identity=df.digest({'schemaVersion':1,'configuration':c,'inputDataVersion':table.data_version,'role':table.role,'graph':graph})
    def pairs():
        number=0;size=c['contextLength'];stride=c['stride']
        for document,parent,tokens in _documents(table,c,check):
            buffer=deque();start=0;position=0
            def emit(valid):
                nonlocal number
                check();number+=1
                values=list(buffer);x=values[:valid]+[c['paddingID']]*(size-valid);y=values[1:valid+1]+[c['paddingID']]*(size-valid)
                row=[number,document,start,parent,x,y,valid]
                if c['incomplete']=='pad':row.append([1]*valid+[0]*(size-valid))
                return row,identity+':'+str(number)
            for token in tokens:
                if position%256==0:check()
                if position>=start:buffer.append(token)
                position+=1
                if len(buffer)==size+1:
                    yield emit(size);start+=stride
                    for _ in range(min(stride,len(buffer))):buffer.popleft()
            if c['incomplete']=='pad':
                while len(buffer)>=2 and start<position-1:
                    yield emit(min(size,len(buffer)-1));start+=stride
                    for _ in range(min(stride,len(buffer))):buffer.popleft()
    # This creates a new observation population, unlike identity-preserving transforms.
    store=df.row_store(columns,check,identity_prefix=identity+':')
    try:
        for index,(row,rid) in enumerate(pairs()):
            if index>=df.MAX_TABLE_CELLS//len(columns):raise ValueError('Sequence windows exceeds the five million cell storage budget; reduce overlap or prepare smaller source regions.')
            store.append(tuple(row),rid)
        rows,ids=store.finish()
        return df.derive(table,rows=rows,ids=ids,columns=columns)
    except BaseException:store.close();raise


def apply(args,c,graph,check=lambda:None):
    validate_config(c);check();allowed,required,_=ports(c,args)
    if not required<=set(args)<=allowed:raise ValueError('Connect Table, or Training with optional Validation/Test tables.')
    if c['mode']=='partitions':
        base=args['training'];members=df.DiskStore(check)
        try:
            db=members.connection;db.execute('CREATE TABLE members (id TEXT PRIMARY KEY) WITHOUT ROWID')
            for role,table in args.items():
                if table.role!=role or not table.partition_digest or table.source_digest!=base.source_digest or table.partition_digest!=base.partition_digest or table.columns!=base.columns or table.fitted_states!=base.fitted_states:raise ValueError('Sequence partitions require matching source, schema, fitted-state and partition authority.')
                for i,rid in enumerate(table.row_ids):
                    check()
                    try:db.execute('INSERT INTO members VALUES (?)',(rid,))
                    except sqlite3.IntegrityError as error:raise ValueError('Sequence input partitions overlap or repeat source rows.') from error
                    if i%128==0:members._allocation_limit();members._account()
            db.commit();members._account()
        finally:members.close()
    return {role:_one(table,c,graph,check) for role,table in args.items()}


def verify_output(args,c,graph,outputs,check):
    expected=apply(args,c,graph,check)
    if set(expected)!=set(outputs):raise ValueError('Cached sequence output ports differ.')
    for role,table in expected.items():
        stored=outputs[role]
        if table.columns!=stored.columns or table.data_version!=stored.data_version or table.evidence()!=stored.evidence():raise ValueError('Cached sequence values, masks or lineage differ from their captured inputs.')

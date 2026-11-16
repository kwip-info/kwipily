"""Fixed, bounded database output adapter, executed only by the authenticated broker.

No arbitrary SQL, connection URI, credential artifacts, implicit schema changes or retries.
Every returned status describes this ONE row transaction; MySQL DDL is separate.
"""
from contextlib import contextmanager,nullcontext
from pathlib import Path
import datetime,math,os,re,sqlite3,ssl,stat,struct,time,uuid
if __package__:
    from .database_source import options as source_options,credentials,quoted
else:
    from database_source import options as source_options,credentials,quoted

TYPES={'sqlite':['INTEGER','REAL','TEXT'], 'postgresql':['BIGINT','DOUBLE PRECISION','TEXT','BOOLEAN','DATE','TIMESTAMPTZ','TIME'], 'mysql':['BIGINT','DOUBLE','VARCHAR(255)','TEXT','BOOLEAN','DATE','DATETIME(6)','TIME(6)']}
DEFAULTS=dict(schemaVersion=1,backend='sqlite',host='localhost',port=5432,database='',schema='main',table='',tlsMode='verifyFull',mode='append',batchRows=2048,timeoutSeconds=30,mapping=[],keys=[])
class OutputError(ValueError):pass

def name(s):
    if type(s) is not str or not s or '\0' in s or len(s.encode())>200:raise OutputError('Use a nonempty database identifier of at most 200 bytes.')
    return s

def options(value,*,require_mapping=True):
    if type(value) is not dict or set(value)-set(DEFAULTS):raise OutputError('Unsupported database output setting.')
    c={**DEFAULTS,**value}
    if type(c['schemaVersion']) is not int or c['schemaVersion']!=1 or c['mode'] not in ('create','append','upsert'):raise OutputError('Choose Create table, Append rows or Upsert.')
    if type(c['batchRows']) is not int or not 1<=c['batchRows']<=8192:raise OutputError('Rows per batch must be between 1 and 8192.')
    try:source_options({**{k:c[k] for k in ('backend','host','port','database','schema','table','tlsMode','timeoutSeconds')},'table':c['table'] or ('_metadata' if not require_mapping else '')})
    except ValueError:raise OutputError('Choose a valid database endpoint, table, TLS and timeout.') from None
    mapping=c['mapping']
    if type(mapping) is not list or not (1 if require_mapping else 0)<=len(mapping)<=128:raise OutputError('Include between 1 and 128 mapped columns.')
    for row in mapping:
        if type(row) is not dict or set(row)!={'source','target','type','nullable'} or row['type'] not in TYPES[c['backend']] or type(row['nullable']) is not bool:raise OutputError('Mapping requires known source/destination names, a supported database type and nullability.')
        name(row['source']);name(row['target'])
        if c['backend']=='postgresql' and len(row['target'].encode())>63 or c['backend']=='mysql' and len(row['target'])>64:raise OutputError('Destination identifier exceeds the engine limit.')
    if len({r['source'] for r in mapping})!=len(mapping) or len({r['target'] if c['backend']=='postgresql' else r['target'].casefold() for r in mapping})!=len(mapping):raise OutputError('Each source and destination column must be mapped once.')
    keys=c['keys']
    if type(keys) is not list or len(keys)>128 or any(type(k) is not str or k not in {r['target'] for r in mapping} for k in keys) or len(set(keys))!=len(keys):raise OutputError('Map every member of the selected key exactly once.')
    if require_mapping and c['mode']=='upsert' and not keys:raise OutputError('Upsert requires a verified unique key.')
    if c['backend']=='postgresql' and c['schema']=='main':c['schema']='public'
    if c['backend']=='mysql':c['schema']=c['database']
    for identifier in (c['schema'],c['table']):
        if c['backend']=='postgresql' and len(identifier.encode())>63 or c['backend']=='mysql' and len(identifier)>64:raise OutputError('Destination identifier exceeds the engine limit.')
    return c

def target(c):return (quoted(c['schema'],c['backend'])+'.' if c['schema'] else '')+quoted(c['table'],c['backend'])

@contextmanager
def connection(c,secret,sqlite_path=None,read_only=False):
    secret=credentials(secret);con=None
    try:
        b=c['backend'];seconds=c['timeoutSeconds']
        if b=='sqlite':
            from urllib.parse import quote
            p=Path(sqlite_path) if sqlite_path is not None else None
            if p is None or p.is_symlink() or not p.is_file() or p.stat().st_nlink!=1:raise OutputError('Choose an existing regular SQLite database with write permission.')
            # Ancestors must not contain user-controlled symlinks. The app holds
            # the explicitly selected directory grant through child completion.
            raw=str(p.absolute());canonical=os.path.realpath(raw)
            expected='/private'+raw if raw.startswith(('/var/','/tmp/','/etc/')) else raw
            if canonical!=expected:raise OutputError('SQLite destination cannot traverse symlinks.')
            for suffix in ('-journal','-wal','-shm'):
                sibling=Path(raw+suffix)
                if sibling.is_symlink() or sibling.exists() and not sibling.is_file():raise OutputError('SQLite journal entries must be regular files.')
            con=sqlite3.connect('file:'+quote(canonical,safe='/')+'?mode='+('ro' if read_only else 'rw'),uri=True,timeout=seconds)
            if hasattr(con,"enable_load_extension"):con.enable_load_extension(False)
            con.execute('PRAGMA foreign_keys=ON')
            con.execute('PRAGMA trusted_schema=OFF')
            if read_only:con.execute('PRAGMA query_only=ON')
        elif b=='postgresql':
            import psycopg
            con=psycopg.connect(host=c['host'],port=c['port'],dbname=c['database'],user=secret['username'],password=secret['password'],connect_timeout=seconds,sslmode='verify-full' if c['tlsMode']=='verifyFull' else 'disable',**({'sslrootcert':'system'} if c['tlsMode']=='verifyFull' else {}))
            con.read_only=read_only
            with con.cursor() as cur:
                cur.execute("SELECT set_config('statement_timeout', %s, true)",(str(seconds*1000),))
                cur.execute("SELECT set_config('lock_timeout', %s, true)",(str(seconds*1000),))
        else:
            import pymysql
            context=ssl.create_default_context() if c['tlsMode']=='verifyFull' else None
            con=pymysql.connect(host=c['host'],port=c['port'],database=c['database'],user=secret['username'],password=secret['password'],connect_timeout=seconds,read_timeout=seconds,write_timeout=seconds,ssl=context,ssl_disabled=context is None,charset='utf8mb4',local_infile=False,autocommit=False)
            with con.cursor() as cur:
                cur.execute('SET SESSION innodb_lock_wait_timeout=%s',(seconds,))
                if read_only:cur.execute('START TRANSACTION READ ONLY')
        yield con
    finally:
        if con is not None:
            try:con.rollback()
            finally:con.close()

def family(dtype):
    t=dtype.upper()
    if 'UNSIGNED' in t:return 'unsupported'
    if t in ('BOOL','BOOLEAN'):return 'bool'
    if any(x in t for x in ('INT','SERIAL')):return 'integer'
    if any(x in t for x in ('DOUBLE','REAL','FLOAT')):return 'float'
    if any(x in t for x in ('TEXT','CHAR','CLOB')):return 'text'
    if t=='DATE':return 'date'
    if any(x in t for x in ('TIMESTAMP','DATETIME')):return 'datetime'
    if t.startswith('TIME'):return 'time'
    return 'unsupported'

def metadata(con,c):
    b=c['backend'];cols=[];keys=[];tables=[];engine=None
    with con.cursor() if b!='sqlite' else nullcontext(con) as cur:
        if b=='sqlite':
            tables=[r[0] for r in con.execute("SELECT name FROM main.sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 129")]
            raw=con.execute('SELECT name,type,"notnull",dflt_value,pk,hidden FROM pragma_table_xinfo(?)',(c['table'],)).fetchall()
            definition=con.execute("SELECT sql FROM main.sqlite_schema WHERE name=? AND type='table'",(c['table'],)).fetchone()
            rowid=bool(definition) and 'WITHOUT ROWID' not in definition[0].upper() and sum(bool(r[4]) for r in raw)==1
            for n,t,nn,default,pk,hidden in raw:cols.append(dict(name=n,type=t.upper(),family=family(t),nullable=not bool(nn or pk),default=default is not None or rowid and bool(pk) and t.upper()=='INTEGER',generated=hidden!=0))
            primary=[r[0] for r in sorted(raw,key=lambda r:r[4]) if r[4]]
            if primary:keys.append(dict(name='PRIMARY',columns=primary))
            for row in con.execute('SELECT name,"unique",partial FROM pragma_index_list(?)',(c['table'],)):
                if row[1] and not row[2]:
                    fields=[r[0] for r in con.execute('SELECT name FROM pragma_index_info(?) ORDER BY seqno',(row[0],))]
                    if fields and all(fields) and fields!=primary:keys.append(dict(name=row[0],columns=fields))
        elif b=='postgresql':
            cur.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_type=%s ORDER BY table_name LIMIT 129',(c['schema'],'BASE TABLE'));tables=[r[0] for r in cur.fetchall()]
            cur.execute('SELECT column_name,data_type,is_nullable,column_default,is_generated,is_identity,identity_generation,character_maximum_length FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position',(c['schema'],c['table']))
            for n,t,nullable,default,generated,identity,identity_generation,length in cur.fetchall():cols.append(dict(maxCharacters=length,name=n,type=t.upper(),family=family(t),nullable=nullable=='YES',default=default is not None or identity=='YES',generated=generated!='NEVER' or identity_generation=='ALWAYS'))
            cur.execute("SELECT tc.constraint_name,k.column_name,k.ordinal_position FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage k ON tc.constraint_catalog=k.constraint_catalog AND tc.constraint_schema=k.constraint_schema AND tc.constraint_name=k.constraint_name AND tc.table_name=k.table_name WHERE tc.table_schema=%s AND tc.table_name=%s AND tc.constraint_type IN ('PRIMARY KEY','UNIQUE') AND tc.is_deferrable='NO' ORDER BY tc.constraint_name,k.ordinal_position",(c['schema'],c['table']))
            for n,col,_ in cur.fetchall():
                if not keys or keys[-1]['name']!=n:keys.append(dict(name=n,columns=[]))
                keys[-1]['columns'].append(col)
        else:
            cur.execute('SELECT TABLE_NAME,ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_TYPE=%s ORDER BY TABLE_NAME LIMIT 129',(c['database'],'BASE TABLE'))
            for n,e in cur.fetchall():
                tables.append(n)
                if n==c['table']:engine=e
            cur.execute('SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLUMN_DEFAULT,EXTRA FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION',(c['database'],c['table']))
            for n,t,nullable,default,extra in cur.fetchall():cols.append(dict(name=n,type=t.upper(),family=family(t),nullable=nullable=='YES',default=default is not None or 'auto_increment' in extra,generated='GENERATED' in extra and 'DEFAULT_GENERATED' not in extra))
            cur.execute("SELECT tc.CONSTRAINT_NAME,k.COLUMN_NAME,k.ORDINAL_POSITION FROM information_schema.TABLE_CONSTRAINTS tc JOIN information_schema.KEY_COLUMN_USAGE k ON tc.CONSTRAINT_SCHEMA=k.CONSTRAINT_SCHEMA AND tc.CONSTRAINT_NAME=k.CONSTRAINT_NAME AND tc.TABLE_NAME=k.TABLE_NAME WHERE tc.TABLE_SCHEMA=%s AND tc.TABLE_NAME=%s AND tc.CONSTRAINT_TYPE IN ('PRIMARY KEY','UNIQUE') ORDER BY tc.CONSTRAINT_NAME,k.ORDINAL_POSITION",(c['database'],c['table']))
            for n,col,_ in cur.fetchall():
                if not keys or keys[-1]['name']!=n:keys.append(dict(name=n,columns=[]))
                keys[-1]['columns'].append(col)
    if len(cols)>128 or len(keys)>128 or any(len(k['columns'])>128 for k in keys):raise OutputError('Destination metadata exceeds its 128-column/key budget.')
    exists=bool(cols)
    if b=='sqlite':exists=con.execute("SELECT 1 FROM main.sqlite_schema WHERE name=? AND type='table'",(c['table'],)).fetchone() is not None
    elif b=='postgresql':
        with con.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s AND table_type='BASE TABLE'",(c['schema'],c['table']));exists=cur.fetchone() is not None
    else:
        with con.cursor() as cur:
            cur.execute("SELECT ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND TABLE_TYPE='BASE TABLE'",(c['database'],c['table']));row=cur.fetchone();exists=row is not None;engine=row[0] if row else None
    return dict(schemaVersion=1,kind='databaseMetadata',exists=exists,columns=cols,keys=keys,tables=tables[:128],tablesTruncated=len(tables)>128,engine=engine,upsertSupported=b!='mysql' or len(keys)==1)

def inspect(value,secret,*,sqlite_path=None):
    c=options(value,require_mapping=False)
    with connection(c,secret,sqlite_path,read_only=True) as con:return metadata(con,c)

def convert(value,source,dest):
    if value is None:
        if not dest['nullable']:raise OutputError('A required destination/key column has a missing value.')
        return None
    sf=source['dtype'];f=dest['family']
    if sf in ('image','tokenSequence','token_sequence'):raise OutputError('Image/token-sequence columns require an explicit scalar representation before database output.')
    if f=='integer' and type(value) in (int,bool):
        kind=dest['type'].split('(')[0]
        bits=8 if kind=='TINYINT' else 16 if kind in ('SMALLINT','INT2') else 24 if kind=='MEDIUMINT' else 32 if kind in ('INT','INTEGER','INT4','SERIAL') and dest.get('backend')!='sqlite' else 64
        if not -(2**(bits-1))<=value<2**(bits-1):raise OutputError('A value exceeds the destination integer range.')
        return int(value)
    if f=='float' and type(value) in (int,float):
        if not math.isfinite(value) or type(value) is int and abs(value)>2**53:raise OutputError('A numeric conversion would lose precision or is not finite.')
        if (dest['type'] in ('REAL','FLOAT','FLOAT4') and dest.get('backend')=='postgresql' or dest['type'].startswith('FLOAT') and dest.get('backend')=='mysql') and struct.unpack('f',struct.pack('f',value))[0]!=value:raise OutputError('A numeric conversion would lose precision.')
        return value
    if f=='bool' and type(value) is bool:return value
    if f=='text' and type(value) is str:
        length=dest.get('maxCharacters')
        match=re.search(r'(?:VAR)?CHAR\((\d+)\)',dest['type'])
        if length is None and match:length=int(match[1])
        if length is not None and len(value)>length:raise OutputError('A text value exceeds the destination character limit.')
        if dest.get('backend')=='mysql' and dest['type']=='TEXT' and len(value.encode())>65535:raise OutputError('A text value exceeds the destination byte limit.')
        return value
    if sf in ('date','datetime','time') and type(value) is int:
        if sf=='time' and not 0<=value<86_400_000_000:raise OutputError('Time must be within one day.')
        epoch=datetime.datetime(1970,1,1,tzinfo=datetime.timezone.utc)
        v=(epoch+datetime.timedelta(days=value)).date() if sf=='date' else epoch+datetime.timedelta(microseconds=value) if sf=='datetime' else (epoch+datetime.timedelta(microseconds=value)).timetz().replace(tzinfo=None)
        if f=='text' and dest.get('backend')=='sqlite':return v.isoformat()
        if f==sf:return v if dest.get('backend')!='mysql' or sf!='datetime' else v.replace(tzinfo=None)
    raise OutputError('A mapped value is incompatible with the destination type. Use Columns for an explicit conversion.')

def write(value,secret,columns,rows,*,operation_id,sqlite_path=None,check_cancelled=lambda:None,after_commit=lambda:None):
    """No retry. Commit failures/ack loss are unknown, even if rollback later succeeds."""
    receipt=dict(schemaVersion=1,kind='databaseWrite',operationID=operation_id,status='notStarted',inputRows=len(rows),createdTable=False)
    con=None;began=False;commit_entered=False;c=None
    try:
        if str(uuid.UUID(operation_id)).upper()!=operation_id:raise OutputError('Invalid database operation identity.')
        c=options(value);receipt.update(backend=c['backend'],table=c['table'],schema=c['schema'],mode=c['mode'])
        if not 1<=len(columns)<=128 or len(rows)>5_000_000//len(columns) or len({x['name'] for x in columns})!=len(columns):raise OutputError('Input table exceeds its row/column budget.')
        source={x['name']:x for x in columns}
        if any(x['source'] not in source for x in c['mapping']):raise OutputError('A mapped source column is absent.')
        indexes=[next(i for i,col in enumerate(columns) if col['name']==m['source']) for m in c['mapping']]
        check_cancelled()
        with connection(c,secret,sqlite_path) as con:
            b=c['backend']
            # Hold schema/row authority through metadata validation and commit.
            with con.cursor() if b!='sqlite' else nullcontext(con) as lock:
                if b=='sqlite':lock.execute('BEGIN IMMEDIATE')
                elif c['mode']!='create':lock.execute(('LOCK TABLE '+target(c)+' IN ROW EXCLUSIVE MODE') if b=='postgresql' else ('SELECT * FROM '+target(c)+' LIMIT 0'))
            m=metadata(con,c)
            if c['mode']=='create':
                if m['exists']:raise OutputError('Destination table already exists; Create never replaces it.')
                dest=[dict(name=r['target'],type=r['type'],family=family(r['type']),nullable=r['nullable'] and r['target'] not in c['keys'],generated=False,backend=b) for r in c['mapping']]
            else:
                if not m['exists']:raise OutputError('Choose an existing destination table or Create table.')
                if b=='mysql' and m['engine']!='InnoDB':raise OutputError('Database row transactions require an InnoDB target.')
                known={x['name']:x for x in m['columns']}
                if any(r['target'] not in known or known[r['target']]['generated'] for r in c['mapping']):raise OutputError('Map known writable destination columns; generated columns cannot be written.')
                dest=[dict(known[r['target']],backend=b) for r in c['mapping']]
                targets={r['target'] for r in c['mapping']}
                if any(not col['nullable'] and not col['default'] and not col['generated'] and col['name'] not in targets for col in m['columns']):raise OutputError('Map every required destination column without a server default.')
                if c['mode']=='upsert' and (not any(k['columns']==c['keys'] for k in m['keys']) or not m['upsertSupported']):raise OutputError('Choose an existing unique key. MySQL upsert requires exactly one unique key.')
            keyindexes=[next(i for i,r in enumerate(c['mapping']) if r['target']==k) for k in c['keys']]
            seen=set();prepared=[]
            for row in rows:
                check_cancelled()
                if type(row) not in (list,tuple) or len(row)!=len(columns):raise OutputError('Captured table has inconsistent row widths.')
                values=tuple(convert(row[i],source[r['source']],d) for i,r,d in zip(indexes,c['mapping'],dest))
                if c['keys'] and c['mode'] in ('create','upsert'):
                    key=tuple(values[i] for i in keyindexes)
                    if None in key or key in seen:raise OutputError('Upsert input keys must be nonmissing and unique.')
                    seen.add(key)
                prepared.append(values)
            check_cancelled()
            cur=con.cursor()
            try:
                began=True
                if c['mode']=='create':
                    definitions=[quoted(r['target'],b)+' '+r['type']+(' NOT NULL' if not d['nullable'] else '') for r,d in zip(c['mapping'],dest)]
                    if c['keys']:definitions.append('PRIMARY KEY ('+', '.join(quoted(k,b) for k in c['keys'])+')')
                    if b=='mysql':commit_entered=True # DDL acknowledgement can itself be ambiguous.
                    cur.execute('CREATE TABLE '+target(c)+' ('+', '.join(definitions)+')'+(' ENGINE=InnoDB' if b=='mysql' else ''))
                    receipt['createdTable']=True;commit_entered=False
                fields=[r['target'] for r in c['mapping']];placeholder='?' if b=='sqlite' else '%s'
                sql='INSERT INTO '+target(c)+' ('+', '.join(quoted(n,b) for n in fields)+') VALUES ('+', '.join([placeholder]*len(fields))+')'
                if c['mode']=='upsert':
                    updates=[n for n in fields if n not in c['keys']]
                    if not updates:raise OutputError('Upsert must include at least one non-key column to update.')
                    if b=='mysql':sql+=' ON DUPLICATE KEY UPDATE '+', '.join(quoted(n,b)+'=VALUES('+quoted(n,b)+')' for n in updates)
                    else:sql+=' ON CONFLICT ('+', '.join(quoted(n,b) for n in c['keys'])+') DO UPDATE SET '+', '.join(quoted(n,b)+'=excluded.'+quoted(n,b) for n in updates)
                for first in range(0,len(prepared),c['batchRows']):
                    check_cancelled();cur.executemany(sql,prepared[first:first+c['batchRows']])
                check_cancelled();commit_entered=True;con.commit();after_commit()
                receipt.update(status='committed')
            except BaseException:
                if commit_entered:receipt['status']='unknown'
                else:
                    try:
                        con.rollback();receipt.update(status='rolledBack',committedRows=0)
                        if b!='mysql':receipt['createdTable']=False
                    except BaseException:receipt['status']='unknown'
                raise
            finally:cur.close()
    except BaseException as error:
        if receipt['status'] not in ('unknown','rolledBack') and began:receipt['status']='unknown'
        receipt['message']=str(error) if isinstance(error,OutputError) else 'Database write interrupted; check the recorded outcome before retrying.'
    return receipt

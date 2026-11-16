"""Exact-table, read-only database source capture. Never used by normal ML workers.

Remote adapters belong to the separate signed network-capable source broker.
The public configuration has no credentials, connection URI, or arbitrary SQL.
"""
from contextlib import contextmanager
import base64
import datetime
import decimal
import ipaddress
import math
from pathlib import Path
import re
import sqlite3
import ssl
import uuid

DEFAULTS = dict(backend='sqlite', host='localhost', port=5432, database='', schema='main', table='',
    columns=[], orderBy=[], rowLimit=0, rowOffset=0, fetchBatchRows=2048, timeoutSeconds=30,
    tlsMode='verifyFull', decimalMode='text', dateTimeMode='iso8601', binaryMode='error')
MAX_CELLS=5_000_000
MAX_CAPTURE_BYTES=10_000_000
MAX_ROW_BYTES=1_048_576

class DatabaseSourceError(ValueError):
    """Messages must never include driver exception text or credential material."""


def options(value):
    if type(value) is not dict or set(value)-set(DEFAULTS):raise DatabaseSourceError('Unknown database source option.')
    result={**DEFAULTS, **value}
    if result['backend'] not in ('sqlite','postgresql','mysql'):raise DatabaseSourceError('Choose SQLite, PostgreSQL, or MySQL.')
    for name in ('host','database','schema','table'):
        text=result[name]
        if type(text) is not str or '\0' in text or len(text.encode('utf-8'))>256:raise DatabaseSourceError('Database identifiers must be bounded text.')
    if not result['table']:raise DatabaseSourceError('Choose one exact database table.')
    for name,lower,upper in [('port',1,65535),('rowLimit',0,MAX_CELLS),('rowOffset',0,2147483647),('fetchBatchRows',1,8192),('timeoutSeconds',1,300)]:
        if type(result[name]) is not int or not lower<=result[name]<=upper:raise DatabaseSourceError('Database source numeric option is outside its supported range.')
    for name in ('columns','orderBy'):
        values=result[name]
        if type(values) is not list or len(values)>128 or any(type(v) is not str or not v or '\0' in v or len(v.encode('utf-8'))>256 for v in values) or len(set(values))!=len(values):raise DatabaseSourceError('Choose unique, bounded database columns.')
    for key,choices in [('tlsMode',('verifyFull','disabledLoopback')),('decimalMode',('text','error')),('dateTimeMode',('iso8601','error')),('binaryMode',('error','base64','hex'))]:
        if result[key] not in choices:raise DatabaseSourceError('Unsupported database decoding or TLS policy.')
    if result['backend']=='sqlite':
        if result['schema'] not in ('','main'):raise DatabaseSourceError('SQLite capture reads only the selected file’s main schema.')
    else:
        if not result['database'] or not result['host'] or not re.fullmatch(r'[A-Za-z0-9_.:\-]+',result['host']):raise DatabaseSourceError('Enter an exact database and a hostname or IP address, without a URI.')
        if result['tlsMode']=='disabledLoopback':
            try:loopback=ipaddress.ip_address(result['host']).is_loopback
            except ValueError:loopback=False
            if not loopback:raise DatabaseSourceError('Unencrypted database connections require a literal loopback IP address.')
        if result['backend']=='mysql' and result['schema'] not in ('',result['database'],'main'):raise DatabaseSourceError('MySQL selected schema must match its database.')
    return result


def credentials(value):
    if type(value) is not dict or set(value)-{'username','password'}:raise DatabaseSourceError('Invalid private database credentials.')
    result={name:value.get(name,'') for name in ('username','password')}
    if any(type(text) is not str or '\0' in text or len(text.encode('utf-8'))>4096 for text in result.values()):raise DatabaseSourceError('Private database credential exceeds its byte limit.')
    return result


def quoted(identifier,backend):
    marker='`' if backend=='mysql' else '"'
    return marker+identifier.replace(marker,marker*2)+marker


def selection(config,names,*,limit_override=None):
    """Only identifiers are composed; LIMIT/OFFSET values remain bound parameters."""
    known=set(names);selected=config['columns'] or list(names)
    if not selected or len(selected)>128 or len(set(selected))!=len(selected) or not set(selected)<=known:raise DatabaseSourceError('Selected columns are absent from this table or exceed 128 columns.')
    ordering=[]
    for value in config['orderBy']:
        direction='DESC' if value.startswith('-') else 'ASC';name=value[1:] if value[:1] in ('-','+') else value
        if name not in known:raise DatabaseSourceError('Ordering column is absent from the selected table.')
        ordering.append(quoted(name,config['backend'])+' '+direction)
    backend=config['backend'];schema=config['schema']
    if backend=='postgresql' and schema=='main':schema='public'
    if backend=='mysql':schema=config['database']
    target=(quoted(schema,backend)+'.' if schema else '')+quoted(config['table'],backend)
    # Guard values on the server before the driver can allocate a huge record.
    # The extra flag rejects oversize rows instead of silently dropping them.
    def value_size(name):
        field=quoted(name,backend)
        return ('COALESCE(octet_length('+field+'::text),0)' if backend=='postgresql' else
                'COALESCE(OCTET_LENGTH('+field+'),0)' if backend=='mysql' else
                'COALESCE(length(CAST('+field+' AS BLOB)),0)')
    guard='('+' + '.join(value_size(name) for name in selected)+') <= '+str(MAX_ROW_BYTES//2)
    fields=['CASE WHEN '+guard+' THEN '+quoted(name,backend)+' ELSE NULL END AS '+quoted(name,backend) for name in selected]
    sql='SELECT '+', '.join(fields)+', CASE WHEN '+guard+' THEN 0 ELSE 1 END FROM '+target
    if ordering:sql+=' ORDER BY '+', '.join(ordering)
    limit=config['rowLimit'] if limit_override is None else min(limit_override,config['rowLimit']) if config['rowLimit'] else limit_override
    placeholder='?' if backend=='sqlite' else '%s';params=[]
    if limit or config['rowOffset']:
        effective=limit if limit else -1 if backend=='sqlite' else 18446744073709551615 if backend=='mysql' else None
        if effective is not None:sql+=' LIMIT '+placeholder;params.append(effective)
        if config['rowOffset']:sql+=' OFFSET '+placeholder;params.append(config['rowOffset'])
    return sql,tuple(params),selected


def _coerce(value,config):
    if value is None or type(value) in (str,bool):return value
    if type(value) is int:
        if not -(2**63)<=value<2**63:raise DatabaseSourceError('A database integer exceeds int64; select a compatible column.')
        return value
    if type(value) is float:
        if not math.isfinite(value):raise DatabaseSourceError('A database number is not finite.')
        return value
    if isinstance(value,decimal.Decimal):
        if config['decimalMode']=='error' or not value.is_finite():raise DatabaseSourceError('Decimal values require the explicit exact-text decoding option.')
        return str(value)
    if isinstance(value,(datetime.date,datetime.datetime,datetime.time)):
        if config['dateTimeMode']=='error':raise DatabaseSourceError('Date/time values require the explicit ISO8601 text decoding option.')
        return value.isoformat()
    if isinstance(value,datetime.timedelta) and config['backend']=='mysql':
        if config['dateTimeMode']=='error':raise DatabaseSourceError('Date/time values require the explicit ISO8601 text decoding option.')
        # MySQL TIME includes signed durations longer than 24 hours. Preserve
        # exact microseconds without a floating-point total_seconds conversion.
        micros=(value.days*86400+value.seconds)*1_000_000+value.microseconds
        sign='-' if micros<0 else '';micros=abs(micros)
        days,micros=divmod(micros,86_400_000_000)
        hours,micros=divmod(micros,3_600_000_000)
        minutes,micros=divmod(micros,60_000_000)
        seconds,fraction=divmod(micros,1_000_000)
        clock=(str(hours)+'H' if hours else '')+(str(minutes)+'M' if minutes else '')
        if seconds or fraction or not (days or clock):
            clock+=str(seconds)+('.'+f'{fraction:06d}'.rstrip('0') if fraction else '')+'S'
        return sign+'P'+(str(days)+'D' if days else '')+('T'+clock if clock else '')
    if isinstance(value,(bytes,bytearray,memoryview)):
        if config['binaryMode']=='base64':return base64.b64encode(value).decode('ascii')
        if config['binaryMode']=='hex':return bytes(value).hex()
        raise DatabaseSourceError('Binary values require an explicit base64 or hexadecimal decoding option.')
    if isinstance(value,uuid.UUID):return str(value)
    raise DatabaseSourceError('This table contains an unsupported database value type; choose compatible columns.')


def _target(config):
    backend=config['backend'];schema=config['schema']
    if backend=='postgresql' and schema=='main':schema='public'
    if backend=='mysql':schema=config['database']
    return (quoted(schema,backend)+'.' if schema else '')+quoted(config['table'],backend)


@contextmanager
def selected_cursor(config,secret,*,sqlite_path=None,check_cancelled=lambda:None,limit_override=None):
    """One consistent read-only transaction; metadata belongs only to the selected table."""
    config=options(config);secret=credentials(secret);connection=cursor=stream=None
    check_cancelled()
    try:
        backend=config['backend']
        if backend=='sqlite':
            if sqlite_path is None:raise DatabaseSourceError('Choose a local SQLite file before capture.')
            from urllib.parse import quote
            path=Path(sqlite_path)
            if path.is_symlink() or not path.is_file():raise DatabaseSourceError('SQLite source requires a selected regular file.')
            connection=sqlite3.connect('file:'+quote(str(path.resolve()),safe='/')+'?mode=ro',uri=True,timeout=config['timeoutSeconds'])
            if hasattr(connection,"enable_load_extension"):connection.enable_load_extension(False)
            connection.execute('PRAGMA query_only=ON');connection.execute('BEGIN')
            cancellation=[]
            def progress():
                try:check_cancelled();return 0
                except BaseException as error:cancellation.append(error);return 1
            connection.set_progress_handler(progress,1000)
            row=connection.execute('SELECT type FROM main.sqlite_schema WHERE name=?',(config['table'],)).fetchone()
            if row is None or row[0]!='table':raise DatabaseSourceError('The selected SQLite table is absent or is not a base table.')
            cursor=connection.cursor()
        elif backend=='postgresql':
            import psycopg
            from psycopg import IsolationLevel
            connection=psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username'],password=secret['password'],connect_timeout=config['timeoutSeconds'],sslmode='verify-full' if config['tlsMode']=='verifyFull' else 'disable',**({'sslrootcert':'system'} if config['tlsMode']=='verifyFull' else {}))
            connection.read_only=True;connection.isolation_level=IsolationLevel.REPEATABLE_READ
            with connection.cursor() as setup:
                setup.execute("SELECT set_config('statement_timeout', %s, true)",(str(config['timeoutSeconds']*1000),))
                setup.execute("SELECT set_config('lock_timeout', %s, true)",(str(config['timeoutSeconds']*1000),))
            cursor=connection.cursor()
        else:
            import pymysql
            context=ssl.create_default_context() if config['tlsMode']=='verifyFull' else None
            connection=pymysql.connect(host=config['host'],port=config['port'],database=config['database'],user=secret['username'],password=secret['password'],charset='utf8mb4',connect_timeout=config['timeoutSeconds'],read_timeout=config['timeoutSeconds'],write_timeout=config['timeoutSeconds'],ssl=context,ssl_disabled=context is None,local_infile=False,autocommit=False)
            with connection.cursor() as setup:
                setup.execute('SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ')
                setup.execute('START TRANSACTION READ ONLY')
                setup.execute('SET SESSION max_execution_time=%s',(config['timeoutSeconds']*1000,))
            cursor=connection.cursor(pymysql.cursors.SSCursor)
        check_cancelled()
        # Introspection is a zero-row projection from this exact selected table.
        metadata=connection.cursor()
        try:
            metadata.execute('SELECT * FROM '+_target(config)+' LIMIT 0')
            names=[field[0] for field in metadata.description]
            typecodes={field[0]:field[1] for field in metadata.description}
            if backend=='sqlite':
                declared=connection.execute('SELECT name, type FROM pragma_table_info(?)',(config['table'],)).fetchall()
                typecodes={name:kind.upper() for name,kind in declared}
            if len(names)>128 or any(type(name) is not str or not name or len(name)>200 for name in names) or len(set(names))!=len(names):raise DatabaseSourceError('Selected table schema exceeds supported column bounds.')
        finally:metadata.close()
        sql,params,names=selection(config,names,limit_override=limit_override)
        # libpq single-row mode streams one bounded record without a network
        # round-trip per FETCH; MySQL's SSCursor is likewise unbuffered.
        if backend=='postgresql':stream=cursor.stream(sql,params,size=1)
        else:cursor.execute(sql,params)
        class SelectedCursor:
            types=[typecodes.get(name) for name in names]
            def fetchmany(self,size):
                if stream is None:values=cursor.fetchmany(size)
                else:
                    values=[]
                    for _ in range(size):
                        try:values.append(next(stream))
                        except StopIteration:break
                if any(row[-1] for row in values):raise DatabaseSourceError('A database record exceeds its bounded transfer budget. Select fewer or smaller columns.')
                return [row[:-1] for row in values]
        yield SelectedCursor(),names,config
    except DatabaseSourceError:raise
    except (KeyboardInterrupt,SystemExit):raise
    except Exception:
        check_cancelled()
        if config['backend']=='sqlite' and 'cancellation' in locals() and cancellation:raise cancellation[0]
        raise DatabaseSourceError('Database read failed. Check the selected table, permissions, TLS, and connection settings.') from None
    finally:
        if stream is not None:
            try:stream.close()
            except Exception:pass
        if cursor is not None:
            try:cursor.close()
            except Exception:pass
        if connection is not None:
            try:connection.rollback()
            except Exception:pass
            connection.close()


def capture_selection_identity(config):
    """Public source/query authority only; never credentials or sampling policy."""
    import hashlib,json
    config=options(config)
    value={key:value for key,value in config.items() if key not in ('fetchBatchRows','timeoutSeconds')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest().encode('ascii')


def capture(config,secret,sink,*,sqlite_path=None,check_cancelled=lambda:None,preview=False):
    """Write an Arrow IPC file; callers own the private, bounded output descriptor."""
    import hashlib
    import json
    import pyarrow as pa
    import pyarrow.ipc as ipc
    written=0;hasher=hashlib.sha256()
    class BoundedSink:
        closed=False
        def writable(self):return True
        def tell(self):return written
        def write(self,block):
            nonlocal written
            check_cancelled();block=bytes(block)
            if written+len(block)>MAX_CAPTURE_BYTES:raise DatabaseSourceError('Database capture exceeds its 10 MB source budget. Select fewer columns or limit rows.')
            result=sink.write(block)
            if result is not None and result!=len(block):raise DatabaseSourceError('Database capture could not write all selected bytes.')
            written+=len(block);hasher.update(block);return len(block)
        def flush(self):sink.flush()
    writer=None;count=0;schema=None
    try:
        with selected_cursor(config,secret,sqlite_path=sqlite_path,check_cancelled=check_cancelled,limit_override=8 if preview else None) as (cursor,names,config):
            while True:
                check_cancelled();raw=[];batch_bytes=0
                # Fetch one server-bounded record at a time. A large text column
                # cannot turn a row-count batch setting into unbounded client RAM.
                for _ in range(min(config['fetchBatchRows'],8) if preview else config['fetchBatchRows']):
                    selected=cursor.fetchmany(1)
                    if not selected:break
                    record=tuple(_coerce(value,config) for value in selected[0])
                    record_bytes=len(json.dumps(record,ensure_ascii=True,allow_nan=False).encode())
                    if record_bytes>MAX_ROW_BYTES:raise DatabaseSourceError('A database record exceeds its 1 MiB row budget.')
                    raw.append(record);batch_bytes+=record_bytes
                    if batch_bytes>=1_048_576:break
                if not raw:break
                rows=[]
                for record in raw:
                    row=tuple(_coerce(value,config) for value in record)
                    if len(row)!=len(names):raise DatabaseSourceError('Database row shape differs from the selected columns.')
                    if len(json.dumps(row,ensure_ascii=True,allow_nan=False).encode())>MAX_ROW_BYTES:raise DatabaseSourceError('A database record exceeds its 1 MiB row budget.')
                    count+=1
                    if count>MAX_CELLS//len(names):raise DatabaseSourceError('Database capture exceeds its five million cell budget.')
                    rows.append(row)
                def field_type(code):
                    if config['backend']=='sqlite':
                        if 'INT' in code:return pa.int64()
                        if any(part in code for part in ('REAL','FLOA','DOUB')):return pa.float64()
                        if any(part in code for part in ('CHAR','CLOB','TEXT','DATE','TIME')):return pa.string()
                        return None
                    integer_codes=(20,21,23,26) if config['backend']=='postgresql' else (1,2,3,8,9,13)
                    float_codes=(700,701) if config['backend']=='postgresql' else (4,5)
                    text_codes=(25,1042,1043,1082,1083,1114,1184,1266,1700,2950) if config['backend']=='postgresql' else (0,7,10,11,12,15,246,253,254)
                    if code in integer_codes:return pa.int64()
                    if code in float_codes:return pa.float64()
                    if config['backend']=='postgresql' and code==16:return pa.bool_()
                    if code in text_codes:return pa.string()
                    return None
                arrays=[]
                for index in range(len(names)):
                    values=[row[index] for row in rows];hint=field_type(cursor.types[index])
                    present=[value for value in values if value is not None]
                    if hint is not None:
                        if pa.types.is_int64(hint) and any(type(value) is not int for value in present):raise DatabaseSourceError('Database values differ from their declared integer type; no lossy conversion was applied.')
                        if pa.types.is_string(hint) and any(type(value) is not str for value in present):raise DatabaseSourceError('Database values differ from their declared text type; no implicit conversion was applied.')
                        if pa.types.is_boolean(hint) and any(type(value) is not bool for value in present):raise DatabaseSourceError('Database values differ from their declared boolean type.')
                        if pa.types.is_float64(hint):
                            if any(type(value) not in (int,float) or (type(value) is int and int(float(value))!=value) for value in present):raise DatabaseSourceError('Database float conversion would lose numeric precision.')
                    try:array=pa.array(values,type=hint)
                    except Exception:raise DatabaseSourceError('Database column values do not share a supported scalar type; choose compatible columns.') from None
                    if pa.types.is_null(array.type):array=pa.array(values,type=pa.string())
                    arrays.append(array)
                for array in arrays:
                    if not (pa.types.is_string(array.type) or pa.types.is_boolean(array.type) or pa.types.is_int64(array.type) or pa.types.is_float64(array.type)):raise DatabaseSourceError('Database columns must decode to text, boolean, int64, or float64.')
                if schema is None:
                    schema=pa.schema([pa.field(name,array.type,nullable=True) for name,array in zip(names,arrays)],
                                     metadata={b'kwip.database.selection.v1':capture_selection_identity(config)})
                    writer=ipc.new_file(BoundedSink(),schema)
                if any(array.type!=field.type and any(value is not None for value in array.to_pylist()) for array,field in zip(arrays,schema)):
                    raise DatabaseSourceError('Database column types changed within the selected table; no implicit conversion was applied.')
                try:batch=pa.RecordBatch.from_arrays([array.cast(field.type,safe=True) for array,field in zip(arrays,schema)],schema=schema)
                except Exception:raise DatabaseSourceError('Database column types changed within the selected table. Choose compatible columns.') from None
                writer.write_batch(batch)
            if writer is None:raise DatabaseSourceError('The selected database table has no observations.')
        writer.close();writer=None;check_cancelled()
        return {'schemaVersion':1,'kind':'databaseCapture','format':'ipc','sha256':hasher.hexdigest(),'bytes':written,'rows':count,'columns':len(schema),'sampleOnly':preview}
    finally:
        if writer is not None:
            try:writer.close()
            except Exception:pass


def _configuration(value):
    if type(value) is dict and value.get('format')=='database':value=value.get('options',{})
    return options(value)


def _is_ipc(path):
    with Path(path).open('rb') as stream:return stream.read(6)==b'ARROW1'


def _source_hash(path,check):
    import hashlib,os,stat
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_CAPTURE_BYTES:raise DatabaseSourceError('SQLite source exceeds its regular-file byte budget.')
        hasher=hashlib.sha256();count=0
        while True:
            check();block=os.read(descriptor,262144)
            if not block:break
            count+=len(block)
            if count>MAX_CAPTURE_BYTES:raise DatabaseSourceError('SQLite source grew beyond its byte budget.')
            hasher.update(block)
        after=os.fstat(descriptor)
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns) or count!=before.st_size:raise DatabaseSourceError('SQLite source changed during capture.')
        return hasher.hexdigest()
    finally:os.close(descriptor)


def _typed_local(path,config,check,*,preview,sampling):
    if __package__:
        from . import table_sources
        from .dataflow_disk import current_scope,disk_scope
    else:
        import table_sources
        from dataflow_disk import current_scope,disk_scope
    def read(target):
        if preview:return table_sources.preview_typed_snapshot(target,check_cancelled=check)
        return table_sources.source_typed_snapshot(target,check_cancelled=check,sampling=sampling)
    if _is_ipc(path):
        import pyarrow.ipc as ipc
        with Path(path).open('rb') as source:
            metadata=ipc.open_file(source).schema.metadata or {}
        if metadata.get(b'kwip.database.selection.v1')!=capture_selection_identity(config):
            raise DatabaseSourceError('Database settings changed. Reconnect or refresh the database before using its output.')
        return read(path)
    if config['backend']!='sqlite':raise DatabaseSourceError('Remote databases require a verified private broker snapshot; network access is unavailable in ordinary workers.')
    import tempfile
    original_sha=_source_hash(path,check)
    # The transient typed file shares the operation's scratch admission.
    with disk_scope(check_cancelled=check):
        context=current_scope();budget=context[1];reserved=MAX_CAPTURE_BYTES
        budget.resize(0,reserved)
        try:
            with tempfile.TemporaryDirectory(prefix='.sqlite-capture-',dir=context[0]) as directory:
                target=Path(directory)/'capture.ipc'
                capture_config=config
                if sampling is not None and not preview:
                    if __package__:from .preparation_demand_configuration import normalize_sampling,sampling_row_limit
                    else:from preparation_demand_configuration import normalize_sampling,sampling_row_limit
                    policy=normalize_sampling(sampling)
                    if policy['mode']=='rowCount':
                        limit=sampling_row_limit(policy)
                        capture_config={**config,'rowLimit':min(config['rowLimit'],limit) if config['rowLimit'] else limit}
                with target.open('xb') as sink:
                    target.chmod(0o600)
                    capture(capture_config,{},sink,sqlite_path=path,check_cancelled=check,preview=preview)
                result=read(target)
                if _source_hash(path,check)!=original_sha:raise DatabaseSourceError('SQLite source changed while reading the selected table.')
                if preview:
                    columns,rows,_=result
                    return columns,rows,original_sha
                # This brand-new private store has never been exposed. Its IDs
                # contain ordinals only; bind their prefix to the authorized raw
                # snapshot and selected-table/query identity. Different tables
                # in one SQLite file must never share observation identities.
                if __package__:from .dataflow import derive,digest
                else:from dataflow import derive,digest
                identity_keys={'backend','schema','table','columns','orderBy','rowLimit','rowOffset','decimalMode','dateTimeMode','binaryMode'}
                selection_identity=digest(['kwip.database.rows.v1',original_sha,{key:config[key] for key in identity_keys}])
                result.row_ids.store.identity_prefix=selection_identity+':'
                return derive(result,source_digest=original_sha)
        finally:budget.resize(reserved,0)


def source_table(path,configuration,check_cancelled=lambda:None,*,sampling=None):
    return _typed_local(path,_configuration(configuration),check_cancelled,preview=False,sampling=sampling)


def preview_table(path,configuration,check_cancelled=lambda:None,*,sampling=None):
    return _typed_local(path,_configuration(configuration),check_cancelled,preview=True,sampling=sampling)


def preview_evidence(path,configuration,check_cancelled=lambda:None):
    """Typed bounded inspection for SQLite files and authority-bound broker IPC."""
    table=source_table(path,configuration,check_cancelled,sampling={'schemaVersion':1,'mode':'rowCount','rowCount':8})
    try:
        return {'columns':list(table.columns),'rows':[list(row) for row in table.rows[:8]],
                'sourceSHA256':table.source_digest,'sampleOnly':True}
    finally:table.rows.store.close()

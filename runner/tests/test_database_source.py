import io
import json
import sqlite3
import pytest
from runner import database_source as db

@pytest.fixture
def table(tmp_path):
    path=tmp_path/'selected.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE chosen (id INTEGER, label TEXT, score REAL)')
        connection.executemany('INSERT INTO chosen VALUES (?,?,?)',[(1,'one',1.5),(2,'two',None),(3,'three',3.5)])
        connection.execute('CREATE TABLE private (secret TEXT)')
        connection.execute("INSERT INTO private VALUES ('unselected')")
    return path

def decode(sink):
    import pyarrow.ipc as ipc
    return ipc.open_file(sink.getvalue()).read_all().to_pydict()

def test_sqlite_selected_projection_order_limits_and_receipt(table):
    sink=io.BytesIO()
    receipt=db.capture(dict(table='chosen',columns=['label','id'],orderBy=['-id'],rowLimit=2,fetchBatchRows=1),{},sink,sqlite_path=table)
    assert decode(sink)=={'label':['three','two'],'id':[3,2]}
    assert receipt['format']=='ipc' and receipt['rows']==2 and receipt['bytes']==len(sink.getvalue())
    import hashlib
    assert receipt['sha256']==hashlib.sha256(sink.getvalue()).hexdigest()
    assert b'unselected' not in sink.getvalue()

def test_sqlite_source_never_writes_and_selected_metadata_does_not_crawl(table):
    before=table.read_bytes();sink=io.BytesIO()
    db.capture(dict(table='chosen',rowOffset=1),{},sink,sqlite_path=table)
    assert decode(sink)['id']==[2,3]
    assert table.read_bytes()==before
    with pytest.raises(db.DatabaseSourceError,match='absent'):
        db.capture(dict(table='missing'),{},io.BytesIO(),sqlite_path=table)

@pytest.mark.parametrize('config',[{'table':'chosen','tlsMode':'insecure'},{'table':'chosen','query':'SELECT * FROM private'},{'table':'chosen','port':True},{'table':'chosen','backend':'postgresql','database':'x','host':'example.com','tlsMode':'disabledLoopback'}])
def test_public_options_reject_unscoped_or_unsafe_settings(config):
    with pytest.raises(db.DatabaseSourceError):db.options(config)

def test_identifiers_are_quoted_parameters_are_bound():
    config=db.options(dict(table='a";DROP TABLE private;--',columns=['x"y'],rowLimit=4,rowOffset=2))
    sql,params,names=db.selection(config,['x"y'])
    assert '"a"";DROP TABLE private;--"' in sql
    assert sql.endswith('LIMIT ? OFFSET ?') and params==(4,2)
    assert names==['x"y']

def test_binary_decoding_is_explicit_and_error_messages_redact_credentials(table):
    with sqlite3.connect(table) as connection:connection.execute('CREATE TABLE blob_values (x BLOB)');connection.execute("INSERT INTO blob_values VALUES (X'0102')")
    with pytest.raises(db.DatabaseSourceError,match='Binary'):
        db.capture(dict(table='blob_values'),{},io.BytesIO(),sqlite_path=table)
    sink=io.BytesIO();db.capture(dict(table='blob_values',binaryMode='hex'),{},sink,sqlite_path=table)
    assert decode(sink)=={'x':['0102']}
    with pytest.raises(db.DatabaseSourceError) as error:
        db.capture(dict(table='missing'),dict(username='DO_NOT_PRINT_USER',password='DO_NOT_PRINT_SECRET'),io.BytesIO(),sqlite_path=table)
    assert 'DO_NOT_PRINT' not in str(error.value)

def test_oversize_record_rejected_before_client_payload_and_cancellation(table):
    with sqlite3.connect(table) as connection:
        connection.execute('CREATE TABLE huge (x TEXT)');connection.execute('INSERT INTO huge VALUES (?)',('x'*600_000,))
    with pytest.raises(db.DatabaseSourceError,match='transfer budget'):
        db.capture(dict(table='huge'),{},io.BytesIO(),sqlite_path=table)
    calls=[]
    def cancel():
        calls.append(1)
        if len(calls)>2:raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):db.capture(dict(table='chosen'),{},io.BytesIO(),sqlite_path=table,check_cancelled=cancel)

def test_preview_bounds_read_to_eight_rows(tmp_path):
    path=tmp_path/'preview.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE chosen (id INTEGER)')
        connection.executemany('INSERT INTO chosen VALUES (?)',[(i,) for i in range(100)])
    sink=io.BytesIO();receipt=db.capture(dict(table='chosen'),{},sink,sqlite_path=path,preview=True)
    assert receipt['rows']==8 and receipt['sampleOnly'] is True

@pytest.fixture
def postgres(tmp_path):
    import getpass
    import shutil
    import socket
    import subprocess
    binary=shutil.which('postgres')
    if not binary:pytest.skip('Local PostgreSQL binaries unavailable')
    folder=__import__('pathlib').Path(binary).resolve().parent
    root=tmp_path/'postgres-data';log=tmp_path/'postgres.log'
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    subprocess.run([str(folder/'initdb'),'-D',str(root),'-A','trust','--no-locale','-E','UTF8'],check=True,capture_output=True,timeout=30)
    subprocess.run([str(folder/'pg_ctl'),'-D',str(root),'-l',str(log),'-o',f'-h 127.0.0.1 -p {port} -k /tmp','-w','start'],check=True,capture_output=True,timeout=30)
    try:
        import psycopg
        with psycopg.connect(host='127.0.0.1',port=port,dbname='postgres',user=getpass.getuser()) as connection:
            connection.execute('CREATE TABLE chosen (id BIGINT, label TEXT, amount NUMERIC, moment DATE)')
            connection.execute("INSERT INTO chosen VALUES (1,'one',1.25,'2026-10-13'),(2,'two',2.50,'2026-10-14')")
            connection.execute('CREATE TABLE private (secret TEXT)')
        yield dict(backend='postgresql',host='127.0.0.1',port=port,database='postgres',schema='public',table='chosen',tlsMode='disabledLoopback'),dict(username=getpass.getuser(),password='')
    finally:subprocess.run([str(folder/'pg_ctl'),'-D',str(root),'-m','immediate','-w','stop'],capture_output=True,timeout=20)

def test_actual_postgres_bounded_cursor_readonly_types_and_ipc(postgres):
    config,secret=postgres;sink=io.BytesIO()
    receipt=db.capture(dict(config,columns=['label','amount','moment'],orderBy=['-id'],fetchBatchRows=1),secret,sink)
    assert receipt['rows']==2
    assert decode(sink)=={'label':['two','one'],'amount':['2.50','1.25'],'moment':['2026-10-14','2026-10-13']}
    with db.selected_cursor(config,secret) as (cursor,names,options):
        assert cursor.fetchmany(1)[0][0]==1
    import psycopg
    with psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username']) as connection:
        connection.execute("CREATE TABLE huge AS SELECT repeat('x',600000) AS content")
    with pytest.raises(db.DatabaseSourceError,match='transfer budget'):
        db.capture(dict(config,table='huge'),secret,io.BytesIO())
    bad=dict(config,table='nonexistent')
    with pytest.raises(db.DatabaseSourceError):db.capture(bad,secret,io.BytesIO())

def test_sqlite_worker_adapter_preserves_raw_file_authority_and_ids(table):
    import hashlib
    config={'format':'database','options':{'table':'chosen'}}
    value=db.source_table(table,config)
    expected=hashlib.sha256(table.read_bytes()).hexdigest()
    assert value.source_digest==expected
    assert all(identity.endswith(':'+str(i)) for i,identity in enumerate(value.row_ids))
    duplicate=db.source_table(table,{'format':'database','options':{'table':'chosen','fetchBatchRows':1}})
    assert list(value.row_ids)==list(duplicate.row_ids)
    reordered=db.source_table(table,{'format':'database','options':{'table':'chosen','orderBy':['-id']}})
    assert set(value.row_ids).isdisjoint(reordered.row_ids)
    names,rows,source=db.preview_table(table,config)
    assert source==expected and names==['id','label','score'] and len(rows)==3

def test_real_broker_child_framed_credentials_and_native_parent_lifetime(postgres,tmp_path):
    import subprocess,sys,time
    from pathlib import Path
    config,secret=postgres
    script=Path(__file__).resolve().parents[1]/'database_source_capture.py'
    payload=json.dumps({'schemaVersion':1,'options':config,'credentials':secret,'preview':False}).encode()
    process=subprocess.Popen([sys.executable,'-I','-B',str(script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        process.stdin.write(len(payload).to_bytes(4,'big')+payload);process.stdin.flush()
        assert process.wait(timeout=10)==0
        output=process.stdout.read();receipt=json.loads(process.stderr.read())
        assert receipt['format']=='ipc' and receipt['rows']==2
        import hashlib
        assert receipt['sha256']==hashlib.sha256(output).hexdigest()
    finally:
        process.stdin.close()
        if process.poll() is None:process.kill();process.wait()
        process.stdout.close();process.stderr.close()
    # A blocking database function cannot hold the Python GIL against revocation.
    import psycopg
    with psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username']) as connection:
        connection.execute('CREATE VIEW slowly AS SELECT i, pg_sleep(0.05)::text AS delayed FROM generate_series(1,1000) AS i')
    payload=json.dumps({'schemaVersion':1,'options':dict(config,table='slowly'),'credentials':secret,'preview':False}).encode()
    output_path=tmp_path/'partial.ipc'
    with output_path.open('wb') as sink:
        process=subprocess.Popen([sys.executable,'-I','-B',str(script)],stdin=subprocess.PIPE,stdout=sink,stderr=subprocess.PIPE)
        try:
            process.stdin.write(len(payload).to_bytes(4,'big')+payload);process.stdin.flush();time.sleep(.5)
            process.stdin.close()
            assert process.wait(timeout=3)==125
            assert b'databaseCapture' not in process.stderr.read()
        finally:
            if not process.stdin.closed:process.stdin.close()
            if process.poll() is None:process.kill();process.wait()
            process.stderr.close()

@pytest.fixture
def mysql(tmp_path):
    import os,socket,subprocess,time,uuid
    from pathlib import Path
    binary=Path('/opt/homebrew/opt/mysql@8.4/bin/mysqld')
    if not binary.is_file():pytest.skip('Local MySQL8.4 binary unavailable')
    folder=tmp_path/'mysql-data';folder.mkdir()
    log=tmp_path/'mysql.log';socket_path=Path('/tmp')/('kwip-db-'+uuid.uuid4().hex[:12]+'.sock')
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    subprocess.run([str(binary),'--no-defaults','--initialize-insecure','--datadir='+str(folder)],check=True,capture_output=True,timeout=45)
    with log.open('wb') as output:
        process=subprocess.Popen([str(binary),'--no-defaults','--datadir='+str(folder),'--bind-address=127.0.0.1','--port='+str(port),'--socket='+str(socket_path),'--mysqlx=OFF','--local-infile=0','--skip-name-resolve'],stdout=output,stderr=output)
        try:
            import pymysql
            deadline=__import__('time').monotonic()+40;connection=None
            while time.monotonic()<deadline:
                if process.poll() is not None:raise AssertionError('Ephemeral MySQL exited: '+log.read_text()[-1000:])
                try:connection=pymysql.connect(unix_socket=str(socket_path),user='root',password='',ssl_disabled=True,connect_timeout=1);break
                except pymysql.err.OperationalError:time.sleep(.1)
            if connection is None:raise AssertionError('Ephemeral MySQL did not become ready')
            try:
                with connection.cursor() as cursor:
                    cursor.execute('CREATE DATABASE learning')
                    cursor.execute('CREATE TABLE learning.chosen (id BIGINT, label VARCHAR(40), amount DECIMAL(10,2), moment DATE)')
                    cursor.execute("INSERT INTO learning.chosen VALUES (1,'one',1.25,'2026-10-13'),(2,'two',2.50,'2026-10-14')")
                    cursor.execute('CREATE TABLE learning.private (secret TEXT)')
                    cursor.execute("CREATE USER 'kwip_test'@'127.0.0.1' IDENTIFIED BY ''")
                    cursor.execute("GRANT SELECT ON learning.chosen TO 'kwip_test'@'127.0.0.1'")
                connection.commit()
            finally:connection.close()
            yield dict(backend='mysql',host='127.0.0.1',port=port,database='learning',schema='learning',table='chosen',tlsMode='disabledLoopback'),dict(username='kwip_test',password='')
        finally:
            if process.poll() is None:process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            for suffix in ('','.lock'):
                try:Path(str(socket_path)+suffix).unlink()
                except FileNotFoundError:pass


def test_actual_mysql_bounded_selected_table_and_types(mysql):
    config,secret=mysql;sink=io.BytesIO()
    receipt=db.capture(dict(config,columns=['label','amount','moment'],orderBy=['-id'],fetchBatchRows=1),secret,sink)
    assert receipt['rows']==2
    assert decode(sink)=={'label':['two','one'],'amount':['2.50','1.25'],'moment':['2026-10-14','2026-10-13']}


def test_actual_native_broker_postgres_if_requested(postgres,tmp_path):
    import os,subprocess
    executable=os.environ.get('KWIPML_DB_PROBE_EXECUTABLE')
    if not executable:pytest.skip('Root runs packaged native XPC probe separately')
    config,secret=postgres
    result=subprocess.run([executable,str(tmp_path/'native-capture')],input=json.dumps({'options':config,'credentials':secret,'preview':False}),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,(result.stdout,result.stderr)
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]['event']=='databaseCaptureVerified' and events[-1]['receipt']['rows']==2

@pytest.mark.parametrize('revocation',['cancel','parentDeath','helperDeath'])
def test_actual_native_broker_join_and_crash_revocation_if_requested(postgres,tmp_path,revocation):
    import os,subprocess,time,selectors,signal
    executable=os.environ.get('KWIPML_DB_PROBE_EXECUTABLE')
    if not executable:pytest.skip('Root runs packaged native XPC probe separately')
    config,secret=postgres
    import psycopg
    with psycopg.connect(host=config['host'],port=config['port'],dbname=config['database'],user=secret['username']) as connection:
        connection.execute('CREATE VIEW slowly AS SELECT i, pg_sleep(0.05)::text AS delayed FROM generate_series(1,1000) AS i')
    request={'options':dict(config,table='slowly'),'credentials':secret,'preview':False}
    if revocation=='cancel':request['cancelAfterMilliseconds']=1500
    process=subprocess.Popen([executable,str(tmp_path/'native-capture')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    worker=None
    try:
        process.stdin.write(json.dumps(request));process.stdin.close()
        selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
        deadline=time.monotonic()+10;events=[]
        while time.monotonic()<deadline:
            if not selector.select(.2):continue
            line=process.stdout.readline()
            if not line:break
            value=json.loads(line);events.append(value)
            if value['event']=='databaseWorker':
                worker=value['workerPID']
                if revocation=='parentDeath':process.kill()
                elif revocation=='helperDeath':os.kill(value['servicePID'],signal.SIGKILL)
                break
        assert worker is not None,events
        process.wait(timeout=5)
        events.extend(json.loads(line) for line in process.stdout.read().splitlines())
        assert not any(value['event']=='databaseCaptureVerified' for value in events)
        if revocation=='cancel':assert {'event':'cancelAcknowledged','joined':True} in events
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            try:os.kill(worker,0)
            except ProcessLookupError:break
            time.sleep(.05)
        else:pytest.fail('Revoked native database worker survived its owner')
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=5)
        process.stdout.close();process.stderr.close()

def test_actual_native_broker_rejects_mismatched_signature_if_requested(postgres,tmp_path):
    import os,subprocess
    executable=os.environ.get('KWIPML_DB_PROBE_EXECUTABLE')
    if not executable:pytest.skip('Root runs packaged native XPC probe separately')
    config,secret=postgres
    request={'options':config,'credentials':secret,'testMismatchedPeerSignature':True}
    result=subprocess.run([executable,str(tmp_path/'rejected-capture')],input=json.dumps(request),capture_output=True,text=True,timeout=10)
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode!=0 and events[-1]['event']=='captureFailed',events
    assert not any(value['event'] in ('databaseWorker','databaseCaptureVerified') for value in events)

def test_sqlite_prefix_sampling_pushes_read_limit_without_changing_ids(table,monkeypatch):
    config={'format':'database','options':{'table':'chosen'}}
    full=db.source_table(table,config)
    calls=[];original=db.capture
    def measured(options,*args,**kwargs):calls.append(options['rowLimit']);return original(options,*args,**kwargs)
    monkeypatch.setattr(db,'capture',measured)
    sampled=db.source_table(table,config,sampling={'schemaVersion':1,'mode':'rowCount','rowCount':2})
    assert calls==[2] and len(sampled.rows)==2
    assert list(sampled.row_ids)==list(full.row_ids)[:2]

def test_sqlite_weak_type_integer_never_silently_truncates_float(table):
    with sqlite3.connect(table) as connection:
        connection.execute('CREATE TABLE inconsistent (value INTEGER)')
        connection.execute('INSERT INTO inconsistent VALUES (1.5)')
    with pytest.raises(db.DatabaseSourceError,match='no lossy conversion'):
        db.capture(dict(table='inconsistent'),{},io.BytesIO(),sqlite_path=table)

def test_ipc_snapshot_is_bound_to_exact_public_source_selection(table,tmp_path):
    config=dict(table='chosen',columns=['id'],orderBy=['+id'])
    path=tmp_path/'capture.ipc'
    with path.open('wb') as sink:db.capture(config,{},sink,sqlite_path=table)
    assert db.preview_table(path,config)[1]==[[1],[2],[3]]
    for change in ({'table':'private'},{'columns':['label']},{'orderBy':['-id']},
                   {'rowLimit':1},{'host':'other.example'},{'tlsMode':'disabledLoopback','host':'127.0.0.1'},
                   {'binaryMode':'hex'},{'database':'different'}):
        with pytest.raises(db.DatabaseSourceError,match='Reconnect or refresh'):
            db.preview_table(path,{**config,**change})
    assert db.preview_table(path,{**config,'fetchBatchRows':1,'timeoutSeconds':90})[1]==[[1],[2],[3]]

def test_dedicated_capture_constrains_native_engine_pools_before_imports():
    import subprocess,sys
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'database_source_capture.py'
    script='''import runpy,sys,os,json
module=runpy.run_path(sys.argv[1])
assert not any(name in sys.modules for name in ('polars','numpy','pyarrow','psycopg','pymysql'))
assert all(os.environ[name]=='1' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS','VECLIB_MAXIMUM_THREADS'))
module['_constrain_arrow']()
import pyarrow,polars
print(json.dumps({'arrowCPU':pyarrow.cpu_count(),'arrowIO':pyarrow.io_thread_count(),'polars':polars.thread_pool_size()}))
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(path)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'arrowCPU':1,'arrowIO':1,'polars':1}

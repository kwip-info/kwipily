"""Disposable real database write effects; no production credentials or destinations."""
import json,sqlite3,uuid
import pytest
from runner.workload_policy import WorkloadPolicy
from runner import database_output as db
from runner.tests.test_database_source import postgres,mysql

ROWS=[[1,'first',1.25],[2,'two',None]]
COLS=[{'name':'id','dtype':'int64','nullable':False},{'name':'text','dtype':'string','nullable':False},{'name':'score','dtype':'float64','nullable':True}]
def config(backend='sqlite',**kw):
    types=['INTEGER','TEXT','REAL'] if backend=='sqlite' else ['BIGINT','TEXT','DOUBLE PRECISION'] if backend=='postgresql' else ['BIGINT','TEXT','DOUBLE']
    return dict(backend=backend,table='predictions',mode='create',keys=['record_id'],mapping=[dict(source=x['name'],target=t,type=k,nullable=x['nullable']) for x,t,k in zip(COLS,['record_id','label','amount'],types)],**kw)
def write(path,c,rows=ROWS,secret=None,**kw):
    return db.write(c,secret or {},COLS,rows,operation_id=str(uuid.uuid4()).upper(),sqlite_path=path,**kw)

def test_sqlite_create_append_upsert_and_readonly_metadata(tmp_path):
    path=tmp_path/'out.sqlite';sqlite3.connect(path).close();c=config(batchRows=1)
    assert write(path,c)['status']=='committed'
    before=path.read_bytes();m=db.inspect(c,{},sqlite_path=path)
    assert m['exists'] and [x['name'] for x in m['columns']]==['record_id','label','amount']
    assert m['keys'][0]['columns']==['record_id'] and path.read_bytes()==before
    assert write(path,{**c,'mode':'append'},[[3,'three',3.5]])['status']=='committed'
    assert write(path,{**c,'mode':'upsert'},[[2,'updated',9.5]])['status']=='committed'
    with sqlite3.connect(path) as con:assert con.execute('SELECT * FROM predictions ORDER BY record_id').fetchall()==[(1,'first',1.25),(2,'updated',9.5),(3,'three',3.5)]

def test_create_never_replaces_existing_and_null_duplicate_keys_are_refused(tmp_path):
    p=tmp_path/'db.sqlite';sqlite3.connect(p).close();c=config();write(p,c)
    for rows in ([[1,'a',1.0],[1,'b',2.0]],[[None,'x',1.0]]):
        r=write(p,{**c,'mode':'upsert'},rows);assert r['status']=='notStarted'
    assert write(p,c)['status']=='notStarted'
    with sqlite3.connect(p) as con:assert con.execute('SELECT count(*) FROM predictions').fetchone()[0]==2

def test_transaction_rolls_back_all_batches_and_cancel_does_not_commit(tmp_path):
    p=tmp_path/'db.sqlite';sqlite3.connect(p).close();c=config(batchRows=1);write(p,c)
    with sqlite3.connect(p) as con:con.execute("CREATE TRIGGER refuse BEFORE INSERT ON predictions WHEN NEW.record_id=4 BEGIN SELECT RAISE(ABORT,'private-driver-secret'); END")
    r=write(p,{**c,'mode':'append'},[[3,'x',1.0],[4,'y',2.0]])
    assert r['status']=='rolledBack' and 'private-driver-secret' not in json.dumps(r)
    calls=[]
    def cancel():
        calls.append(1)
        if len(calls)>5:raise InterruptedError()
    r=write(p,{**c,'mode':'append'},[[5,'x',1.0],[6,'y',2.0]],check_cancelled=cancel)
    assert r['status'] in ('notStarted','rolledBack')
    with sqlite3.connect(p) as con:assert con.execute('SELECT count(*) FROM predictions').fetchone()[0]==2

def test_lost_ack_is_unknown_even_when_rows_committed(tmp_path):
    p=tmp_path/'db.sqlite';sqlite3.connect(p).close()
    def lost():raise ConnectionError('do-not-print-this')
    r=write(p,config(),after_commit=lost)
    assert r['status']=='unknown' and 'do-not-print' not in json.dumps(r)
    with sqlite3.connect(p) as con:assert con.execute('SELECT count(*) FROM predictions').fetchone()[0]==2

@pytest.mark.parametrize('bad',[{'mode':'replace'},{'query':'DROP TABLE predictions'},{'batchRows':True},{'mapping':[dict(source='id',target='x',type='TEXT; DROP TABLE x',nullable=False)]}])
def test_strict_options(bad):
    with pytest.raises(ValueError):db.options({**config(),**bad})

def test_mapping_known_types_lossless_nulls_and_no_implicit_schema_alter(tmp_path):
    p=tmp_path/'db.sqlite';sqlite3.connect(p).close();c=config();write(p,c)
    bad=[{**c,'mode':'append','mapping':[dict(source='id',target='missing',type='INTEGER',nullable=False)]},{**c,'mode':'upsert','keys':['label']}]
    for b in bad:assert write(p,b)['status']=='notStarted'
    assert write(p,{**c,'mode':'append'},[[2**63,'overflow',1.0]])['status']=='notStarted'
    assert write(p,{**c,'mode':'append'},[[3,'inf',float('inf')]])['status']=='notStarted'
    link=tmp_path/'link.sqlite';link.symlink_to(p)
    assert write(link,c)['status']=='notStarted'

@pytest.mark.parametrize('fixture,backend',[('postgres','postgresql'),('mysql_writer','mysql')])
def test_real_remote_round_trip_and_rollback(request,fixture,backend):
    source,secret=request.getfixturevalue(fixture)
    connection={k:v for k,v in source.items() if k in ('host','port','database','schema','tlsMode')}
    c=config(backend,**connection,batchRows=1)
    assert write(None,c,secret=secret)['status']=='committed'
    assert write(None,{**c,'mode':'append'},[[3,'three',3.0]],secret)['status']=='committed'
    assert write(None,{**c,'mode':'upsert'},[[2,'changed',8.0]],secret)['status']=='committed'
    m=db.inspect(c,secret);assert m['exists'] and m['keys']
    r=write(None,{**c,'mode':'append'},[[4,'four',4.0],[1,'duplicate',5.0]],secret)
    assert r['status']=='rolledBack'
    from runner.database_source import selected_cursor
    with selected_cursor({**source,'table':'predictions','columns':['record_id','label','amount'],'orderBy':['record_id']},secret) as (cursor,_,_):
        got=cursor.fetchmany(10)
    assert [tuple(row) for row in got]==[(1,'first',1.25),(2,'changed',8.0),(3,'three',3.0)]

@pytest.fixture
def mysql_writer(tmp_path):
    # Reuse the disposable source fixture, adding rights ONLY to that test
    # server over its generated private Unix socket. Source fixture stays readonly.
    import pymysql
    from pathlib import Path
    fixture=mysql.__wrapped__(tmp_path)
    config,secret=next(fixture)
    try:
        pid=(tmp_path/'mysql-data'/'mysqld.pid')
        # The server advertises its fixed test socket in its private data directory.
        logs=(tmp_path/'mysql.log').read_text(errors='replace')
        import re
        matches=re.findall(r"socket: '([^']+)'",logs)
        assert matches,'Disposable MySQL socket was not recorded'
        con=pymysql.connect(unix_socket=matches[-1],user='root',password='',ssl_disabled=True)
        with con.cursor() as cur:cur.execute("GRANT SELECT,INSERT,UPDATE,CREATE ON learning.* TO 'kwip_test'@'127.0.0.1'")
        con.close()
        yield config,secret
    finally:
        try:next(fixture)
        except StopIteration:pass

def test_destination_identifier_limits_and_case_collisions():
    c=config();c['mapping'][1]['target']='RECORD_ID'
    with pytest.raises(ValueError):db.options(c)
    c=config('postgresql');c['table']='x'*64
    with pytest.raises(ValueError):db.options(c)

def test_omitted_sqlite_rowid_uses_server_default(tmp_path):
    p=tmp_path/'db.sqlite'
    with sqlite3.connect(p) as con:con.execute('CREATE TABLE items(id INTEGER PRIMARY KEY, label TEXT NOT NULL)')
    c=dict(backend='sqlite',table='items',mode='append',mapping=[dict(source='text',target='label',type='TEXT',nullable=False)])
    assert write(p,c)['status']=='committed'
    with sqlite3.connect(p) as con:assert con.execute('SELECT * FROM items ORDER BY id').fetchall()==[(1,'first'),(2,'two')]

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('workload',[None,'legacy','balanced'])
def test_preparation_defers_database_effects_but_retains_complete_table(tmp_path,engine,workload):
    from runner.dataflow_cli import run_dataflow
    source=tmp_path/'source.csv';source.write_text('id,text,score\n'+''.join(f'{i},row{i},1.25\n' for i in range(12)))
    g={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}},{'id':'save','operator':'kwip.data.outputDelimited','config':dict(destination='database',database=config(),filename='old.csv',columns=[],separator=',',overwrite=False)}],'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'save','port':'table'}}]}
    graph=tmp_path/'graph.json';graph.write_text(json.dumps(g));events=[]
    run_dataflow(graph,{'source':str(source)},tmp_path/'archive',events.append,engine=engine,defer_database_writes=True,retain_outputs=False,sampling={'mode':'rowCount','rowCount':3},
                 **({} if workload is None else {'workload_policy':WorkloadPolicy(workload).to_document()}))
    manifest=json.loads((tmp_path/'archive'/'manifest.json').read_text())
    artifact=next(a for a in manifest['artifacts'] if a.get('nodeID')=='save' and a['kind']=='table')
    captured=json.loads((tmp_path/'archive'/artifact['path']).read_text())
    assert len(captured['rows'])==12
    assert not manifest.get('outputs')
    # An ordinary exported script must refuse rather than claim a write.
    with pytest.raises(Exception,match='(?i)(database|broker)'):
        run_dataflow(graph,{'source':str(source)},tmp_path/'refused',lambda e:None,engine=engine)

def test_actual_broker_child_pins_readonly_table_and_never_rewrites_input(tmp_path):
    import hashlib,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    # Uses the real native parent-lifetime watchdog; retain stdin through child exit.
    if not (root/'.local/native-watchdog/lifetime_watchdog.dylib').exists():pytest.skip('Build the native development lifetime watchdog first')
    p=tmp_path/'out.sqlite';sqlite3.connect(p).close()
    table=tmp_path/'input.json';table.write_text(json.dumps(dict(kind='table',columns=COLS,rows=ROWS)))
    digest=hashlib.sha256(table.read_bytes()).hexdigest()
    body=dict(schemaVersion=1,action='write',options=config(),credentials={},operationID=str(uuid.uuid4()).upper(),tableSHA256=digest,sqlitePath=str(p))
    raw=json.dumps(body).encode()
    with table.open('rb') as read:
        child=subprocess.Popen([sys.executable,'-I','-B',str(root/'runner/database_output_service.py')],stdin=subprocess.PIPE,stdout=read,stderr=subprocess.PIPE)
        try:
            child.stdin.write(len(raw).to_bytes(4,'big')+raw);child.stdin.flush()
            child.wait(timeout=15);receipt=json.loads(child.stderr.read())
        finally:
            if child.poll() is None:child.kill();child.wait()
            child.stdin.close();child.stderr.close()
    assert receipt['status']=='committed',receipt
    assert receipt['sourceSHA256']==digest
    assert hashlib.sha256(table.read_bytes()).hexdigest()==digest
    with sqlite3.connect(p) as con:assert con.execute('SELECT count(*) FROM predictions').fetchone()[0]==2

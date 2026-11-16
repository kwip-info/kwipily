import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from runner import api_tables as api
from runner.workload_policy import WorkloadPolicy

@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        calls=[]
        def log_message(self,*args):pass
        def do_GET(self):
            Handler.calls.append(self.path)
            if self.path.startswith('/redirect'):
                self.send_response(302);self.send_header('Location','http://169.254.169.254/latest');self.end_headers();return
            if self.path.startswith('/auth-check') and self.headers.get('Authorization') not in ('Bearer synthetic','Basic dXNlcjpzeW50aGV0aWM=') and self.headers.get('X-API-Key')!='synthetic':
                self.send_response(401);self.end_headers();return
            if self.path.startswith('/rate') and len(Handler.calls)==1:
                self.send_response(429);self.send_header('Retry-After','0');self.end_headers();return
            if self.path.startswith('/malformed'):
                self.send_response(200);self.end_headers();self.wfile.write(b'not JSON');return
            if self.path.startswith('/empty'):
                self.send_response(200);self.end_headers();self.wfile.write(b'{"items":[]}');return
            if self.path.startswith('/cross'):
                self.send_response(200);self.end_headers();self.wfile.write(b'{"items":[],"next":"https://other.example.com/page"}');return
            body={'items':[{'id':len(Handler.calls),'name':'Alice','created_at':'2026-10-19T12:00:00Z'}],'next':None if 'cursor=two' in self.path else 'two'}
            raw=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_POST(self):
            body=self.rfile.read(int(self.headers['Content-Length']));Handler.calls.append(json.loads(body))
            if self.path=='/lost':self.connection.shutdown(2);self.connection.close();return
            self.send_response(400 if self.path=='/reject-second' and len(Handler.calls)==2 else 202);self.end_headers();self.wfile.write(b'{"accepted":true}')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    yield f'http://127.0.0.1:{server.server_port}',Handler.calls
    server.shutdown();server.server_close();t.join()

def config(url,**extra):return api.options(dict(url=url,allowLoopback=True,recordsPath='/items',mapping=[dict(source='/id',target='order_id',type='int64')],**extra))

def test_cursor_snapshot_and_types(endpoint):
    url,calls=endpoint;c=config(url,pagination='cursor',cursorPath='/next',cursorParameter='cursor')
    columns,rows,receipt=api.fetch(c,{})
    assert rows==[[1],[2]] and columns[0]['name']=='order_id' and receipt['pages']==2 and receipt['complete'] is True
    assert calls==['/','/?cursor=two']

def test_pagination_limit_is_not_complete(endpoint):
    url,_=endpoint
    with pytest.raises(api.APIError):api.fetch(config(url,pagination='cursor',cursorPath='/next',maximumPages=1),{})

def test_no_redirect_or_metadata_destination(endpoint):
    url,_=endpoint
    with pytest.raises(api.APIError):api.fetch(config(url+'/redirect'),{})
    for address in ['http://example.com','https://169.254.169.254/latest','https://user:secret@example.com','file:///etc/passwd']:
        with pytest.raises(api.APIError):api.options({'url':address})

def test_missing_field_is_not_null_unless_selected(endpoint):
    url,_=endpoint;c=config(url);c['mapping'][0]['source']='/absent'
    with pytest.raises(api.APIError):api.fetch(c,{})
    c['missing']='null';assert api.fetch(c,{})[1]==[[None]]

def test_output_mapping_batch_accepted_and_unknown(endpoint):
    url,calls=endpoint;c=api.options(dict(url=url,method='POST',allowLoopback=True,direction='output',delivery='batch',batchRows=2,arrayPath='/predictions',mapping=[dict(source='score',target='/result/score',type='float64')]))
    r=api.deliver(c,{},[{'name':'score','dtype':'float64'}],[[.5],[.75],[.9]],operation_id='op')
    assert r['status']=='committed' and r['acceptedRows']==3 and len(r['requests'])==2
    assert calls[0]=={'predictions':[{'result':{'score':.5}},{'result':{'score':.75}}]}
    c['url']=url+'/lost';r=api.deliver(c,{},[{'name':'score','dtype':'float64'}],[[.5]],operation_id='op2')
    assert r['status']=='unknown' and r['requests'][0]['outcome']=='unknown'
    assert len(calls)==3 # never retry POST

@pytest.mark.parametrize('extra',[{'maximumPages':True},{'headers':{'Authorization':'secret'}},{'query':{'token':'secret'}},{'timeoutSeconds':0},{'mapping':[{'source':'/id','target':'x','type':'image'}]},{'retries':99}])
def test_strict_contract(extra):
    with pytest.raises(api.APIError):api.options(dict(url='https://api.example.com',**extra))

def test_json_pointer_escape_and_cast_is_lossless():
    assert api.path({'a/b':{'~c':[4]}},'/a~1b/~0c/0')==4
    for value,dtype in [(True,'int64'),(1.5,'int64'),(2**54,'float64'),(float('inf'),'float64'),('2026-01-01T00:00:00','datetime')]:
        with pytest.raises(api.APIError):api.scalar(value,dtype)
    assert api.scalar('2026-01-01T01:00:00+01:00','datetime')=='2026-01-01T00:00:00+00:00'

def test_validate_all_output_rows_before_first_side_effect(endpoint):
    url,calls=endpoint;c=api.options(dict(direction='output',method='POST',url=url,allowLoopback=True,mapping=[dict(source='id',target='/id',type='int64')]))
    with pytest.raises(api.APIError):api.deliver(c,{},[{'name':'id','dtype':'int64'}],[[1],['bad']],operation_id='never')
    assert calls==[]
    with pytest.raises(api.APIError):api.deliver({**c,'maximumRequests':1},{},[{'name':'id','dtype':'int64'}],[[1],[2]],operation_id='never')
    assert calls==[]

def test_output_cancellation_stops_without_retry(endpoint):
    url,calls=endpoint;c=api.options(dict(direction='output',method='POST',url=url,allowLoopback=True,mapping=[dict(source='id',target='/id',type='int64')]))
    def stop():raise InterruptedError()
    with pytest.raises(InterruptedError):api.deliver(c,{},[{'name':'id','dtype':'int64'}],[[1]],'cancel',stop)
    assert calls==[]

def test_source_snapshot_metadata_pins_config(tmp_path):
    import pyarrow as pa,pyarrow.ipc as ipc
    from runner.api_table_snapshot import identity,verified,preview_evidence
    c=config('http://127.0.0.1:12345');target=tmp_path/'api.arrow'
    table=pa.table({'order_id':[1,2]}).replace_schema_metadata({b'kwip.api.selection.v1':identity(c)})
    with ipc.new_file(target,table.schema) as w:w.write_table(table)
    verified(target,c)
    with pytest.raises(api.APIError):verified(target,{**c,'recordsPath':'/different'})
    assert preview_evidence(target,{'format':'api','options':c})['rows']==[[1],[2]]

@pytest.mark.parametrize('workload',[None,'legacy','balanced'])
def test_api_graph_defers_side_effects_and_captures_complete_input(tmp_path,workload):
    from runner.dataflow_cli import run_dataflow
    c=api.options(dict(direction='output',method='POST',url='https://api.example.com',mapping=[dict(source='id',target='/id',type='string')]))
    source=tmp_path/'source.csv';source.write_text('id\n1\n2\n3\n4\n')
    graph=tmp_path/'graph.json';graph.write_text(json.dumps({'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}},{'id':'save','operator':'kwip.data.outputDelimited','config':dict(destination='api',api=c,filename='old.csv',columns=[],separator=',',overwrite=False)}],'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'save','port':'table'}}]}))
    events=[]
    run_dataflow(graph,{'source':str(source)},tmp_path/'archive',events.append,engine='polars',defer_database_writes=True,retain_outputs=False,sampling={'mode':'rowCount','rowCount':1},
                 **({} if workload is None else {'workload_policy':WorkloadPolicy(workload).to_document()}))
    manifest=json.loads((tmp_path/'archive'/'manifest.json').read_text())
    assert any(e.get('event')=='dataflowCompleted' for e in events)
    # Output must request complete upstream data even with capture off / sampled work.
    artifact=next(a for a in manifest["artifacts"] if a.get("nodeID")=="save" and a["kind"]=="table")
    assert len(json.loads((tmp_path/"archive"/artifact["path"]).read_text())["rows"])==4
    with pytest.raises(Exception):run_dataflow(graph,{'source':str(source)},tmp_path/'refused',lambda _:None,engine='polars')

def test_actual_broker_child_captures_pinned_snapshot(endpoint,tmp_path):
    import subprocess,sys,hashlib
    from pathlib import Path
    url,calls=endpoint;c=config(url,pagination='cursor',cursorPath='/next',cursorParameter='cursor')
    c['mapping'].append(dict(source='/created_at',target='created_at',type='datetime'))
    payload=json.dumps(dict(schemaVersion=1,options=c,credentials={'username':'','password':''},preview=False)).encode()
    runner=Path(__file__).parents[1];target=tmp_path/'capture.arrow'
    with target.open('xb') as sink:
        p=subprocess.Popen([sys.executable,'-I','-B',str(runner/'api_table_service.py')],stdin=subprocess.PIPE,stdout=sink,stderr=subprocess.PIPE)
        try:
            p.stdin.write(len(payload).to_bytes(4,'big')+payload);p.stdin.flush();raw=p.stderr.read(65537);p.wait(timeout=15)
        finally:
            p.stdin.close();p.stderr.close()
            if p.poll() is None:p.kill();p.wait()
    assert p.returncode==0,raw
    receipt=json.loads(raw);assert receipt['sha256']==hashlib.sha256(target.read_bytes()).hexdigest() and receipt['rows']==2 and receipt['apiFetch']['pages']==2
    from runner.api_table_snapshot import verified
    verified(target,c)
    from runner.data import inspect_data
    preview=inspect_data(target,table_options={'format':'api','options':c})
    assert preview['apiFetch']['pages']==2 and preview['columnSchema'][1]['dtype']=='datetime'
    assert len(preview['rows'])==2
    from runner.api_table_snapshot import source_table
    table=source_table(target,{'format':'api','options':c})
    try:
        output=api.options(dict(direction='output',method='POST',url=url,allowLoopback=True,mapping=[dict(source='created_at',target='/created_at',type='datetime')]))
        assert api.deliver(output,{},list(table.columns),[list(r) for r in table.rows],'temporal-roundtrip')['acceptedRows']==2
        assert calls[-1]=={'created_at':'2026-10-19T12:00:00+00:00'}
    finally:table.rows.store.close()

def test_actual_broker_child_posts_complete_readonly_capture(endpoint,tmp_path):
    import subprocess,sys,hashlib
    from pathlib import Path
    url,calls=endpoint;c=api.options(dict(url=url,method='POST',direction='output',allowLoopback=True,mapping=[dict(source='id',target='/id',type='int64')]))
    table=json.dumps(dict(kind='table',columns=[dict(name='id',dtype='int64')],rows=[[1],[2]])).encode();source=tmp_path/'table.json';source.write_bytes(table)
    payload=json.dumps(dict(schemaVersion=1,action='write',options=c,operationID='00000000-0000-4000-8000-000000000001',credentials={'username':'','password':''},tableSHA256=hashlib.sha256(table).hexdigest(),sqlitePath=None)).encode()
    with source.open('rb') as stream:
        p=subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).parents[1]/'api_table_service.py')],stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.PIPE)
        try:
            p.stdin.write(len(payload).to_bytes(4,'big')+payload);p.stdin.flush();raw=p.stderr.read(65537);p.wait(timeout=15)
        finally:
            p.stdin.close();p.stderr.close()
            if p.poll() is None:p.kill();p.wait()
    assert p.returncode==0,raw
    receipt=json.loads(raw);assert receipt['status']=='committed' and receipt['acceptedRows']==2 and calls==[{'id':1},{'id':2}]


@pytest.mark.parametrize('auth,secret',[('bearer',{'password':'synthetic'}),('basic',{'username':'user','password':'synthetic'}),('apiKey',{'password':'synthetic'})])
def test_private_auth_modes_and_expired_credential(endpoint,auth,secret):
    url,calls=endpoint;c=config(url+'/auth-check',auth=auth)
    assert len(api.fetch(c,secret)[1])==1
    with pytest.raises(api.APIError):api.fetch(c,{'password':'expired'})
    assert len(calls)==2 # no retry for authentication failure

def test_rate_limit_retry_and_malformed_response(endpoint):
    url,calls=endpoint
    assert api.fetch(config(url+'/rate'),{})[1]==[[2]]
    assert len(calls)==2
    with pytest.raises(api.APIError):api.fetch(config(url+'/malformed'),{})
    assert len(calls)==3

def test_empty_snapshot_and_same_origin_next_link_policy(endpoint):
    url,_=endpoint
    columns,rows,receipt=api.fetch(config(url+'/empty'),{})
    assert rows==[] and columns[0]['name']=='order_id' and receipt['complete'] is True
    with pytest.raises(api.APIError):api.fetch(config(url+'/cross',pagination='nextLink',nextPath='/next'),{})

def test_confirmed_partial_failure_preserves_accepted_request_evidence(endpoint):
    url,calls=endpoint;c=api.options(dict(url=url+'/reject-second',method='POST',direction='output',allowLoopback=True,mapping=[dict(source='id',target='/id',type='int64')]))
    receipt=api.deliver(c,{},[dict(name='id',dtype='int64')],[[1],[2],[3]],'partial')
    assert receipt['acceptedRows']==1 and receipt['status']=='unknown' and len(calls)==2
    assert [r['outcome'] for r in receipt['requests']]==['accepted','rejected']

def test_constants_and_nested_json_payloads(endpoint):
    url,calls=endpoint;c=api.options(dict(url=url,method='POST',direction='output',allowLoopback=True,mapping=[dict(source='id',target='/record/id',type='int64')],constants=[dict(target='/record/source',json='"canvas"'),dict(target='/version',json='1')]))
    assert api.deliver(c,{},[dict(name='id',dtype='int64')],[[1]],'constant')['status']=='committed'
    assert calls==[{'record':{'id':1,'source':'canvas'},'version':1}]
    with pytest.raises(api.APIError):api.options({**c,'constants':[dict(target='/record',json='null')]})
    with pytest.raises(api.APIError):api.options({**c,'constants':[dict(target='/bad',json='not-json')]})


def test_broker_memory_budget_stops_before_a_request(endpoint,tmp_path):
    import subprocess,sys
    from pathlib import Path
    url,calls=endpoint;c=config(url)
    raw=json.dumps(dict(schemaVersion=1,options=c,credentials={'username':'','password':''},preview=False,memoryLimit=1_000_000,operationSeconds=5)).encode()
    with (tmp_path/'capture.arrow').open('xb') as target:
        p=subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).parents[1]/'api_table_service.py')],stdin=subprocess.PIPE,stdout=target,stderr=subprocess.PIPE)
        try:
            p.stdin.write(len(raw).to_bytes(4,'big')+raw);p.stdin.flush();p.stderr.read(65537);p.wait(timeout=10)
        finally:
            p.stdin.close();p.stderr.close()
            if p.poll() is None:p.kill();p.wait()
    assert p.returncode==137 and calls==[]


def test_output_preserves_canonical_temporal_values_and_refuses_artifact_columns(endpoint):
    url,calls=endpoint
    c=api.options(dict(direction='output',method='POST',url=url,allowLoopback=True,mapping=[dict(source='created',target='/created',type='datetime'),dict(source='day',target='/day',type='string'),dict(source='clock',target='/clock',type='string')]))
    columns=[dict(name='created',dtype='datetime',temporal={'timezone':'UTC'}),dict(name='day',dtype='date'),dict(name='clock',dtype='time')]
    receipt=api.deliver(c,{},columns,[[0,0,12_345_678]],'temporal')
    assert receipt['status']=='committed'
    assert calls==[{'created':'1970-01-01T00:00:00+00:00','day':'1970-01-01','clock':'00:00:12.345678'}]
    c['mapping']=[dict(source='asset',target='/asset',type='json')]
    for dtype,value in [('image',{'path':'private.png'}),('int64Sequence',[1,2])]:
        with pytest.raises(api.APIError,match='scalar representation'):
            api.deliver(c,{},[dict(name='asset',dtype=dtype)],[[value]],'refuse')
    assert len(calls)==1


def test_output_json_encoding_preserves_scalar_types(endpoint):
    url,calls=endpoint
    c=api.options(dict(direction='output',method='POST',url=url,allowLoopback=True,mapping=[dict(source='value',target='/value',type='json')]))
    for dtype,value in [('int64',42),('bool',True),('float64',.5),('string','{"nested":[1,2]}')]:
        assert api.deliver(c,{},[dict(name='value',dtype=dtype)],[[value]],'scalar-json')['status']=='committed'
    assert calls==[{'value':42},{'value':True},{'value':.5},{'value':{'nested':[1,2]}}]


@pytest.mark.parametrize("roots,platform,override,loads", [(0,"darwin",False,True),(1,"darwin",False,False),(0,"linux",False,False),(0,"darwin",True,False)])
def test_verified_tls_uses_platform_roots_only_when_default_roots_absent(monkeypatch,roots,platform,override,loads):
    class Context:
        verify_mode=api.ssl.CERT_REQUIRED
        check_hostname=True
        loaded=[]
        def get_ca_certs(self):return [{}]*roots
        def load_verify_locations(self,*,cafile):self.loaded.append(cafile)
    context=Context()
    monkeypatch.setattr(api.ssl,"create_default_context",lambda:context)
    monkeypatch.setattr(api.sys,"platform",platform)
    monkeypatch.delenv("SSL_CERT_FILE",raising=False);monkeypatch.delenv("SSL_CERT_DIR",raising=False)
    if override:monkeypatch.setenv("SSL_CERT_FILE","/explicit/ca.pem")
    assert api.verified_tls_context() is context
    assert context.loaded == (["/etc/ssl/cert.pem"] if loads else [])
    assert context.verify_mode==api.ssl.CERT_REQUIRED and context.check_hostname


def test_platform_root_load_failure_remains_closed(monkeypatch):
    class Context:
        def get_ca_certs(self):return []
        def load_verify_locations(self,**kwargs):raise OSError("platform roots unavailable")
    monkeypatch.setattr(api.ssl,"create_default_context",lambda:Context())
    monkeypatch.setattr(api.sys,"platform","darwin")
    monkeypatch.delenv("SSL_CERT_FILE",raising=False);monkeypatch.delenv("SSL_CERT_DIR",raising=False)
    with pytest.raises(OSError,match="platform roots unavailable"):api.verified_tls_context()

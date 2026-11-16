import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from runner.table_view import build_index
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


@pytest.fixture
def view_operation(operation_authority):
    f=operation_authority;request=f[1]
    for key in ('runID','directoryID','phase'):del request[key]
    folder=f[0]/'input';folder.mkdir()
    table=SimpleNamespace(columns=({'name':'value','dtype':'string','nullable':False},),rows=[('hello',),('world',)],row_ids=['0','1'])
    receipt=build_index(table,folder/'view.sqlite3',source_sha256='a'*64)
    query=dict(schemaVersion=1,offset=0,limit=100,search='',sourceSHA256='a'*64,indexSHA256=receipt['index']['sha256'])
    raw=encoded(query);(folder/'view-query.json').write_bytes(raw)
    request.update(schemaVersion=2,toolMode='tableViewQuery',arguments=['--query-table-view',str(folder/'view-query.json')],inputFiles=[dict(path='input/view-query.json',bytes=len(raw),sha256=digest(raw)),dict(path='input/view.sqlite3',**receipt['index'])])
    publish(f);return f


def test_view_authority_rechecks_pinned_index(view_operation):
    f=view_operation
    with claim(f) as authority:
        path=f[0]/'input/view.sqlite3'
        with path.open('r+b') as stream:stream.seek(-1,2);stream.write(b'!')
        with pytest.raises(ValueError):authority.check()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('with_registry',[False,True])
def test_real_guarded_query_is_pure_and_releases_scratch(view_operation,tmp_path,libraries,with_registry):
    f=view_operation;guard,_=libraries
    registry=tmp_path/'registry.json';registry.write_text('{}')
    path=entry(tmp_path,guard,registry=registry if with_registry else None)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    replacement="    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    status=main(check_cancelled=result.authority.check)\n    assert 'torch' not in sys.modules\n    sys.exit(status)"
    text=path.read_text();assert marker in text;path.write_text(text.replace(marker,replacement))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,(events,error)
    assert events[0]['toolMode']=='tableViewQuery'
    assert events[-1]['event']=='tableViewPage' and events[-1]['rows']==[['hello'],['world']]
    assert not list(f[0].glob('kwip-table-query-*'))


def test_view_refuses_excessive_time(view_operation):
    f=view_operation;f[1].update(executionBudget=dict(schemaVersion=1,requestedSeconds=60,maximumRunSeconds=60,source='application'),remainingSeconds=60);publish(f)
    with pytest.raises(ValueError):claim(f)


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('with_registry',[False,True])
def test_real_guarded_lazy_table_query_pins_full_recorded_population(view_operation,tmp_path,libraries,with_registry):
    from runner import dataflow as df
    f=view_operation;folder=f[0]/'input'
    value=df.derive(df.Table(({'name':'value','dtype':'int64','nullable':False},),tuple((i,) for i in range(251)),tuple('source:'+str(i) for i in range(251)),'a'*64))
    lineage={**value.evidence(),'dataVersion':value.data_version,'partitionDigest':'','fittedStateDigests':[],'diagnostics':[]}
    raw=encoded({'schemaVersion':1,'kind':'table','columns':value.columns,'rows':value.rows,'lineage':lineage})
    pin=digest(raw);(folder/'view.table.json').write_bytes(raw);(folder/'view.sqlite3').unlink()
    query=encoded(dict(schemaVersion=3,inputKind='table',offset=200,limit=100,search='',sourceSHA256=pin,indexSHA256=pin))
    (folder/'view-query.json').write_bytes(query)
    f[1]['inputFiles']=[dict(path='input/view-query.json',bytes=len(query),sha256=digest(query)),dict(path='input/view.table.json',bytes=len(raw),sha256=pin)]
    publish(f)
    guard,_=libraries
    registry=tmp_path/'registry.json';registry.write_text('{}')
    path=entry(tmp_path,guard,registry=registry if with_registry else None)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    replacement="    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    status=main(check_cancelled=result.authority.check)\n    assert 'torch' not in sys.modules\n    sys.exit(status)"
    text=path.read_text();assert marker in text;path.write_text(text.replace(marker,replacement))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,(events,error)
    page=events[-1]
    assert page['event']=='tableViewPage' and page['totalRows']==251 and page['rows']==[[i] for i in range(200,251)]
    index=page['materializedIndex'];assert index['path']=='artifacts/view.sqlite3'
    assert digest((f[0]/index['path']).read_bytes())==index['sha256']
    assert not list(f[0].glob('.table-rows-*')) and not list(f[0].glob('kwip-table-query-*'))

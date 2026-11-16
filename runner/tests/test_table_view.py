from contextlib import closing
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from runner.table_view import build_index, query_index


def table(count=250):
    return SimpleNamespace(columns=({'name':'id','dtype':'int64','nullable':False},{'name':'note','dtype':'string','nullable':True}),
                           rows=[(i, 'Café example' if i%2 else None) for i in range(count)],row_ids=[str(i) for i in range(count)])


def index(tmp_path,count=250):
    path=tmp_path/'view.sqlite3'
    receipt=build_index(table(count),path,source_sha256='a'*64)
    return path,receipt


def test_page_and_literal_casefold_search(tmp_path):
    path,receipt=index(tmp_path)
    pin=receipt['index']['sha256']
    first=query_index(path,pin,limit=100)
    assert first['totalRows']==250 and first['nextOffset']==100 and first['rows'][0]==[0,None]
    last=query_index(path,pin,offset=200)
    assert len(last['rows'])==50 and last['nextOffset'] is None
    searched=query_index(path,pin,search='CAFÉ')
    assert searched['matchedRows']==125 and searched['positions'][:3]==[1,3,5]
    assert query_index(path,pin,search="' OR 1=1 --")['rows']==[]


def test_tampered_replaced_symlink_and_wrong_table_are_refused(tmp_path):
    path,receipt=index(tmp_path)
    pin=receipt['index']['sha256']
    with pytest.raises(ValueError):query_index(path,pin,source_sha256='b'*64)
    raw=path.read_bytes();changed=bytearray(raw);changed[-1]^=1;path.write_bytes(changed)
    with pytest.raises(ValueError,match='integrity'):query_index(path,pin)
    path.write_bytes(raw)
    link=tmp_path/'alias.sqlite3';link.symlink_to(path)
    with pytest.raises((OSError,ValueError)):query_index(link,pin)
    with pytest.raises(ValueError):build_index(table(),path,source_sha256='a'*64)


def test_cancelled_build_and_query_leave_no_partial_output(tmp_path):
    def cancel():raise InterruptedError('cancelled')
    path=tmp_path/'view.sqlite3'
    with pytest.raises(InterruptedError):build_index(table(),path,source_sha256='a'*64,check_cancelled=cancel)
    assert not path.exists() and list(tmp_path.iterdir())==[]
    path,receipt=index(tmp_path)
    with pytest.raises(InterruptedError):query_index(path,receipt['index']['sha256'],check_cancelled=cancel)


def test_display_limits_do_not_change_index_or_search(tmp_path):
    value=table(1);value.rows=[(1,'x'*9000+'needle')]
    path=tmp_path/'view.sqlite3';receipt=build_index(value,path,source_sha256='a'*64)
    result=query_index(path,receipt['index']['sha256'],search='needle')
    assert result['matchedRows']==1 and result['cellTruncated'] is True
    assert len(result['rows'][0][1])==2000
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['index']['sha256']

def test_large_cells_do_not_duplicate_the_entire_table_for_search(tmp_path,monkeypatch):
    from runner import table_view
    monkeypatch.setattr(table_view,'MAX_INDEX_BYTES',8_000_000)
    value=table(100);value.rows=[(i,'x'*50000+'Straße needle') for i in range(100)]
    path=tmp_path/'large.sqlite3'
    receipt=build_index(value,path,source_sha256='a'*64)
    assert receipt['schemaVersion']==2 and path.stat().st_size<8_000_000
    page=query_index(path,receipt['index']['sha256'],search='STRASSE NEEDLE',offset=90,limit=10,sort_column=0)
    assert page['matchedRows']==100 and page['positions']==list(range(90,100))
    assert page['cellTruncated'] and len(page['rows'][0][1])==2000
    monkeypatch.setattr(table_view,'MAX_INDEX_BYTES',1_000_000)
    with pytest.raises(ValueError,match='index storage byte budget'):
        build_index(value,tmp_path/'refused.sqlite3',source_sha256='a'*64)
    assert not (tmp_path/'refused.sqlite3').exists() and not list(tmp_path.glob('.table-view-*'))

def test_legacy_index_search_still_uses_its_saved_casefold_text(tmp_path):
    import sqlite3,json
    path=tmp_path/'legacy.sqlite3'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE metadata(value TEXT NOT NULL)')
        db.execute('CREATE TABLE records(position INTEGER PRIMARY KEY,values_json TEXT NOT NULL,search_text TEXT NOT NULL)')
        db.execute('INSERT INTO metadata VALUES(?)',(json.dumps(dict(schemaVersion=1,kind='tableViewIndex',sourceSHA256='a'*64,columns=list(table(1).columns),rowCount=1)),))
        db.execute('INSERT INTO records VALUES(0,?,?)',(json.dumps([1,'Straße']),'1\nstrasse'))
    page=query_index(path,hashlib.sha256(path.read_bytes()).hexdigest(),search='STRASSE')
    assert page['rows']==[[1,'Straße']]


def test_in_memory_rows_are_not_required(tmp_path):
    class Rows:
        def __len__(self):return 10001
        def __iter__(self):
            for i in range(10001):yield (i, 'value')
    class IDs:
        def __len__(self):return 10001
        def __iter__(self):
            for i in range(10001):yield str(i)
    value=table(0);value.rows=Rows();value.row_ids=IDs()
    path=tmp_path/'view.sqlite3';receipt=build_index(value,path,source_sha256='a'*64)
    page=query_index(path,receipt['index']['sha256'],offset=10000)
    assert page['rows']==[[10000,'value']]


@pytest.mark.parametrize('kwargs',[{'limit':101},{'limit':True},{'offset':-1},{'search':'x'*257}])
def test_query_bounds(tmp_path,kwargs):
    path,receipt=index(tmp_path)
    with pytest.raises(ValueError):query_index(path,receipt['index']['sha256'],**kwargs)


def test_mid_build_cancel_cleans_owned_scratch(tmp_path):
    checks=0
    def cancel():
        nonlocal checks
        checks+=1
        if checks==5:raise InterruptedError('cancelled while batching')
    with pytest.raises(InterruptedError):build_index(table(10000),tmp_path/'view.sqlite3',source_sha256='a'*64,check_cancelled=cancel)
    assert list(tmp_path.iterdir())==[]


def test_mutation_during_pinned_copy_is_rejected(tmp_path):
    path,receipt=index(tmp_path,10000)
    checks=0
    def mutate():
        nonlocal checks
        checks+=1
        if checks==2:
            with path.open('r+b') as stream:
                stream.seek(-1,2);stream.write(b'!')
    with pytest.raises(ValueError,match='integrity'):
        query_index(path,receipt['index']['sha256'],check_cancelled=mutate)


def test_symlink_ancestor_is_rejected(tmp_path):
    real=tmp_path/'real';real.mkdir()
    path,receipt=index(real)
    link=tmp_path/'link';link.symlink_to(real,target_is_directory=True)
    with pytest.raises((OSError,ValueError)):query_index(link/path.name,receipt['index']['sha256'])


def test_cli_query_and_captured_inventory_contract(tmp_path):
    import json
    from runner.table_view import run_query,parse_contract,validate_query
    owned=tmp_path/'operation';inputs=owned/'input';inputs.mkdir(parents=True)
    path=inputs/'view.sqlite3';receipt=build_index(table(2),path,source_sha256='a'*64)
    query={'schemaVersion':1,'offset':0,'limit':100,'search':'','sourceSHA256':'a'*64,'indexSHA256':receipt['index']['sha256']}
    raw=json.dumps(query).encode();request=inputs/'view-query.json';request.write_bytes(raw)
    files=[{'path':'input/view.sqlite3',**receipt['index']},{'path':'input/view-query.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]
    contract={'arguments':['--query-table-view',str(request)],'inputFiles':files}
    assert len(parse_contract(contract,owned))==2
    events=[];run_query(request,events.append)
    assert len(events)==1 and events[0]['event']=='tableViewPage' and len(events[0]['rows'])==2
    with pytest.raises(ValueError):validate_query({**query,'offset':True})
    with pytest.raises(ValueError):parse_contract({**contract,'arguments':['--query-table-view',str(tmp_path/'foreign.json')]},owned)


def test_sort_is_global_typed_stable_and_composes_with_search_and_pages(tmp_path):
    values=SimpleNamespace(columns=({'name':'amount','dtype':'int64','nullable':True},{'name':'note','dtype':'string','nullable':False}),
        rows=[(10,'keep'),(2,'keep'),(None,'keep'),(2,'keep'),(1,'other')],row_ids=list(map(str,range(5))))
    path=tmp_path/'sorted.sqlite3';receipt=build_index(values,path,source_sha256='a'*64);pin=receipt['index']['sha256']
    first=query_index(path,pin,limit=2,search='keep',sort_column=0)
    second=query_index(path,pin,offset=2,limit=2,search='keep',sort_column=0)
    assert first['positions']==[1,3] and first['nextOffset']==2
    assert second['positions']==[0,2] and second['nextOffset'] is None
    assert query_index(path,pin,sort_column=0,sort_descending=True)['positions']==[0,1,3,4,2]
    assert query_index(path,pin,sort_descending=True)['positions']==[4,3,2,1,0]
    for column in [-1,2,True,"0); DROP TABLE records;--"]:
        with pytest.raises(ValueError):query_index(path,pin,sort_column=column)


def test_sort_disk_cap_fails_closed_and_cleans_owned_scratch(tmp_path,monkeypatch):
    import sqlite3
    import runner.table_view as module
    path,receipt=index(tmp_path,1000)
    scratch=tmp_path/'scratch';scratch.mkdir()
    monkeypatch.setattr(module,'MAX_SORT_BYTES',8192)
    with pytest.raises(sqlite3.OperationalError,match='full'):
        query_index(path,receipt['index']['sha256'],sort_column=1,scratch_directory=scratch)
    assert list(scratch.iterdir())==[]
    assert query_index(path,receipt['index']['sha256'],limit=1)['positions']==[0]


def test_sorted_managed_request_version_and_sort_bounds():
    from runner.table_view import validate_query
    value={'schemaVersion':2,'offset':0,'limit':100,'search':'','sourceSHA256':'a'*64,'indexSHA256':'b'*64,'sortColumn':0,'sortDescending':False}
    assert validate_query(value)==value
    for change in [{'sortColumn':True},{'sortColumn':128},{'sortDescending':1},{'schemaVersion':1}]:
        with pytest.raises(ValueError):validate_query({**value,**change})


def captured_table_query(tmp_path,count=251,*,sampled=False):
    import json
    from runner import dataflow as df
    owned=tmp_path/'operation';inputs=owned/'input';inputs.mkdir(parents=True)
    selected=7 if sampled else count
    value=df.derive(df.Table(({'name':'id','dtype':'int64','nullable':False},),tuple((i,) for i in range(selected)),tuple('source:'+str(i) for i in range(selected)),'a'*64))
    lineage={**value.evidence(),'dataVersion':value.data_version,'partitionDigest':value.partition_digest,'fittedStateDigests':[],'diagnostics':[]}
    raw=json.dumps({'schemaVersion':1,'kind':'table','columns':value.columns,'rows':value.rows,'lineage':lineage}).encode()
    table_path=inputs/'view.table.json';table_path.write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
    query={'schemaVersion':3,'inputKind':'table','offset':100 if not sampled else 0,'limit':100,'search':'','sourceSHA256':pin,'indexSHA256':pin}
    query_raw=json.dumps(query).encode();request=inputs/'view-query.json';request.write_bytes(query_raw)
    inventory=[{'path':'input/view-query.json','bytes':len(query_raw),'sha256':hashlib.sha256(query_raw).hexdigest()},{'path':'input/view.table.json','bytes':len(raw),'sha256':pin}]
    return owned,request,query,{'arguments':['--query-table-view',str(request)],'inputFiles':inventory}


def test_lazy_table_query_materializes_one_reusable_pinned_index(tmp_path):
    from runner.table_view import run_query,parse_contract,verify_inputs
    owned,request,query,contract=captured_table_query(tmp_path)
    assert len(parse_contract(contract,owned))==2
    verify_inputs(contract,owned,{})
    events=[];run_query(request,events.append,scratch_directory=owned)
    page=events[0]
    assert page['rows']==[[i] for i in range(100,200)] and page['totalRows']==251
    index=page['materializedIndex'];assert index['path']=='artifacts/view.sqlite3'
    assert hashlib.sha256((owned/index['path']).read_bytes()).hexdigest()==index['sha256']
    assert not list(owned.glob('.table-rows-*'))
    reused=query_index(owned/index['path'],index['sha256'],offset=200,source_sha256=query['sourceSHA256'],scratch_directory=owned)
    assert reused['rows']==[[i] for i in range(200,251)]


def test_lazy_sampled_table_exposes_only_recorded_population(tmp_path):
    from runner.table_view import run_query
    owned,request,query,_=captured_table_query(tmp_path,sampled=True)
    events=[];run_query(request,events.append,scratch_directory=owned)
    assert events[0]['totalRows']==7 and events[0]['rows']==[[i] for i in range(7)]


def test_lazy_table_rejects_forged_pin_and_invalid_input_kind(tmp_path):
    from runner.table_view import run_query,validate_query,verify_inputs
    owned,request,query,contract=captured_table_query(tmp_path)
    for changes in ({'inputKind':'source'},{'sourceSHA256':'b'*64},{'schemaVersion':2}):
        with pytest.raises(ValueError):validate_query({**query,**changes})
    (owned/'input/view.table.json').write_bytes(b'{}')
    with pytest.raises(ValueError):verify_inputs(contract,owned,{})
    with pytest.raises(ValueError):run_query(request,lambda _:None,scratch_directory=owned)
    assert not (owned/'artifacts/view.sqlite3').exists()


def test_lazy_table_cancellation_does_not_publish_an_index(tmp_path):
    from runner.table_view import run_query
    owned,request,_,_=captured_table_query(tmp_path)
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):run_query(request,lambda _:None,check_cancelled=cancel,scratch_directory=owned)
    assert not (owned/'artifacts/view.sqlite3').exists()


def test_lazy_table_generated_index_is_removed_when_cancelled_before_page(tmp_path,monkeypatch):
    import runner.table_view as module
    owned,request,_,_=captured_table_query(tmp_path)
    original=module.build_index;published=False
    def build(*args,**kwargs):
        nonlocal published
        result=original(*args,**kwargs);published=True;return result
    def cancel():
        if published:raise InterruptedError('cancelled after indexing')
    monkeypatch.setattr(module,'build_index',build)
    with pytest.raises(InterruptedError):module.run_query(request,lambda _:None,check_cancelled=cancel,scratch_directory=owned)
    assert published and not (owned/'artifacts/view.sqlite3').exists()
    assert not list(owned.glob('.table-rows-*'))


def test_index_build_shares_aggregate_scratch_budget_and_releases_reservation(tmp_path):
    from runner.dataflow_disk import ScratchBudget
    import sqlite3
    budget=ScratchBudget(30_000);budget.resize(0,20_000)
    destination=tmp_path/'limited.sqlite3'
    with pytest.raises((ValueError,sqlite3.DatabaseError)):
        build_index(table(1000),destination,source_sha256='a'*64,scratch_budget=budget)
    assert budget.used==20_000 and not destination.exists()


def test_lazy_contract_refuses_mixed_or_crossed_owned_inputs(tmp_path):
    from runner.table_view import parse_contract,verify_inputs
    owned,_,_,contract=captured_table_query(tmp_path)
    extra={'path':'input/view.sqlite3','bytes':10,'sha256':'b'*64}
    with pytest.raises(ValueError):parse_contract({**contract,'inputFiles':contract['inputFiles']+[extra]},owned)
    crossed={**contract,'inputFiles':[contract['inputFiles'][0],extra]}
    with pytest.raises((ValueError,OSError)):verify_inputs(crossed,owned,{})


def test_index_bytes_are_identical_after_canonical_table_cache_roundtrip(tmp_path):
    import json
    original = table(120)
    # Canonical table serialization sorts object keys; cache decoding must not
    # alter the index artifact for an identical typed population/source pin.
    restored = table(120)
    restored.columns = tuple(json.loads(json.dumps(original.columns, sort_keys=True)))
    first, second = tmp_path / 'first.sqlite3', tmp_path / 'restored.sqlite3'
    first_receipt = build_index(original, first, source_sha256='a'*64)
    second_receipt = build_index(restored, second, source_sha256='a'*64)
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt == second_receipt
    assert query_index(second, second_receipt['index']['sha256'], offset=100)['rows'] == [[i, 'Café example' if i%2 else None] for i in range(100,120)]


def test_explicit_cell_inspection_reads_original_position_and_full_value(tmp_path):
    value='Line one\n'+('Long OCR text 🔎 '*700)
    data=table(3);data.rows=[(0,'short'),(1,value),(2,'other')]
    path=tmp_path/'inspect.sqlite3';receipt=build_index(data,path,source_sha256='a'*64)
    pin=receipt['index']['sha256']
    assert query_index(path,pin)['cellTruncated'] is True
    detail=query_index(path,pin,offset=1,limit=1,full_cell_column=1)
    assert detail['positions']==[1] and detail['rows']==[[value]]
    assert detail['columns']==[data.columns[1]] and not detail['cellTruncated']
    assert detail['nextOffset'] is None
    for kwargs in [dict(offset=1,limit=2,full_cell_column=1),dict(limit=1,full_cell_column=128),dict(limit=1,full_cell_column=1,search='query')]:
        with pytest.raises(ValueError):query_index(path,pin,**kwargs)


def test_cell_inspection_pages_unicode_without_returning_entire_value(tmp_path):
    value='OCR 🔎 line\n'*5000
    data=table(1);data.rows=[(0,value)]
    path=tmp_path/'paged.sqlite3';receipt=build_index(data,path,source_sha256='a'*64)
    chunks=[]
    for start in range(0,len(value),16384):
        page=query_index(path,receipt['index']['sha256'],limit=1,full_cell_column=1,full_cell_start=start)
        assert page['cellOffset']==start and page['cellTotalCharacters']==len(value)
        assert len(page['rows'][0][0])<=16384
        chunks.append(page['rows'][0][0])
    assert ''.join(chunks)==value
    with pytest.raises(ValueError):query_index(path,receipt['index']['sha256'],limit=1,full_cell_column=1,full_cell_start=-1)


def test_cell_inspection_refuses_oversize_evidence_without_loading_file(tmp_path):
    import hashlib
    path=tmp_path/'oversize.sqlite3'
    with path.open('wb') as file:file.truncate(10*1024**3)
    with pytest.raises(ValueError):query_index(path,'a'*64,limit=1,full_cell_column=0)
    path.unlink()
    data=table(1);receipt=build_index(data,path,source_sha256='a'*64)
    import sqlite3,json
    with closing(sqlite3.connect(path)) as connection, connection:connection.execute('UPDATE records SET values_json=?',(json.dumps([0,'x'*(2*1024**2)]),))
    pin=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError,sqlite3.DataError)):query_index(path,pin,limit=1,full_cell_column=1)


def test_wide_index_and_query_follow_captured_workload_without_changing_page_frames(tmp_path):
    from runner.workload_policy import WorkloadPolicy,workload_scope
    value=SimpleNamespace(columns=tuple({'name':'column'+str(i),'dtype':'int64','nullable':False} for i in range(129)),
        rows=[tuple(range(129))],row_ids=['one'])
    path=tmp_path/'wide.sqlite3'
    receipt=build_index(value,path,source_sha256='d'*64)
    with pytest.raises(ValueError):query_index(path,receipt['index']['sha256'])
    with workload_scope(WorkloadPolicy('balanced')):
        page=query_index(path,receipt['index']['sha256'],sort_column=128)
        assert page['rows']==[list(range(129))] and len(page['columns'])==129
    with workload_scope(WorkloadPolicy('balanced').overriding({'tableColumns':128})):
        with pytest.raises(ValueError):query_index(path,receipt['index']['sha256'])


def test_index_reserves_actual_retained_bytes_and_releases_failure(tmp_path):
    from runner.artifact_storage import ArtifactStorageBudget
    budget=ArtifactStorageBudget(200_000)
    receipt=build_index(table(10),tmp_path/'index.sqlite3',source_sha256='a'*64,retention_budget=budget)
    assert budget.used_bytes==receipt['index']['bytes']<budget.maximum_bytes
    other=ArtifactStorageBudget(8192)
    with pytest.raises(ValueError):build_index(table(10),tmp_path/'refused.sqlite3',source_sha256='a'*64,retention_budget=other)
    assert other.used_bytes==0 and not (tmp_path/'refused.sqlite3').exists()
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if failing.used_bytes>0:raise InterruptedError('cancel after some writes')
    failing=ArtifactStorageBudget(1_000_000)
    with pytest.raises(InterruptedError):build_index(table(300),tmp_path/'cancelled.sqlite3',source_sha256='a'*64,retention_budget=failing,check_cancelled=cancel)
    assert failing.used_bytes==0 and not (tmp_path/'cancelled.sqlite3').exists()


def test_real_index_larger_than_legacy_file_allowance_is_paged_under_captured_policy(tmp_path):
    from runner.workload_policy import WorkloadPolicy,workload_scope
    text='x'*800_000
    value=SimpleNamespace(columns=({'name':'text','dtype':'string','nullable':False},),
        rows=[(text,)]*66,row_ids=[str(i) for i in range(66)])
    path=tmp_path/'large.sqlite3'
    with workload_scope(WorkloadPolicy('balanced')):
        receipt=build_index(value,path,source_sha256='e'*64)
        assert receipt['index']['bytes']>50_000_000
        page=query_index(path,receipt['index']['sha256'],limit=1)
        assert page['totalRows']==66 and page['rows']==[['x'*2000]] and page['cellTruncated']
    with pytest.raises(ValueError):query_index(path,receipt['index']['sha256'],limit=1)
    with workload_scope(WorkloadPolicy('balanced').overriding({'retainedBytes':receipt['index']['bytes']-1})):
        with pytest.raises(ValueError):query_index(path,receipt['index']['sha256'],limit=1)

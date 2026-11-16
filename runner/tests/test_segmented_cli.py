"""New captured policies use live segmented tables; legacy keeps its bytes."""
import hashlib,json
from pathlib import Path
from runner.dataflow_cli import run_dataflow
from runner.dataflow_storage import completed_inventory
from runner.dataflow_table_reader import read_table_artifact
from runner.workload_policy import WorkloadPolicy,workload_scope

ROOT=Path(__file__).resolve().parents[2]

def test_explicit_model_inputs_keep_canonical_rows_without_expanding_analytics(tmp_path):
    policy=WorkloadPolicy('balanced');out=tmp_path/'model-boundary'
    manifest=run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},out,lambda _:None,
        workload_policy=policy.to_document(),required_nodes=['training','heldout'],model_input_nodes=['training','heldout'])
    assert manifest['modelInputNodes']==['heldout','training']
    tables=[e for e in manifest['artifacts'] if e['kind']=='table']
    for entry in tables:
        value=json.loads((out/entry['path']).read_bytes())
        assert value['kind']==('table' if entry['nodeID'] in ('training','heldout') else 'segmentedTable')
    assert any(e['kind']=='tableSegment' for e in manifest['artifacts'])
    assert "model_input_nodes=['training', 'heldout']" in (out/'prep.py').read_text()


def test_captured_workload_retains_versioned_segments_and_reloads_cache(tmp_path):
    policy=WorkloadPolicy('balanced')
    output=tmp_path/'first'
    with workload_scope(policy):
        manifest=run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},output,lambda _:None,workload_policy=policy.to_document(),cache_policy='materialize')
        assert manifest['tableStorageVersion']==1
        assert any(entry['kind']=='tableSegment' for entry in manifest['artifacts'])
        prior,files,pin=completed_inventory(output/'manifest.json',lazy=True)
        tables=[entry for entry in prior['artifacts'] if entry['kind']=='table']
        table=read_table_artifact(output/tables[-1]['path'],tables[-1]['sha256'])
        data_version=table.data_version;table.rows.store.close()
        reused=run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},tmp_path/'reused',lambda _:None,
                            workload_policy=policy.to_document(),cache_policy='reuse',reuse_manifest=output/'manifest.json',reuse_manifest_sha256=pin)
        assert any(node['status']=='reused' for node in reused['cache']['nodes'])
        selected=[entry for entry in reused['artifacts'] if entry['kind']=='table'][-1]
        restored=read_table_artifact(tmp_path/'reused'/selected['path'],selected['sha256'])
        assert restored.data_version==data_version;restored.rows.store.close()


def test_portable_replay_of_segmented_capture_uses_frozen_capacity(tmp_path):
    import subprocess,sys
    policy=WorkloadPolicy('balanced');source=ROOT/'examples/dataflow/observations.csv'
    output=tmp_path/'captured';bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}))
    with workload_scope(policy):
        manifest=run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(source)},output,lambda _:None,workload_policy=policy.to_document())
    replay=tmp_path/'portable'
    result=subprocess.run([sys.executable,str(output/'prep.py'),'--source-bindings',str(bindings),'--output',str(replay)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr[-2000:]
    restored=json.loads((replay/'manifest.json').read_bytes())
    assert restored['workloadPolicy']==policy.to_document() and restored['tableStorageVersion']==1
    assert [(entry['nodeID'],entry['port']) for entry in manifest['artifacts'] if entry['kind']=='table']==[(entry['nodeID'],entry['port']) for entry in restored['artifacts'] if entry['kind']=='table']
    with workload_scope(policy):
        for entry in restored['artifacts']:
            if entry['kind']!='table':continue
            counterpart=next(item for item in manifest['artifacts'] if item.get('nodeID')==entry['nodeID'] and item.get('port')==entry['port'] and item['kind']=='table')
            first=read_table_artifact(output/counterpart['path'],counterpart['sha256']);second=read_table_artifact(replay/entry['path'],entry['sha256'])
            assert first.data_version==second.data_version
            first.rows.store.close();second.rows.store.close()


def test_segmented_table_page_captures_dependencies_and_refuses_tamper(tmp_path):
    import shutil
    import pytest
    from runner import table_view
    policy=WorkloadPolicy('balanced');output=tmp_path/'prepared'
    with workload_scope(policy):
        manifest=run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},output,lambda _:None,workload_policy=policy.to_document())
        selected=next(entry for entry in manifest['artifacts'] if entry['kind']=='table')
        root=tmp_path/'query';inputs=root/'input';inputs.mkdir(parents=True)
        shutil.copyfile(output/selected['path'],inputs/'view.table.json')
        segments=[entry for entry in manifest['artifacts'] if entry['kind']=='tableSegment']
        for entry in segments:shutil.copyfile(output/entry['path'],inputs/entry['path'])
        query=dict(schemaVersion=3,inputKind='table',offset=0,limit=100,search='',sourceSHA256=selected['sha256'],indexSHA256=selected['sha256'])
        raw=json.dumps(query).encode();path=inputs/'view-query.json';path.write_bytes(raw)
        inventory=[dict(path='input/view-query.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),dict(path='input/view.table.json',bytes=selected['bytes'],sha256=selected['sha256'])]
        inventory += [dict(path='input/'+entry['path'],bytes=entry['bytes'],sha256=entry['sha256']) for entry in segments]
        contract=dict(arguments=['--query-table-view',str(path)],inputFiles=inventory,executionBudget={'resources':{'workloadPolicy':policy.to_document()}})
        table_view.verify_inputs(contract,root,{})
        events=[];table_view.run_query(path,events.append)
        assert events[-1]['event']=='tableViewPage' and events[-1]['totalRows']==40
        with pytest.raises(ValueError):table_view.parse_contract({**contract,'executionBudget':{}},root)
        target=inputs/segments[0]['path'];content=target.read_bytes();target.write_bytes(content[:-1]+b'!')
        with pytest.raises(ValueError):table_view.verify_inputs(contract,root,{})


def test_selected_snapshot_standalone_replay_keeps_complete_data_and_identity(tmp_path):
    import subprocess,sys
    from runner.segmented_table_store import SegmentResolver
    from runner.tests.test_filter_rows import config as filter_config
    source=tmp_path/'input.csv';source.write_text('keep,n\n'+''.join(('true' if i%2==0 else 'false')+','+str(i)+'\n' for i in range(600)))
    cast={'rows':[{'source':name,'outputName':name,'included':True,'targetType':dtype} for name,dtype in [('keep','bool'),('n','int64')]],'newColumns':'drop','castFailure':'error','missingPolicy':'preserve','trimText':False}
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}},{'id':'typed','operator':'kwip.data.columns','config':cast},{'id':'selected','operator':'kwip.data.filterRows','config':filter_config(mode='expression',expression='[keep]',nullPolicy='nonmatching')}],
           'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'typed','port':'table'}},{'from':{'node':'typed','port':'table'},'to':{'node':'selected','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}));policy=WorkloadPolicy('balanced')
    with workload_scope(policy):
        manifest=run_dataflow(path,{'source':str(source)},tmp_path/'first',lambda _:None,engine='polars',workload_policy=policy.to_document())
        selected=next(e for e in manifest['artifacts'] if e.get('nodeID')=='selected' and e.get('port')=='matching')
        envelope=json.loads((tmp_path/'first'/selected['path']).read_text())
        with SegmentResolver(tmp_path/'first',maximum_bytes=policy.limit('retainedBytes')) as resolver:
            metadata,_=resolver.read_node(envelope['table']);assert metadata['schemaVersion']==2
        replay=tmp_path/'replay'
        result=subprocess.run([sys.executable,str(tmp_path/'first/prep.py'),'--source-bindings',str(bindings),'--output',str(replay)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr[-2000:]
        second=json.loads((replay/'manifest.json').read_text());other=next(e for e in second['artifacts'] if e.get('nodeID')=='selected' and e.get('port')=='matching')
        a=read_table_artifact(tmp_path/'first'/selected['path'],selected['sha256']);b=read_table_artifact(replay/other['path'],other['sha256'])
        try:
            assert list(a.rows)==list(b.rows)==[(True,i) for i in range(0,600,2)]
            assert list(a.row_ids)==list(b.row_ids) and a.data_version==b.data_version
        finally:a.rows.store.close();b.rows.store.close()

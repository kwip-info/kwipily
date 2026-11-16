"""Canonical identities stay exact at durable boundaries, not every live step."""
import json
import pytest
from dataclasses import replace
from runner import dataflow as df
from runner.dataflow_columnar import ColumnarStore
from runner.dataflow_disk import disk_scope


def test_deferred_identity_is_computed_once_and_keeps_original_digest(tmp_path,monkeypatch):
    from runner.table_identity import deferred_identity_scope
    columns=({'name':'x','dtype':'string','nullable':False},)
    with disk_scope(directory=tmp_path):
        store=ColumnarStore(columns,directory=tmp_path);store.append(('a',),'r:1');rows,ids=store.finish()
        calls=[];original=df.digest
        def measured(value):
            if isinstance(value,dict) and 'rows' in value:calls.append(1)
            return original(value)
        monkeypatch.setattr(df,'digest',measured)
        with deferred_identity_scope():
            table=df.derive(df.Table(columns,rows,ids,'a'*64))
            renamed=df.derive(table,role='training')
            assert calls==[]
            assert renamed.preview(include_version=False)['dataVersion'] is None
            assert calls==[]
            expected=original({'columns':columns,'rows':rows,'ids':ids})
            assert table.data_version==renamed.data_version==expected
            assert len(calls)==1
            assert df.derive(table).data_version==expected
            assert len(calls)==1
        store.close()


def test_deferred_identity_does_not_change_legacy_preview_or_unowned_rows():
    from runner.table_identity import deferred_identity_scope
    columns=({'name':'x','dtype':'int64','nullable':False},)
    with deferred_identity_scope():table=df.derive(df.Table(columns,((1,),),('a',),'a'*64))
    assert table.preview()['dataVersion']==df.digest({'columns':columns,'rows':((1,),),'ids':('a',)})


def test_explicit_live_preview_deferred_identity_never_impersonates_canonical(tmp_path):
    from runner.table_identity import deferred_identity_scope
    from runner.dataflow_cli import output_preview
    columns=({'name':'x','dtype':'string','nullable':False},)
    with disk_scope(directory=tmp_path):
        store=ColumnarStore(columns,directory=tmp_path);store.append(('a',),'r:1');rows,ids=store.finish()
        with deferred_identity_scope():
            table=df.derive(df.Table(columns,rows,ids,'a'*64))
            preview=output_preview(table,canonical_identity=False)
            assert preview['schemaVersion']==2
            assert preview['preview']['dataVersion'] is None
            assert preview['lineage']['dataVersion'] is None
            assert preview['preview']['rows']==[['a']]
            legacy=output_preview(table)
            assert 'schemaVersion' not in legacy
            assert legacy['preview']['dataVersion']==table.data_version
        store.close()


def test_unknown_stored_sequence_is_not_treated_as_an_immutable_producer():
    from runner.table_identity import deferred_identity_scope
    from runner.table_storage import StoredRows,StoredIDs
    class Rows(StoredRows):
        def __len__(self):return 1
        def __getitem__(self,index):
            if index!=0:raise IndexError(index)
            return (1,)
    class IDs(StoredIDs):
        def __len__(self):return 1
        def __getitem__(self,index):
            if index!=0:raise IndexError(index)
            return 'a'
    with deferred_identity_scope():
        table=df.derive(df.Table(({'name':'x','dtype':'int64','nullable':False},),Rows(),IDs(),'a'*64))
        assert type(object.__getattribute__(table,'data_version')) is str


@pytest.mark.parametrize('retain',[False,True])
def test_live_workflow_hashes_only_its_retained_tables_and_keeps_old_readers(tmp_path,monkeypatch,retain):
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_native_query_fusion import steps
    from runner.workload_policy import WorkloadPolicy
    source=tmp_path/'source.csv';source.write_text('x\n a \nb\n')
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{'separator':',','parserProfile':'polars'}},*[
        {'id':key,'operator':'kwip.data.'+op,'config':config} for key,op,config in steps()]],'edges':[
        {'from':{'node':'source','port':'table'},'to':{'node':'label','port':'table'}},
        {'from':{'node':'label','port':'table'},'to':{'node':'rename','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));calls=[];original=df.digest
    def measured(value):
        if isinstance(value,dict) and set(value)=={'columns','rows','ids'}:
            calls.append(1)
        return original(value)
    monkeypatch.setattr(df,'digest',measured);events=[]
    run_dataflow(path,{'source':str(source)},tmp_path/'artifacts',events.append,engine='polars',workload_policy=WorkloadPolicy('balanced').to_document(),retain_outputs=retain,live_previews=True)
    assert calls==[1]*(3 if retain else 1)  # Durable boundaries keep canonical identity.
    previews=[event for event in events if event.get('event')=='dataflowPreview']
    assert len(previews)==3
    assert [event['outputs']['table'].get('schemaVersion',1) for event in previews]==([1,1,1] if retain else [2,2,1])
    assert all(event['outputs']['table']['preview']['rows'] for event in previews)

    from runner.dataflow_table_reader import read_table_artifact
    manifest=json.loads((tmp_path/'artifacts/manifest.json').read_text())
    entries=[entry for entry in manifest['artifacts'] if entry['kind']=='table']
    assert len(entries)==(3 if retain else 1)
    for entry in entries:
        table=read_table_artifact(tmp_path/'artifacts'/entry['path'],entry['sha256'])
        assert table.data_version==original({'columns':table.columns,'rows':table.rows,'ids':table.row_ids})
        table.rows.store.close()

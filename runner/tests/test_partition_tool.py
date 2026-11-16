import hashlib
import json
import pytest
from runner import dataflow as df
from runner.dataflow_partition_tool import apply, validate_config, verify_output


def table(n=30):
    columns=({'name':'entity','dtype':'string','nullable':False},{'name':'time','dtype':'int64','nullable':False},{'name':'class','dtype':'int64','nullable':False})
    return df.derive(df.Table(columns,tuple((f'e{i//3}',i*1000,i%2) for i in range(n)),tuple(f'row-{i:03}' for i in range(n)),'a'*64))


def random_config(**values):return {'strategy':'random','validationPercent':20,'testPercent':20,'seed':42,**values}
def entity_config(**values):return {'strategy':'entity','groupBy':['entity'],'validationPercent':20,'testPercent':20,'seed':42,**values}
def time_config(**values):return {'strategy':'time','timeColumn':'time','timeEncoding':'epochMilliseconds','trainingEnd':9000,'validationEnd':19000,'embargoMilliseconds':2000,'entityColumns':[],'entityPolicy':'allowAcrossTime',**values}


def test_random_three_way_has_independently_ranked_membership_and_preserves_order():
    source=table();outputs=apply(source,random_config(),lambda:None)
    ordered=sorted(range(30),key=lambda i:(hashlib.sha256(json.dumps(['kwip.partition.random.v1',42,[],source.row_ids[i]],sort_keys=True,separators=(',',':')).encode()).hexdigest(),source.row_ids[i]))
    expected={'training':set(ordered[:18]),'validation':set(ordered[18:24]),'test':set(ordered[24:])}
    assert set(outputs)=={'training','validation','test','exclusions'}
    for role,indices in expected.items():
        assert list(outputs[role].row_ids)==[source.row_ids[i] for i in range(30) if i in indices]
        assert outputs[role].role==role
    assert len(outputs['exclusions'].rows)==0
    verify_output(source,random_config(),outputs,lambda:None)


def test_entities_stay_whole_and_seeded_allocation_is_by_entity():
    source=table();outputs=apply(source,entity_config(),lambda:None)
    assert [len(outputs[p].rows) for p in ('training','validation','test')]==[18,6,6]
    groups=[{r[0] for r in outputs[p].rows} for p in ('training','validation','test')]
    assert all(a.isdisjoint(b) for i,a in enumerate(groups) for b in groups[i+1:])
    shuffled=df.derive(source,rows=tuple(reversed(source.rows)),ids=tuple(reversed(source.row_ids)))
    again=apply(shuffled,entity_config(),lambda:None)
    assert all(set(outputs[p].row_ids)==set(again[p].row_ids) for p in ('training','validation','test'))


def test_time_boundaries_embargo_and_audit_rows_are_exact():
    source=table();outputs=apply(source,time_config(),lambda:None)
    assert [len(outputs[p].rows) for p in ('training','validation','test','exclusions')]==[10,8,8,4]
    assert [r[1] for r in outputs['exclusions'].rows]==[10000,11000,20000,21000]
    assert outputs['exclusions'].role=='unassigned'
    assert outputs['exclusions'].partition_digest==outputs['training'].partition_digest
    assert outputs['training'].diagnostics[-1]['excludedRows']==4
    verify_output(source,time_config(),outputs,lambda:None)


def test_stratification_and_rare_classes_refuse_instead_of_silent_adjustment():
    outputs=apply(table(),random_config(stratifyColumn='class'),lambda:None)
    assert all({r[2] for r in outputs[p].rows}=={0,1} for p in ('training','validation','test'))
    with pytest.raises(ValueError,match='nonempty|three|strat'):
        apply(table(3),random_config(stratifyColumn='class'),lambda:None)


@pytest.mark.parametrize('config',[random_config(testPercent=0),entity_config(groupBy=[]),time_config(trainingEnd=19000),time_config(entityPolicy='requireDisjoint')])
def test_invalid_configuration_refuses(config):
    with pytest.raises(ValueError):validate_config(config)


def test_cache_forgery_and_repartition_refuse():
    source=table();outputs=apply(source,random_config(),lambda:None)
    forged=dict(outputs);forged['training']=outputs['validation']
    with pytest.raises(ValueError):verify_output(source,random_config(),forged,lambda:None)
    with pytest.raises(ValueError):apply(outputs['training'],random_config(),lambda:None)


def test_cancellation_is_observed_during_assignment():
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):apply(table(),random_config(),cancel)

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('config,counts',[(random_config(),[18,6,6,0]),(entity_config(),[18,6,6,0]),(time_config(),[10,8,8,4])])
def test_actual_fixture_capture_and_independent_package_replay(engine,config,counts,tmp_path):
    from pathlib import Path
    from runner.polars_csv_configuration import DEFAULTS
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/partition-practice-observations.csv'
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{**DEFAULTS,'schemaOverrides':{'time':'Int64','class':'Int64'}}},{'id':'p','operator':'kwip.data.partition','config':config}], 'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'p','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));binding=tmp_path/'bindings.json';binding.write_text(json.dumps({'s':str(source)}))
    export=tmp_path/'export';manifest=run_dataflow(path,{'s':str(source)},export,lambda _:None,engine=engine)
    entries={e['port']:e for e in manifest['artifacts'] if e.get('nodeID')=='p'}
    captured={role:json.loads((export/entry['path']).read_text()) for role,entry in entries.items()}
    assert [len(captured[p]['rows']) for p in ('training','validation','test','exclusions')]==counts
    assert captured['exclusions']['lineage']['role']=='unassigned'
    assert (export/'dataflow_partition_tool.py').is_file()
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    reproduced=replay(export,binding,tmp_path/'replayed',elsewhere)
    assert manifest['artifacts']==reproduced['artifacts']
    forged=json.loads(json.dumps(graph));forged['nodes'].append({'id':'v','operator':'kwip.data.view','config':{}})
    forged['edges'].append({'from':{'node':'p','port':'exclusions'},'to':{'node':'v','port':'table'}})
    with pytest.raises(ValueError):df.validate(forged)

@pytest.mark.parametrize('engine',['disk','polars'])
def test_streamed_population_above_legacy_cap_is_not_eagerly_loaded(engine):
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_disk import disk_scope
    columns=({'name':'value','dtype':'int64','nullable':False},)
    with disk_scope(),preparation_engine(engine):
        writer=df.row_store(columns,lambda:None,identity_prefix='source:'+'a'*64+':')
        for i in range(100001):writer.append((i,),writer.identity_prefix+str(i))
        rows,ids=writer.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64))
        outputs=apply(source,random_config(),lambda:None)
        try:
            assert [len(outputs[p].rows) for p in ('training','validation','test')]==[60001,20000,20000]
            assert all(isinstance(result.rows,df.StoredRows) for result in outputs.values())
            assert sum(sum(row[0] for row in outputs[p].rows) for p in ('training','validation','test'))==100001*100000//2
        finally:
            for result in outputs.values():result.rows.store.close()
            writer.close()


def test_missing_keys_duplicate_identities_and_cross_time_entities_refuse():
    source=table()
    missing=df.derive(source,rows=((None,*source.rows[0][1:]),*source.rows[1:]),columns=({'name':'entity','dtype':'string','nullable':True},*source.columns[1:]))
    with pytest.raises(ValueError,match='missing'):apply(missing,entity_config(),lambda:None)
    duplicate=df.derive(source,ids=('duplicate',)*30)
    with pytest.raises(ValueError,match='unique'):apply(duplicate,random_config(),lambda:None)
    with pytest.raises(ValueError,match='crosses'):apply(source,time_config(embargoMilliseconds=0,entityColumns=['entity'],entityPolicy='requireDisjoint'),lambda:None)


def test_verified_cache_reuse_retains_audit_and_all_roles(tmp_path):
    from pathlib import Path
    from runner.polars_csv_configuration import DEFAULTS
    from runner.dataflow_cli import run_dataflow
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/partition-practice-observations.csv'
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{**DEFAULTS,'schemaOverrides':{'time':'Int64','class':'Int64'}}},{'id':'p','operator':'kwip.data.partition','config':time_config()}], 'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'p','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    first=tmp_path/'first';run_dataflow(path,{'s':str(source)},first,lambda _:None,cache_policy='materialize')
    manifest=first/'manifest.json';events=[]
    second=tmp_path/'second';cached=run_dataflow(path,{'s':str(source)},second,events.append,cache_policy='reuse',reuse_manifest=manifest,reuse_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    previews=[e for e in events if e.get('event')=='dataflowPreview' and e.get('nodeID')=='p']
    assert previews
    assert 'p' in cached['cache']['reusedNodeIDs']

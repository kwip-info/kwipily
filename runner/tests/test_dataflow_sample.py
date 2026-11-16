"""Seeded row budgets preserve observation identity and partition ownership."""
import pytest
from runner.dataflow import Table, derive
from runner.dataflow_sample import sample_rows


def table():
    return derive(Table(({'name':'x','dtype':'int64','nullable':False},),
                        tuple((i,) for i in range(100)),tuple(f'row-{i}' for i in range(100)),
                        'source',role='training',partition_digest='partition',fitted_states=('fit',)))


def test_repeatable_nested_budget_and_source_order():
    source=table();small=sample_rows(source,10,42);large=sample_rows(source,20,42)
    assert len(small.rows)==10 and len(set(small.row_ids))==10
    assert sample_rows(source,10,42)==small
    assert set(small.row_ids)<set(large.row_ids)
    assert [row[0] for row in small.rows]==sorted(row[0] for row in small.rows)
    assert sample_rows(source,10,43).row_ids!=small.row_ids
    assert sample_rows(source,100,42).rows==source.rows
    assert source.rows==tuple((i,) for i in range(100))


def test_membership_does_not_depend_on_input_order_and_metadata_is_preserved():
    source=table();result=sample_rows(source,12,7)
    reordered=derive(source,rows=source.rows[::-1],ids=source.row_ids[::-1])
    assert set(sample_rows(reordered,12,7).row_ids)==set(result.row_ids)
    assert result.columns==source.columns and result.role=='training'
    assert result.partition_digest=='partition' and result.fitted_states==('fit',)
    assert result.source_digest==source.source_digest
    assert result.data_version!=source.data_version
    assert result.evidence()['membershipDigest']!=source.evidence()['membershipDigest']
    assert all(identity==f'row-{row[0]}' for identity,row in zip(result.row_ids,result.rows))


@pytest.mark.parametrize('count,seed',[(0,1),(101,1),(True,1),(1.5,1),(10,-1),(10,2**32),(10,True)])
def test_invalid_budget_or_seed_rejected_without_silent_clamping(count,seed):
    with pytest.raises(ValueError):sample_rows(table(),count,seed)


def test_duplicate_observation_identity_is_rejected():
    source=table();bad=derive(source,ids=('same',)*100)
    with pytest.raises(ValueError,match='identit'):sample_rows(bad,10,42)


def graph_fixture(tmp_path,after_split=False):
    source=tmp_path/'rows.csv'
    source.write_text('x,label\n'+''.join(f'{i},{i%2}\n' for i in range(100)))
    nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={'separator':','}),
           dict(id='sample',operator='kwip.data.sampleRows',config={'count':20,'seed':42})]
    def wire(a,port,b):return {'from':{'node':a,'port':port},'to':{'node':b,'port':'table'}}
    edges=[wire('source','table','sample')]
    if after_split:
        nodes.insert(1,dict(id='split',operator='kwip.data.partition',config={'strategy':'random','seed':17,'validationPercent':20,'testPercent':20}))
        edges=[wire('source','table','split'),wire('split','training','sample')]
    return dict(schemaVersion=1,nodes=nodes,edges=edges),{'source':source}


@pytest.mark.parametrize('after_split',[False,True])
def test_sample_component_executes_with_explicit_lineage(tmp_path,after_split):
    from runner.dataflow import execute
    graph,bindings=graph_fixture(tmp_path,after_split)
    values=execute(graph,bindings);sample=values['sample']['table']
    assert len(sample.rows)==20
    assert sample.role==('training' if after_split else 'unassigned')
    if after_split:
        assert set(sample.row_ids)<=set(values['split']['training'].row_ids)
        assert set(sample.row_ids).isdisjoint(values['split']['validation'].row_ids)
        assert sample.partition_digest==values['split']['training'].partition_digest
    assert execute(graph,bindings)['sample']['table']==sample


@pytest.mark.parametrize('config',[{'count':0,'seed':42},{'count':20,'seed':True},
                                   {'count':20},{'count':20,'seed':42,'replacement':True}])
def test_invalid_component_configuration_precedes_source_read(tmp_path,config):
    from runner.dataflow import execute
    graph,bindings=graph_fixture(tmp_path)
    graph['nodes'][-1]['config']=config
    bindings['source'].unlink()
    with pytest.raises(ValueError,match='configuration|Configuration'):execute(graph,bindings)


def test_sampling_materialization_reuses_only_identical_seed_and_count(tmp_path):
    import hashlib,json
    from runner.dataflow_cli import run_dataflow
    graph,bindings=graph_fixture(tmp_path)
    bindings={key:str(value) for key,value in bindings.items()}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    first=tmp_path/'first'
    run_dataflow(path,bindings,first,lambda event:None,cache_policy='materialize')
    prior=first/'manifest.json';pin=hashlib.sha256(prior.read_bytes()).hexdigest()
    def replay(name):
        path.write_text(json.dumps(graph))
        return run_dataflow(path,bindings,tmp_path/name,lambda event:None,cache_policy='reuse',
                            reuse_manifest=prior,reuse_manifest_sha256=pin)
    same=replay('same')
    assert same['cache']['reusedNodeIDs']==['sample']
    graph['nodes'][-1]['config']['count']=10
    changed=replay('count')
    assert 'sample' in changed['cache']['executedNodeIDs'] and not changed['cache']['reusedNodeIDs']
    graph['nodes'][-1]['config'].update(count=20,seed=43)
    changed=replay('seed')
    assert 'sample' in changed['cache']['executedNodeIDs'] and not changed['cache']['reusedNodeIDs']

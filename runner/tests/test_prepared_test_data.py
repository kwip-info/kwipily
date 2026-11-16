"""Protected-test tables use archived training mapping and frozen transforms only."""
import copy
import hashlib
import json

import pytest
import numpy as np
import torch
from runner import dataflow, prepared_data, dataflow_preprocess
from runner.preprocess_parameters import rule_defaults
from runner.dataflow_cli import run_dataflow


def filter_config(threshold):
    return dict(mode='simple',conditions=[dict(column='a',comparison='gt',value=threshold,caseSensitive=True)],match='all',expression='',column='a',pattern='.+',matchMode='search',flags=[],nullPolicy='error')


def fixture(tmp_path, *, transformed=False, integer=False, single_training=False, test_descendant=None, partition_config=None, engine=None, segmented=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source=tmp_path/'rows.csv'
    source.write_text('time,a,b,label\n'+''.join(
        f'{i},{i+1},{i*i+3},{1 if i>=45 or single_training else i%2}\n' for i in range(60)))
    dtype='int64' if integer else 'float64'
    column_config={'rows':[dict(source=c,included=True,outputName=c,targetType=t) for c,t in [('time','int64'),('a',dtype),('b',dtype),('label','int64')]],
                   'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}
    nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={'separator':','}),
           dict(id='cast',operator='kwip.data.columns',config=column_config),
           dict(id='split',operator='kwip.data.partition',config={'strategy':'time','timeColumn':'time','timeEncoding':'epochMilliseconds','trainingEnd':30,'validationEnd':45,'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'})]
    if partition_config is not None:nodes[-1]['config']=partition_config
    def edge(a,port,b,input='table'):return {'from':{'node':a,'port':port},'to':{'node':b,'port':input}}
    edges=[edge('source','table','cast'),edge('cast','table','split')]
    if not integer:
        rules=[dict(rule_defaults(c),missing='error',scaling='standard') for c in ['a','b']]
        nodes += [dict(id='fit',operator='kwip.data.preprocess',config={'mode':'partitions','columns':rules}),
                  dict(id='training',operator='kwip.data.view',config={}),dict(id='validation',operator='kwip.data.view',config={})]
        edges += [edge('split','training','fit','training'),edge('split','validation','fit','validation'),edge('fit','training','training'),edge('fit','validation','validation')]
        if transformed:
            nodes.append(dict(id='test',operator='kwip.data.view',config={}))
            edges += [edge('split','test','fit','test'),edge('fit','test','test')]
    if test_descendant is not None:
        operator,config=test_descendant
        nodes.append(dict(id='test-descendant',operator='kwip.data.'+operator,config=config))
        edges.append(edge('split','test','test-descendant'))
    graph=tmp_path/'flow.json';graph.write_text(json.dumps(dict(schemaVersion=1,nodes=nodes,edges=edges)))
    from runner.workload_policy import LEGACY
    output=tmp_path/'prepared';run_dataflow(graph,{'source':str(source)},output,lambda _:None,engine=engine,workload_policy=LEGACY.to_document() if segmented else None)
    manifest=output/'manifest.json'
    options={'input_dtype':'int64','sample_shape':[1,2]} if integer else {}
    selection=('split','training','split','validation') if integer else ('training','table','validation','table')
    _,provenance,files=prepared_data.load_prepared(manifest,*selection,['b','a'],'label',**options)
    return manifest,provenance,files,source


def evaluate(manifest,provenance,node='split',port='test',**options):
    return prepared_data.load_test_partition(manifest,node,port,provenance,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),**options)


def artifact(manifest,node,port):
    value=json.loads(manifest.read_bytes())
    entry=next(a for a in value['artifacts'] if a.get('nodeID')==node and a.get('port')==port)
    return value,entry,json.loads((manifest.parent/entry['path']).read_bytes())


def rewrite(manifest,provenance,node,port,change):
    value,entry,table=artifact(manifest,node,port);change(table)
    if table.get('kind')=='table':
        ids=table['lineage']['observationIDs']
        table['lineage']['membershipDigest']=dataflow.membership_digest(ids)
        table['lineage']['dataVersion']=dataflow.digest({'columns':table['columns'],'rows':table['rows'],'ids':ids})
    raw=json.dumps(table).encode();(manifest.parent/entry['path']).write_bytes(raw)
    entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    manifest.write_text(json.dumps(value))
    # Re-pin only the archive container to isolate semantic leakage checks.
    provenance['preparation']['manifestSHA256']=hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('storage',['memory','streamed'])
def test_raw_test_replays_frozen_scaler_without_sources_or_refit(tmp_path,monkeypatch,engine,storage):
    manifest,provenance,files,source=fixture(tmp_path,engine=engine)
    if storage=='memory':
        from dataclasses import replace
        transform=dataflow.transform
        def memory_transform(*args,**kwargs):
            table=transform(*args,**kwargs)
            return replace(table,rows=tuple(table.rows),row_ids=tuple(table.row_ids))
        monkeypatch.setattr(dataflow,'transform',memory_transform)
    _,_,state=artifact(manifest,'fit','state')
    _,_,raw=artifact(manifest,'split','test')
    _,_,training=artifact(manifest,'split','training')
    numeric=np.array([row[1:3] for row in training['rows']],dtype=np.float64)
    np.testing.assert_allclose([r['mean'] for r in state['rules']],numeric.mean(axis=0),rtol=1e-12)
    np.testing.assert_allclose([r['scale'] for r in state['rules']],numeric.std(axis=0),rtol=1e-12)
    source.unlink()
    monkeypatch.setattr(dataflow,'fit_scaler',lambda *a,**k:pytest.fail('Test data must never be fitted'))
    monkeypatch.setattr(dataflow._fitted_module(),'fit',lambda *a,**k:pytest.fail('Test data must never be fitted'))
    monkeypatch.setattr(dataflow_preprocess,'_fit',lambda *a,**k:pytest.fail('Test data must never be fitted'))
    (x,y),test,verified_files=evaluate(manifest,provenance)
    expected=torch.tensor([[(row[2]-state['rules'][1]['mean'])/state['rules'][1]['scale'],
                            (row[1]-state['rules'][0]['mean'])/state['rules'][0]['scale']] for row in raw['rows']],dtype=torch.float32)
    # validationEnd=45 is inclusive; test contains integer times46 through59.
    assert torch.equal(x,expected) and y.tolist()==[1]*14
    assert test['testRowIDs']==raw['lineage']['observationIDs']
    assert test['featureColumns']==['b','a'] and test['targetColumn']=='label'
    assert test['preparation']['selectedDataVersion']==raw['lineage']['dataVersion']
    assert test['preparation']['evaluatedDataVersion']!=raw['lineage']['dataVersion']
    assert test['preparation']['appliedStateDigests']==provenance['preparation']['fittedStateDigests']
    assert verified_files==files


def test_inventoried_transformed_test_is_not_transformed_twice(tmp_path):
    manifest,provenance,_,_=fixture(tmp_path,transformed=True)
    raw,raw_provenance,_=evaluate(manifest,provenance)
    transformed,result,_=evaluate(manifest,provenance,'test','table')
    assert all(torch.equal(a,b) for a,b in zip(raw,transformed))
    assert result['preparation']['appliedStateDigests']==[]
    assert result['preparation']['selectedDataVersion']==result['preparation']['evaluatedDataVersion']
    assert result['preparation']['evaluatedDataVersion']==raw_provenance['preparation']['evaluatedDataVersion']


@pytest.mark.parametrize('field,value',[('role','validation'),('sourceDigest','0'*64),('partitionDigest','0'*64),('fittedStateDigests',['0'*64])])
def test_wrong_test_lineage_rejects_even_when_inventory_is_resealed(tmp_path,field,value):
    manifest,provenance,_,_=fixture(tmp_path)
    rewrite(manifest,provenance,'split','test',lambda table:table['lineage'].__setitem__(field,value))
    with pytest.raises(ValueError):evaluate(manifest,provenance)


@pytest.mark.parametrize('partition',['trainingRowIDs','validationRowIDs'])
def test_test_ids_must_not_overlap_either_archived_partition(tmp_path,partition):
    manifest,provenance,_,_=fixture(tmp_path)
    rewrite(manifest,provenance,'split','test',lambda table:table['lineage']['observationIDs'].__setitem__(0,provenance[partition][0]))
    with pytest.raises(ValueError):evaluate(manifest,provenance)


@pytest.mark.parametrize('mutation',['mapping','rows','state','dataVersion','graph'])
def test_training_provenance_cannot_relabel_the_selected_archive(tmp_path,mutation):
    manifest,provenance,_,_=fixture(tmp_path)
    modified=copy.deepcopy(provenance)
    if mutation=='mapping':modified['featureColumns']=['label','a']
    elif mutation=='rows':modified['trainingRowIDs']=[]
    elif mutation=='state':modified['preparation']['fittedStateDigests']=[]
    elif mutation=='dataVersion':modified['preparation']['trainingDataVersion']='0'*64
    else:modified['preparation']['graphDigest']='0'*64
    with pytest.raises(ValueError):evaluate(manifest,modified)


def test_missing_port_and_changed_manifest_fail_before_evaluation(tmp_path):
    manifest,provenance,_,_=fixture(tmp_path)
    with pytest.raises(ValueError):evaluate(manifest,provenance,'not-materialized','table')
    with pytest.raises(ValueError):prepared_data.load_test_partition(manifest,'split','test',provenance,expected_manifest_sha256='0'*64)
    value,entry,_=artifact(manifest,'split','test')
    (manifest.parent/entry['path']).write_text('{}')
    with pytest.raises(ValueError):evaluate(manifest,provenance)


def test_integer_test_preserves_shape_order_and_training_class_requirement(tmp_path):
    manifest,provenance,_,_=fixture(tmp_path,integer=True)
    (x,y),_,_=evaluate(manifest,provenance,input_dtype='int64',sample_shape=[1,2])
    assert x.dtype==torch.int64 and tuple(x.shape)==(14,1,2)
    assert x[0].tolist()==[[46*46+3,47]] and y.tolist()==[1]*14
    with pytest.raises(ValueError):evaluate(manifest,provenance,input_dtype='float32')
    with pytest.raises(ValueError,match='both binary classes'):
        fixture(tmp_path/'single-training',single_training=True)


def test_state_fitting_evidence_cannot_include_test_observations(tmp_path):
    manifest,provenance,_,_=fixture(tmp_path)
    _,_,test=artifact(manifest,'split','test')
    def contaminate(state):
        state['fitEvidence']['observationIDs'][0]=test['lineage']['observationIDs'][0]
        state['fitEvidence']['membershipDigest']=dataflow.membership_digest(state['fitEvidence']['observationIDs'])
        state['contentDigest']=dataflow.digest({k:v for k,v in state.items() if k!='contentDigest'})
    rewrite(manifest,provenance,'fit','state',contaminate)
    with pytest.raises(ValueError):evaluate(manifest,provenance)


def test_raw_port_cannot_claim_transform_already_applied(tmp_path):
    manifest,provenance,_,_=fixture(tmp_path)
    # Schema is deliberately unchanged by scaler; blindly trusting only a chain
    # label would make raw values masquerade as the trained model's inputs.
    rewrite(manifest,provenance,'split','test',lambda table:table['lineage'].__setitem__('fittedStateDigests',provenance['preparation']['fittedStateDigests']))
    with pytest.raises(ValueError):evaluate(manifest,provenance)


@pytest.mark.parametrize('descendant',[
    ('sampleRows',{'count':5,'seed':42}),
    ('filterRows',filter_config(52.0)),
])
def test_selected_test_descendant_cannot_shrink_protected_population(tmp_path,descendant):
    manifest,provenance,_,_=fixture(tmp_path,test_descendant=descendant)
    port='matching' if descendant[0]=='filterRows' else 'table'
    _,_,selected=artifact(manifest,'test-descendant',port)
    _,_,canonical=artifact(manifest,'split','test')
    assert selected['lineage']['role']=='test'
    assert selected['lineage']['partitionDigest']==canonical['lineage']['partitionDigest']
    assert set(selected['lineage']['observationIDs'])<set(canonical['lineage']['observationIDs'])
    with pytest.raises(ValueError,match='population|membership|complete|subset'):
        evaluate(manifest,provenance,'test-descendant',port)


def test_row_preserving_test_descendant_retains_full_population(tmp_path):
    manifest,provenance,_,_=fixture(tmp_path,test_descendant=('filterRows',filter_config(0.0)))
    raw,original,_=evaluate(manifest,provenance)
    selected,selected_provenance,_=evaluate(manifest,provenance,'test-descendant','matching')
    assert all(torch.equal(a,b) for a,b in zip(raw,selected))
    assert selected_provenance['testRowIDs']==original['testRowIDs']


@pytest.mark.parametrize('explicit_optimizer', [False, True])
def test_actual_composite_training_metadata_is_separate_from_exact_data_provenance(tmp_path, explicit_optimizer):
    from runner.engine import run
    from runner.composite_source import load_source
    from runner.tests.test_composite_source import fixture as composite_fixture
    manifest, provenance, _, _ = fixture(tmp_path)
    prepared = prepared_data.load_prepared(manifest, 'training', 'table', 'validation', 'table', ['b', 'a'], 'label')
    graph, source_path, source_sha, _ = composite_fixture(tmp_path)
    output = tmp_path / 'trained'
    options = {}
    if explicit_optimizer:
        from runner.tests.test_optimizer_runtime import recipe
        options['optimizer_recipe'] = recipe('optimizer_adam')
    run(graph, output, epochs=1, device='cpu', prepared=prepared,
        composite_source=load_source(source_path, source_sha, graph), **options)
    archived = json.loads((output / 'manifest.json').read_bytes())['provenance']
    assert archived['composite']['sha256'] == source_sha
    model_fields = {'composite', 'optimizer'} if explicit_optimizer else {'composite'}
    assert {key: value for key, value in archived.items() if key not in model_fields} == provenance
    expected, expected_provenance, _ = evaluate(manifest, provenance)
    actual, actual_provenance, _ = evaluate(manifest, archived)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert actual_provenance == expected_provenance
    for mutation in ['unknown', 'partition', 'rows']:
        changed = copy.deepcopy(archived)
        if mutation == 'unknown':
            changed['unrecognizedModelMetadata'] = {'trusted': True}
        elif mutation == 'partition':
            changed['preparation']['partitionDigest'] = '0' * 64
        else:
            changed['trainingRowIDs'] = []
        with pytest.raises(ValueError):
            evaluate(manifest, changed)
    # Separating model metadata must not mutate the supplied pinned provenance.
    assert archived == json.loads((output / 'manifest.json').read_bytes())['provenance']


@pytest.mark.parametrize('field', ['scheduler', 'initialization', 'gradientPolicy', 'trainability'])
def test_known_training_metadata_does_not_change_data_population(tmp_path, field):
    manifest, provenance, _, _ = fixture(tmp_path)
    archived = copy.deepcopy(provenance)
    # This loader receives already pinned archive metadata. Test only the data
    # comparison boundary here; real recipe validation belongs to model loading.
    archived[field] = {'recipe': {'schemaVersion': 1}, 'torchVersion': str(torch.__version__)}
    expected, expected_provenance, _ = evaluate(manifest, provenance)
    actual, actual_provenance, _ = evaluate(manifest, archived)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert actual_provenance == expected_provenance
    for key in ['trainingRowIDs', 'featureColumns']:
        changed = copy.deepcopy(archived)
        changed[key] = []
        with pytest.raises(ValueError):
            evaluate(manifest, changed)

@pytest.mark.parametrize('config',[
    {'strategy':'random','seed':42,'validationPercent':20,'testPercent':20},
    {'strategy':'entity','seed':42,'validationPercent':20,'testPercent':20,'groupBy':['time']},
    {'strategy':'time','timeColumn':'time','timeEncoding':'epochMilliseconds','trainingEnd':30,'validationEnd':45,'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'}])
def test_unified_partition_frozen_test_replays_without_source_or_refitting(config,tmp_path,monkeypatch):
    manifest,provenance,_,source=fixture(tmp_path,partition_config=config)
    _,_,state=artifact(manifest,'fit','state');_,_,raw=artifact(manifest,'split','test')
    _,_,training=artifact(manifest,'split','training')
    numeric=np.array([row[1:3] for row in training['rows']],dtype=np.float64)
    np.testing.assert_allclose([r['mean'] for r in state['rules']],numeric.mean(axis=0),rtol=1e-12)
    np.testing.assert_allclose([r['scale'] for r in state['rules']],numeric.std(axis=0),rtol=1e-12)
    source.unlink()
    monkeypatch.setattr(dataflow,'fit_scaler',lambda *a,**k:pytest.fail('Protected test must never fit'))
    monkeypatch.setattr(dataflow._fitted_module(),'fit',lambda *a,**k:pytest.fail('Protected test must never fit'))
    monkeypatch.setattr(dataflow_preprocess,'_fit',lambda *a,**k:pytest.fail('Protected test must never fit'))
    (x,y),test,_=evaluate(manifest,provenance)
    expected=torch.tensor([[(row[2]-state['rules'][1]['mean'])/state['rules'][1]['scale'],(row[1]-state['rules'][0]['mean'])/state['rules'][0]['scale']] for row in raw['rows']],dtype=torch.float32)
    assert torch.equal(x,expected) and y.tolist()==[row[3] for row in raw['rows']]
    assert test['testRowIDs']==raw['lineage']['observationIDs']
    with pytest.raises(ValueError):evaluate(manifest,provenance,'split','exclusions')


def test_frozen_test_adapter_rejects_oversized_stream_before_materialization(tmp_path,monkeypatch):
    from dataclasses import replace
    manifest,provenance,_,_=fixture(tmp_path)
    class OversizedRows:
        def __len__(self):return 100001
        def __iter__(self):pytest.fail('Oversized frozen test rows were materialized')
    class UnreadableIDs:
        def __iter__(self):pytest.fail('Oversized frozen test IDs were materialized')
    transform=dataflow.transform
    def oversized_transform(*args,**kwargs):
        return replace(transform(*args,**kwargs),rows=OversizedRows(),row_ids=UnreadableIDs())
    monkeypatch.setattr(dataflow,'transform',oversized_transform)
    with pytest.raises(ValueError,match='bounds'):
        evaluate(manifest,provenance)


def test_segmented_protected_test_preserves_frozen_values_and_identity(tmp_path):
    old,provenance,_,_=fixture(tmp_path/'legacy')
    new,segmented,_,source=fixture(tmp_path/'segmented',segmented=True)
    source.unlink()
    (x,y),expected,_=evaluate(old,provenance)
    (actual,labels),observed,_=evaluate(new,segmented)
    assert torch.equal(x,actual) and torch.equal(y,labels)
    assert expected['testRowIDs']==observed['testRowIDs']
    assert expected['preparation']['selectedDataVersion']==observed['preparation']['selectedDataVersion']

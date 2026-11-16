import copy
import json
from pathlib import Path
import numpy as np
import pytest
from runner.dataflow import execute, membership_digest, load_state,fit_observation_ids
from runner.dataflow_preprocess import rule_defaults


def fixture(tmp_path):
    source=tmp_path/'data.csv'
    source.write_text('raw,label\n'+''.join(f'{i},{i%2}\n' for i in range(20)))
    rename=dict(rows=[dict(source='raw',included=True,outputName='x',targetType='keep'),dict(source='label',included=True,outputName='label',targetType='keep')],castFailure='error',trimText=False,newColumns='error',missingPolicy='error')
    cast=dict(rows=[dict(source='x',included=True,outputName='x',targetType='float64'),dict(source='label',included=True,outputName='label',targetType='int64')],castFailure='error',trimText=False,newColumns='error',missingPolicy='error')
    nodes=[('s','sourceCSV',{'separator':','}),('r','columns',rename),('c','columns',cast),('p','partition',dict(strategy='random',seed=7,validationPercent=20,testPercent=20,stratifyColumn='label')),('f','preprocess',dict(mode='partitions',columns=[dict(rule_defaults('x'),scaling='standard')])),('t','view',{})]
    edges=[('s','table','r','table'),('r','table','c','table'),('c','table','p','table'),('p','training','f','training'),('p','validation','f','validation'),('p','test','f','test'),('f','validation','t','table')]
    return {'schemaVersion':1,'nodes':[{'id':i,'operator':'kwip.data.'+op,'config':cfg} for i,op,cfg in nodes],'edges':[{'from':{'node':a,'port':b},'to':{'node':c,'port':d}} for a,b,c,d in edges]}, {'s':source}


def test_reference_scaler_and_stratified_disjoint_lineage(tmp_path):
    graph,bindings=fixture(tmp_path); result=execute(graph,bindings)
    train=result['p']['training']; val=result['p']['validation']; out=result['t']['table']; state=result['f']['state']
    x=np.array(train.values('x')); held=np.array(val.values('x'))
    np.testing.assert_allclose(out.values('x'),(held-x.mean())/x.std())
    assert set(train.row_ids).isdisjoint(val.row_ids)
    assert set(train.values('label'))==set(val.values('label'))=={0,1}
    assert state['fitEvidence']['membershipDigest']==membership_digest(train.row_ids)
    assert execute(graph,bindings)['p']['validation'].row_ids==val.row_ids
    assert out.row_ids==val.row_ids and out.role=='validation'
    assert load_state(json.dumps(state))==state


def test_heldout_outlier_never_changes_fitted_statistics(tmp_path):
    graph,bindings=fixture(tmp_path); result=execute(graph,bindings)
    train=result['p']['training']; state=result['f']['state']
    assert state['rules'][0]['mean']==pytest.approx(np.mean(train.values('x')))
    assert list(fit_observation_ids(state))==list(train.row_ids)
    graph['edges'][3]['from']['port']='validation'
    with pytest.raises(ValueError,match='[Tt]raining'):execute(graph,bindings)


def test_validation_precedes_source_loading_and_rejects_unknown_fields(tmp_path):
    graph,bindings=fixture(tmp_path); graph['nodes'][1]['config']['typo']=True
    with pytest.raises(ValueError,match='configuration'):execute(graph,{'s':tmp_path/'absent'})
    graph,bindings=fixture(tmp_path);graph['edges'].append(copy.deepcopy(graph['edges'][0]))
    with pytest.raises(ValueError,match='connected'):execute(graph,bindings)


def test_missing_cast_filter_and_bounds(tmp_path):
    graph,bindings=fixture(tmp_path)
    bindings['s'].write_text('raw,label\n,0\n2,1\n3,0\n4,1\n5,0\n6,1\n7,0\n8,1\n')
    with pytest.raises(ValueError,match='missing'):execute(graph,bindings)
    graph['nodes'].insert(2,{'id':'m','operator':'kwip.data.preprocess','config':dict(mode='table',columns=[dict(rule_defaults('x'),missing='constant',value='1')])})
    graph['edges'][1]['to']['node']='m';graph['edges'].append({'from':{'node':'m','port':'table'},'to':{'node':'c','port':'table'}})
    result=execute(graph,bindings);assert 1.0 in result['c']['table'].values('x')
    assert len(result['s']['table'].preview()['rows'])<=8
    bindings['s'].write_text('raw,label\nNaN,0\n2,1\n')
    with pytest.raises(ValueError,match='finite'):execute(graph,bindings)


def test_state_corruption_and_stale_source_rejected(tmp_path):
    graph,bindings=fixture(tmp_path);state=execute(graph,bindings)['f']['state'];state['rules'][0]['mean']+=1
    with pytest.raises(ValueError,match='digest'):load_state(json.dumps(state))


def test_outlier_transform_state_and_overlap_guard(tmp_path):
    from runner.dataflow import derive, transform, digest
    graph,bindings=fixture(tmp_path);result=execute(graph,bindings)
    state=result['f']['state'];held=result['p']['validation'];before=copy.deepcopy(state)
    changed=derive(held,rows=[(1e9,r[1]) for r in held.rows])
    transformed=transform(changed,state,digest(graph))
    np.testing.assert_allclose(transformed.values('x'),(1e9-state['rules'][0]['mean'])/state['rules'][0]['scale'])
    assert state==before
    overlap=derive(held,ids=result['p']['training'].row_ids[:len(held.rows)])
    with pytest.raises(ValueError,match='overlaps training'):transform(overlap,state,digest(graph))
    stale=derive(held,source_digest='other')
    with pytest.raises(ValueError,match='differs'):transform(stale,state,digest(graph))


def test_filter_preserves_ids_and_lossy_cast_rejected(tmp_path):
    graph,bindings=fixture(tmp_path)
    graph['nodes'].insert(3,{'id':'q','operator':'kwip.data.filterRows','config':dict(mode='simple',conditions=[dict(column='x',comparison='ge',value=4,caseSensitive=True)],match='all',expression='',column='value',pattern='',matchMode='search',flags=[],nullPolicy='error')})
    graph['edges'][2]['to']['node']='q';graph['edges'].append({'from':{'node':'q','port':'matching'},'to':{'node':'p','port':'table'}})
    result=execute(graph,bindings)
    assert result['q']['matching'].row_ids==result['c']['table'].row_ids[4:]
    from runner.dataflow import cast_value
    for value in ['1.2','1e3',1.5,2**63]:
        with pytest.raises(ValueError):cast_value(value,'int64')


def test_constant_scale_and_unstratified_default(tmp_path):
    graph,bindings=fixture(tmp_path);del graph['nodes'][3]['config']['stratifyColumn']
    bindings['s'].write_text('raw,label\n'+''.join(f'3,{i}\n' for i in range(20)))
    result=execute(graph,bindings)
    assert result['f']['state']['rules'][0]['scale']==1.0
    assert result['t']['table'].values('x')==[0.0]*4


def test_cycle_and_missing_required_port_rejected(tmp_path):
    graph,bindings=fixture(tmp_path);graph['edges'][0]['from']={'node':'c','port':'table'}
    with pytest.raises(ValueError,match='cycle'):execute(graph,bindings)
    graph,bindings=fixture(tmp_path);graph['edges'].pop()
    with pytest.raises(ValueError,match='not connected'):execute(graph,bindings)


def test_tampered_training_values_cannot_use_original_fit_state(tmp_path):
    from runner.dataflow import derive, transform, digest
    graph,bindings=fixture(tmp_path);result=execute(graph,bindings);train=result['p']['training']
    changed=derive(train,rows=[(999.,r[1]) for r in train.rows])
    with pytest.raises(ValueError,match='Training data differs'):transform(changed,result['f']['state'],digest(graph))

@pytest.mark.parametrize('payload',['[]','null','{"contentDigest":"x"}','{"means":[NaN]}'])
def test_malformed_state_always_reports_validation_error(payload):
    with pytest.raises(ValueError):load_state(payload)


def test_live_progress_brackets_real_operator_work_and_keeps_numerical_result(tmp_path):
    graph,bindings=fixture(tmp_path)
    timeline=[]
    result=execute(graph,bindings,on_node=lambda ident,out:timeline.append(('finished',ident)),
                   on_node_started=lambda ident:timeline.append(('started',ident)))
    assert timeline == [event for ident in ['s','r','c','p','f','t'] for event in [('started',ident),('finished',ident)]]
    assert result['s']['table'].rows[0] == ('0','0')


def test_live_progress_starts_failed_operator_without_finishing_or_downstream(tmp_path):
    graph,bindings=fixture(tmp_path)
    bindings['s'].write_text('raw,label\nNaN,0\n2,1\n')
    timeline=[]
    with pytest.raises(ValueError,match='finite'):
        execute(graph,bindings,on_node=lambda ident,out:timeline.append(('finished',ident)),
                on_node_started=lambda ident:timeline.append(('started',ident)))
    assert timeline == [('started','s'),('finished','s'),('started','r'),('finished','r'),('started','c')]

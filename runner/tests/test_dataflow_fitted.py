"""Current Preprocess numerical/leakage tests, plus explicit retired-fit refusal."""
import copy
import json
import numpy as np
import pytest
from runner import dataflow


def table(rows,role='training',ids=None,types=('float64','string')):
    cols=tuple({'name':name,'dtype':dtype,'nullable':any(r[i] is None for r in rows)} for i,(name,dtype) in enumerate(zip(['amount','kind'],types)))
    return dataflow.derive(dataflow.Table(cols,tuple(map(tuple,rows)),tuple(ids or [f'source:{i}' for i in range(len(rows))]),'a'*64,role,partition_digest='b'*64))


def fit(value,operator,config):
    from runner.dataflow_preprocess import apply,rule_defaults
    rules=[]
    for name in config['columns']:
        rule=rule_defaults(name)
        if operator=='standardScalerFit':rule['scaling']='standard'
        elif operator=='imputerFit':rule.update(missing=config['strategy'],emptyPolicy=config['emptyPolicy'],emptyValue=config.get('emptyValue',0),tiePolicy=config.get('tiePolicy','error'))
        else:rule.update(encoding='oneHot',vocabularyOrder=config['vocabularyOrder'],unknownPolicy=config['unknownPolicy'],maxCategories=config['maxCategories'])
        rules.append(rule)
    return apply({'training':value},{'mode':'partitions','columns':rules},'c'*64,lambda:None)['state']


@pytest.mark.parametrize('strategy',['mean','median'])
def test_imputer_uses_training_only_numeric_reference(strategy):
    training=table([[1.,'a'],[None,'b'],[9.,'a'],[3.,'b']]);held=table([[None,'c'],[1e9,'a']],role='validation',ids=['held:0','held:1'])
    # Schemas are declared, so holdout nullability must match the input contract.
    state=fit(training,'imputerFit',{'columns':['amount'],'strategy':strategy,'emptyPolicy':'error'})
    out=dataflow.transform(held,state,'c'*64)
    expected=getattr(np,strategy)([1.,9.,3.])
    assert state['rules'][0]['fill']==expected and out.values('amount')==[expected,1e9]
    assert out.row_ids==held.row_ids and out.role=='validation'
    assert dataflow.load_state(json.dumps(state))==state


def test_most_frequent_tie_empty_and_explicit_constant():
    training=table([[None,'z'],[None,'a']])
    state=fit(training,'imputerFit',{'columns':['kind'],'strategy':'mostFrequent','tiePolicy':'smallest','emptyPolicy':'error'})
    assert state['rules'][0]['fill']=='a'
    with pytest.raises(ValueError,match='tie'):fit(training,'imputerFit',{'columns':['kind'],'strategy':'mostFrequent','tiePolicy':'error','emptyPolicy':'error'})
    with pytest.raises(ValueError,match='Entirely missing'):fit(training,'imputerFit',{'columns':['amount'],'strategy':'mean','emptyPolicy':'error'})
    state=fit(training,'imputerFit',{'columns':['amount'],'strategy':'median','emptyPolicy':'constant','emptyValue':2.5})
    assert state['rules'][0]['fill']==2.5 and state['fitEvidence']['observationIDs']==list(training.row_ids)
    with pytest.raises(ValueError,match='type'):fit(training,'imputerFit',{'columns':['amount'],'strategy':'mean','emptyPolicy':'constant','emptyValue':'bad'})


def test_encoder_vocabulary_output_reference_and_unknown_decisions():
    training=table([[1.,'z'],[2.,'a'],[3.,'z']]);held=table([[4.,'NEW'],[5.,'z']],role='validation',ids=['held:0','held:1'])
    config={'columns':['kind'],'vocabularyOrder':'lexicographic','unknownPolicy':'ignore','maxCategories':10}
    state=fit(training,'oneHotEncoderFit',config)
    assert state['rules'][0]['vocabulary']==['a','z']
    output=dataflow.transform(held,state,'c'*64)
    np.testing.assert_array_equal(np.array(output.rows)[:,1:],[[0.,0.],[0.,1.]])
    assert output.diagnostics[-1]['count']==1 and output.preview()['diagnostics'][-1]['code']=='unknownCategoryIgnored'
    strict=fit(training,'oneHotEncoderFit',{**config,'unknownPolicy':'error'})
    with pytest.raises(ValueError,match='Unknown category'):dataflow.transform(held,strict,'c'*64)
    first=fit(training,'oneHotEncoderFit',{**config,'vocabularyOrder':'firstSeen'})
    assert first['rules'][0]['vocabulary']==['z','a']
    assert all('displayLabel' in col for col in output.columns[1:])


def test_fit_validation_leakage_and_state_chain_rejection():
    training=table([[1.,'a'],[None,'b']]);config={'columns':['amount'],'strategy':'mean','emptyPolicy':'error'}
    with pytest.raises(ValueError,match='[Tt]raining'):fit(dataflow.derive(training,role='validation'),'imputerFit',config)
    imputer=fit(training,'imputerFit',config);filled=dataflow.transform(training,imputer,'c'*64)
    scaler=fit(filled,'standardScalerFit',{'columns':['amount']})
    wrong=dataflow.derive(filled,fitted_states=('other-state',))
    with pytest.raises(ValueError,match='chain'):dataflow.transform(wrong,scaler,'c'*64)


def test_encoder_bounds_and_output_collision():
    value=table([[1.,'a'],[2.,'b']]);config={'columns':['kind'],'vocabularyOrder':'lexicographic','unknownPolicy':'error','maxCategories':1}
    with pytest.raises(ValueError,match='categor'):fit(value,'oneHotEncoderFit',config)
    config['maxCategories']=10;state=fit(value,'oneHotEncoderFit',config)
    collision=dataflow.derive(value,columns=[{**value.columns[0],'name':state['outputSchema'][1]['name']},value.columns[1]])
    with pytest.raises(ValueError,match='collid'):fit(collision,'oneHotEncoderFit',config)


def test_csv_cast_explicitly_preserves_missing_for_statistical_fit(tmp_path):
    from runner.dataflow_preprocess import rule_defaults
    source=tmp_path/'nullable.csv';source.write_text('amount,kind\n,a\n2,b\n4,a\n6,b\n8,a\n10,b\n')
    cast=dict(rows=[dict(source='amount',included=True,outputName='amount',targetType='float64'),dict(source='kind',included=True,outputName='kind',targetType='keep')],castFailure='error',trimText=False,newColumns='error',missingPolicy='preserve')
    nodes=[('s','sourceCSV',{}),('c','columns',cast),('p','partition',dict(strategy='random',seed=42,validationPercent=30,testPercent=20)),('f','preprocess',dict(mode='partitions',columns=[dict(rule_defaults('amount'),missing='mean')]))]
    edges=[('s','table','c','table'),('c','table','p','table'),('p','training','f','training'),('p','validation','f','validation')]
    graph={'schemaVersion':1,'nodes':[{'id':i,'operator':'kwip.data.'+op,'config':config} for i,op,config in nodes],'edges':[{'from':{'node':a,'port':b},'to':{'node':c,'port':d}} for a,b,c,d in edges]}
    result=dataflow.execute(graph,{'s':source})
    assert result['c']['table'].rows[0][0] is None
    assert result['c']['table'].columns[0]['dtype']=='float64'
    assert None not in result['f']['validation'].values('amount')


def test_original_scaler_state_remains_readable():
    from runner.tests.test_portable_fitted import fixture
    from runner import dataflow_fitted as historical
    training,state=fixture(historical.SCALER,{},[1.,3.])
    del state['outputSchema'];del state['inputFittedStateDigests'];del state['contentDigest'];state['contentDigest']=dataflow.digest(state)
    loaded=dataflow.load_state(json.dumps(state));out=dataflow.transform(training,loaded,'PRIVATE-GRAPH')
    np.testing.assert_array_equal(out.values('x'),[-1.,1.])


def test_encoder_total_width_and_names_are_unambiguous():
    value=table([[float(i),str(i)] for i in range(128)])
    config={'columns':['kind'],'vocabularyOrder':'lexicographic','unknownPolicy':'error','maxCategories':128}
    with pytest.raises(ValueError,match='128'):fit(value,'oneHotEncoderFit',config)
    tricky='a",[雪]\\';value=table([[1.,tricky],[2.,'other']]);state=fit(value,'oneHotEncoderFit',config)
    generated=state['outputSchema'][1:]
    assert [json.loads(c['name'].removeprefix('onehot:')) for c in generated]==[['kind',v] for v in state['rules'][0]['vocabulary']]


def test_bundled_chain_artifact_replay_and_prepared_training_integrity(tmp_path):
    from pathlib import Path
    from runner.dataflow_cli import run_dataflow
    from runner.prepared_data import load_prepared
    base=Path(dataflow.__file__).parent/'dataflow_examples';output=tmp_path/'prepared'
    events=[];manifest=run_dataflow(base/'preparation.json',{'source':str(base/'observations.csv')},output,events.append)
    prepared=load_prepared(output/'manifest.json','training','table','validation','table',['amount','other'],'group')
    assert [len(t) for t in prepared[0]]==[24,24,8,8]
    assert len(prepared[1]['preparation']['fittedStateDigests'])==1
    final=next(e for e in events if e.get('nodeID')=='validation' and e.get('event')=='dataflowPreview')['outputs']['table']['preview']
    assert final['rowCount']==8
    # Unknown-category policy has separate held-out-only adversarial fixtures above.
    state_event=next(e for e in events if e.get('nodeID')=='prepare' and e.get('event')=='dataflowPreview')
    assert state_event['outputs']['state']['kind']=='state'
    assert (output/'dataflow_fitted.py').is_file()
    import subprocess,sys
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(base/'observations.csv')}))
    replay=tmp_path/'replayed'
    process=subprocess.run([sys.executable,str(output/'prep.py'),'--source-bindings',str(bindings),'--output',str(replay)],capture_output=True,text=True)
    assert process.returncode==0,process.stdout+process.stderr
    assert json.loads((replay/'manifest.json').read_text())['artifacts']==manifest['artifacts']


def test_retired_fitted_api_cannot_learn_new_states():
    from runner.dataflow_fitted import fit as retired_fit
    with pytest.raises(ValueError,match='Retired fitted execution'):retired_fit(table([[1.,'a']]),'standardScalerFit',{'columns':['amount']},'c'*64)

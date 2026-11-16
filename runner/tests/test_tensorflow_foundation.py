"""Real CPU TensorFlow lifecycle; explicitly opt in via the isolated TF environment."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
pytest.importorskip('tensorflow')
from runner import tensorflow_foundation as adapter

EXAMPLE=Path(__file__).parents[1]/'backend_examples/tensorflow_classifier.json'

def fixture():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(64,2)).astype('float32')
    y=(x[:,0]+x[:,1]>0).astype('int64')
    return x[:48],y[:48],x[48:],y[48:]

def test_real_fit_export_reload_and_relocated_inference(tmp_path):
    definition=json.loads(EXAMPLE.read_text()); x,y,v,w=fixture(); events=[]
    model,history=adapter.fit(definition['graph'],x,y,v,w,definition['settings'],event=events.append)
    before=adapter.predict(model,v)
    assert before.shape==(16,2) and np.isfinite(before).all()
    assert history[-1]['loss']<history[0]['loss']
    assert adapter.evaluate(model,v,w)['accuracy']>=.75
    assert len(events)==definition['settings']['epochs']
    package=tmp_path/'export'; adapter.export(model,definition['graph'],package)
    loaded=adapter.load(package)
    np.testing.assert_array_equal(before,adapter.predict(loaded,v))
    np.save(tmp_path/'input.npy',v)
    completed=subprocess.run([sys.executable,str(package/'inference.py'),str(tmp_path/'input.npy')],cwd=tmp_path,capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    np.testing.assert_allclose(np.array(json.loads(completed.stdout)['logits']),before,rtol=1e-6)
    with pytest.raises(FileExistsError):adapter.export(model,definition['graph'],package)
    with (package/'model.weights.h5').open('ab') as f:f.write(b'altered')
    with pytest.raises(ValueError,match='integrity'):adapter.load(package)

def test_graph_branches_and_dense_transpose_semantics():
    definition=json.loads(EXAMPLE.read_text()); model=adapter.build(definition['graph'],seed=2)
    layers=[layer for layer in model.layers if layer.__class__.__name__=='Dense']
    layers[0].set_weights([np.arange(16,dtype='float32').reshape(2,8)/10,np.zeros(8,dtype='float32')])
    layers[1].set_weights([np.ones((8,2),dtype='float32'),np.array([1,2],dtype='float32')])
    x=np.array([[1,2],[-1,-2]],dtype='float32')
    expected=np.maximum(x@layers[0].get_weights()[0],0)@np.ones((8,2))+[1,2]
    np.testing.assert_allclose(adapter.predict(model,x),expected,rtol=1e-6)

def test_seed_replay_cancel_validation_and_contract():
    d=json.loads(EXAMPLE.read_text()); x,y,v,w=fixture(); settings={**d['settings'],'epochs':2}
    a,_=adapter.fit(d['graph'],x,y,v,w,settings); b,_=adapter.fit(d['graph'],x,y,v,w,settings)
    np.testing.assert_array_equal(adapter.predict(a,v),adapter.predict(b,v))
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):adapter.fit(d['graph'],x,y,v,w,settings,check_cancelled=cancel)
    with pytest.raises(ValueError,match='overlap'):adapter.fit(d['graph'],x,y,x[:2],y[:2],settings)
    with pytest.raises(ValueError):adapter.predict(a,np.array([[np.nan,1]],dtype='float32'))
    with pytest.raises(ValueError):adapter.fit(d['graph'],x,y,v,w,{**settings,'device':'mps'})
    assert adapter.capabilities()['devices']==['cpu']
    for mutation in ['unknown','cycle','width','disconnected']:
        bad=copy.deepcopy(d['graph'])
        if mutation=='unknown':bad['nodes'][1]['type']='custom_code'
        if mutation=='cycle':bad['edges'][0]['source']=bad['edges'][0]['target']
        if mutation=='width':bad['nodes'][1]['config']['in_features']=3
        if mutation=='disconnected':bad['edges'].pop()
        with pytest.raises(ValueError):adapter.build(bad)

def test_export_rejects_wrong_authored_graph_and_nonfinite_weights(tmp_path):
    d=json.loads(EXAMPLE.read_text());model=adapter.build(d['graph'])
    wrong=copy.deepcopy(d['graph']);wrong['nodes'][2]['type']='tanh'
    with pytest.raises(ValueError,match='graph'):adapter.export(model,wrong,tmp_path/'wrong')
    values=model.get_weights();values[0][0,0]=np.inf;model.set_weights(values)
    with pytest.raises(ValueError,match='finite'):adapter.export(model,d['graph'],tmp_path/'invalid')
    assert not (tmp_path/'wrong').exists() and not (tmp_path/'invalid').exists()

def test_cancel_during_batches_and_branch_connections():
    d=json.loads(EXAMPLE.read_text());x,y,v,w=fixture();calls=[];events=[]
    def cancel():
        calls.append(1)
        if len(calls)==3:raise RuntimeError('batch cancellation')
    with pytest.raises(RuntimeError,match='batch cancellation'):
        adapter.fit(d['graph'],x,y,v,w,d['settings'],check_cancelled=cancel,event=events.append)
    assert not events
    graph=d['graph'];graph['nodes'].insert(3,{'id':'skip','type':'add','config':{}})
    graph['edges']=[e for e in graph['edges'] if e['target']!='logits']
    graph['edges'] += [{'id':'a','source':'hidden','sourcePort':'out','target':'skip','targetPort':'a'},
                       {'id':'b','source':'activation','sourcePort':'out','target':'skip','targetPort':'b'},
                       {'id':'c','source':'skip','sourcePort':'out','target':'logits','targetPort':'in'}]
    model=adapter.build(graph);assert adapter.predict(model,v).shape==(16,2)

def test_portable_graph_omits_source_location(tmp_path):
    d=json.loads(EXAMPLE.read_text());d['graph']['execution']={'datasetCSV':'/private/user/data.csv'}
    model=adapter.build(d['graph']);adapter.export(model,d['graph'],tmp_path/'model')
    assert '/private/user' not in (tmp_path/'model/graph.json').read_text()
    assert adapter.predict(adapter.load(tmp_path/'model'),np.ones((1,2),dtype='float32')).shape==(1,2)

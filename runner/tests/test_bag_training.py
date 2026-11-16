"""Actual packet training, empty observation alignment and continuation."""
import dataclasses
import json
import pytest
import torch
from runner import engine
from runner.tests.test_bag_source_graph import registry,graph,packet


@pytest.fixture(autouse=True)
def restore_worker_determinism_after_in_process_test():
    # Production runs own an isolated worker. These tests invoke several worker
    # runs in one process, so their CPU policy must not leak into MPS tests.
    enabled=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    try:yield
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn_only)


def prepared():
    tx=packet('cpu');vx=packet('cpu')
    # Table-derived floating input weights are values, not learned parameters.
    tx=dataclasses.replace(tx,per_sample_weights=tx.per_sample_weights.detach())
    vx=dataclasses.replace(vx,per_sample_weights=vx.per_sample_weights.detach())
    targets=torch.tensor([0,1,1,0])
    return (tx,targets,vx,targets.clone()),dict(sourceHash='a'*64,splitHash='b'*64,
        trainingRowIDs=['t0','t1','t2','t3'],validationRowIDs=['v0','v1','v2','v3'],
        modelInput={'kind':'embeddingBagPacket','schemaVersion':1}),{}


def manual(seed,batch_size,epochs):
    torch.manual_seed(seed)
    model=torch.nn.EmbeddingBag(5,2,mode='sum',padding_idx=0,include_last_offset=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    ids=[[1,2,1],[],[0,3],[4,0]];weights=[[.5,2.,-1.],[],[99.,3.],[.25,88.]]
    labels=torch.tensor([0,1,1,0]);criterion=torch.nn.CrossEntropyLoss();losses=[]
    for _ in range(epochs):
        total=0.
        for begin in range(0,4,batch_size):
            end=min(4,begin+batch_size);selected=ids[begin:end];importance=weights[begin:end]
            starts=[0]
            for row in selected:starts.append(starts[-1]+len(row))
            optimizer.zero_grad()
            output=model(torch.tensor([v for row in selected for v in row],dtype=torch.int64),
                         torch.tensor(starts),per_sample_weights=torch.tensor([v for row in importance for v in row]))
            loss=criterion(output,labels[begin:end]);loss.backward();optimizer.step()
            total+=loss.item()*(end-begin)
        losses.append(total/4)
    return model.weight.detach(),losses


@pytest.mark.parametrize('batch_size',[1,3])
def test_training_uses_whole_bags_empty_targets_and_real_native_reference(tmp_path,registry,batch_size):
    expected,losses=manual(42,batch_size,2);events=[];output=tmp_path/'run'
    result=engine.run(graph(),output,registry=registry,prepared=prepared(),device='cpu',epochs=2,batch_size=batch_size,emit=events.append)
    state=torch.load(output/'checkpoint.pt',weights_only=True,map_location='cpu')
    torch.testing.assert_close(state['modelState']['layers.bag.weight'],expected,rtol=0,atol=0)
    actual=json.loads((output/'metrics.json').read_text())
    assert [row['trainLoss'] for row in actual]==losses
    started=next(event for event in events if event['event']=='started')
    assert started['trainingSamples']==4 and started['validationSamples']==4
    assert result['evaluationCount']==4 and len(result['predictionRows'])==4
    assert result['predictionInput']['kind']=='bagPacket'
    assert result['predictionRows'][1]['input']['indices']==[]
    assert result['predictionRows'][1]['label']==1
    assert result['predictionRows'][0]['input']['per_sample_weights']==[.5,2.,-1.]
    assert result['previews']['source']['outputs']['offsets']['shape']==[5]
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['settings']['lastBatchSize']==(4%batch_size or batch_size)
    assert manifest['predictionPreview']['schemaVersion']==3


def test_exact_resume_matches_full_and_changed_bag_boundaries_refuse(tmp_path,registry):
    common=dict(registry=registry,prepared=prepared(),device='cpu',batch_size=3)
    engine.run(graph(),tmp_path/'full',epochs=4,**common)
    engine.run(graph(),tmp_path/'first',epochs=2,**common)
    engine.run(graph(),tmp_path/'resumed',epochs=4,resume_checkpoint=tmp_path/'first/checkpoint.pt',**common)
    def state(name):return torch.load(tmp_path/name/'checkpoint.pt',weights_only=True,map_location='cpu')
    assert torch.equal(state('full')['modelState']['layers.bag.weight'],state('resumed')['modelState']['layers.bag.weight'])
    assert json.loads((tmp_path/'full/metrics.json').read_text())==json.loads((tmp_path/'resumed/metrics.json').read_text())
    data,provenance,files=prepared()
    changed=dataclasses.replace(data[0],offsets=torch.tensor([0,2,3,5,7]))
    with pytest.raises(ValueError):engine.run(graph(),tmp_path/'bad',epochs=4,resume_checkpoint=tmp_path/'first/checkpoint.pt',**{**common,'prepared':((changed,*data[1:]),provenance,files)})
    assert not (tmp_path/'bad/manifest.json').exists()


@pytest.mark.parametrize('device',['cpu','mps'])
def test_real_csv_preparation_training_and_isolated_saved_export(tmp_path,registry,device):
    import shutil,subprocess,sys
    from runner.tests.test_prepared_bag_loader import fixture,load
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    source=fixture(tmp_path/'source');admitted=load(source)
    value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2
    output=tmp_path/'run';events=[]
    result=engine.run(value,output,registry=registry,prepared=admitted,device=device,epochs=2,batch_size=3,emit=events.append)
    manifest=json.loads((output/'manifest.json').read_bytes());provenance=admitted[1]
    assert manifest['provenance']['bagSources']==provenance['bagSources']
    assert manifest['checkpoint']['contract']['data']['bagInput']['sources']==provenance['bagSources']
    assert (output/'preparation/manifest.json').read_bytes()==(source/'manifest.json').read_bytes()
    val=admitted[0][2];weights=torch.load(output/'checkpoint.pt',weights_only=True,map_location='cpu')['modelState']['layers.bag.weight']
    expected=[]
    for begin,end in zip(val.offsets.tolist(),val.offsets.tolist()[1:]):
        row=torch.zeros(2)
        for i in range(begin,end):
            if val.indices[i]!=0:row+=weights[int(val.indices[i])]*val.per_sample_weights[i]
        expected.append(row)
    expected=torch.stack(expected)
    expected_loss=torch.nn.functional.cross_entropy(expected,admitted[0][3]).item()
    assert result['validationLoss']==pytest.approx(expected_loss,rel=2e-5,abs=2e-5)
    isolated=tmp_path/'isolated';isolated.mkdir()
    shutil.copyfile(output/'model.py',isolated/'model.py')
    torch.save(dict(weights=weights,indices=val.indices,offsets=val.offsets,importance=val.per_sample_weights,expected=expected),isolated/'input.pt')
    (isolated/'check.py').write_text('''import torch
import importlib.util,sys
spec=importlib.util.spec_from_file_location('portable', 'model.py');module=importlib.util.module_from_spec(spec);sys.modules['portable']=module;spec.loader.exec_module(module);Model=module.Model
item=torch.load('input.pt',weights_only=True,map_location='cpu')
model=Model();model.load_state_dict({'layers.bag.weight':item['weights']},strict=True)
actual=model(item['indices'],item['offsets'],item['importance'])
torch.testing.assert_close(actual,item['expected'],rtol=2e-6,atol=2e-6)
assert actual.shape[0]==8
''')
    child=subprocess.run([sys.executable,'-I',str(isolated/'check.py')],cwd=isolated,capture_output=True,text=True,timeout=30)
    assert child.returncode==0,child.stderr
    assert [e['nodeID'] for e in events if e['event']=='modelNodeCompleted']==['source','bag','result']


def test_explicit_objective_and_capture_policies_keep_real_bag_rows(tmp_path,registry):
    from runner.tests.test_prepared_bag_loader import fixture,load
    from runner.loss_configuration import validate_recipe
    source=fixture(tmp_path/'source')
    recipe=validate_recipe({'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':{'reduction':'none'}})
    mapping={'schemaVersion':1,'componentID':'loss_mse','inputColumns':['i1','i2'],
             'roles':[{'role':'target','columns':['desired'],'dtype':'float32','sampleShape':[1]}],'tripletSourceGroups':[]}
    admitted=load(source,embedding_dim=1,target_column=None,objective_mapping=mapping,objective_recipe=recipe,prediction_shape=[1])
    value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2;value['nodes'][1]['config']['embedding_dim']=1
    result=engine.run(value,tmp_path/'run',registry=registry,prepared=admitted,device='cpu',epochs=2,batch_size=3,objective_recipe=recipe,target_mapping=mapping)
    assert 'accuracy' not in result and len(result['predictionRows'])==8
    assert all('targets'in row and 'objectiveOutput'in row and 'label'not in row for row in result['predictionRows'])
    omitted=engine.run(value,tmp_path/'omitted',registry=registry,prepared=admitted,device='cpu',epochs=1,batch_size=3,objective_recipe=recipe,target_mapping=mapping,sample_capture={'schemaVersion':1,'activationValues':False,'predictionRows':False})
    assert omitted['predictionRows']==[]
    assert all(output['values']==[]for output in omitted['previews']['source']['outputs'].values())
    preview=json.loads((tmp_path/'omitted/predictions.json').read_text())
    assert preview['schemaVersion']==3 and preview['captureStatus']=='omittedByPolicy'

"""Standalone exported helper: real learned weights, no KwipML runtime imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
import pytest
from runner.engine import run, GraphModel
from runner.tests.test_runner import graph

HELPER=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/inference.py'

def bundle(tmp_path, graph_value=None, registry=None, prepared=None, input_contract=None):
    value=graph() if graph_value is None else graph_value
    origin=tmp_path/'private-training';run(value,origin,epochs=2,device='cpu',registry=registry,prepared=prepared)
    target=tmp_path/'portable';target.mkdir()
    for name in ('model.py','checkpoint.pt'):shutil.copyfile(origin/name,target/name)
    shutil.copyfile(HELPER,target/'inference.py')
    contract={'schemaVersion':1,'input':{'dtype':'float32','shape':['N',2],'featureColumns':['first','second'],'preprocessing':'alreadyPrepared'},'output':{'kind':'logits','shape':['N',2],'classIndices':[0,1]}}
    if input_contract is not None:contract['input']=input_contract
    (target/'input-contract.json').write_text(json.dumps(contract))
    files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in target.iterdir()]
    (target/'export-manifest.json').write_text(json.dumps({'schemaVersion':1,'kind':'kwipmlTrainedModel','files':files}))
    model=GraphModel(value,registry);model.load_state_dict(torch.load(origin/'checkpoint.pt',weights_only=True)['modelState']);model.eval()
    shutil.rmtree(origin)
    spec=importlib.util.spec_from_file_location('exported_inference',target/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return target,module,model

def test_exported_helper_reloads_trained_weights_without_original_run(tmp_path):
    target,helper,reference=bundle(tmp_path)
    x=torch.tensor([[0.,0.],[1.,-1.],[.125,.75]],dtype=torch.float32)
    expected=reference(x).detach()
    actual=helper.predict(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    result=subprocess.run([sys.executable,str(target/'inference.py'),'--example'],cwd=tmp_path.parent,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout)
    assert output['inputKind']=='synthetic prepared example'
    torch.testing.assert_close(torch.tensor(output['logits']),expected[:2],rtol=1e-6,atol=1e-7)

@pytest.mark.parametrize('bad',[torch.ones(2),torch.ones(0,2),torch.ones(2,3),torch.ones(2,2,dtype=torch.float64),torch.tensor([[float('nan'),0.]]),torch.ones(2,2,dtype=torch.int64)])
def test_input_contract_rejects_invalid_tensors(tmp_path,bad):
    _,helper,_=bundle(tmp_path)
    with pytest.raises(ValueError):helper.predict(bad)

@pytest.mark.parametrize('name',['model.py','checkpoint.pt','input-contract.json'])
def test_modified_artifact_refused_before_model_code_runs(tmp_path,name):
    target,helper,_=bundle(tmp_path)
    with (target/name).open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):helper.predict(torch.zeros(1,2))

def test_manifest_cannot_redirect_to_external_file(tmp_path):
    target,helper,_=bundle(tmp_path)
    manifest=json.loads((target/'export-manifest.json').read_text())
    manifest['files'][0]['path']='../private'
    (target/'export-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):helper.predict(torch.zeros(1,2))

def test_imported_helper_does_not_inherit_host_default_dtype(tmp_path):
    _,helper,reference=bundle(tmp_path)
    x=torch.tensor([[.25,-.5]],dtype=torch.float32)
    expected=reference(x).detach()
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual=helper.predict(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.get_default_dtype()==torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_reusable_predictor_loads_once_and_batches_in_original_order(tmp_path, monkeypatch):
    target, helper, reference = bundle(tmp_path)
    x = torch.linspace(-1, 1, 22).reshape(11, 2)
    expected = reference(x).detach()
    predictor = helper.load_model()
    def no_reload(*args, **kwargs):
        raise AssertionError('Prediction must use the verified loaded snapshot')
    monkeypatch.setattr(helper, 'verified_files', no_reload)
    monkeypatch.setattr(torch, 'load', no_reload)
    # Loaded sessions remain useful even if the original files move or disappear.
    shutil.rmtree(target)
    observed = []
    handle = predictor._model.register_forward_pre_hook(lambda model, inputs: observed.append(inputs[0].shape[0]))
    for batch_size in (1, 3, 11, 20):
        observed.clear()
        actual = predictor.predict(x, batch_size=batch_size)
        assert sum(observed) == len(x) and max(observed) <= batch_size
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        assert not actual.requires_grad
    handle.remove()


def test_model_loading_preserves_host_cpu_random_generator(tmp_path):
    _, helper, _ = bundle(tmp_path)
    before = torch.random.get_rng_state().clone()
    helper.predict(torch.zeros(1, 2))
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5, 100001, None])
def test_reusable_predictor_rejects_invalid_batch_budget(tmp_path, batch_size):
    _, helper, _ = bundle(tmp_path)
    predictor = helper.load_model()
    with pytest.raises(ValueError, match='batch_size'):
        predictor.predict(torch.zeros(2, 2), batch_size=batch_size)


def test_cli_accepts_explicit_prediction_batch_size(tmp_path):
    target, _, reference = bundle(tmp_path)
    result = subprocess.run([sys.executable, str(target/'inference.py'), '--example', '--batch-size', '1'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = reference(torch.tensor([[0., 0.], [1., -1.]])).detach()
    torch.testing.assert_close(torch.tensor(json.loads(result.stdout)['logits']), expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('shape',[[2],[1,2]])
def test_integer_export_preserves_sample_shape_and_cli_values(tmp_path,shape):
    from runner.tests.test_embedding_graph import graph as embedding_graph
    from runner.tests.test_embedding_lifecycle import prepared
    value=embedding_graph();value['nodes'][0]['config']['shape']=shape
    # The same reviewed definitions used by the custom-registry runner tests.
    from runner.tests.test_embedding_graph import registry as fixture
    registry=fixture.__wrapped__(tmp_path)
    data,source,extra=prepared()
    data=(data[0].reshape(-1,*shape),data[1],data[2].reshape(-1,*shape),data[3])
    contract={'dtype':'int64','shape':['N',*shape],'featureColumns':['a','b'],'preprocessing':'alreadyPrepared'}
    target,helper,reference=bundle(tmp_path,value,registry,(data,source,extra),contract)
    predictor=helper.load_model()
    expected=reference(data[0]).detach()
    torch.testing.assert_close(predictor.predict(data[0],batch_size=3),expected)
    with pytest.raises(ValueError):predictor.predict(data[0].float())
    path=tmp_path/'inputs.json';path.write_text(json.dumps(data[0].tolist()))
    result=subprocess.run([sys.executable,str(target/'inference.py'),'--input',str(path),'--batch-size','3'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    torch.testing.assert_close(torch.tensor(json.loads(result.stdout)['logits']),expected)
    bad=data[0].tolist()
    row=bad[0]
    while isinstance(row[0],list):row=row[0]
    row[0]=1.5
    path.write_text(json.dumps(bad))
    rejected=subprocess.run([sys.executable,str(target/'inference.py'),'--input',str(path)],capture_output=True,text=True)
    assert rejected.returncode!=0
    assert 'integer' in rejected.stderr


def test_integer_json_conversion_is_exact_and_bounded(tmp_path):
    _,helper,_=bundle(tmp_path)
    values=[[2**53+1,2**63-1],[-2**63,0]]
    tensor=helper.prepared_json(values,(2,),torch.int64)
    assert tensor.tolist()==values and tensor.dtype==torch.int64
    for bad in [True,1.5,2**63,-2**63-1,None,'1']:
        with pytest.raises(ValueError,match='integer'):
            helper.prepared_json([[bad,0]],(2,),torch.int64)
    for shape in [['N',True],['N',129],['N',128,2],['N'],['N',1,1,1,1,1,1,1,1]]:
        with pytest.raises(ValueError):helper.input_policy({'schemaVersion':1,'input':{'dtype':'int64','shape':shape,'preprocessing':'alreadyPrepared'}})

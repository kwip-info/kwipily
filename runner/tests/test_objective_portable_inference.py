"""Real objective-trained checkpoints exported independently of source data/runtime."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from runner import engine
from runner.tests.test_loss_engine import prepared
from runner.tests.test_loss_prepared_data import CASES

HELPER=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/inference.py'


def seal(target):
    entries=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in target.iterdir() if p.name!='export-manifest.json' and not p.is_dir()]
    (target/'export-manifest.json').write_text(json.dumps({'schemaVersion':1,'kind':'kwipmlTrainedModel','files':entries}))


def package(tmp_path,op='loss_ctc',device='cpu',scalarization='mean',config=None):
    graph,recipe,mapping,data=prepared(tmp_path/'private-source',op,config)
    origin=tmp_path/'private-run'
    engine.run(graph,origin,epochs=1,batch_size=5,seed=41,device=device,prepared=data,
               objective_recipe=recipe,target_mapping=mapping,objective_scalarization=scalarization)
    target=tmp_path/'portable';target.mkdir()
    for name in ('model.py','checkpoint.pt','objective_recipe.py'):shutil.copyfile(origin/name,target/name)
    shutil.copyfile(HELPER,target/'inference.py')
    contract={'schemaVersion':2,'input':{'dtype':'float32','shape':['N',len(mapping['inputColumns'])],
              'featureColumns':mapping['inputColumns'],'preprocessing':'alreadyPrepared'},
              'output':{'kind':'modelOutput','dtype':'float32','shape':['N',*CASES[op][0]]},
              'objective':{'recipe':recipe,'source':{'domain':'control','localID':'objective'},
                           'targetMapping':mapping,'scalarization':scalarization}}
    (target/'input-contract.json').write_text(json.dumps(contract));seal(target)
    namespace={'__name__':'independent_saved_reference'}
    exec(compile((origin/'model.py').read_bytes(),'saved_model.py','exec'),namespace)
    reference=namespace['Model']().to(dtype=torch.float32)
    checkpoint=(origin/'checkpoint.pt').read_bytes()
    reference.load_state_dict(torch.load(origin/'checkpoint.pt',map_location='cpu',weights_only=True)['modelState']);reference.eval()
    held=data[0][2].detach().cpu();roles={k:v.detach().cpu() for k,v in data[0][3].items()}
    shutil.rmtree(tmp_path/'private-source');shutil.rmtree(origin)
    spec=importlib.util.spec_from_file_location('portable_inference',target/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return target,module,reference,held,roles,contract,checkpoint


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',CASES)
def test_all_saved_objectives_generic_model_output_exact_checkpoint_and_fresh_objective(tmp_path,op,device):
    target,helper,reference,x,roles,contract,checkpoint=package(tmp_path,op,device)
    assert (target/'checkpoint.pt').read_bytes()==checkpoint
    predictor=helper.load_model()
    with torch.inference_mode():expected=reference(x)
    assert tuple(expected.shape)==(len(x),*CASES[op][0])
    for batch in (1,len(x)):
        actual=predictor.predict(x,batch_size=batch)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    assert len(x)>1 and x.shape[1]==(6 if op=='loss_triplet_margin' else 2)
    spec=importlib.util.spec_from_file_location('fresh_objective',target/'objective_recipe.py')
    objective=importlib.util.module_from_spec(spec);spec.loader.exec_module(objective)
    assert objective.OBJECTIVE_RECIPE==contract['objective']['recipe']
    from runner.loss_objectives import create_objective,evaluate_objective
    fresh=objective.create_training_objective()
    measured=objective.evaluate_training_objective(fresh,expected,roles)
    independent=evaluate_objective(create_objective(contract['objective']['recipe']),expected,roles)
    torch.testing.assert_close(measured['raw'],independent['raw'],rtol=0,atol=0)
    torch.testing.assert_close(measured['scalar'],independent['scalar'],rtol=0,atol=0)
    assert measured['raw'].shape!=expected.shape or op in ('loss_mse','loss_l1','loss_bce_with_logits')
    exported=set(p.name for p in target.iterdir())
    assert exported=={'model.py','checkpoint.pt','objective_recipe.py','inference.py','input-contract.json','export-manifest.json','__pycache__'}
    text=(target/'input-contract.json').read_text()
    for forbidden in ('sourcePath','trainingRowIDs','evaluationRowIDs','membershipDigest',str(tmp_path)):
        assert forbidden not in text


def test_objective_inference_does_not_execute_optional_training_helpers_and_cli_is_generic(tmp_path):
    target,helper,reference,x,*_=package(tmp_path,'loss_triplet_margin')
    for name in ('objective_recipe.py','training_recipe.py'):
        (target/name).write_text("raise AssertionError('Training helpers must not execute during inference')\n")
    seal(target)
    actual=helper.predict(x,batch_size=1)
    with torch.inference_mode():torch.testing.assert_close(actual,reference(x),rtol=1e-6,atol=1e-7)
    result=subprocess.run([sys.executable,'-I',str(target/'inference.py'),'--example'],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert set(value)=={'inputKind','modelOutput'}
    assert torch.tensor(value['modelOutput']).shape==(2,3,2)


@pytest.mark.parametrize('change',[
    lambda c:c.update(schemaVersion=True),lambda c:c.update(extra=1),
    lambda c:c['output'].update(kind='logits'),lambda c:c['output'].update(dtype='float64'),
    lambda c:c['output'].update(shape=['N',True]),lambda c:c['output'].update(shape=['N',4097]),
    lambda c:c['output'].update(shape=['N',4096,4096,7]),
    lambda c:c['objective'].update(scalarization='median'),lambda c:c['objective'].update(extra=1),
    lambda c:c['objective']['source'].update(domain='model'),
    lambda c:c['objective']['targetMapping'].update(componentID='loss_mse'),
    lambda c:c['input'].update(shape=['N',3]),
])
def test_objective_contract_tamper_refuses_before_model_execution(tmp_path,change,monkeypatch):
    target,helper,*rest=package(tmp_path)
    contract=rest[-2];change(contract)
    (target/'input-contract.json').write_text(json.dumps(contract));seal(target)
    def forbidden(*args,**kwargs):raise AssertionError('Model code must not execute for invalid contract')
    monkeypatch.setattr(helper,'exec',forbidden,raising=False)
    with pytest.raises(ValueError):helper.load_model()


def test_objective_helper_tamper_symlink_and_unknown_inventory_are_rejected(tmp_path):
    target,helper,*_=package(tmp_path,'loss_mse')
    original=(target/'objective_recipe.py').read_bytes()
    (target/'objective_recipe.py').write_bytes(original+b'#changed')
    with pytest.raises(ValueError,match='integrity'):helper.load_model()
    external=tmp_path/'external.py';external.write_bytes(original)
    (target/'objective_recipe.py').unlink();(target/'objective_recipe.py').symlink_to(external)
    with pytest.raises(ValueError,match='regular'):helper.load_model()
    (target/'objective_recipe.py').unlink();(target/'objective_recipe.py').write_bytes(original)
    (target/'private.csv').write_text('secret');seal(target)
    with pytest.raises(ValueError,match='name'):helper.load_model()


def test_none_objective_raw_array_and_scalarization_do_not_change_model_output(tmp_path):
    target,helper,reference,x,roles,contract,_=package(tmp_path,'loss_mse',scalarization='sum',config={'reduction':'none'})
    spec=importlib.util.spec_from_file_location('none_objective',target/'objective_recipe.py')
    objective=importlib.util.module_from_spec(spec);spec.loader.exec_module(objective)
    output=helper.predict(x)
    measured=objective.evaluate_training_objective(objective.create_training_objective(),output,roles)
    expected=(output-roles['target']).square()
    torch.testing.assert_close(measured['raw'],expected,rtol=0,atol=0)
    torch.testing.assert_close(measured['scalar'],expected.sum(),rtol=0,atol=0)
    with torch.inference_mode():torch.testing.assert_close(output,reference(x),rtol=0,atol=0)


def test_objective_shape_budgets_checked_before_allocation_and_model_call(tmp_path,monkeypatch):
    _,helper,*_=package(tmp_path,'loss_mse')
    # Valid per-sample shape but N times sample exceeds 100M. No tensor allocation needed.
    def forbidden(*args,**kwargs):raise AssertionError('Do not allocate before checking element budget')
    monkeypatch.setattr(torch,'tensor',forbidden)
    with pytest.raises(ValueError,match='budget|limit|bound'):
        helper.prepared_json([None]*100000,(4096,),torch.float32)
    class ForbiddenModel:
        def __call__(self,value):raise AssertionError('Reject oversized output before model execution')
    predictor=helper.Predictor(ForbiddenModel(),(2,),torch.float32,output_shape=(4096,4096))
    with pytest.raises(ValueError,match='budget|limit|bound'):
        predictor.predict(torch.zeros(10,2))

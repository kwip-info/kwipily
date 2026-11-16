"""Real trained portable packages; host proof, not native UI acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from runner.tests.test_bag_cli import candidate, invoke, registry
from runner.tests.test_objective_portable_inference import seal

HELPER=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/inference.py'


def package(tmp_path,registry,layout='packed',weighted=True,objective=False):
    command,origin,graph,_,mapping=candidate(tmp_path,registry,layout,weighted,objective)
    completed,_=invoke(command)
    assert completed.returncode==0,(completed.stdout,completed.stderr)
    target=tmp_path/'isolated';target.mkdir()
    for name in ('model.py','checkpoint.pt')+ (('objective_recipe.py',) if objective else ()):
        shutil.copyfile(origin/name,target/name)
    shutil.copyfile(HELPER,target/'inference.py')
    config=graph['nodes'][0]['config'];ports=['indices']+(['offsets']if layout=='packed'else[])+(['per_sample_weights']if weighted else[])
    contract={'schemaVersion':3,'input':{'kind':'bagPacket','schemaVersion':1,'sourceID':'source','configuration':config,
        'ports':{p:{'role':p,'dtype':'float32'if p=='per_sample_weights'else'int64'}for p in ports},
        'mapping':mapping,'numEmbeddings':5,'preprocessing':'alreadyPrepared'},
        'output':{'kind':'modelOutput','dtype':'float32','shape':['N',1]}if objective else{'kind':'logits','shape':['N',2],'classIndices':[0,1]}}
    if objective:
        recipe=json.loads(command[command.index('--objective-recipe-json')+1]);targets=json.loads(command[command.index('--objective-target-mapping-json')+1])
        contract['objective']={'recipe':recipe,'source':{'domain':'control','localID':'objective'},'targetMapping':targets,'scalarization':'mean'}
    (target/'input-contract.json').write_text(json.dumps(contract));seal(target)
    table=torch.load(origin/'checkpoint.pt',map_location='cpu',weights_only=True)['modelState']['layers.bag.weight']
    ids=[[1,2],[],[0,3]]if layout=='packed'else[[1,2],[0,0],[0,3]]
    rows=[dict(indices=row,**({'per_sample_weights':[.5,2.][:len(row)]}if weighted else{}))for row in ids]
    expected=[]
    for row in rows:
        output=torch.zeros(table.shape[1])
        for i,index in enumerate(row['indices']):
            if index:output+=table[index]*(row['per_sample_weights'][i]if weighted else 1.)
        expected.append(output)
    shutil.rmtree(origin);shutil.rmtree(tmp_path/'source')
    spec=importlib.util.spec_from_file_location('bag_native_inference',target/'inference.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return target,helper,rows,torch.stack(expected),contract


@pytest.fixture(scope='module')
def malformed_package_bytes(tmp_path_factory):
    # Only the nine contract-refusal cases reuse a trained package. Scientific
    # prediction, cold CLI and optional-helper/budget cases still train afresh.
    build=tmp_path_factory.mktemp('bag-malformed-capture')
    captured_registry=registry.__wrapped__(build)
    target,_,_,_,_=package(build,captured_registry)
    files=tuple((path.name,path.read_bytes()) for path in sorted(target.iterdir()) if path.is_file())
    inventory=json.loads((target/'export-manifest.json').read_bytes())
    assert {name for name,_ in files}=={'export-manifest.json',*(entry['path'] for entry in inventory['files'])}
    return files


@pytest.fixture
def malformed_package(tmp_path,malformed_package_bytes):
    target=tmp_path/'isolated';target.mkdir()
    for name,raw in malformed_package_bytes:(target/name).write_bytes(raw)
    # Every sibling starts from the exact sealed bytes, including after another
    # case rewrites/reseals its own contract. No mutable helper/model is retained.
    assert tuple((path.name,path.read_bytes()) for path in sorted(target.iterdir()) if path.is_file())==malformed_package_bytes
    spec=importlib.util.spec_from_file_location('bag_native_inference',target/'inference.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    contract=json.loads((target/'input-contract.json').read_bytes())
    return target,helper,contract


@pytest.mark.parametrize('layout',['fixed','packed'])
@pytest.mark.parametrize('weighted',[False,True])
@pytest.mark.parametrize('objective',[False,True])
def test_actual_trained_full_and_whole_group_micro_predictions(tmp_path,registry,layout,weighted,objective):
    target,helper,rows,expected,_=package(tmp_path,registry,layout,weighted,objective)
    predictor=helper.load_model()
    for batch in (1,2,3):torch.testing.assert_close(predictor.predict(rows,batch_size=batch),expected,rtol=2e-6,atol=2e-6)
    path=tmp_path/'groups.json';path.write_text(json.dumps(rows))
    result=subprocess.run([sys.executable,'-I',str(target/'inference.py'),'--input',str(path),'--batch-size','1'],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);key='modelOutput'if objective else'logits'
    torch.testing.assert_close(torch.tensor(value[key]),expected,rtol=2e-6,atol=2e-6)
    assert ('classIndices'in value)is(not objective)


def test_packet_admission_before_model_and_optional_helpers_never_execute(tmp_path,registry):
    target,helper,rows,expected,_=package(tmp_path,registry,objective=True)
    (target/'objective_recipe.py').write_text("raise AssertionError('Do not execute training helpers')\n");seal(target)
    predictor=helper.load_model();torch.testing.assert_close(predictor.predict(rows),expected)
    def forbidden(*args,**kwargs):raise AssertionError('Malformed groups must refuse before model execution')
    predictor._model=forbidden
    for bad in ([{'indices':[True],'per_sample_weights':[1.]}],[{'indices':[5],'per_sample_weights':[1.]}],
                [{'indices':[1,2,3],'per_sample_weights':[1.,1.,1.]}],[{'indices':[1],'per_sample_weights':[]}],
                [{'indices':[1],'per_sample_weights':[float('nan')]}],[{'indices':[],'per_sample_weights':[],'offsets':[0]}],[]):
        with pytest.raises(ValueError):predictor.predict(bad)


@pytest.mark.parametrize('mutation',[
    lambda c:c['input'].update(numEmbeddings=True),lambda c:c['input'].update(numEmbeddings=0),
    lambda c:c['input']['configuration'].update(maximum_items_per_bag=129),
    lambda c:c['input']['configuration'].update(include_last_offset=1),
    lambda c:c['input']['ports']['indices'].update(dtype='float32'),
    lambda c:c['input']['mapping'].update(lengthColumn=None),
    lambda c:c['input'].update(extra='foreign'),lambda c:c['output'].update(shape=['N',3]),
])
def test_rehashed_malformed_packet_contract_refuses_before_code(malformed_package,mutation,monkeypatch):
    target,helper,contract=malformed_package
    mutation(contract);(target/'input-contract.json').write_text(json.dumps(contract));seal(target)
    def forbidden(*args,**kwargs):raise AssertionError('Invalid contract must refuse before exported code')
    monkeypatch.setattr(helper,'exec',forbidden,raising=False)
    with pytest.raises(ValueError):helper.load_model()


def test_boolean_class_indices_refused_before_exported_code(malformed_package,monkeypatch):
    target,helper,contract=malformed_package
    contract['output']['classIndices']=[False,True]
    (target/'input-contract.json').write_text(json.dumps(contract));seal(target)
    def forbidden(*args,**kwargs):raise AssertionError('Boolean class identities are not integer indices')
    monkeypatch.setattr(helper,'exec',forbidden,raising=False)
    with pytest.raises(ValueError):helper.load_model()


def test_fixed_empty_unweighted_extra_weights_and_budgets_refuse_before_allocation(tmp_path,registry,monkeypatch):
    _,helper,rows,_,_=package(tmp_path,registry,layout='fixed',weighted=False)
    predictor=helper.load_model()
    def forbidden(*args,**kwargs):raise AssertionError('Refuse before tensor allocation')
    monkeypatch.setattr(torch,'tensor',forbidden)
    for bad in ([{'indices':[]}],[{'indices':[1,2],'per_sample_weights':[1.,1.]}],rows*1366):
        with pytest.raises(ValueError):predictor.predict(bad)
    for batch in (False,0,100001):
        with pytest.raises(ValueError):predictor.predict(rows,batch_size=batch)

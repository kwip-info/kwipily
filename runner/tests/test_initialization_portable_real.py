"""Actual selected initialization package: learned inference versus explicit fresh use."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from runner import engine
from runner.tests.test_initialization import recipe,rule
from runner.optimizer_configuration import canonical_configuration

ROOT=Path(__file__).resolve().parents[2]


def test_selected_initializer_preserves_real_learned_portable_predictions(tmp_path):
    graph=dict(schemaVersion=1,name='Portable learned affine',nodes=[dict(id='input',type='input',config={'features':2}),dict(id='head',type='linear',config=dict(in_features=2,out_features=2)),dict(id='output',type='output',config={})],edges=[dict(id='a',source='input',sourcePort='out',target='head',targetPort='in'),dict(id='b',source='head',sourcePort='out',target='output',targetPort='in')])
    selected=recipe(rule('head','bias',algorithm='zeros'),rule('head','weight',algorithm='normal',mean=.15,std=.2),seed=42)
    optimizer=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.05}))
    original=tmp_path/'private-run'
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:engine.run(graph,original,epochs=2,batch_size=16,sample_limit=64,device='cpu',optimizer_recipe=optimizer,initialization_recipe=selected)
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)
    trained=torch.load(original/'checkpoint.pt',weights_only=True,map_location='cpu')
    learned=trained['modelState']
    manifest=json.loads((original/'manifest.json').read_text())
    assert manifest['settings']['initializationRecipe']==selected
    assert manifest['provenance']['initialization']['application']=='applied'
    # Independent owner seed + primitive normal stream. No native initializer in this oracle.
    message=json.dumps(['kwip.initialization.owner-seed.v1',42,'head','weight'],ensure_ascii=False,separators=(',',':')).encode()
    seed=int.from_bytes(hashlib.sha256(message).digest()[:8],'big')%(2**63)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    expected_initial=(torch.randn(2,2,generator=generator,dtype=torch.float64)*.2+.15).float()
    assert not torch.equal(learned['layers.head.weight'],expected_initial)
    assert not torch.equal(learned['layers.head.bias'],torch.zeros(2))

    package=tmp_path/'isolated-package';package.mkdir()
    for name in ('model.py','checkpoint.pt','initialization_recipe.py'):shutil.copyfile(original/name,package/name)
    shutil.copyfile(ROOT/'Sources/KwipMLApp/Resources/inference.py',package/'inference.py')
    contract=dict(schemaVersion=1,input=dict(dtype='float32',shape=['N',2],featureColumns=['first','second'],preprocessing='alreadyPrepared'),output=dict(kind='logits',shape=['N',2],classIndices=[0,1]))
    (package/'input-contract.json').write_text(json.dumps(contract))
    helper_source=(package/'initialization_recipe.py').read_bytes()
    def pin():
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(package.iterdir()) if p.is_file() and p.name!='export-manifest.json']
        value=dict(schemaVersion=1,kind='kwipmlTrainedModel',files=files,initializationRecipe=dict(source=dict(domain='control',localID='starting'),configuration=selected))
        (package/'export-manifest.json').write_text(json.dumps(value))
    pin()
    package_manifest=json.loads((package/'export-manifest.json').read_text())
    assert package_manifest['initializationRecipe']['configuration']==selected
    assert next(v for v in package_manifest['files'] if v['path']=='initialization_recipe.py')['sha256']==hashlib.sha256(helper_source).hexdigest()
    shutil.rmtree(original)
    spec=importlib.util.spec_from_file_location('isolated_initialization_inference',package/'inference.py')
    inference=importlib.util.module_from_spec(spec);spec.loader.exec_module(inference)
    inputs=torch.tensor([[i/8-0.5,((i*3)%7)/4-.75] for i in range(11)],dtype=torch.float32)
    weight=learned['layers.head.weight'].tolist();bias=learned['layers.head.bias'].tolist()
    expected=[[sum(float(x[j])*weight[o][j] for j in range(2))+bias[o] for o in range(2)] for x in inputs.tolist()]
    expected=torch.tensor(expected,dtype=torch.float32)
    rng=torch.random.get_rng_state().clone()
    predictor=inference.load_model()
    for batch in (1,3,len(inputs)):
        torch.testing.assert_close(predictor.predict(inputs,batch_size=batch),expected,rtol=1e-6,atol=1e-7)
    assert torch.equal(torch.random.get_rng_state(),rng)
    for key,value in learned.items():assert torch.equal(predictor._model.state_dict()[key],value)
    assert (package/'initialization_recipe.py').read_bytes()==helper_source
    rows=tmp_path/'prepared-rows.json';rows.write_text(json.dumps(inputs.tolist()))
    for batch in (1,len(inputs)):
        command=[sys.executable,str(package/'inference.py'),'--input',str(rows),'--batch-size',str(batch)]
        result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        torch.testing.assert_close(torch.tensor(json.loads(result.stdout)['logits']),expected,rtol=1e-6,atol=1e-7)
    # Explicit fresh use is distinct from loading learned weights. Merely importing
    # the optional helper must also have no model-side effect.
    namespace={};exec((package/'model.py').read_text(),namespace)
    fresh=namespace['Model']();fresh_before={k:v.clone() for k,v in fresh.state_dict().items()}
    helper={};exec(helper_source,helper)
    assert helper['INITIALIZATION_RECIPE']==selected
    assert all(torch.equal(fresh.state_dict()[k],v) for k,v in fresh_before.items())
    helper['apply_initialization'](fresh,helper['INITIALIZATION_RECIPE'])
    assert torch.equal(fresh.state_dict()['layers.head.weight'],expected_initial)
    assert torch.equal(fresh.state_dict()['layers.head.bias'],torch.zeros(2))
    assert not torch.equal(fresh.state_dict()['layers.head.weight'],learned['layers.head.weight'])
    # A consistently pinned, syntactically valid poison proves inference neither
    # imports nor applies the optional helper; this is not a third-party trust claim.
    (package/'initialization_recipe.py').write_text('raise RuntimeError("fresh initialization must never run during inference")\n')
    pin()
    torch.testing.assert_close(inference.predict(inputs,batch_size=1),expected,rtol=1e-6,atol=1e-7)
    (package/'initialization_recipe.py').write_text('# changed without updating inventory\n')
    import pytest
    with pytest.raises(ValueError,match='integrity'):inference.load_model()

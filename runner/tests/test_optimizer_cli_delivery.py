"""Real host CLI and genuine retained macOS watchdog optimizer transport checks."""
import importlib.util,json,subprocess,sys
from pathlib import Path
import pytest,torch
from runner.optimizer_configuration import canonical_configuration,validate_optimizer_state
from runner.tests.test_optimizer_runtime import scalar_update
from runner.tests.test_model_training_operation import training
from runner.tests.test_preview_operation_authorization import preview_operation,source,inventory
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries
ROOT=Path(__file__).resolve().parents[2]
IDS=['optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad']
CONTROLS={'optimizer_sgd':{'lr':.04,'momentum':.7,'nesterov':True},'optimizer_adam':{'lr':.02,'amsgrad':True,'decoupled_weight_decay':True,'weight_decay':.03},'optimizer_adamw':{'lr':.02,'amsgrad':True,'weight_decay':.03},'optimizer_rmsprop':{'lr':.02,'momentum':.4,'centered':True},'optimizer_adagrad':{'lr':.02,'lr_decay':.1,'initial_accumulator_value':.2}}
def recipe(op):return {'schemaVersion':1,'componentID':op,'configurationVersion':1,'config':canonical_configuration(op,CONTROLS[op])}
def cli(args):
 result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),*args],capture_output=True,text=True,timeout=30)
 return result,[json.loads(line) for line in result.stdout.splitlines()]
def artifacts(path,op,device):
 m=json.loads((path/'manifest.json').read_text());s=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True);assert m['status']=='completed' and m['runtime']['device']==device and m['settings']['optimizerRecipe']==recipe(op);assert m['settings']['learningRate']==CONTROLS[op]['lr'] and m['settings']['optimizerRateSource']=='explicitRecipe';assert m['provenance']['optimizer']['recipe']==recipe(op) and s['contract']['optimizer']['recipe']==recipe(op);assert len(s['optimizerUpdates'])==len(s['contract']['optimizer']['parameterOwnership']);assert (path/'training_recipe.py').is_file();assert 'OPTIMIZER_RECIPE' not in (path/'model.py').read_text()
 spec=importlib.util.spec_from_file_location('actual_cli_export',path/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.Model();model.load_state_dict(s['modelState'],strict=True);from runner.optimizer_configuration import create_optimizer
 owned=[p for p in model.parameters() if p.requires_grad];validate_optimizer_state(create_optimizer(owned,recipe(op)),s['optimizerState'],recipe(op),s['optimizerUpdates']);assert s['contract']['optimizer']['parameterOwnership']==[{'name':name,'shape':list(p.shape),'dtype':str(p.dtype).removeprefix('torch.')} for name,p in model.named_parameters() if p.requires_grad];code={};exec((path/'training_recipe.py').read_text(),code);o=code['create_optimizer'](model);params=list(p for p in model.parameters() if p.requires_grad);saved={k:v.clone() for k,v in model.state_dict().items()};first=params[0];values=first.detach().flatten().tolist();gradient=[(i%7-3)/10 for i in range(len(values))];first.grad=torch.tensor(gradient).reshape(first.shape);expected,_=scalar_update(op,CONTROLS_CANONICAL[op],values,gradient,{});o.step();torch.testing.assert_close(first.detach().flatten(),torch.tensor(expected),atol=3e-5,rtol=3e-5);assert any(not torch.equal(saved[k],v) for k,v in model.state_dict().items());return m,s
CONTROLS_CANONICAL={op:canonical_configuration(op,CONTROLS[op]) for op in IDS}

@pytest.mark.parametrize('op',IDS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_standalone_training_cli_recipe_and_saved_fresh_step(op,device,tmp_path):
 from runner.tests.test_runner import graph
 g=tmp_path/'graph.json';g.write_text(json.dumps(graph()));out=tmp_path/'run';r,events=cli(['--graph',str(g),'--output',str(out),'--epochs','1','--sample-limit','32','--batch-size','10','--device',device,'--learning-rate','.8','--optimizer-recipe-json',json.dumps(recipe(op))]);assert r.returncode==0,(events,r.stderr);assert events[-1]['event']=='completed';artifacts(out,op,device)

@pytest.mark.parametrize('flags',[['--capabilities'],['--dataflow-registry'],['--learning-runtime'],['--preview'],['--export-code']])
def test_orphan_recipe_nontraining_cli_refuses_without_success(flags,tmp_path):
 from runner.tests.test_runner import graph
 g=tmp_path/'graph.json';g.write_text(json.dumps(graph()));out=tmp_path/'run';r,events=cli([*flags,'--graph',str(g),'--output',str(out),'--optimizer-recipe-json',json.dumps(recipe('optimizer_sgd'))]);assert r.returncode==1 and events[-1]['event']=='error';assert 'training only' in events[-1]['message'];assert not out.exists()

def test_orphan_recipe_cache_resume_cli_refuses_without_cache_execution(tmp_path):
 from runner.preview import preview
 from runner.tests.test_runner import graph
 g=tmp_path/'graph.json';g.write_text(json.dumps(graph()));prior=tmp_path/'prior';preview(graph(),prior,device='cpu',sample_limit=32);out=tmp_path/'run';r,events=cli(['--graph',str(g),'--output',str(out),'--device','cpu','--resume-from','merge','--cache',str(prior/'activation-cache.pt'),'--sample-limit','32','--optimizer-recipe-json',json.dumps(recipe('optimizer_sgd'))]);assert r.returncode==1 and events[-1]['event']=='error';assert 'training only' in events[-1]['message'];assert not out.exists()

@pytest.mark.parametrize('op',IDS)
def test_actual_guarded_recipe_training_uses_retained_exact_launch(training,tmp_path,libraries,op):
 # The shared training fixture pins a bounded cold-training allowance. The
 # operation deadline and cancellation fixtures keep their short budgets.
 f=training;source(f,'prepared',tmp_path);inventory(f);text=json.dumps(recipe(op),sort_keys=True,separators=(',',':'));f[1]['arguments']+=['--optimizer-recipe-json',text];publish(f);guard,_=libraries;path=entry(tmp_path,guard);marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)';script=path.read_text();assert marker in script;path.write_text(script.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'));code,events,error=run_supervisor(path,control(f),timeout=40);assert code==0,(events,error);assert events[0]['toolMode']=='modelTraining' and events[-1]['event']=='completed';m,s=artifacts(f[0]/'artifacts',op,'cpu');assert m['settings']['partitionMode']=='prepared' and m['provenance']['preparation'];assert (f[0]/'claim.json').is_file()

@pytest.mark.parametrize('op',IDS)
def test_retained_recipe_exact_json_revalidation(training,op):
 # These two fixtures share one operation base; retain separate preview args snapshot.
 f=training;args=list(f[1]['arguments']);text=json.dumps(recipe(op),sort_keys=True,separators=(',',':'));f[1]['arguments']+=['--optimizer-recipe-json',text];publish(f)
 with claim(f) as authority:
  authority.check();f[1]['arguments'][-1]=json.dumps(recipe(op),indent=2);publish(f)
  with pytest.raises(ValueError):authority.check()

@pytest.mark.parametrize('op',IDS)
def test_preview_operation_rejects_orphan_recipe_before_claim(preview_operation,op):
 f=preview_operation;f[1]['arguments']+=['--optimizer-recipe-json',json.dumps(recipe(op))];publish(f)
 with pytest.raises(ValueError):claim(f)
 assert not (f[0]/'claim.json').exists()

@pytest.mark.parametrize('content',[
 b"raise RuntimeError('Inference must never execute optional training recipe')\n",
 b'#'+b'x'*8191,
])
def test_optional_training_recipe_inventory_is_verified_without_execution(tmp_path,content):
 import hashlib
 from runner.tests.test_portable_inference import bundle
 target,helper,reference=bundle(tmp_path)
 path=target/'training_recipe.py';path.write_bytes(content)
 manifest=json.loads((target/'export-manifest.json').read_text())
 manifest['files'].append({'path':path.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
 (target/'export-manifest.json').write_text(json.dumps(manifest))
 x=torch.tensor([[.25,-.5]],dtype=torch.float32)
 torch.testing.assert_close(helper.predict(x),reference(x).detach(),rtol=0,atol=0)
 result=subprocess.run([sys.executable,str(target/'inference.py'),'--example'],capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 path.write_bytes(b'!'+content[1:])
 with pytest.raises(ValueError,match='integrity'):helper.predict(x)

@pytest.mark.parametrize('content',[b'#'+b'x'*8192,b'\xff'])
def test_optional_training_recipe_refuses_oversized_or_non_utf8_content(tmp_path,content):
 import hashlib
 from runner.tests.test_portable_inference import bundle
 target,helper,_=bundle(tmp_path)
 path=target/'training_recipe.py';path.write_bytes(content)
 manifest=json.loads((target/'export-manifest.json').read_text())
 manifest['files'].append({'path':path.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
 (target/'export-manifest.json').write_text(json.dumps(manifest))
 with pytest.raises(ValueError):helper.predict(torch.zeros(1,2))

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pytest,torch
from runner.loss_configuration import validate_recipe
from runner.loss_export import objective_recipe_code
from runner.loss_objectives import create_objective,evaluate_objective
from runner.tests.test_loss_prepared_data import CASES

@pytest.mark.parametrize('op',CASES)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_portable_recipe_matches_real_values_and_gradients_with_fixed_identity(tmp_path,op,device):
 p,roles,c=CASES[op]
 recipe=validate_recipe({'schemaVersion':1,'componentID':op,'configurationVersion':1,'config':c})
 code=objective_recipe_code(recipe,'mean');assert len(code.encode())<=131072
 path=tmp_path/'objective_recipe.py';path.write_text(code)
 spec=importlib.util.spec_from_file_location('portable_objective',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 dtype=torch.float32
 x=torch.linspace(-.8,.9,2*__import__('math').prod(p),device=device).reshape(2,*p)
 if op in ('loss_nll','loss_kl_div','loss_ctc'):x=torch.log_softmax(x,-1)
 x=x.detach().requires_grad_();copy=x.detach().clone().requires_grad_()
 targets={name:torch.tensor([values]*2,device=device,dtype=getattr(torch,dtype_name)).reshape(2,*shape)for name,shape,dtype_name,values in roles}
 expected=evaluate_objective(create_objective(recipe,device),x,targets)
 objective=module.create_training_objective(device=device)
 actual=module.evaluate_training_objective(objective,copy,targets)
 torch.testing.assert_close(actual['raw'],expected['raw'],atol=0,rtol=0)
 torch.testing.assert_close(actual['scalar'],expected['scalar'],atol=0,rtol=0)
 expected['scalar'].backward();actual['scalar'].backward()
 torch.testing.assert_close(copy.grad,x.grad,atol=0,rtol=0)
 assert actual['metadata']==expected['metadata']
 objective.configuration['reduction']='sum'
 with pytest.raises(ValueError):module.evaluate_training_objective(objective,copy,targets)


def test_export_runs_outside_repository_with_only_torch_and_standard_library(tmp_path):
 recipe=validate_recipe({'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':{'reduction':'none'}})
 (tmp_path/'objective_recipe.py').write_text(objective_recipe_code(recipe,'sum'))
 probe=tmp_path/'probe.py';probe.write_text('''import importlib.util,json,torch,sys
spec=importlib.util.spec_from_file_location('portable',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=torch.tensor([[1.,3.]],requires_grad=True)
r=m.evaluate_training_objective(m.create_training_objective(),x,{'target':torch.tensor([[0.,1.]])});r['scalar'].backward()
print(json.dumps({'raw':r['raw'].tolist(),'scalar':r['scalar'].item(),'gradient':x.grad.tolist()}))
''')
 result=subprocess.run([sys.executable,'-I',str(probe),str(tmp_path/'objective_recipe.py')],cwd=tmp_path,capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stderr
 assert json.loads(result.stdout)=={'raw':[[1.,4.]],'scalar':5.,'gradient':[[2.,4.]]}

@pytest.mark.parametrize('scalarization',['median',True,None])
def test_export_refuses_unknown_scalarization(scalarization):
 r=validate_recipe({'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':{}})
 with pytest.raises(ValueError):objective_recipe_code(r,scalarization)

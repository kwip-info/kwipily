"""Fresh saved-package training steps, independent of the original run and worker objectives."""
import importlib.util
import hashlib
import pytest
import torch
from torch.nn import functional as F
from runner.tests.test_objective_portable_inference import package,seal
from runner.tests.test_loss_prepared_data import CASES
from runner.optimizer_configuration import validate_recipe as optimizer_recipe,training_recipe_code
from runner.scheduler_configuration import validate_recipe as scheduler_recipe


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def native_reference(op,c,prediction,roles):
    """Separate public native calls, never the exported or engine objective evaluator."""
    target=roles.get('target');reduction=c['reduction']
    if op=='loss_cross_entropy':
        return F.cross_entropy(prediction.movedim(c['class_axis'],1),target,
            ignore_index=c['ignore_index'],label_smoothing=c['label_smoothing'],reduction=reduction)
    if op=='loss_nll':
        return F.nll_loss(prediction.movedim(c['class_axis'],1),target,ignore_index=c['ignore_index'],reduction=reduction)
    if op=='loss_bce_with_logits':
        return F.binary_cross_entropy_with_logits(prediction,target,weight=roles['weight'],reduction=reduction)
    if op=='loss_mse':return F.mse_loss(prediction,target,reduction=reduction)
    if op=='loss_l1':return F.l1_loss(prediction,target,reduction=reduction)
    if op=='loss_kl_div':return F.kl_div(prediction,target,reduction=reduction,log_target=c['log_target'])
    if op=='loss_ctc':
        # Independent CPU native reference: to(cpu) preserves the input/model gradient
        # back to MPS. This is not a claim that the saved model ran on CPU.
        lengths=roles['target_lengths'].cpu()
        labels=torch.cat([roles['target'][i,:int(n)].cpu() for i,n in enumerate(lengths)])
        return F.ctc_loss(prediction.to('cpu').transpose(0,1),labels,
            roles['input_lengths'].cpu(),lengths,blank=c['blank'],reduction=reduction,zero_infinity=c['zero_infinity'])
    a,p,n=prediction.unbind(1)
    return F.triplet_margin_loss(a,p,n,margin=c['margin'],p=c['p'],eps=c['eps'],swap=c['swap'],reduction=reduction)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',CASES)
def test_saved_package_actual_device_fresh_objective_optimizer_step(tmp_path,op,device):
    old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        target,helper,cpu_model,x,roles,contract,checkpoint=package(tmp_path,op,device)
        assert not (tmp_path/'private-run').exists() and not (tmp_path/'private-source').exists()
        checkpoint_hash=hashlib.sha256(checkpoint).hexdigest()
        model_module=load(target/'model.py','saved_step_model')
        state=torch.load(target/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=model_module.Model().to(device);model.load_state_dict(state['modelState'],strict=True);model.eval()
        assert all(p.device.type==device for p in model.parameters())
        torch.use_deterministic_algorithms(device=='cpu')
        source=x.to(device)
        with torch.no_grad():
            expected=cpu_model(x)
            torch.testing.assert_close(model(source).cpu(),expected,rtol=3e-4,atol=3e-5)
            micro=torch.cat([model(row[None]) for row in source])
            torch.testing.assert_close(micro.cpu(),expected,rtol=3e-4,atol=3e-5)
        objective=load(target/'objective_recipe.py','saved_step_objective')
        opt_recipe=optimizer_recipe({'schemaVersion':1,'componentID':'optimizer_sgd','configurationVersion':1,
            'config':{'lr':.03,'momentum':.5,'weight_decay':.01}})
        schedule=scheduler_recipe({'schemaVersion':1,'componentID':'scheduler_step_lr','configurationVersion':1,
            'config':{'step_size':1,'gamma':.5}})
        (target/'training_recipe.py').write_text(training_recipe_code(opt_recipe,schedule));seal(target)
        training=load(target/'training_recipe.py','saved_step_training')
        optimizer=training.create_optimizer(model);scheduler=training.create_scheduler(optimizer)
        assert not optimizer.state # Fresh exported helper is deliberately not checkpoint resume.
        assert training.OPTIMIZER_RECIPE==opt_recipe and training.SCHEDULER_RECIPE==schedule
        reference=model_module.Model().to(device);reference.load_state_dict(state['modelState'],strict=True);reference.eval()
        actual_input=source.detach().clone().requires_grad_();expected_input=source.detach().clone().requires_grad_()
        typed_roles={k:v.to(device) for k,v in roles.items()}
        actual_prediction=model(actual_input);reference_prediction=reference(expected_input)
        measured=objective.evaluate_training_objective(objective.create_training_objective(device),actual_prediction,typed_roles)
        assert measured['metadata']['device']==device and actual_prediction.device.type==device
        assert measured['metadata']['observations']==len(source)
        assert actual_prediction.numel()<=100_000_000
        raw=native_reference(op,contract['objective']['recipe']['config'],reference_prediction,typed_roles)
        scalar=raw.mean() if raw.ndim else raw
        torch.testing.assert_close(measured['raw'].cpu(),raw.cpu(),rtol=3e-4,atol=3e-5)
        torch.testing.assert_close(measured['scalar'].cpu(),scalar.cpu(),rtol=3e-4,atol=3e-5)
        measured['scalar'].backward();scalar.backward()
        torch.testing.assert_close(actual_input.grad.cpu(),expected_input.grad.cpu(),rtol=4e-4,atol=4e-5)
        actual_parameters=dict(model.named_parameters());expected_parameters=dict(reference.named_parameters())
        assert actual_parameters.keys()==expected_parameters.keys() and actual_parameters
        before={k:p.detach().clone() for k,p in actual_parameters.items()}
        for key,p in actual_parameters.items():
            assert p.grad is not None and expected_parameters[key].grad is not None
            torch.testing.assert_close(p.grad.cpu(),expected_parameters[key].grad.cpu(),rtol=4e-4,atol=4e-5)
        # Exercise the generated fresh training helper on a new graph. Its zero_grad
        # must not accumulate the independently checked derivative above.
        fresh=objective.evaluate_training_objective(objective.create_training_objective(device),model(source),typed_roles)
        training.training_step(fresh['scalar'],optimizer,scheduler)
        for key,p in actual_parameters.items():
            derivative=expected_parameters[key].grad+.01*before[key]
            torch.testing.assert_close(p.detach().cpu(),(before[key]-.03*derivative).cpu(),rtol=4e-4,atol=4e-5)
            torch.testing.assert_close(optimizer.state[p]['momentum_buffer'].cpu(),derivative.cpu(),rtol=4e-4,atol=4e-5)
        training.completed_epoch_step(scheduler,validation_loss=float(scalar.detach()))
        assert optimizer.param_groups[0]['lr']==pytest.approx(.015)
        assert hashlib.sha256((target/'checkpoint.pt').read_bytes()).hexdigest()==checkpoint_hash
    finally:torch.use_deterministic_algorithms(old,warn_only=warn)

import math
import pytest
from runner.training_control import TrainingControl,validate_policy


def policy(selection='bestValidation',enabled=True,patience=2,delta=.1):
    return dict(schemaVersion=1,modelSelection=selection,monitor='validationLoss',earlyStopping=dict(enabled=enabled,patience=patience,minimumImprovement=delta))


def test_selection_and_stopping_use_independent_improvement_rules():
    copies=[]
    control=TrainingControl(policy(),capture=lambda:copies.append('copy') or {'weight':len(copies)})
    assert not control.observe(1,1.)
    assert not control.observe(2,.95)
    assert control.observe(3,.94)
    # Best weights follow any strict improvement; stopping requires > minimum delta.
    assert control.selected_epoch==3 and control.best_model=={'weight':3}
    assert control.stop_epoch==3 and len(copies)==3


def test_final_epoch_selection_never_copies_weights_and_can_stop_independently():
    control=TrainingControl(policy('finalEpoch',True,2,0),capture=lambda:pytest.fail('Unexpected copy'))
    for epoch,value in enumerate([1.,.5,.5,.6],1):stopped=control.observe(epoch,value)
    assert stopped and control.selected_epoch==4 and control.best_model is None
    assert control.report()['stopReason']=='earlyStopping'


def test_ties_do_not_replace_selected_model_and_invalid_observations_refuse():
    control=TrainingControl(policy(enabled=False),capture=lambda:{'weight':42})
    control.observe(1,.5);control.observe(2,.5)
    assert control.selected_epoch==1 and control.stop_epoch is None
    for epoch,value in [(2,.4),(3,math.nan),(3,math.inf)]:
        with pytest.raises(ValueError):control.observe(epoch,value)


@pytest.mark.parametrize('value',[None,{},dict(policy(),extra=True),policy(patience=True),policy(patience=0),policy(delta=-1),policy(delta=math.nan),dict(policy(),monitor='trainingLoss')])
def test_strict_control_policy(value):
    with pytest.raises(ValueError):validate_policy(value)


def test_resume_retains_patience_and_selected_model_state():
    first=TrainingControl(policy(),capture=lambda:{'weight':7})
    first.observe(1,1.);first.observe(2,.95)
    resumed=TrainingControl(policy(),capture=lambda:{'weight':8},state=first.state(),best_model=first.best_model)
    assert resumed.observe(3,.94)
    assert resumed.report()==dict(first.report(),selectedEpoch=3,completedEpoch=3,validationLoss=.94,stopReason='earlyStopping')


@pytest.mark.parametrize('device',['cpu','mps'])
def test_real_engine_selects_weights_and_retains_terminal_continuation(tmp_path,device):
    import json
    import torch
    from runner.engine import run
    from runner.tests.test_runner import graph
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    options=dict(device=device,seed=7,batch_size=128,learning_rate=.5)
    selected_policy=policy(enabled=False)
    run(graph(),tmp_path/'reference',epochs=8,**options)
    history=json.loads((tmp_path/'reference/metrics.json').read_text())
    best=min(range(len(history)),key=lambda i:history[i]['validationLoss'])+1
    assert best<8, 'Fixture must exercise a selected model earlier than the terminal epoch'
    run(graph(),tmp_path/'selected',epochs=8,training_control=selected_policy,**options)
    run(graph(),tmp_path/'prefix',epochs=best,**options)
    chosen=torch.load(tmp_path/'selected/checkpoint.pt',weights_only=True)
    reference=torch.load(tmp_path/'reference/checkpoint.pt',weights_only=True)
    prefix=torch.load(tmp_path/'prefix/checkpoint.pt',weights_only=True)
    for key,value in prefix['modelState'].items():assert torch.equal(value,chosen['modelState'][key])
    for key,value in reference['modelState'].items():assert torch.equal(value,chosen['continuationModelState'][key])
    manifest=json.loads((tmp_path/'selected/manifest.json').read_text())
    assert manifest['trainingControl']['selectedEpoch']==best
    assert manifest['evaluation']['validationLoss']==pytest.approx(history[best-1]['validationLoss'])


def test_early_stopping_records_actual_epoch_and_capture_off_keeps_policy_state(tmp_path):
    import json
    import torch
    from runner.engine import run
    from runner.tests.test_runner import graph
    run(graph(),tmp_path/'stop',epochs=10,device='cpu',training_control=policy(delta=1_000_000),sample_capture=dict(schemaVersion=3,activationValues=False,predictionRows=False,performanceMeasurements=False,retainComponentOutputs=False,retainDiagnosticDetails=False))
    manifest=json.loads((tmp_path/'stop/manifest.json').read_text())
    assert manifest['checkpoint']['completedEpoch']==3
    assert manifest['settings']['epochLimit']==10 and manifest['settings']['epochsExecuted']==3
    assert len(json.loads((tmp_path/'stop/metrics.json').read_text()))==3
    assert manifest['trainingControl']['stopReason']=='earlyStopping'
    checkpoint=torch.load(tmp_path/'stop/checkpoint.pt',weights_only=True)
    assert checkpoint['trainingControlState']['unimprovedEpochs']==2
    assert not (tmp_path/'stop/activations.json').exists()


def test_exact_resume_uses_terminal_state_while_fine_tuning_uses_selected_weights(tmp_path):
    import torch
    from runner.engine import run
    from runner.tests.test_runner import graph
    from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
    options=dict(device='cpu',seed=7,batch_size=128,learning_rate=.5,training_control=policy(enabled=False))
    run(graph(),tmp_path/'first',epochs=4,**options)
    run(graph(),tmp_path/'resumed',epochs=8,resume_checkpoint=tmp_path/'first/checkpoint.pt',**options)
    run(graph(),tmp_path/'whole',epochs=8,**options)
    resumed=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
    whole=torch.load(tmp_path/'whole/checkpoint.pt',weights_only=True)
    for key in ('modelState','continuationModelState','optimizerState','rngState','trainingControlState'):assert_nested_equal(resumed[key],whole[key])
    run(graph(),tmp_path/'fine',epochs=1,finetune_checkpoint=tmp_path/'whole/checkpoint.pt',device='cpu',training_control=policy(enabled=False))


@pytest.mark.parametrize('scheduler',['scheduler_step_lr','scheduler_one_cycle_lr'])
def test_stopping_preserves_actual_gradient_schedule_and_selected_checkpoint_loading(tmp_path,scheduler):
    import json
    from runner.engine import run
    from runner.tests.test_runner import graph
    from runner.tests.test_scheduler_runtime import recipe,EXPECTED
    from runner.optimizer_configuration import canonical_configuration
    optimizer=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5}))
    configuration=dict(EXPECTED[scheduler])
    if scheduler=='scheduler_one_cycle_lr':configuration['planned_total_batches']=24
    options=dict(device='cpu',sample_limit=32,batch_size=10,optimizer_recipe=optimizer,scheduler_recipe=recipe(scheduler,configuration))
    run(graph(),tmp_path/'stopped',epochs=8,training_control=policy(delta=1_000_000),gradient_capture=dict(schemaVersion=1,everyEpochs=2,maxBatchesPerEpoch=1,maxCaptures=16),**options)
    gradient=json.loads((tmp_path/'stopped/gradient-summary.json').read_text())
    trace=json.loads((tmp_path/'stopped/rate-trace.json').read_text())
    assert gradient['finalEpoch']==3 and [row['epoch'] for row in gradient['summaries']]==[1,2,3]
    assert len(trace['records'])==(9 if scheduler=='scheduler_one_cycle_lr' else 3)
    # Loading selected weights validates terminal scheduler/gradient state independently.
    run(graph(),tmp_path/'fine',epochs=1,finetune_checkpoint=tmp_path/'stopped/checkpoint.pt',**options)

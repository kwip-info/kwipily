"""Real token masks, gradients, continuation and safe independent exports."""
import json
import pytest
import torch
from runner.tests.test_pretrained_models import folder
from runner.tests.test_pretrained_training import settings, state
from runner import pretrained_training as ft


def test_response_only_loss_masks_prompt_and_padding(folder):
    config=settings(exampleFormat='promptResponse',promptColumn='question',responseColumn='answer',separator=' ',lossCoverage='response')
    batch=ft.prepare_documents(folder,[{'prompt':'hello','response':'world .'},{'prompt':'world','response':'hello'}],config)[0]
    assert batch['labels'][0].tolist()==[4,5,2]
    # With a two-token prompt, the prompt's next token must not enter the loss.
    batch=ft.prepare_documents(folder,[{'prompt':'hello world','response':'.'}],config)[0]
    assert batch['labels'][0].tolist()==[-100,5,2]
    assert batch['attention_mask'][0].tolist()==[1,1,1]
    with pytest.raises(ValueError,match='response'):
        ft.prepare_documents(folder,[{'prompt':'hello','response':''}],config)


def test_response_loss_training_and_token_weighted_evaluation(folder,tmp_path):
    c=settings(exampleFormat='promptResponse',promptColumn='question',responseColumn='answer',separator=' ',lossCoverage='response')
    texts=[{'prompt':'hello world','response':'.'},{'prompt':'hello','response':'world .'}]
    result=ft.fine_tune(folder,tmp_path/'trained',texts,texts,c)
    rows=ft.prediction_rows(tmp_path/'trained/model',texts,c)
    assert rows[0][4]==2 and rows[1][4]==3
    summary=ft.evaluate(tmp_path/'trained/model',texts,c)
    assert summary['scoredTokens']==5
    assert summary['loss']==pytest.approx(sum(r[4]*r[5] for r in rows)/5)
    assert result['settings']['exampleFormat']=='promptResponse'


def test_cpu_lora_dropout_resume_matches_original_horizon(folder,tmp_path):
    c=settings(mode='lora',targets=['attentionInput'],dropout=.2,epochs=3)
    c['scheduler']=dict(schemaVersion=1,componentID='scheduler_cosine_annealing_lr',configurationVersion=1,config={'T_max':3,'eta_min':.0001})
    texts=['hello world .','world hello .']*2
    ft.fine_tune(folder,tmp_path/'whole',texts,['hello .'],c)
    first=ft.fine_tune(folder,tmp_path/'first',texts,['hello .'],c,pause_requested=lambda:True)
    assert first['completedEpoch']==1 and first['paused']
    ft.fine_tune(folder,tmp_path/'resumed',texts,['hello .'],c,resume_from=tmp_path/'first')
    a=state(tmp_path/'whole/model');b=state(tmp_path/'resumed/model')
    assert all(torch.equal(a[k],b[k]) for k in a)
    with pytest.raises(ValueError,match='horizon|settings|identical'):
        ft.fine_tune(folder,tmp_path/'changed',texts,['hello .'],{**c,'epochs':4},resume_from=tmp_path/'first')


def test_lora_resume_package_contains_pinned_base_and_terminal_state(folder,tmp_path):
    c=settings(mode='lora',targets=['attentionInput'],epochs=2)
    ft.fine_tune(folder,tmp_path/'trained',['hello world .']*2,['world hello .'],c,pause_requested=lambda:True)
    ft.export_fine_tuned(tmp_path/'trained',tmp_path/'package',kind='continuation')
    assert (tmp_path/'package/base/model.safetensors').exists()
    assert (tmp_path/'package/latest/kwip-continuation.json').exists()
    ft.fine_tune(folder,tmp_path/'resumed',['hello world .']*2,['world hello .'],c,resume_from=tmp_path/'package')
    (tmp_path/'package/base/config.json').write_text('{}')
    with pytest.raises(ValueError,match='inventory|changed|identity'):
        ft.fine_tune(folder,tmp_path/'tampered',['hello world .']*2,['world hello .'],c,resume_from=tmp_path/'package')


def test_activation_recomputation_preserves_lora_gradients(folder,tmp_path):
    c=settings(mode='lora',targets=['attentionInput'],dropout=.1)
    ft.fine_tune(folder,tmp_path/'reference',['hello world .']*2,['world .'],c)
    ft.fine_tune(folder,tmp_path/'checkpointed',['hello world .']*2,['world .'],{**c,'activationCheckpointing':True})
    a=state(tmp_path/'reference/model');b=state(tmp_path/'checkpointed/model')
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_epoch_retention_and_budget_are_bounded(folder,tmp_path):
    c=settings(epochs=3,checkpointCount=2,checkpointBytes=100_000_000)
    ft.fine_tune(folder,tmp_path/'trained',['hello world .']*2,['world .'],c)
    assert sorted(p.name for p in (tmp_path/'trained/checkpoints').iterdir())==['epoch-000002','epoch-000003']
    with pytest.raises(ValueError,match='checkpoint|storage|budget'):
        ft.fine_tune(folder,tmp_path/'small',['hello world .']*2,['world .'],{**c,'checkpointBytes':1})


def test_interrupted_epoch_keeps_a_verified_resumable_boundary(folder,tmp_path):
    c=settings(mode='lora',targets=['attentionInput'],dropout=.15,epochs=2,checkpointCount=1,checkpointBytes=100_000_000)
    docs=['hello world .']*2;valid=['world .']
    ft.fine_tune(folder,tmp_path/'whole',docs,valid,c)
    checkpoint=tmp_path/'interrupted/checkpoints/epoch-000001'
    def stop():
        if checkpoint.exists():raise InterruptedError('stop after committed epoch')
    with pytest.raises(InterruptedError):ft.fine_tune(folder,tmp_path/'interrupted',docs,valid,c,check_cancelled=stop)
    assert ft.verify_result(checkpoint)['completedEpoch']==1
    assert not (tmp_path/'interrupted/model').exists()
    ft.fine_tune(folder,tmp_path/'resumed',docs,valid,c,resume_from=checkpoint)
    a=state(tmp_path/'whole/model');b=state(tmp_path/'resumed/model')
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_prepared_preview_checks_actual_targets_memory_and_gradients(folder):
    c=settings(mode='lora',targets=['attentionInput'],exampleFormat='promptResponse',separator=' ')
    texts=[dict(prompt='hello',response='world .')]
    value=ft.prepared_preview(folder,texts,texts,c)
    assert value['trainableParameters']>0 and value['trainableParameters']<value['totalParameters']
    assert value['estimatedResidentBytes']>0 and 'no optimizer update' in value['gradientCheck']
    with pytest.raises(ValueError,match='does not exist'):
        ft.prepared_preview(folder,texts,texts,{**c,'targets':['imaginary']})
    with pytest.raises(ValueError,match='memory'):
        ft.prepared_preview(folder,texts,texts,{**c,'maximumResidentBytes':100})


def test_mps_activation_recomputation_and_weights_only_boundary(folder,tmp_path):
    if not torch.backends.mps.is_available():pytest.skip('Requires actual Apple GPU')
    c=settings(mode='lora',targets=['attentionInput'],device='mps',activationCheckpointing=True,epochs=2)
    report=ft.fine_tune(folder,tmp_path/'gpu',['hello world .']*4,['world hello .'],c,pause_requested=lambda:True)
    assert report['continuation']=='weightsOnly' and report['completedEpoch']==1
    with pytest.raises(ValueError,match='CPU epoch boundary'):
        ft.fine_tune(folder,tmp_path/'exact',['hello world .']*4,['world hello .'],c,resume_from=tmp_path/'gpu')

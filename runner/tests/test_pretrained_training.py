"""Independent safe-format round trips and real learned/frozen parameter proofs."""
import hashlib
import json
import pytest
import torch
from runner.tests.test_pretrained_models import folder
from runner.pretrained_training import fine_tune, training_inventory, training_settings, prepare_documents
from runner.pretrained_models import load_model, inspect_model


def settings(**changes):
    value=dict(mode='full',groups=[],targets=[],rank=2,alpha=4.,dropout=0.,bias='none',additionalModules=[],
                textColumn='text',maximumTokens=8,stride=7,appendEOS=True,epochs=1,batchSize=2,accumulation=1,
                seed=23,device='cpu',maximumResidentBytes=200_000_000,
                optimizer=dict(schemaVersion=1,componentID='optimizer_adamw',configurationVersion=1,config={'lr':.01}),
                trainingControl=dict(schemaVersion=1,modelSelection='finalEpoch',monitor='validationLoss',earlyStopping=dict(enabled=False,patience=5,minimumImprovement=0.)),
                reporting=dict(everyBatches=1,everySeconds=.1))
    value.update(changes);return value


def state(folder):return {k:v.detach().cpu().clone() for k,v in load_model(folder).state_dict().items()}
def pins(folder):return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def test_full_updates_weights_emits_measured_progress_and_reopens(folder,tmp_path):
    before=state(folder);original=pins(folder);events=[]
    result=fine_tune(folder,tmp_path/'result',['hello world .','world hello .']*2,['hello world .'],settings(),emit=events.append)
    after=state(tmp_path/'result/model')
    assert any(not torch.equal(v,after[k]) for k,v in before.items())
    assert pins(folder)==original
    assert result['completedEpoch']==1 and result['updates']>0
    assert any(e['kind']=='training' for e in events) and any(e['kind']=='validation' for e in events)
    assert result['selectedEpoch']==1
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(tmp_path/'result/model',local_files_only=True,trust_remote_code=False,use_safetensors=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in after.items())


def test_selected_group_preserves_frozen_and_tied_parameters(folder,tmp_path):
    inventory=training_inventory(folder)
    assert any(g['id']=='block:0' for g in inventory['groups'])
    tied=next(g for g in inventory['groups'] if 'transformer.wte.weight' in g['parameters'])
    assert 'lm_head.weight' in tied['parameters']
    before=state(folder)
    fine_tune(folder,tmp_path/'result',['hello world .']*2,['world hello .'],settings(mode='selected',groups=['block:0']))
    after=state(tmp_path/'result/model')
    assert torch.equal(before['transformer.wte.weight'],after['transformer.wte.weight'])
    assert any(not torch.equal(before[k],v) for k,v in after.items() if k.startswith('transformer.h.0.'))


def test_padding_uses_positions_not_eos_id_and_shifts_once(folder):
    config=json.loads((folder/'tokenizer_config.json').read_text());config['pad_token']='[EOS]'
    (folder/'tokenizer_config.json').write_text(json.dumps(config))
    batches=prepare_documents(folder,['hello [EOS] world','hello'],settings())
    batch=batches[0]
    assert batch['labels'][0].tolist()==[2,4,2]
    assert batch['labels'][1].tolist()==[2,-100,-100]
    assert batch['attention_mask'][1].tolist()==[1,0,0]


def test_lora_has_frozen_base_adapter_and_merged_roundtrips(folder,tmp_path):
    from runner.pretrained_training import export_fine_tuned, load_adapter
    before=pins(folder)
    result=fine_tune(folder,tmp_path/'result',['hello world .']*2,['world hello .'],settings(mode='lora',targets=['attentionInput']))
    assert result['trainableParameters']<result['totalParameters']
    assert pins(folder)==before
    export_fine_tuned(tmp_path/'result',tmp_path/'adapter',kind='adapter')
    assert not (tmp_path/'adapter/model.safetensors').exists()
    adapter=load_adapter(tmp_path/'adapter',folder)
    export_fine_tuned(tmp_path/'result',tmp_path/'merged',kind='merged')
    merged=load_model(tmp_path/'merged')
    ids=torch.tensor([[3,4]])
    with torch.no_grad():assert torch.allclose(adapter(input_ids=ids).logits,merged(input_ids=ids).logits,atol=1e-6)
    (tmp_path/'adapter/adapter_config.json').write_text('{}')
    with pytest.raises(ValueError,match='changed|identity|provenance'):load_adapter(tmp_path/'adapter',folder)


def test_failure_and_cancel_remove_partial_output(folder,tmp_path):
    with pytest.raises(ValueError,match='memory'):
        fine_tune(folder,tmp_path/'result',['hello'],['hello'],settings(maximumResidentBytes=1))
    assert not (tmp_path/'result').exists()
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):fine_tune(folder,tmp_path/'result',['hello'],['hello'],settings(),check_cancelled=cancel)
    assert not (tmp_path/'result').exists()
    with pytest.raises(ValueError,match='group'):fine_tune(folder,tmp_path/'result',['hello'],['hello'],settings(mode='selected',groups=['invented']))
    assert not (tmp_path/'result').exists()


def test_strict_settings_and_target_inventory(folder):
    with pytest.raises(ValueError):training_settings(settings(epochs=True))
    with pytest.raises(ValueError):training_settings({**settings(),'remoteCode':True})
    with pytest.raises(ValueError):training_settings(settings(stride=8))
    with pytest.raises(ValueError,match='Validation accuracy'):
        training_settings(settings(scheduler=dict(schemaVersion=1,componentID='scheduler_reduce_on_plateau',configurationVersion=1,config={'metric':'validation_accuracy'})))
    inventory=training_inventory(folder)
    assert next(g for g in inventory['targets'] if g['id']=='attentionInput')['modules']==['transformer.h.0.attn.c_attn']


def test_cpu_exact_continuation_matches_uninterrupted_run(folder,tmp_path):
    texts=['hello world .','world hello .']*3;validation=['hello .']
    fine_tune(folder,tmp_path/'whole',texts,validation,settings(epochs=2))
    fine_tune(folder,tmp_path/'first',texts,validation,settings())
    fine_tune(folder,tmp_path/'continued',texts,validation,settings(epochs=2),resume_from=tmp_path/'first')
    a=state(tmp_path/'whole/model');b=state(tmp_path/'continued/model')
    assert all(torch.equal(a[k],b[k]) for k in a)
    with pytest.raises(ValueError,match='identical'):
        fine_tune(folder,tmp_path/'wrong',['different'],validation,settings(epochs=2),resume_from=tmp_path/'first')


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Requires actual Apple MPS')
@pytest.mark.parametrize('mode,options',[('full',{}),('selected',{'groups':['block:0']}),('lora',{'targets':['attentionInput']})])
def test_real_mps_learned_weights_roundtrip(folder,tmp_path,mode,options):
    before=state(folder)
    fine_tune(folder,tmp_path/'result',['hello world .']*2,['world hello .'],settings(mode=mode,device='mps',**options))
    after=state(tmp_path/'result/model')
    assert any(not torch.equal(before[k],v) for k,v in after.items())
    if mode=='selected':assert torch.equal(before['transformer.wte.weight'],after['transformer.wte.weight'])
    assert json.loads((tmp_path/'result/training.json').read_text())['continuation']=='weightsOnly'


def test_shared_scheduler_and_best_selection_keep_terminal_state(folder,tmp_path):
    schedule=dict(schemaVersion=1,componentID='scheduler_step_lr',configurationVersion=1,config={'step_size':1,'gamma':.5})
    c=settings(epochs=2);c['scheduler']=schedule
    report=fine_tune(folder,tmp_path/'result',['hello world .']*2,['world hello .'],c)
    terminal=json.loads((tmp_path/'result/latest/kwip-continuation.json').read_text())
    assert terminal['epoch']==report['completedEpoch']==2
    assert (tmp_path/'result/latest/training-state.safetensors').exists()


@pytest.mark.parametrize('mode,choices',[('full',{}),('selected',{'groups':['block:0']}),('lora',{'targets':['query','value']})])
def test_llama_training_and_independent_reopen(folder,tmp_path,mode,choices):
    from transformers import LlamaConfig,LlamaForCausalLM,AutoModelForCausalLM
    model=LlamaForCausalLM(LlamaConfig(vocab_size=6,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32,pad_token_id=1,eos_token_id=2,bos_token_id=None))
    model.save_pretrained(folder)
    before=state(folder)
    fine_tune(folder,tmp_path/'result',['hello world .']*2,['world hello .'],settings(mode=mode,**choices))
    restored=AutoModelForCausalLM.from_pretrained(tmp_path/'result/model',local_files_only=True,trust_remote_code=False,use_safetensors=True)
    assert any(not torch.equal(v,restored.state_dict()[k]) for k,v in before.items())
    if mode=='selected':assert torch.equal(before['model.embed_tokens.weight'],restored.state_dict()['model.embed_tokens.weight'])


def test_gradient_accumulation_uses_token_weighted_window(folder,tmp_path):
    config=json.loads((folder/'config.json').read_text())
    for key in ('attn_pdrop','embd_pdrop','resid_pdrop'):config[key]=0.
    (folder/'config.json').write_text(json.dumps(config))
    docs=['hello world .','world .','hello hello world .','world hello .']
    optimizer=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config={'lr':.01})
    fine_tune(folder,tmp_path/'batch',docs,['hello .'],settings(batchSize=4,optimizer=optimizer))
    fine_tune(folder,tmp_path/'micro',docs,['hello .'],settings(batchSize=1,accumulation=4,optimizer=optimizer))
    a=state(tmp_path/'batch/model');b=state(tmp_path/'micro/model')
    assert all(torch.allclose(a[k],b[k],atol=1e-6,rtol=1e-5) for k in a)


def test_predictions_are_complete_measured_next_token_rows(folder,tmp_path):
    from runner.pretrained_training import prediction_rows
    fine_tune(folder,tmp_path/'result',['hello world .'],['world hello .'],settings())
    rows=prediction_rows(tmp_path/'result/model',['hello world .','world .'],settings())
    assert len(rows)==2
    assert rows[0][1]==[3,4,5] and rows[0][2]==[4,5,2]
    assert len(rows[0][3])==rows[0][4]==3
    assert rows[0][5]>0 and rows[0][6]>1


def test_exported_cpu_boundary_can_resume_independently(folder,tmp_path):
    from runner.pretrained_training import export_fine_tuned
    docs=['hello world .']*2;valid=['world hello .']
    fine_tune(folder,tmp_path/'result',docs,valid,settings())
    export_fine_tuned(tmp_path/'result',tmp_path/'boundary',kind='continuation')
    report=fine_tune(folder,tmp_path/'next',docs,valid,settings(epochs=2),resume_from=tmp_path/'boundary')
    assert report['completedEpoch']==2
    assert json.loads((tmp_path/'boundary/kwip-continuation.json').read_text())['epoch']==1


def test_sharded_safe_optimizer_state_and_plateau_continue_exactly(folder,tmp_path,monkeypatch):
    import runner.pretrained_training as task
    original=task._save_safe_state
    def sharded(*args,**kwargs):return original(*args,**kwargs,shard_bytes=4096)
    monkeypatch.setattr(task,'_save_safe_state',sharded)
    schedule=dict(schemaVersion=1,componentID='scheduler_reduce_on_plateau',configurationVersion=1,config={})
    docs=['hello world .','world hello .']*2
    fine_tune(folder,tmp_path/'whole',docs,['hello .'],settings(epochs=2,scheduler=schedule))
    fine_tune(folder,tmp_path/'first',docs,['hello .'],settings(scheduler=schedule))
    assert len(list((tmp_path/'first/latest').glob('training-state-*.safetensors')))>1
    fine_tune(folder,tmp_path/'continued',docs,['hello .'],settings(epochs=2,scheduler=schedule),resume_from=tmp_path/'first')
    a=state(tmp_path/'whole/model');b=state(tmp_path/'continued/model')
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_adapter_requires_exact_matching_base(folder,tmp_path):
    import shutil
    from runner.pretrained_training import load_adapter
    fine_tune(folder,tmp_path/'result',['hello world .']*2,['hello .'],settings(mode='lora',targets=['attentionInput']))
    shutil.copytree(folder,tmp_path/'different')
    config=json.loads((tmp_path/'different/config.json').read_text());config['layer_norm_epsilon']=.0002
    (tmp_path/'different/config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='base|identity|changed'):load_adapter(tmp_path/'result/adapter',tmp_path/'different')


def test_overlapping_windows_are_admitted_by_materialized_memory(folder):
    from runner.pretrained_training import _windows
    # Individual tokenization fits, but stride-one overlap does not.
    with pytest.raises(ValueError,match='Overlapping token windows'):
        _windows(folder,['hello '*4000],settings(maximumTokens=512,stride=1,maximumResidentBytes=8_000_000),lambda:None)
    with pytest.raises(ValueError,match='Document tokenization'):
        _windows(folder,['hello '*4000],settings(maximumResidentBytes=1_000_000),lambda:None)


def test_lora_additional_modules_are_independent_and_reopen(folder,tmp_path):
    from runner.pretrained_training import load_adapter
    with pytest.raises(ValueError,match='overlap'):
        fine_tune(folder,tmp_path/'overlap',['hello world .'],['hello .'],settings(mode='lora',targets=['attentionInput'],additionalModules=['block:0']))
    fine_tune(folder,tmp_path/'valid',['hello world .']*2,['hello .'],settings(mode='lora',targets=['attentionInput'],additionalModules=['finalNorm'],bias='all'))
    restored=load_adapter(tmp_path/'valid/adapter',folder)
    assert any('modules_to_save' in n for n,_ in restored.named_parameters())


def test_auto_device_uses_shared_resolver_and_records_effective_device(folder,tmp_path,monkeypatch):
    """Automatic must not silently force CPU when the shared resolver selects MPS."""
    from runner import execution_device
    calls=[]
    def resolve(requested):
        calls.append(requested)
        return 'cpu'
    monkeypatch.setattr(execution_device,'resolve_device',resolve)
    report=fine_tune(folder,tmp_path/'trained',['hello world .'],['world hello .'],settings(device='auto'))
    assert calls==['auto']
    assert report['settings']['device']=='cpu'


def test_automatic_training_uses_the_actual_available_runtime_device(folder,tmp_path):
    expected='mps' if torch.backends.mps.is_available() else 'cpu'
    before=pins(folder)
    result=fine_tune(folder,tmp_path/'automatic',['hello world .'],['world hello .'],settings(device='auto'))
    assert result['settings']['device']==expected
    assert result['continuation']==('weightsOnly' if expected=='mps' else 'cpuEpochBoundary')
    assert pins(folder)==before

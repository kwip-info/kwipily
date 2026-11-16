import json
from pathlib import Path
import pytest
import torch
from runner.pretrained_models import inspect_model, tokenize, load_model, export_model

@pytest.fixture
def folder(tmp_path):
    from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,pre_tokenizers
    torch.manual_seed(17)
    p=tmp_path/'model';p.mkdir()
    t=Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'hello':3,'world':4,'.':5},unk_token='[UNK]'))
    t.pre_tokenizer=pre_tokenizers.Whitespace()
    t=PreTrainedTokenizerFast(tokenizer_object=t,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]',model_max_length=32)
    t.save_pretrained(p)
    m=GPT2LMHeadModel(GPT2Config(vocab_size=6,n_positions=32,n_embd=8,n_layer=1,n_head=2,bos_token_id=None,eos_token_id=2,pad_token_id=1))
    m.save_pretrained(p)
    return p

def test_metadata_and_token_shapes_do_not_flatten(folder):
    value=inspect_model(folder)
    assert value['architecture']=='gpt2' and value['vocabularySize']==6
    assert value['capabilities']==['inspect','tokenize','generate','export']
    packet=tokenize(folder,['hello world','hello'],max_tokens=8,padding='longest',truncation='reject',padding_side='right')
    assert packet['input_ids']==[[3,4],[3,1]] and packet['attention_mask']==[[1,1],[1,0]]
    assert packet['tokens'][0]==['hello','world']

def test_round_trip_same_logits_and_tokenizer(folder,tmp_path):
    model=load_model(folder,device='cpu',maximum_resident_bytes=100_000_000)
    ids=torch.tensor([[3,4]])
    with torch.no_grad():before=model(input_ids=ids).logits
    out=tmp_path/'saved';export_model(folder,out,shard_bytes=1024)
    assert (out/'kwip-manifest.json').is_file()
    after_model=load_model(out,device='cpu',maximum_resident_bytes=100_000_000)
    with torch.no_grad():after=after_model(input_ids=ids).logits
    assert torch.equal(before,after)
    assert tokenize(out,['hello'])['input_ids']==[[3]]
    with pytest.raises(ValueError,match='exist'):export_model(folder,out)

def test_refuse_code_and_mismatched_vocabulary(folder):
    config=json.loads((folder/'config.json').read_text());config['auto_map']={'AutoModel':'evil.Model'}
    (folder/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='code'):inspect_model(folder)
    del config['auto_map'];config['vocab_size']=7;(folder/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='vocabulary'):inspect_model(folder)

def test_budget_missing_shard_symlink_and_changed_pin(folder,tmp_path):
    with pytest.raises(ValueError,match='budget'):inspect_model(folder,maximum_bytes=10)
    weights=folder/'model.safetensors';weights.rename(tmp_path/'weights.safetensors');weights.symlink_to(tmp_path/'weights.safetensors')
    with pytest.raises(ValueError,match='regular|symbolic'):inspect_model(folder)
    weights.unlink()
    with pytest.raises(ValueError,match='weights'):inspect_model(folder)

def test_lengths_padding_and_resource_admission(folder):
    with pytest.raises(ValueError,match='length'):tokenize(folder,['hello world'],max_tokens=1,truncation='reject')
    assert tokenize(folder,['hello world'],max_tokens=1,truncation='truncate')['input_ids']==[[3]]
    with pytest.raises(ValueError,match='memory'):load_model(folder,maximum_resident_bytes=1)
    with pytest.raises(ValueError,match='length|context'):tokenize(folder,['hello'],max_tokens=1000)

def test_inspection_cancellation(folder):
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):inspect_model(folder,check_cancelled=cancel)

def test_llama_safe_adapter_and_offline_export(tmp_path):
    from transformers import LlamaConfig,LlamaForCausalLM,AutoModelForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models
    p=tmp_path/'llama';p.mkdir()
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'hello':3},unk_token='[UNK]'))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]').save_pretrained(p)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=4,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32,pad_token_id=1,eos_token_id=2,bos_token_id=None))
    model.save_pretrained(p)
    assert inspect_model(p)['architecture']=='llama'
    out=tmp_path/'exported';export_model(p,out,shard_bytes=1024)
    # Verify independently with upstream's offline safe loader, not just our wrapper.
    restored=AutoModelForCausalLM.from_pretrained(out,local_files_only=True,trust_remote_code=False,use_safetensors=True)
    with torch.no_grad():assert torch.equal(model(torch.tensor([[3]])).logits,restored(torch.tensor([[3]])).logits)

def test_tampered_shapes_and_unsupported_architecture_refused(folder):
    config=json.loads((folder/'config.json').read_text());config['n_embd']=16
    (folder/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='shape|weight'):inspect_model(folder)
    config['model_type']='bert';(folder/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='adapter|architecture|supported'):inspect_model(folder)

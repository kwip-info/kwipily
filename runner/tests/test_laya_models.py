import json,os,shutil
import numpy as np
import pytest
import torch
from runner.laya_models import Laya,inspect_model,export_model
from runner.tests.test_typed_decisions import QUESTIONS

@pytest.fixture
def model_folder(tmp_path):
 from transformers import ModernBertConfig,ModernBertModel,PreTrainedTokenizerFast
 from tokenizers import Tokenizer,models,pre_tokenizers
 from safetensors.torch import save_file
 from runner.laya_architecture import DecisionModel
 p=tmp_path/'model';(p/'encoder').mkdir(parents=True);(p/'tokenizer').mkdir()
 with torch.random.fork_rng():
  torch.manual_seed(23)
  cfg=ModernBertConfig(vocab_size=64,pad_token_id=0,bos_token_id=2,eos_token_id=3,cls_token_id=2,sep_token_id=3,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=1,max_position_embeddings=512,global_attn_every_n_layers=3,local_attention=32,attention_dropout=0.)
  cfg._attn_implementation='sdpa';model=DecisionModel(ModernBertModel(cfg),1,2).eval()
  save_file({k:v.contiguous() for k,v in model.state_dict().items()},p/'model.safetensors');cfg.save_pretrained(p/'encoder')
 vocabulary=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','billing','technical','yes','no','refund','crash','account','Low','High','Medium','team','review','urgency','Which','ticket','handles','this','?','choice','score','noul','question',':','Billing','Technical','No','Yes','false','true','level','0','1','2']
 tokenizer=Tokenizer(models.WordLevel({w:i for i,w in enumerate(vocabulary)},unk_token='[UNK]'));tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
 PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]').save_pretrained(p/'tokenizer')
 (p/'rl_agent_config.json').write_text(json.dumps(dict(encoder='answerdotai/ModernBERT-large',head_layers=1,act_costs={'escalate':.5},max_len=512,head_max_len=128,temperature=[1.,1.,1.],temperature_by_options={'choice:11+':.1})))
 return p

def test_real_safe_model_and_complete_probability_roundtrip(model_folder,tmp_path):
 info=inspect_model(model_folder);assert info['task']=='typedDecisions' and info['architecture']=='laya-modernbert'
 assert info['calibrationWarnings'] and 'typedDecisions' in info['capabilities']
 engine=Laya(model_folder);before=engine.predict([{'message':'refund'}],QUESTIONS)
 assert len(before)==1 and len(before[0])==3
 assert sum(before[0][0]['probabilities'])==pytest.approx(1.)
 target=tmp_path/'export';export_model(model_folder,target)
 after=Laya(target).predict([{'message':'refund'}],QUESTIONS)
 assert before==after
 with pytest.raises(ValueError):export_model(model_folder,target)

def test_no_silent_truncation_no_network_or_tokenizer_class_execution(model_folder):
 cfg=model_folder/'tokenizer/tokenizer_config.json';v=json.loads(cfg.read_text());v['auto_map']={'AutoTokenizer':'evil.Tokenizer'};cfg.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='code'):inspect_model(model_folder)
 v.pop('auto_map');cfg.write_text(json.dumps(v))
 engine=Laya(model_folder)
 with pytest.raises(ValueError,match='budget'):engine.predict(['refund '*1000],QUESTIONS)
 def cancel():raise InterruptedError('cancel')
 with pytest.raises(InterruptedError):engine.predict(['refund'],QUESTIONS,check=cancel)

def test_nested_path_symlink_and_bad_geometry_are_refused(model_folder,tmp_path):
 p=model_folder/'tokenizer/tokenizer.json';copy=tmp_path/'tokenizer.json';p.rename(copy);p.symlink_to(copy)
 with pytest.raises(ValueError,match='regular|symbolic'):inspect_model(model_folder)
 p.unlink();copy.rename(p)
 p=model_folder/'encoder/config.json';v=json.loads(p.read_text());v['hidden_size']=100000;p.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='geometry'):inspect_model(model_folder)

def test_reserved_mask_and_mismatched_calibration_are_refused(model_folder):
 engine=Laya(model_folder)
 with pytest.raises(ValueError,match='MASK|reserved'):engine.predict(['refund [MASK]'],QUESTIONS)
 (model_folder/'kwip-decisions.json').write_text(json.dumps(dict(schemaVersion=1,questionSet=QUESTIONS,calibration=dict(questionHash='0'*64,questions={}))))
 with pytest.raises(ValueError,match='revision'):inspect_model(model_folder)

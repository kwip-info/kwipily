import json
import numpy as np
import pytest
import torch
from runner.audio_classification_models import AST,extract_features
from runner.pretrained_models import inspect_model,export_model,load_model,tokenize

@pytest.fixture
def ast_folder(tmp_path):
 from transformers import ASTConfig,ASTForAudioClassification,ASTFeatureExtractor
 p=tmp_path/'ast';p.mkdir();torch.manual_seed(42)
 c=ASTConfig(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,id2label={0:'Speech',1:'Music',2:'Other'},label2id={'Speech':0,'Music':1,'Other':2},problem_type='multi_label_classification')
 ASTForAudioClassification(c).save_pretrained(p)
 ASTFeatureExtractor().save_pretrained(p)
 return p

def test_processor_matches_independent_reference_and_retains_short_samples():
 from transformers import ASTFeatureExtractor
 rng=np.random.default_rng(7);waves=[rng.normal(0,.1,16000).astype('float32'),np.zeros(160000,dtype='float32')]
 reference=ASTFeatureExtractor()
 # Force the documented NumPy path independently of optional torchaudio installation.
 import transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer as module
 original=module.is_speech_available
 module.is_speech_available=lambda:False
 try:expected=reference(waves,sampling_rate=16000,return_tensors='np').input_values
 finally:module.is_speech_available=original
 actual=extract_features(waves,16000)
 np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
 assert extract_features([np.zeros(100,dtype='float32')],16000).shape==(1,1024,128)
 with pytest.raises(ValueError):extract_features(waves,8000)
 with pytest.raises(ValueError):extract_features([np.zeros(160001,dtype='float32')],16000)

def test_actual_safe_ast_inference_export_reload(ast_folder,tmp_path):
 info=inspect_model(ast_folder);assert info['task']=='audioClassification' and info['labels']==['Speech','Music','Other']
 assert not (ast_folder/'tokenizer.json').exists()
 engine=AST(ast_folder);waves=[np.zeros(16000,dtype='float32')];before=engine.predict(waves,16000)
 assert before.shape==(1,3) and np.isfinite(before).all()
 out=tmp_path/'export';export_model(ast_folder,out,shard_bytes=1000000)
 np.testing.assert_array_equal(before,AST(out).predict(waves,16000))
 for action in [lambda:load_model(ast_folder),lambda:tokenize(ast_folder,['hello'])]:
  with pytest.raises(ValueError):action()

def test_refuse_task_processor_labels_and_remote_code(ast_folder):
 for name,key,value in [('config.json','auto_map',{'AutoModel':'evil.Model'}),('config.json','id2label',{'1':'Missing zero'}),('preprocessor_config.json','sampling_rate',8000),('preprocessor_config.json','mean',0)]:
  p=ast_folder/name;old=p.read_text();c=json.loads(old);c[key]=value;p.write_text(json.dumps(c))
  with pytest.raises(ValueError):inspect_model(ast_folder)
  p.write_text(old)
 p=ast_folder/'preprocessor_config.json';p.unlink()
 with pytest.raises(ValueError):inspect_model(ast_folder)

"""Reviewed offline AST adapter. No Hub access, tokenizer, repository code or pickle."""
import hashlib,json,math,os,re
from pathlib import Path
import numpy as np
REVISION='f826b80d28226b62986cc218e5cec390b1096902'
def require(ok,message):
 if not ok:raise ValueError(message)
def inspect(folder,files,weights,total,check=lambda:None):
 from safetensors import safe_open
 if __package__:from .pretrained_models import _json
 else:from pretrained_models import _json
 c=_json(folder/'config.json');p=_json(folder/'preprocessor_config.json')
 require(c.get('architectures')==['ASTForAudioClassification'] and not any(c.get(k) for k in ('auto_map','custom_pipelines','quantization_config')),'Only safe AST audio-classification architectures are supported.')
 require(not p.get('auto_map') and p.get('feature_extractor_type')=='ASTFeatureExtractor' and p.get('sampling_rate')==16000 and p.get('num_mel_bins')==128 and p.get('max_length')==1024 and p.get('do_normalize') is True and p.get('mean')==-4.2677393 and p.get('std')==4.5689974 and p.get('feature_size')==1 and p.get('padding_side')=='right' and p.get('padding_value')==0 and p.get('return_attention_mask') is False,'AST requires the reviewed 16 kHz/128 mel/1024 frame AudioSet processor.')
 require(all(type(c.get(k)) is int and 1<=c[k]<=bound for k,bound in [('hidden_size',1536),('num_hidden_layers',24),('num_attention_heads',24),('intermediate_size',6144),('patch_size',32),('frequency_stride',32),('time_stride',32)]) and c.get('num_mel_bins',128)==128 and c.get('max_length',1024)==1024 and c['hidden_size']%c['num_attention_heads']==0,'AST geometry exceeds supported admission.')
 labels=c.get('id2label');require(type(labels) is dict and 1<=len(labels)<=1024 and set(labels)=={str(i) for i in range(len(labels))},'AST class IDs must be contiguous and complete.')
 ordered=[labels[str(i)] for i in range(len(labels))];require(all(type(x) is str and 0<len(x)<=200 for x in ordered) and len(set(ordered))==len(ordered),'AST class labels must be unique bounded strings.')
 require(c.get('problem_type') in (None,'multi_label_classification'),'Only the multi-label AST adapter is qualified.')
 if 'label2id' in c:require(c['label2id']=={label:i for i,label in enumerate(ordered)},'AST label maps disagree.')
 shapes={};parameters=0;weight_bytes=0;dtypes=set()
 for name in weights:
  check()
  with safe_open(folder/name,framework='pt',device='cpu') as f:
   for key in f.keys():
    canonical=canonical_weight_key(key);require(canonical not in shapes,'Duplicate AST weight.');v=f.get_slice(key);shape=tuple(v.get_shape());dtype=v.get_dtype()
    require(all(type(n) is int and n>0 for n in shape) and dtype in ('F32','F16','BF16'),'Invalid AST weight type/shape.')
    shapes[canonical]=shape;count=math.prod(shape);parameters+=count;weight_bytes+=count*(4 if dtype=='F32' else 2);dtypes.add(dtype)
 require(parameters<=200000000 and weight_bytes<=1400000000,'AST resident weights exceed admission.')
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
 import torch
 from transformers import ASTConfig,ASTForAudioClassification,ASTFeatureExtractor
 check()
 with torch.device('meta'):model=ASTForAudioClassification(ASTConfig.from_pretrained(folder,local_files_only=True))
 expected={k:tuple(v.shape) for k,v in model.state_dict().items()}
 require(expected==shapes,'AST weight keys/shapes differ from the declared architecture.')
 del model
 processor=ASTFeatureExtractor.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
 # Always qualify and record the NumPy processor path: optional torchaudio cannot change captured semantics.
 processor_path='numpy-kaldi-v1'
 identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return dict(schemaVersion=1,architecture='audio-spectrogram-transformer',task='audioClassification',capabilities=['inspect','classify','export'],vocabularySize=0,maximumTokens=0,parameterCount=parameters,weightBytes=weight_bytes,totalBytes=total,dtypes=sorted(dtypes),identity=identity,files=files,labels=ordered,sampleRate=16000,maximumSeconds=10,processorBackend=processor_path,scoreSemantics='independentSigmoid',inputs=[dict(name='audio',shape=['channels','samples'],dtype='float32')],outputs=[dict(name=n,shape=['rows'],dtype='table') for n in ('predictions','scores')],specialTokens={})

class AST:
 def __init__(self,folder,check=lambda:None):
  if __package__:from .pretrained_models import inspect_model
  else:from pretrained_models import inspect_model
  self.info=inspect_model(folder,check_cancelled=check);require(self.info['task']=='audioClassification','Connect a verified AST classification model.')
  from transformers import ASTForAudioClassification
  import torch
  check();self.model,loading=ASTForAudioClassification.from_pretrained(folder,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
  require(not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'AST safe weights did not load completely.')
  self.model=self.model.cpu().eval();self.check=check
 def predict(self,waves,rate):
  import torch
  features=extract_features(waves,rate,self.check)
  self.check()
  with torch.inference_mode():out=self.model(input_values=torch.from_numpy(features)).logits.cpu().numpy()
  self.check();require(out.shape==(len(waves),len(self.info['labels'])) and np.isfinite(out).all(),'AST produced invalid class logits.');return out

def extract_features(waves,rate,check=lambda:None):
 require(rate==16000 and 1<=len(waves)<=8,'AST needs bounded mono 16 kHz windows.')
 from transformers.audio_utils import mel_filter_bank,spectrogram,window_function
 features=[]
 filters=mel_filter_bank(num_frequency_bins=257,num_mel_filters=128,min_frequency=20,max_frequency=8000,norm=None,mel_scale='kaldi',triangularize_in_mel_space=True,sampling_rate=16000)
 window=window_function(400,'hann',periodic=False)
 for samples in waves:
  check();require(samples.dtype==np.float32 and samples.ndim==1 and 0<len(samples)<=160000 and np.isfinite(samples).all(),'AST needs finite float32 mono windows up to10 seconds.')
  # The reference extractor needs a complete25ms frame. Pad only computation; source length remains in table lineage.
  samples=np.pad(samples,(0,max(0,400-len(samples))))
  values=spectrogram(samples,window,frame_length=400,hop_length=160,fft_length=512,power=2.,center=False,preemphasis=.97,mel_filters=filters,log_mel='log',mel_floor=1.192092955078125e-07,remove_dc_offset=True).T
  require(len(values)<=1024,'AST processor would truncate actual samples.')
  values=np.pad(values,((0,1024-len(values)),(0,0)))
  features.append(((values-(-4.2677393))/(4.5689974*2)).astype(np.float32))
 return np.stack(features)

def canonical_weight_key(key):
 # Transformers5 AST uses a reviewed rename of the serialized encoder keys.
 # Only this exact conversion is accepted; dimensions and complete keys still match.
 key=re.sub(r'^audio_spectrogram_transformer\.encoder\.layer\.(\d+)\.',r'audio_spectrogram_transformer.layers.\1.',key)
 for old,new in [('attention.attention.query.','attention.q_proj.'),('attention.attention.key.','attention.k_proj.'),('attention.attention.value.','attention.v_proj.'),('attention.output.dense.','attention.o_proj.'),('intermediate.dense.','mlp.fc1.'),('output.dense.','mlp.fc2.')]:
  key=key.replace(old,new)
 return key

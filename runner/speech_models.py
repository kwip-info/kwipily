"""Offline Whisper adapter: immutable safetensors only, CPU inference, bounded chunks."""
import hashlib,json,os
from pathlib import Path
import numpy as np

REVISION='169d4a4341b33bc18d8881c4b69c2e104e1cc0af'

def require(ok,message):
    if not ok:raise ValueError(message)

def inspect(folder,files,weights,total,check=lambda:None):
    from safetensors import safe_open
    if __package__:from .pretrained_models import _json
    else:from pretrained_models import _json
    c=_json(folder/'config.json');p=_json(folder/'preprocessor_config.json');t=_json(folder/'tokenizer_config.json')
    require(not p.get('auto_map') and not t.get('auto_map') and p.get('processor_class') in (None,'WhisperProcessor') and t.get('processor_class') in (None,'WhisperProcessor') and t.get('tokenizer_class') in ('WhisperTokenizer','WhisperTokenizerFast'),'Speech processors/tokenizers cannot request repository code or another task adapter.')
    require(c.get('architectures')==['WhisperForConditionalGeneration'] and not c.get('auto_map') and not c.get('quantization_config'),'Only the reviewed Whisper conditional-generation adapter is supported.')
    require(p.get('feature_extractor_type')=='WhisperFeatureExtractor' and p.get('sampling_rate')==16000 and p.get('chunk_length')==30 and p.get('hop_length')==160 and p.get('n_fft')==400 and p.get('feature_size') in (80,128),'Whisper requires its matching 16 kHz / 30 second processor.')
    require(all(type(c.get(k)) is int and 1<=c[k]<=bound for k,bound in [('d_model',2048),('encoder_layers',32),('decoder_layers',32),('vocab_size',100000),('max_source_positions',1500),('max_target_positions',1024)]),'Whisper geometry exceeds the supported allowance.')
    require(c['max_source_positions']==1500 and c.get('num_mel_bins')==p['feature_size'],'Whisper encoder geometry and audio processor do not match.')
    shapes={};parameters=0;weight_bytes=0;dtypes=set()
    for name in weights:
        check()
        with safe_open(folder/name,framework='pt',device='cpu') as f:
            for key in f.keys():
                require(key not in shapes,'Duplicate speech weight.');v=f.get_slice(key);shape=tuple(v.get_shape());dtype=v.get_dtype()
                require(all(type(n) is int and n>0 for n in shape) and dtype in ('F32','F16','BF16'),'Invalid speech weight type/shape.')
                shapes[key]=shape;count=int(np.prod(shape));parameters+=count;weight_bytes+=count*(4 if dtype=='F32' else 2);dtypes.add(dtype)
    require(weight_bytes<=2_000_000_000,'Speech weights exceed the model byte allowance.')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0')
    import torch
    from transformers import WhisperConfig,WhisperForConditionalGeneration,WhisperProcessor
    check()
    with torch.device('meta'):model=WhisperForConditionalGeneration(WhisperConfig.from_pretrained(folder,local_files_only=True))
    expected={k:tuple(v.shape) for k,v in model.state_dict().items()};missing=set(expected)-set(shapes)
    if model.proj_out.weight is model.model.decoder.embed_tokens.weight:missing.discard('proj_out.weight')
    require(not missing and not(set(shapes)-set(expected)) and all(expected.get(k)==v for k,v in shapes.items()),'Speech weights do not match the declared architecture.')
    del model
    processor=WhisperProcessor.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    require(len(processor.tokenizer)==c['vocab_size'] and processor.feature_extractor.feature_size==c['num_mel_bins'],'Speech tokenizer/processor differ from model dimensions.')
    identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(schemaVersion=1,architecture='whisper',task='automaticSpeechRecognition',capabilities=['inspect','transcribe'],vocabularySize=c['vocab_size'],maximumTokens=c['max_target_positions'],parameterCount=parameters,weightBytes=weight_bytes,totalBytes=total,dtypes=sorted(dtypes),identity=identity,files=files,inputs=[dict(name='audio',shape=['channels','samples'],dtype='float32')],outputs=[dict(name='transcripts',shape=['rows'],dtype='table'),dict(name='segments',shape=['segments'],dtype='table')],specialTokens={},sampleRate=16000,chunkSeconds=30)

class Whisper:
    def __init__(self,folder,check=lambda:None):
        if __package__:from .pretrained_models import inspect_model
        else:from pretrained_models import inspect_model
        self.info=inspect_model(folder,check_cancelled=check);require(self.info['task']=='automaticSpeechRecognition','Connect a verified Whisper speech model, not a text-generation model.')
        import torch
        from transformers import WhisperForConditionalGeneration,WhisperProcessor
        check();self.processor=WhisperProcessor.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        self.model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='eager').cpu().eval();self.check=check
    def transcribe(self,samples,rate,config):
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        require(samples.dtype==np.float32 and samples.ndim==1 and len(samples)>0 and rate==16000 and np.isfinite(samples).all(),'Speech adapter requires finite mono float32 at 16 kHz. Use Prepare media for explicit mixing/resampling.')
        check=self.check
        class Cancel(StoppingCriteria):
            def __init__(self):self.steps=0
            def __call__(self,*args,**kwargs):check();self.steps+=1;return False
        output=[];chunk=16000*config['chunkSeconds'];maximum=config['maximumTokens']
        require(maximum+4<=self.model.config.max_target_positions,'Token allowance plus the speech prompt exceeds this model decoder context.')
        for start in range(0,len(samples),chunk):
            check();part=samples[start:start+chunk]
            features=self.processor.feature_extractor(part,sampling_rate=rate,return_tensors='pt',return_attention_mask=True)
            stopping=Cancel()
            kwargs=dict(task='transcribe',language=None if config['language']=='auto' else config['language'],return_timestamps=True,max_new_tokens=maximum,num_beams=config['beamWidth'],do_sample=False,stopping_criteria=StoppingCriteriaList([stopping]))
            with torch.inference_mode():tokens=self.model.generate(**features,**kwargs)[0].tolist()
            check();text=self.processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            # Decode timestamp token boundaries, clamped to the actual chunk (never padded 30 s).
            timestamp=self.processor.tokenizer.convert_tokens_to_ids('<|0.00|>');begin=None;words=[];segments=[]
            for token in tokens:
                if token>=timestamp:
                    t=min(len(part)/rate,max(0,(token-timestamp)*0.02))
                    if begin is not None and words:
                        value=self.processor.tokenizer.decode(words,skip_special_tokens=True).strip()
                        if value and t>begin:segments.append((start/rate+begin,start/rate+t,value))
                    begin=t;words=[]
                elif begin is not None:words.append(token)
            aligned=bool(segments)
            # Empty/non-timestamp decoder output is honest chunk-level coverage, not invented word alignment.
            if text and not segments:segments=[(start/rate,(start+len(part))/rate,text)]
            output.append(dict(start=start/rate,end=(start+len(part))/rate,text=text,segments=segments,timestampCoverage='modelSegments' if aligned else 'chunk',truncated=stopping.steps>=maximum))
        return output

"""Offline, allow-listed pretrained model adapters. No remote code, pickle or Hub access.

The native broker owns downloads and immutable snapshots. This module only sees a
retained operation's hash-pinned files; every expensive action is explicit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

MAX_BYTES = 2_000_000_000
MAX_FILES = 128
ADAPTERS = {'gpt2': 'GPT2LMHeadModel', 'llama': 'LlamaForCausalLM'}
JSON_FILES = {'preprocessor_config.json','normalizer.json','kwip-training.json','config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
              'generation_config.json', 'added_tokens.json', 'vocab.json', 'model.safetensors.index.json'}
AUX_FILES = {'merges.txt', 'tokenizer.model', 'README.md', 'LICENSE', 'LICENSE.txt', 'NOTICE', 'NOTICE.txt'}


def allowed_file(name):
    return name in {'rl_agent_config.json','encoder/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json','tokenizer/added_tokens.json','kwip-decisions.json'} or name in JSON_FILES | AUX_FILES or bool(re.fullmatch(r'model(?:-\d{5}-of-\d{5})?\.safetensors', name))


def _json(path, limit=8_000_000):
    if path.stat().st_size > limit:raise ValueError('Model metadata exceeds its byte budget.')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('Duplicate model metadata key.')
            out[k]=v
        return out
    value=json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite model metadata.')))
    if not isinstance(value,dict):raise ValueError('Expected model metadata object.')
    return value


def _inventory(folder, maximum_bytes, check_cancelled):
    folder=Path(folder)
    if folder.is_symlink() or not folder.is_dir():raise ValueError('Choose a regular model folder, not a symbolic link.')
    files=[];total=0
    for path in sorted(folder.iterdir()):
        check_cancelled()
        if not allowed_file(path.name):continue
        before=path.lstat()
        if not stat.S_ISREG(before.st_mode):raise ValueError('Model files must be regular files, never symbolic links.')
        total+=before.st_size
        if before.st_size<=0 or total>maximum_bytes or len(files)>=MAX_FILES:raise ValueError('Model files exceed the captured byte/file budget.')
        digest=hashlib.sha256()
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            opened=os.fstat(stream.fileno())
            if (opened.st_dev,opened.st_ino,opened.st_size)!=(before.st_dev,before.st_ino,before.st_size):raise ValueError('Model file changed during inspection.')
            while chunk:=stream.read(1_048_576):check_cancelled();digest.update(chunk)
            after=os.fstat(stream.fileno())
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Model file changed during inspection.')
        files.append(dict(path=path.name,bytes=before.st_size,sha256=digest.hexdigest()))
    names={x['path'] for x in files}
    if 'config.json' not in names:raise ValueError('Model folder needs config.json.')
    config=_json(folder/'config.json')
    required={'config.json','preprocessor_config.json'} if config.get('model_type')=='audio-spectrogram-transformer' and config.get('architectures')==['ASTForAudioClassification'] else {'config.json','tokenizer.json'}
    if not required<=names:raise ValueError('Model folder lacks its task-matched processor or tokenizer.json.')
    weights=[x['path'] for x in files if x['path'].endswith('.safetensors')]
    if not weights:raise ValueError('Safe model weights are missing. Pickle/.bin checkpoints are not imported.')
    if 'model.safetensors.index.json' in names:
        index=_json(folder/'model.safetensors.index.json')
        mapping=index.get('weight_map')
        if not isinstance(mapping,dict) or not mapping or not all(isinstance(k,str) and isinstance(v,str) and v in weights for k,v in mapping.items()) or set(mapping.values())!=set(weights):raise ValueError('Model weight shards/index are incomplete or unsafe.')
    elif weights!=['model.safetensors']:raise ValueError('Sharded model weights require their exact index.')
    return files,weights,total


def inspect_model(folder,maximum_bytes=MAX_BYTES,check_cancelled=lambda:None):
    if (Path(folder)/'rl_agent_config.json').exists():
        if __package__:from .laya_models import inspect_model as inspect_laya
        else:from laya_models import inspect_model as inspect_laya
        return inspect_laya(folder,maximum_bytes,check_cancelled)
    from safetensors import safe_open
    from tokenizers import Tokenizer
    if type(maximum_bytes) is not int or not 0<maximum_bytes<=MAX_BYTES:raise ValueError('Invalid model byte budget.')
    folder=Path(folder);files,weights,total=_inventory(folder,maximum_bytes,check_cancelled)
    config=_json(folder/'config.json')
    kind=config.get('model_type')
    if config.get('auto_map') or config.get('custom_pipelines'):raise ValueError('Remote repository code is not supported.')
    if kind=='audio-spectrogram-transformer':
        if __package__:from .audio_classification_models import inspect
        else:from audio_classification_models import inspect
        result=inspect(folder,files,weights,total,check_cancelled)
        manifest=folder/'kwip-manifest.json'
        if manifest.exists():
            value=_json(manifest)
            if value.get('schemaVersion')!=1 or value.get('kind')!='kwipPretrainedModel' or value.get('files')!=files:raise ValueError('Exported AST model changed from its provenance.')
        return result
    if kind=='whisper':
        if __package__:from .speech_models import inspect
        else:from speech_models import inspect
        return inspect(folder,files,weights,total,check_cancelled)
    if kind not in ADAPTERS or config.get('quantization_config'):raise ValueError('Unsupported model architecture or quantized weights. Supported: GPT-2 and Llama causal language models.')
    architectures=config.get('architectures')
    if architectures and architectures!=[ADAPTERS[kind]]:raise ValueError('Model task/head does not match the supported causal language adapter.')
    if config.get('add_cross_attention') or config.get('is_encoder_decoder'):raise ValueError('Cross-attention/encoder-decoder model requires a different adapter.')
    vocabulary=config.get('vocab_size');context=config.get('n_positions') if kind=='gpt2' else config.get('max_position_embeddings')
    if type(vocabulary) is not int or not 1<=vocabulary<=2_000_000 or type(context) is not int or not 1<=context<=131_072:raise ValueError('Model vocabulary or context exceeds supported bounds.')
    tc=_json(folder/'tokenizer_config.json') if (folder/'tokenizer_config.json').exists() else {}
    if tc.get('auto_map'):raise ValueError('Tokenizer repository code is not supported.')
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
    if tokenizer.get_vocab_size(with_added_tokens=True)!=vocabulary:raise ValueError('Tokenizer vocabulary does not match the model embedding rows.')
    if set(tokenizer.get_vocab(with_added_tokens=True).values())!=set(range(vocabulary)):raise ValueError('Tokenizer IDs must match contiguous model vocabulary rows.')
    width=config.get('n_embd') if kind=='gpt2' else config.get('hidden_size')
    layers=config.get('n_layer') if kind=='gpt2' else config.get('num_hidden_layers')
    heads=config.get('n_head') if kind=='gpt2' else config.get('num_attention_heads')
    if any(type(v) is not int for v in (width,layers,heads)) or not 1<=width<=16384 or not 1<=layers<=128 or not 1<=heads<=256 or width%heads:raise ValueError('Model architecture dimensions exceed supported bounds.')
    parameters=0;weight_bytes=0;names=set();dtypes=set();actual_shapes={}
    for name in weights:
        check_cancelled()
        with safe_open(folder/name,framework='pt',device='cpu') as handle:
            for key in handle.keys():
                if key in names:raise ValueError('Duplicate tensor across model shards.')
                names.add(key);view=handle.get_slice(key);shape=view.get_shape();count=1
                for dimension in shape:
                    if type(dimension) is not int or dimension<=0:raise ValueError('Invalid model weight shape.')
                    count*=dimension
                actual_shapes[key]=tuple(shape)
                dtype=view.get_dtype();size={'F32':4,'F16':2,'BF16':2}.get(dtype)
                if size is None:raise ValueError('Only float32, float16 and bfloat16 weights are supported.')
                parameters+=count;weight_bytes+=count*size;dtypes.add(dtype)
                if weight_bytes>maximum_bytes:raise ValueError('Model tensors exceed the byte budget.')
    if 'model.safetensors.index.json' in {x['path'] for x in files} and set(_json(folder/'model.safetensors.index.json')['weight_map'])!=names:raise ValueError('Weight index and actual tensors disagree.')
    # Match architecture keys/shapes on the meta device without allocating model weights.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['USE_TF']='0'
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    check_cancelled()
    with torch.device('meta'):
        skeleton=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False),trust_remote_code=False,attn_implementation='eager')
    expected={name:tuple(value.shape) for name,value in skeleton.state_dict().items()}
    missing=set(expected)-set(actual_shapes)
    tied=skeleton.get_output_embeddings().weight is skeleton.get_input_embeddings().weight
    if tied:missing.discard('lm_head.weight')
    if missing or set(actual_shapes)-set(expected) or any(expected.get(k)!=v for k,v in actual_shapes.items()):raise ValueError('Safe weights do not match the declared architecture dimensions/keys.')
    del skeleton
    # Existing exported provenance is data, never an execution instruction.
    manifest=folder/'kwip-manifest.json'
    if manifest.exists():
        if manifest.is_symlink():raise ValueError('Model provenance must be a regular file.')
        pinned=_json(manifest)
        if pinned.get('schemaVersion')!=1 or pinned.get('kind')!='kwipPretrainedModel' or pinned.get('files')!=files:raise ValueError('Exported model files changed from their saved provenance.')
    identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(schemaVersion=1,architecture=kind,task='causalLanguageModel',capabilities=['inspect','tokenize','generate','export'],
                vocabularySize=vocabulary,maximumTokens=context,parameterCount=parameters,weightBytes=weight_bytes,totalBytes=total,
                dtypes=sorted(dtypes),identity=identity,files=files,
                inputs=[dict(name='input_ids',shape=['N','T'],dtype='int64'),dict(name='attention_mask',shape=['N','T'],dtype='int64')],
                outputs=[dict(name='logits',shape=['N','T','V'],dtype='float')],specialTokens=_tokenizer(folder)[1])


def _tokenizer(folder):
    from tokenizers import Tokenizer
    folder=Path(folder);tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
    cfg=_json(folder/'tokenizer_config.json') if (folder/'tokenizer_config.json').exists() else {}
    special={}
    for name in ('pad_token','eos_token','bos_token','unk_token'):
        value=cfg.get(name)
        if isinstance(value,dict):value=value.get('content')
        special[name]=tokenizer.token_to_id(value) if isinstance(value,str) else None
    return tokenizer,special


def tokenize(folder,texts,max_tokens=None,padding='longest',truncation='reject',padding_side='right',check_cancelled=lambda:None):
    info=inspect_model(folder,check_cancelled=check_cancelled)
    if info['task']!='causalLanguageModel':raise ValueError('Speech models use their matching audio processor through Transcribe audio, not the causal token preview.')
    if not isinstance(texts,list) or not 1<=len(texts)<=128 or any(not isinstance(t,str) or len(t.encode())>262144 for t in texts):raise ValueError('Token preview needs 1–128 bounded text rows.')
    maximum=info['maximumTokens'] if max_tokens is None else max_tokens
    if type(maximum) is not int or not 1<=maximum<=info['maximumTokens']:raise ValueError('Requested token length exceeds model context.')
    if padding not in ('none','longest','maximum') or truncation not in ('reject','truncate') or padding_side not in ('left','right'):raise ValueError('Unsupported tokenizer settings.')
    tokenizer,special=_tokenizer(folder);tokenizer.no_padding();tokenizer.no_truncation()
    encoded=[]
    for text in texts:
        check_cancelled();value=tokenizer.encode(text,add_special_tokens=True)
        if len(value.ids)>maximum and truncation=='reject':raise ValueError('Text exceeds maximum token length; select explicit truncation or reduce the text.')
        encoded.append(value)
    ids=[e.ids[:maximum] for e in encoded];tokens=[e.tokens[:maximum] for e in encoded]
    if any(not row for row in ids):raise ValueError('Text must produce at least one token.')
    masks=[[1]*len(row) for row in ids]
    width=max(map(len,ids)) if padding=='longest' else maximum if padding=='maximum' else None
    if width is not None:
        pad=special['pad_token']
        if pad is None and any(len(row)<width for row in ids):raise ValueError('Tokenizer has no padding token. Use unpadded rows or a matching tokenizer with padding defined.')
        for i,row in enumerate(ids):
            n=width-len(row)
            if n:
                pieces=[tokenizer.id_to_token(pad)]*n
                if padding_side=='left':ids[i]=[pad]*n+row;tokens[i]=pieces+tokens[i];masks[i]=[0]*n+masks[i]
                else:ids[i]=row+[pad]*n;tokens[i]+=pieces;masks[i]+=[0]*n
    return dict(input_ids=ids,attention_mask=masks,tokens=tokens,modelIdentity=info['identity'])


def load_model(folder,device='cpu',maximum_resident_bytes=None,check_cancelled=lambda:None):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['USE_TF']='0'
    info=inspect_model(folder,check_cancelled=check_cancelled)
    if info['task']!='causalLanguageModel':raise ValueError('Speech models use Transcribe audio; causal generation/fine-tuning is unavailable for this task.')
    estimate=info['weightBytes']*4+64_000_000
    if maximum_resident_bytes is not None and estimate>maximum_resident_bytes:raise ValueError('Estimated model loading memory exceeds the captured memory allowance.')
    if device not in ('cpu','mps'):raise ValueError('Choose CPU or MPS for this adapter.')
    import torch
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS is unavailable on this Mac.')
    from transformers import AutoModelForCausalLM
    check_cancelled()
    model,loading=AutoModelForCausalLM.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False,use_safetensors=True,output_loading_info=True,attn_implementation='eager',dtype='auto')
    if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Model weights do not exactly match the supported architecture.')
    check_cancelled();model=model.to(device);model.eval();return model


def export_model(folder,destination,shard_bytes=2_000_000_000,check_cancelled=lambda:None):
    if (Path(folder)/'rl_agent_config.json').exists():
        if __package__:from .laya_models import export_model as export_laya
        else:from laya_models import export_model as export_laya
        return export_laya(folder,destination,check_cancelled)
    info=inspect_model(folder,check_cancelled=check_cancelled)
    if type(shard_bytes) is not int or not 1024<=shard_bytes<=MAX_BYTES:raise ValueError('Invalid model shard byte limit.')
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise ValueError('Export destination already exists; choose a new folder.')
    destination.mkdir(mode=0o700)
    try:
        # Copy safe data; no model construction, dtype conversion or caller code execution.
        weights=[]
        for entry in info['files']:
            check_cancelled();path=entry['path']
            if path.endswith('.safetensors'):weights.append(path);continue
            if path=='model.safetensors.index.json':continue
            shutil.copyfile(Path(folder)/path,destination/path)
        from safetensors import safe_open
        from safetensors.torch import save_file
        shards=[];pending={};size=0;mapping={}
        def flush():
            nonlocal pending,size
            if pending:
                name=f'pending-{len(shards):05}.safetensors';save_file(pending,destination/name,metadata={'format':'pt'})
                shards.append((name,list(pending)));pending={};size=0
        for name in weights:
            with safe_open(Path(folder)/name,framework='pt',device='cpu') as handle:
                for key in handle.keys():
                    check_cancelled();tensor=handle.get_tensor(key);n=tensor.numel()*tensor.element_size()
                    if size and size+n>shard_bytes:flush()
                    pending[key]=tensor;size+=n
        flush()
        for i,(old,keys) in enumerate(shards):
            name='model.safetensors' if len(shards)==1 else f'model-{i+1:05}-of-{len(shards):05}.safetensors'
            (destination/old).rename(destination/name)
            mapping.update({key:name for key in keys})
        if len(shards)>1:(destination/'model.safetensors.index.json').write_text(json.dumps(dict(metadata={'total_size':info['weightBytes']},weight_map=mapping),sort_keys=True))
        verified=inspect_model(destination,check_cancelled=check_cancelled)
        manifest=dict(schemaVersion=1,kind='kwipPretrainedModel',sourceIdentity=info['identity'],files=verified['files'])
        (destination/'kwip-manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2))
        if inspect_model(folder,check_cancelled=check_cancelled)['identity']!=info['identity']:raise ValueError('Source model changed during export.')
        return inspect_model(destination,check_cancelled=check_cancelled)
    except BaseException:
        shutil.rmtree(destination);raise

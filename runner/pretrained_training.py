"""Reviewed offline causal-LM task adapter, sharing optimizer/reporting/control authorities.

Inputs are documents, never pre-shifted labels. Every window stays in one document.
The source model is immutable. Selected inference weights and terminal continuation
state are different artifacts, captured at explicit completed-epoch boundaries.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil

if __package__:
    from . import pretrained_models as models
    from .optimizer_configuration import validate_recipe,create_optimizer
    from .training_control import validate_policy as control_policy,TrainingControl
    from .training_progress import validate_policy as reporting_policy,TrainingProgress
else:
    import pretrained_models as models
    from optimizer_configuration import validate_recipe,create_optimizer
    from training_control import validate_policy as control_policy,TrainingControl
    from training_progress import validate_policy as reporting_policy,TrainingProgress


def training_settings(value):
    keys={'mode','groups','targets','rank','alpha','dropout','bias','additionalModules','textColumn','maximumTokens','stride','appendEOS','epochs','batchSize','accumulation','seed','device','maximumResidentBytes','optimizer','trainingControl','reporting'}
    if type(value) is not dict or set(value)-{'scheduler','capturePredictions','exampleFormat','promptColumn','responseColumn','separator','lossCoverage','activationCheckpointing','checkpointCount','checkpointBytes'}!=keys:raise ValueError('Fine-tuning needs exactly its declared settings.')
    c=dict(value);c.setdefault('scheduler',None);c.setdefault('capturePredictions',False)
    if type(c['capturePredictions']) is not bool:raise ValueError('Prediction capture needs a Boolean.')
    if c['mode'] not in ('full','selected','lora') or c['device'] not in ('auto','cpu','mps'):raise ValueError('Choose a supported fine-tuning mode and Automatic, CPU or MPS device.')
    for key,lo,hi in [('rank',1,1024),('maximumTokens',2,131072),('stride',1,131071),('epochs',1,10000),('batchSize',1,1024),('accumulation',1,1024),('seed',0,2**32-1),('maximumResidentBytes',1,2**63-1)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError('Fine-tuning '+key+' is outside its supported bounds.')
    if c['stride']>=c['maximumTokens']:raise ValueError('Stride must be smaller than the context so windows retain a next-token target.')
    for key,lo,hi in [('alpha',.00001,1_000_000),('dropout',0.,.999999)]:
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or not lo<=c[key]<=hi:raise ValueError('Invalid LoRA '+key+'.')
    for key in ('groups','targets','additionalModules'):
        if type(c[key]) is not list or len(c[key])>512 or any(type(s) is not str or not s or len(s)>256 for s in c[key]) or len(set(c[key]))!=len(c[key]):raise ValueError('Choose unique known '+key+'.')
    if c['bias'] not in ('none','lora_only','all') or type(c['appendEOS']) is not bool or type(c['textColumn']) is not str or not c['textColumn'].strip() or len(c['textColumn'].encode())>200:raise ValueError('Invalid text mapping or LoRA bias policy.')
    for key,default in dict(exampleFormat='text',promptColumn='prompt',responseColumn='response',separator='\n',lossCoverage='response',activationCheckpointing=False,checkpointCount=0,checkpointBytes=1_000_000_000).items():c.setdefault(key,default)
    if c['exampleFormat'] not in ('text','promptResponse') or c['lossCoverage'] not in ('response','all'):raise ValueError('Choose supported text preparation and loss coverage.')
    if any(type(c[k]) is not str or not c[k].strip() or len(c[k].encode())>200 for k in ('promptColumn','responseColumn')) or c['promptColumn']==c['responseColumn']:raise ValueError('Choose distinct prompt and response columns.')
    if type(c['separator']) is not str or len(c['separator'].encode())>256:raise ValueError('Use a bounded explicit prompt separator.')
    if type(c['activationCheckpointing']) is not bool:raise ValueError('Activation checkpointing needs a Boolean.')
    if type(c['checkpointCount']) is not int or not 0<=c['checkpointCount']<=10 or type(c['checkpointBytes']) is not int or not 1<=c['checkpointBytes']<=8_000_000_000:raise ValueError('Invalid checkpoint retention count or storage budget.')
    c['optimizer']=validate_recipe(c['optimizer']);c['trainingControl']=control_policy(c['trainingControl']);c['reporting']=reporting_policy(c['reporting'])
    if c['scheduler'] is not None:
        if __package__:from .scheduler_configuration import validate_compatibility
        else:from scheduler_configuration import validate_compatibility
        c['scheduler'],_=validate_compatibility(c['scheduler'],c['optimizer'])
        if c['scheduler']['componentID']=='scheduler_reduce_on_plateau' and c['scheduler']['config']['metric']!='validation_loss':raise ValueError('This causal task reports Validation loss for plateau scheduling; Validation accuracy is not captured.')
    if c['mode']=='selected' and not c['groups']:raise ValueError('Choose at least one trainable layer group.')
    if c['mode']=='lora' and not c['targets']:raise ValueError('Choose at least one known LoRA target.')
    return c


def _group(name,kind):
    prefix='transformer.h.' if kind=='gpt2' else 'model.layers.'
    if name.startswith(prefix):return 'block:'+name[len(prefix):].split('.')[0]
    if name in ('transformer.wte.weight','model.embed_tokens.weight','lm_head.weight'):return 'embeddings' if name!='lm_head.weight' else 'head'
    if name.startswith('transformer.wpe.'):return 'positions'
    return 'finalNorm'


def _inventory(model,kind):
    # Parameters are grouped by actual ownership. Tied embedding/output aliases
    # belong to one selectable row; selecting either cannot silently untie them.
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
    by_id={id(p):p for p in model.parameters()};groups={}
    for ident,names in aliases.items():
        group=_group(names[0],kind)
        if len({_group(n,kind) for n in names})>1:group='embeddings'
        g=groups.setdefault(group,dict(id=group,title=('Decoder block '+str(int(group.split(':')[1])+1) if group.startswith('block:') else {'embeddings':'Token embeddings / tied output','head':'Output head','positions':'Position embeddings','finalNorm':'Final normalization'}[group]),parameters=[],count=0,tied=False))
        g['parameters']+=names;g['count']+=by_id[ident].numel();g['tied']|=len(names)>1
    targets={}
    categories=({'c_attn':'attentionInput','c_proj':'attentionOutput','c_fc':'feedForward'} if kind=='gpt2' else {'q_proj':'query','k_proj':'key','v_proj':'value','o_proj':'attentionOutput','gate_proj':'feedForward','up_proj':'feedForward','down_proj':'feedForward'})
    for name,module in model.named_modules():
        leaf=name.rsplit('.',1)[-1];category=categories.get(leaf)
        if kind=='gpt2' and leaf=='c_proj' and '.mlp.' in name:category='feedForward'
        if category:targets.setdefault(category,[]).append(name)
    titles={'attentionInput':'Attention · fused Q/K/V','attentionOutput':'Attention · output','query':'Attention · query','key':'Attention · key','value':'Attention · value','feedForward':'Feed-forward projections'}
    return dict(groups=list(groups.values()),targets=[dict(id=k,title=titles[k],modules=v) for k,v in targets.items()],totalParameters=sum(p.numel() for p in model.parameters()))


def training_inventory(folder,check_cancelled=lambda:None,info=None):
    info=info or models.inspect_model(folder,check_cancelled=check_cancelled)
    import torch
    from transformers import AutoModelForCausalLM,AutoConfig
    with torch.device('meta'):model=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False),trust_remote_code=False,attn_implementation='eager')
    result=_inventory(model,info['architecture']);result['supportedModes']=['full','selected']
    import importlib.metadata
    try:
        if importlib.metadata.version('peft')=='0.21.2':result['supportedModes'].append('lora')
    except importlib.metadata.PackageNotFoundError:pass
    result['capabilities']=dict(objectives=['text','promptResponse'],lossCoverage=['all','response'],activationCheckpointing=True,evaluation=['tokenNLL','perplexity'],resume=dict(cpu=['full','selected','lora'],mps='weightsOnly'),exports=['full','adapter','merged','continuation'])
    return result


def _ids(window):return window['ids'] if isinstance(window,dict) else window

def _scored(window):
    return sum(window['score'][1:]) if isinstance(window,dict) else len(window)-1

def _windows(folder,texts,c,check_cancelled):
    c=training_settings(c)
    tokenizer,special=models._tokenizer(folder);tokenizer.no_padding();tokenizer.no_truncation()
    if not isinstance(texts,list) or not texts:raise ValueError('Training and Validation both need nonempty captured text rows.')
    maximum=c['maximumTokens'];windows=[];token_count=0;window_bytes=0
    for text in texts:
        check_cancelled();response=c['exampleFormat']=='promptResponse'
        if response:
            if type(text) is not dict or set(text)!={'prompt','response'} or any(type(v) is not str for v in text.values()) or not text['response'].strip():raise ValueError('Each prompt/response row needs a nonempty response string.')
            prefix=text['prompt']+c['separator'];combined=prefix+text['response']
        else:
            if type(text) is not str:raise ValueError('The selected text column must contain only non-null strings.')
            combined=text
        if len(combined.encode())>8_000_000 or len(combined.encode())*64>c['maximumResidentBytes']//4:raise ValueError('Document tokenization exceeds the captured memory allowance. Split upstream.')
        encoding=tokenizer.encode(combined,add_special_tokens=True);ids=list(encoding.ids);scores=[True]*len(ids)
        if response and c['lossCoverage']=='response':
            scores=[];boundary=len(prefix)
            for start,end in encoding.offsets:
                if end<=boundary or end<=start:scores.append(False)
                elif start>=boundary or combined[start:boundary].isspace():scores.append(True)
                else:raise ValueError('A token crosses the prompt/response boundary. Choose a separator that preserves response token boundaries.')
        if c['appendEOS']:
            if special['eos_token'] is None:raise ValueError('Append EOS needs an EOS token in the matching model tokenizer.')
            if not ids or ids[-1]!=special['eos_token']:ids.append(special['eos_token']);scores.append(True)
        token_count+=len(ids)
        if token_count*64>c['maximumResidentBytes']//4:raise ValueError('Tokenized documents exceed the captured memory allowance.')
        if response and len(ids)>maximum:raise ValueError('Prompt/response exceeds context. Split upstream; the response is never silently truncated.')
        for start in range(0,max(0,len(ids)-1),c['stride']):
            values=ids[start:start+maximum];mask=scores[start:start+maximum]
            if len(values)>1:
                window_bytes+=len(values)*16+64
                if window_bytes>c['maximumResidentBytes']//8:raise ValueError('Overlapping token windows exceed the captured memory allowance.')
                if not any(mask[1:]):raise ValueError('This example has no supervised response tokens after shifting.')
                windows.append(dict(ids=values,score=mask) if response else values)
            if response or start+maximum>=len(ids):break
    if not windows:raise ValueError('Each partition needs at least one document with a next-token target.')
    return windows,special


def _batch(windows,pad,device):
    import torch
    width=max(len(_ids(w))-1 for w in windows)
    if pad is None and any(len(_ids(w))-1!=width for w in windows):raise ValueError('Variable-length batches require the model tokenizer padding token; use batch size 1 otherwise.')
    ids=[];labels=[];masks=[]
    for window in windows:
        values=_ids(window);scores=window['score'] if isinstance(window,dict) else [True]*len(values)
        n=len(values)-1;ids.append(values[:-1]+[pad or 0]*(width-n));labels.append([v if score else -100 for v,score in zip(values[1:],scores[1:])]+[-100]*(width-n));masks.append([1]*n+[0]*(width-n))
    return {k:torch.tensor(v,dtype=torch.long,device=device) for k,v in [('input_ids',ids),('labels',labels),('attention_mask',masks)]}


def prepare_documents(folder,texts,settings,check_cancelled=lambda:None):
    c=training_settings(settings);windows,special=_windows(folder,texts,c,check_cancelled)
    return [_batch(windows[i:i+c['batchSize']],special['pad_token'],c['device']) for i in range(0,len(windows),c['batchSize'])]


def _configure(model,kind,c):
    inventory=_inventory(model,kind)
    group_ids={g['id']:g for g in inventory['groups']};targets={g['id']:g for g in inventory['targets']}
    if c['mode']=='selected' and set(c['groups'])-set(group_ids):raise ValueError('A selected layer group does not exist in this model.')
    if c['mode']=='lora' and set(c['targets'])-set(targets):raise ValueError('A selected LoRA target does not exist in this model.')
    if c['mode']=='selected':
        names={n for key in c['groups'] for n in group_ids[key]['parameters']}
        for name,p in model.named_parameters():p.requires_grad_(name in names)
    elif c['mode']=='lora':
        import importlib.metadata
        if importlib.metadata.version('peft')!='0.21.2':raise ValueError('LoRA requires the reviewed PEFT 0.21.2 runtime.')
        from peft import LoraConfig,TaskType,get_peft_model
        modules=[n for key in c['targets'] for n in targets[key]['modules']]
        additional={n for key in c['additionalModules'] if key in group_ids for n in group_ids[key]['parameters']}
        if set(c['additionalModules'])-set(group_ids):raise ValueError('Additional trainable modules must be known layer groups.')
        # Tied weights and whole-block modules_to_save need a distinct ownership
        # adapter. Expose only actual independent leaf modules here.
        module_names=sorted({n.rsplit('.',1)[0] for n in additional})
        if set(module_names)&set(modules):raise ValueError('Additional trainable modules overlap LoRA targets. Choose independent groups.')
        if any(group_ids[key]['tied'] for key in c['additionalModules']):raise ValueError('Tied embedding/output modules cannot be independently saved as LoRA additional modules.')
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=c['rank'],lora_alpha=c['alpha'],lora_dropout=c['dropout'],bias=c['bias'],target_modules=modules,modules_to_save=module_names or None,fan_in_fan_out=kind=='gpt2'))
    return model,inventory


def _write_json(path,value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>8_000_000:raise ValueError('Training metadata exceeds its byte allowance.')
    with Path(path).open('xb') as f:f.write(raw)


def _copy_metadata(source,destination):
    for p in Path(source).iterdir():
        if models.allowed_file(p.name) and not p.name.endswith('.safetensors') and p.name not in ('config.json','generation_config.json','model.safetensors.index.json','kwip-training.json'):
            shutil.copyfile(p,destination/p.name)


def _pins(folder):
    entries=[]
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file() or p.is_symlink():raise ValueError('Artifact must contain only regular flat files.')
        h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(1_048_576):h.update(chunk)
        entries.append(dict(path=p.name,bytes=p.stat().st_size,sha256=h.hexdigest()))
    return entries


def _save_model(model,folder,source,c):
    folder.mkdir(mode=0o700)
    model.config._name_or_path="kwip-base-sha256:"+models.inspect_model(source)["identity"]
    if hasattr(model,"peft_config"):
        for adapter in model.peft_config.values():adapter.base_model_name_or_path="kwip-base-sha256:"+models.inspect_model(source)["identity"]
    model.save_pretrained(folder,safe_serialization=True,max_shard_size='2GB',**({'save_embedding_layers':False} if hasattr(model,'peft_config') else {}))
    _copy_metadata(source,folder)


def _training_admission(folder,c,info):
    import torch
    cfg=models._json(Path(folder)/'config.json');heads=cfg.get('n_head',cfg.get('num_attention_heads',1));width=cfg.get('n_embd',cfg.get('hidden_size'));layers=cfg.get('n_layer',cfg.get('num_hidden_layers'))
    # Float32 master weights, gradients and Adam slots plus conservative eager
    # attention/logit/activation bounds. Admission precedes token/model loading.
    from transformers import AutoConfig,AutoModelForCausalLM
    with torch.device('meta'):
        skeleton=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False),trust_remote_code=False,attn_implementation='eager')
        planned,_=_configure(skeleton,info['architecture'],c)
    all_parameters=sum(p.numel() for p in planned.parameters());learned_parameters=sum(p.numel() for p in planned.parameters() if p.requires_grad)
    del planned,skeleton
    weights=all_parameters*4;learned=learned_parameters*12;best=weights if c['trainingControl']['modelSelection']=='bestValidation' else 0
    estimate=max(info['weightBytes']*4,weights+learned+best)+64_000_000+c['batchSize']*(c['maximumTokens']*info['vocabularySize']*12+layers*(c['maximumTokens']**2*heads*16+c['maximumTokens']*width*64))
    if estimate>c['maximumResidentBytes']:raise ValueError('Estimated fine-tuning memory exceeds the captured allowance. Reduce context/batch size, or increase the project memory limit.')
    return estimate,all_parameters,learned_parameters


def fine_tune(folder,destination,training,validation,settings,*,emit=lambda _:None,check_cancelled=lambda:None,run_id='fine-tuning',resume_from=None,pause_requested=lambda:False):
    import torch
    import torch.nn.functional as F
    c=training_settings(settings)
    if __package__:from .execution_device import resolve_device
    else:from execution_device import resolve_device
    c['device']=resolve_device(c['device'])
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise ValueError('Training destination already exists.')
    info=models.inspect_model(folder,check_cancelled=check_cancelled)
    if info['parameterCount']*4>models.MAX_BYTES:raise ValueError('Float32 trained weights exceed the current 2 GB safe-model admission. Choose a smaller model for this adapter.')
    if c['maximumTokens']>info['maximumTokens']:raise ValueError('Training context exceeds the loaded model context.')
    estimate,_,_=_training_admission(folder,c,info)
    data_settings={**c,'maximumResidentBytes':c['maximumResidentBytes']-estimate}
    train,pad=_windows(folder,training,data_settings,check_cancelled);valid,_=_windows(folder,validation,data_settings,check_cancelled)
    torch.manual_seed(c['seed'])
    import platform,importlib.metadata
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),transformers=importlib.metadata.version('transformers'),peft=importlib.metadata.version('peft'),cpuThreads=torch.get_num_threads(),interopThreads=torch.get_num_interop_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),matmulPrecision=torch.get_float32_matmul_precision(),machine=platform.machine(),system=platform.release())
    prior=None
    resume_directory=Path(resume_from)/'latest' if resume_from is not None and (Path(resume_from)/'latest').is_dir() else Path(resume_from) if resume_from is not None else None
    if resume_from is not None:
        prior=verify_result(resume_from)
        if prior.get('runtime')!=runtime:raise ValueError('Exact continuation requires the same captured Python/Torch/task runtime.')
        if c['device']!='cpu' or prior['settings']['device']!='cpu':raise ValueError('Exact continuation is supported only at a verified CPU epoch boundary.')
        if prior['baseIdentity']!=info['identity'] or prior['trainingDigest']!=_digest(training) or prior['validationDigest']!=_digest(validation) or {k:v for k,v in training_settings(prior['settings']).items() if k!='epochs'}!={k:v for k,v in c.items() if k!='epochs'}:raise ValueError('Continuation requires identical base, tokenizer, data and training settings.')
        if (prior.get('paused') or prior['settings'].get('checkpointCount',0)>0) and prior['settings']['epochs']!=c['epochs']:raise ValueError('Exact continuation preserves the original scheduler horizon.')
        if c['epochs']<=prior['completedEpoch']:raise ValueError('Continuation target epochs must exceed the completed boundary.')
    model=models.load_model(resume_directory if prior and c['mode']!='lora' else folder,device=c['device'],maximum_resident_bytes=c['maximumResidentBytes'],check_cancelled=check_cancelled).float()
    model,inventory=_configure(model,info['architecture'],c)
    if prior and c['mode']=='lora':
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        from safetensors.torch import load_file
        saved=load_file(resume_directory/'adapter_model.safetensors');expected=get_peft_model_state_dict(model,save_embedding_layers=False)
        if set(saved)!=set(expected) or any(saved[k].shape!=expected[k].shape or not torch.isfinite(saved[k]).all().item() for k in saved):raise ValueError('Continuation adapter differs from actual parameter inventory.')
        set_peft_model_state_dict(model,saved)
        actual=get_peft_model_state_dict(model,save_embedding_layers=False)
        if any(not torch.equal(saved[k].float(),actual[k].detach().cpu().float()) for k in saved):raise ValueError('Continuation adapter did not restore exactly.')
    if c['activationCheckpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    total_parameters=sum(p.numel() for p in model.parameters())
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=create_optimizer(parameters,c['optimizer'])
    schedule=None
    if c['scheduler'] is not None:
        if __package__:from .scheduler_configuration import SchedulerController,preflight
        else:from scheduler_configuration import SchedulerController,preflight
        preflight(c['scheduler'],c['optimizer'],c['epochs'],math.ceil(len(train)/(c['batchSize']*c['accumulation'])),len(parameters))
        schedule=SchedulerController(optimizer,c['scheduler'],c['optimizer'],len(parameters))
    snapshot=lambda:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    control=TrainingControl(c['trainingControl'],capture=snapshot)
    progress=TrainingProgress(run_id,emit,c['reporting'])
    progress.epoch_batches=math.ceil(len(train)/c['batchSize'])
    updates=0;measurements=[];initial_epoch=0
    if prior:
        restored=_load_safe_state(resume_directory,optimizer,c,schedule)
        control=TrainingControl(c['trainingControl'],capture=snapshot,state=restored['control'],best_model=restored['best'])
        torch.set_rng_state(restored['rng']);initial_epoch=restored['epoch'];updates=restored['updates'];measurements=list(prior['metrics'])
    progress.batches=initial_epoch*progress.epoch_batches;progress.updates=updates
    destination.mkdir(mode=0o700)
    paused=False
    def report_now():return {**control.report(),'schemaVersion':2,'kind':'kwipFineTuning','baseIdentity':info['identity'],'settings':c,'updates':updates,'trainableParameters':sum(p.numel() for p in parameters),'totalParameters':total_parameters,'metrics':list(measurements),'continuation':'cpuEpochBoundary' if c['device']=='cpu' else 'weightsOnly','trainingDigest':_digest(training),'validationDigest':_digest(validation),'runtime':runtime,'paused':paused,'initialEpoch':initial_epoch}
    try:
        for epoch in range(initial_epoch+1,c['epochs']+1):
            check_cancelled();model.train();optimizer.zero_grad(set_to_none=True)
            if schedule:schedule.begin_epoch()
            # Epoch-specific local shuffle keeps continuation independent of
            # validation/reporting RNG consumption and window materialization.
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(c['seed']+epoch)).tolist()
            numerator=0.;denominator=0;pending=0
            for offset in range(0,len(order),c['batchSize']*c['accumulation']):
                group=order[offset:offset+c['batchSize']*c['accumulation']]
                group_tokens=sum(_scored(train[i]) for i in group)
                for start in range(0,len(group),c['batchSize']):
                    check_cancelled();batch=_batch([train[i] for i in group[start:start+c['batchSize']]],pad['pad_token'],c['device'])
                    labels=batch.pop('labels');logits=model(**batch,use_cache=False).logits
                    # Labels are explicitly shifted once above. Calling the HF
                    # labels= loss here would shift a second time.
                    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')
                    count=int((labels!=-100).sum().item());value=float(loss.detach().item())
                    if not math.isfinite(value):raise ValueError('Training loss is nonfinite; no artifact was published.')
                    (loss/group_tokens).backward();numerator+=value;denominator+=count;pending+=1
                    updated=start+c['batchSize']>=len(group)
                    if updated:
                        if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in parameters):raise ValueError('Training gradients are nonfinite.')
                        if schedule:schedule.before_update(list(range(len(parameters))))
                        optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                        if schedule and schedule.is_batch:schedule.tick(epoch,batch=updates,updated_indices=list(range(len(parameters))))
                    progress.batch(epoch=epoch,epoch_batch=pending,epoch_batches=progress.epoch_batches,numerator=value,denominator=count,updated=updated)
            model.eval();vsum=0.;vcount=0
            with torch.no_grad():
                for offset in range(0,len(valid),c['batchSize']):
                    check_cancelled();batch=_batch(valid[offset:offset+c['batchSize']],pad['pad_token'],c['device']);labels=batch.pop('labels');logits=model(**batch,use_cache=False).logits
                    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')
                    vsum+=float(loss.item());vcount+=int((labels!=-100).sum().item())
            vloss=vsum/vcount
            if not math.isfinite(vloss):raise ValueError('Validation loss is nonfinite.')
            if schedule and not schedule.is_batch:schedule.tick(epoch,metric_value=vloss)
            control.observe(epoch,vloss);progress.validation(epoch,numerator/denominator,vloss,None)
            measurements.append(dict(epoch=epoch,trainingLoss=numerator/denominator,validationLoss=vloss))
            paused=bool(pause_requested()) and epoch<c['epochs'] and control.stop_epoch is None
            if c['checkpointCount']:
                _commit_epoch(model,destination,folder,c,optimizer,control,epoch,updates,info,schedule,report_now(),check_cancelled)
            if paused or control.stop_epoch is not None:break
        report=report_now()
        # Capture terminal state BEFORE selecting inference weights. Continuation
        # is a separate boundary, never best weights paired with final moments.
        _save_model(model,destination/'latest',folder,c)
        _save_safe_state(destination/'latest',optimizer.state_dict(),torch.get_rng_state(),control.state(),epoch,updates,c,info,control.best_model,schedule.checkpoint() if schedule else None)
        if control.best_model is not None and c['trainingControl']['modelSelection']=='bestValidation':model.load_state_dict(control.best_model,strict=True)
        if c['mode']=='lora':
            _save_model(model,destination/'adapter',folder,c)
            _write_json(destination/'adapter/kwip-training.json',report)
            _write_json(destination/'adapter/kwip-adapter.json',dict(schemaVersion=1,kind='kwipAdapter',baseIdentity=info['identity'],baseFiles=info['files'],files=_pins(destination/'adapter')))
            # Retain the exact base for offline use; export adapter-only is an
            # explicit separate action, never a misleading standalone model.
            models.export_model(folder,destination/'base',check_cancelled=check_cancelled)
            merged=model.merge_and_unload(safe_merge=True)
            _save_model(merged,destination/'model',folder,c)
        else:_save_model(model,destination/'model',folder,c)
        _write_json(destination/'model/kwip-training.json',report)
        _write_json(destination/'latest/training.json',report)
        _write_json(destination/'latest/result-manifest.json',dict(schemaVersion=1,files=_tree_pins(destination/'latest')))
        _write_json(destination/'training.json',report)
        _write_json(destination/'result-manifest.json',dict(schemaVersion=1,files=_tree_pins(destination)))
        if models.inspect_model(folder,check_cancelled=check_cancelled)['identity']!=info['identity']:raise ValueError('Source model changed during training.')
        check_cancelled();return report
    except BaseException:
        # Keep only atomically renamed epochs; never expose a partial inference result.
        if c["checkpointCount"] and (destination/"checkpoints").exists():
            for path in list(destination.iterdir()):
                if path.name!="checkpoints":
                    if path.is_dir():shutil.rmtree(path)
                    else:path.unlink()
            for path in list((destination/"checkpoints").iterdir()):
                if path.name.startswith("."):shutil.rmtree(path)
        else:shutil.rmtree(destination)
        raise


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _save_safe_state(folder,optimizer,rng,control,epoch,updates,settings,info,best,scheduler,shard_bytes=1_000_000_000):
    from safetensors.torch import save_file
    tensors={}
    def encode(value):
        import torch
        if isinstance(value,torch.Tensor):
            name='tensor'+str(len(tensors));tensors[name]=value.detach().cpu().contiguous();return {'tensor':name}
        if type(value) in (list,tuple):return {'sequence':[encode(v) for v in value],'tuple':type(value) is tuple}
        if type(value) is dict:return {'mapping':[[encode(k),encode(v)] for k,v in value.items()]}
        if type(value) is float and math.isinf(value):return {'infinity':1 if value>0 else -1}
        if value is None or type(value) in (str,int,float,bool):return value
        raise ValueError('Unreviewed checkpoint state value.')
    state=encode(dict(optimizer=optimizer,rng=rng,best=best,scheduler=scheduler))
    chunks=[];pending={};size=0
    for key,tensor in tensors.items():
        count=tensor.numel()*tensor.element_size()
        if size and size+count>shard_bytes:chunks.append(pending);pending={};size=0
        pending[key]=tensor;size+=count
    if pending:chunks.append(pending)
    names=[]
    for i,chunk in enumerate(chunks):
        name='training-state.safetensors' if len(chunks)==1 else f'training-state-{i+1:05}-of-{len(chunks):05}.safetensors'
        save_file(chunk,folder/name);names.append(name)
    _write_json(folder/'kwip-continuation.json',dict(schemaVersion=1,kind='kwipFineTuningContinuation',device=settings['device'],epoch=epoch,updates=updates,control=control,settings=settings,baseIdentity=info['identity'],stateFiles=names,state=state))


def load_adapter(adapter,base,device='cpu',check_cancelled=lambda:None,maximum_resident_bytes=2_000_000_000):
    adapter=Path(adapter);info=models.inspect_model(base,check_cancelled=check_cancelled)
    pin=models._json(adapter/'kwip-adapter.json')
    files=[p for p in _pins(adapter) if p['path']!='kwip-adapter.json']
    if pin.get('schemaVersion')!=1 or pin.get('kind')!='kwipAdapter' or pin.get('baseIdentity')!=info['identity'] or pin.get('baseFiles')!=info['files'] or pin.get('files')!=files:raise ValueError('Adapter files/base identity changed from saved provenance. Choose its exact matching base.')
    config=models._json(adapter/'adapter_config.json')
    inventory=training_inventory(base,check_cancelled=check_cancelled,info=info)
    known={name for group in inventory['targets'] for name in group['modules']}
    targets=config.get('target_modules')
    if config.get('peft_type')!='LORA' or config.get('task_type')!='CAUSAL_LM' or config.get('auto_mapping') or not isinstance(targets,list) or not targets or set(targets)-known:raise ValueError('Adapter needs reviewed causal LoRA targets from its exact base inventory.')
    if config.get('use_dora') or config.get('use_qalora') or config.get('alora_invocation_tokens') or config.get('layer_replication'):raise ValueError('This adapter variant needs a separate reviewed capability.')
    from peft import PeftModel,LoraConfig,TaskType,get_peft_model_state_dict
    if type(config.get('r')) is not int or not 1<=config['r']<=1024 or type(config.get('lora_alpha')) not in (int,float) or not .00001<=config['lora_alpha']<=1_000_000 or type(config.get('lora_dropout')) not in (int,float) or not 0<=config['lora_dropout']<1 or config.get('bias') not in ('none','all','lora_only'):raise ValueError('Unsupported LoRA settings.')
    defaults=LoraConfig(task_type=TaskType.CAUSAL_LM).to_dict()
    variable={'r','lora_alpha','lora_dropout','bias','target_modules','modules_to_save','fan_in_fan_out','inference_mode','base_model_name_or_path','peft_version'}
    if set(config)-set(defaults) or any(config.get(key)!=value for key,value in defaults.items() if key not in variable):raise ValueError('Adapter contains options outside the reviewed LoRA configuration.')
    independent={n.rsplit('.',1)[0] for g in inventory['groups'] if not g['tied'] for n in g['parameters']}
    additional=config.get('modules_to_save')
    if additional is not None and (type(additional) is not list or set(additional)-independent):raise ValueError('Additional adapter modules differ from known independent base layers.')
    if type(config.get('fan_in_fan_out')) is not bool or config['fan_in_fan_out']!=(info['architecture']=='gpt2'):raise ValueError('Adapter projection layout differs from the verified architecture.')
    adapter_bytes=sum(f['bytes'] for f in files)
    if info['parameterCount']*16+adapter_bytes*4+64_000_000>maximum_resident_bytes:raise ValueError('Adapter/base memory exceeds the captured allowance.')
    model=models.load_model(base,device=device,check_cancelled=check_cancelled).float()
    model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=False)
    from safetensors.torch import load_file
    import torch
    saved=load_file(adapter/'adapter_model.safetensors');actual=get_peft_model_state_dict(model,save_embedding_layers=False)
    if set(saved)!=set(actual) or any(saved[k].shape!=actual[k].shape or not torch.isfinite(saved[k]).all().item() or not torch.equal(saved[k].float(),actual[k].detach().cpu().float()) for k in saved):raise ValueError('Adapter safe weights do not exactly match the declared module inventory.')
    model.eval();return model


def export_fine_tuned(result,destination,kind='full',shard_bytes=2_000_000_000,check_cancelled=lambda:None):
    result=Path(result);destination=Path(destination)
    report=verify_result(result)
    if report.get('kind')!='kwipFineTuning':raise ValueError('Choose a verified fine-tuning result.')
    if destination.exists() or destination.is_symlink():raise ValueError('Export destination already exists.')
    lora=report['settings']['mode']=='lora'
    if kind=='full' and not lora:return models.export_model(result/'model',destination,shard_bytes,check_cancelled)
    if kind not in ('adapter','merged','continuation') or (kind in ('adapter','merged') and not lora):raise ValueError('This result does not support that save content.')
    try:
        if kind=='adapter':
            load_adapter(result/'adapter',result/'base',check_cancelled=check_cancelled)
            destination.mkdir(mode=0o700)
            for entry in _pins(result/'adapter'):
                check_cancelled();shutil.copyfile(result/'adapter'/entry['path'],destination/entry['path'])
            return dict(kind='adapter',baseIdentity=report['baseIdentity'],files=_pins(destination))
        if kind=='continuation':
            if report.get('continuation')!='cpuEpochBoundary':raise ValueError('This device result supports weights reuse, not exact CPU continuation.')
            destination.mkdir(mode=0o700)
            if not lora:
                for entry in _pins(result/'latest'):shutil.copyfile(result/'latest'/entry['path'],destination/entry['path'])
                return dict(kind='continuation',files=_pins(destination))
            groups=['latest','base']
            for group in groups:
                shutil.copytree(result/group,destination/group)
            _write_json(destination/'training.json',report)
            _write_json(destination/'result-manifest.json',dict(schemaVersion=1,files=_tree_pins(destination)))
            return dict(kind='continuation',files=_tree_pins(destination))
        model=load_adapter(result/'adapter',result/'base',check_cancelled=check_cancelled)
        merged=model.merge_and_unload(safe_merge=True);destination.mkdir(mode=0o700)
        merged.save_pretrained(destination,safe_serialization=True,max_shard_size=shard_bytes)
        _copy_metadata(result/'base',destination)
        verified=models.inspect_model(destination,check_cancelled=check_cancelled)
        _write_json(destination/'kwip-manifest.json',dict(schemaVersion=1,kind='kwipPretrainedModel',sourceIdentity=report['baseIdentity'],files=verified['files']))
        return models.inspect_model(destination,check_cancelled=check_cancelled)
    except BaseException:
        if destination.exists():shutil.rmtree(destination)
        raise


def _tree_pins(folder):
    files=[]
    for p in sorted(Path(folder).rglob('*')):
        if p.is_symlink():raise ValueError('Training artifact contains a symbolic link.')
        if p.is_dir():continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(1_048_576):h.update(chunk)
        files.append(dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    return files


def verify_result(folder):
    folder=Path(folder);pin=models._json(folder/'result-manifest.json')
    current=[f for f in _tree_pins(folder) if f['path']!='result-manifest.json']
    if pin.get('schemaVersion')!=1 or pin.get('files')!=current:raise ValueError('Fine-tuning result changed from its verified inventory.')
    return models._json(folder/'training.json')


def _load_safe_state(folder,optimizer,settings,schedule):
    import torch
    from safetensors.torch import load_file
    if __package__:from .optimizer_configuration import restore_optimizer
    else:from optimizer_configuration import restore_optimizer
    meta=models._json(folder/'kwip-continuation.json');tensors={}
    names=meta.get('stateFiles')
    import re
    if type(names) is not list or not 1<=len(names)<=32 or len(set(names))!=len(names) or any(type(n) is not str or re.fullmatch(r'training-state(?:-\d{5}-of-\d{5})?\.safetensors',n) is None for n in names):raise ValueError('Invalid continuation tensor shard inventory.')
    for name in names:
        values=load_file(folder/name)
        if set(values)&set(tensors):raise ValueError('Duplicate tensor in continuation shards.')
        tensors.update(values)
    if meta.get('schemaVersion')!=1 or meta.get('device')!='cpu':raise ValueError('Unsupported continuation state.')
    used=set()
    def decode(value,depth=0):
        if depth>32:raise ValueError('Checkpoint state is too deeply nested.')
        if type(value) is dict:
            if set(value)=={'infinity'} and type(value['infinity']) is int and value['infinity'] in (-1,1):return math.inf*value['infinity']
            if set(value)=={'tensor'}:
                name=value['tensor']
                if name not in tensors or name in used:raise ValueError('Checkpoint tensor identity is missing or duplicated.')
                used.add(name);return tensors[name]
            if set(value)=={'sequence','tuple'} and type(value['sequence']) is list and type(value['tuple']) is bool:
                seq=[decode(v,depth+1) for v in value['sequence']];return tuple(seq) if value['tuple'] else seq
            if set(value)=={'mapping'} and type(value['mapping']) is list:
                out={}
                for pair in value['mapping']:
                    if type(pair) is not list or len(pair)!=2:raise ValueError('Invalid checkpoint mapping.')
                    key=decode(pair[0],depth+1)
                    if type(key) not in (str,int) or key in out:raise ValueError('Invalid checkpoint key.')
                    out[key]=decode(pair[1],depth+1)
                return out
            raise ValueError('Unknown checkpoint state encoding.')
        if value is None or type(value) in (str,int,float,bool):return value
        raise ValueError('Invalid checkpoint scalar.')
    state=decode(meta['state'])
    if used!=set(tensors) or set(state)!={'optimizer','rng','best','scheduler'}:raise ValueError('Checkpoint state inventory differs.')
    params=[p for g in optimizer.param_groups for p in g['params']]
    if schedule:
        schedule.restore(state['scheduler'],[meta['updates']]*len(params))
    elif state['scheduler'] is not None:raise ValueError('Checkpoint has a different schedule.')
    restore_optimizer(optimizer,state['optimizer'],settings['optimizer'],[meta['updates']]*len(params),schedule.context() if schedule else None)
    rng=state['rng']
    if not isinstance(rng,torch.Tensor) or rng.dtype!=torch.uint8 or rng.shape!=torch.get_rng_state().shape:raise ValueError('Invalid CPU RNG state.')
    return dict(rng=rng,best=state['best'],control=meta['control'],epoch=meta['epoch'],updates=meta['updates'])


def prediction_rows(folder,texts,settings,check_cancelled=lambda:None):
    import torch
    import torch.nn.functional as F
    c=training_settings(settings)
    if __package__:from .execution_device import resolve_device
    else:from execution_device import resolve_device
    c['device']=resolve_device(c['device'])
    info=models.inspect_model(folder,check_cancelled=check_cancelled)
    if c['maximumTokens']>info['maximumTokens']:raise ValueError('Scoring context exceeds the loaded model context.')
    estimate,_,_=_training_admission(folder,c,info)
    windows,special=_windows(folder,texts,{**c,'maximumResidentBytes':c['maximumResidentBytes']-estimate},check_cancelled)
    if sum(len(_ids(w)) for w in windows)*72>7_000_000:raise ValueError('Captured token prediction table exceeds the 8 MB result budget. Reduce the evaluated population or context.')
    model=models.load_model(folder,device=c['device'],maximum_resident_bytes=c['maximumResidentBytes'],check_cancelled=check_cancelled)
    rows=[]
    with torch.no_grad():
        for start in range(0,len(windows),c['batchSize']):
            check_cancelled();batch=_batch(windows[start:start+c['batchSize']],special['pad_token'],c['device']);labels=batch.pop('labels');logits=model(**batch,use_cache=False).logits
            losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='none').reshape(labels.shape);predicted=logits.argmax(-1)
            for i in range(labels.shape[0]):
                n=int(batch['attention_mask'][i].sum().item());count=int((labels[i,:n]!=-100).sum().item());loss=float(losses[i,:n].sum().item())/count
                rows.append([start+i,batch['input_ids'][i,:n].cpu().tolist(),labels[i,:n].cpu().tolist(),predicted[i,:n].cpu().tolist(),count,loss,math.exp(loss) if loss<700 else None])
    return rows


def evaluate(folder,texts,settings,check_cancelled=lambda:None):
    rows=prediction_rows(folder,texts,settings,check_cancelled)
    tokens=sum(row[4] for row in rows);loss=sum(row[4]*row[5] for row in rows)/tokens
    return dict(rows=rows,scoredTokens=tokens,loss=loss,perplexity=math.exp(loss) if loss<700 else None)


def _commit_epoch(model,destination,source,c,optimizer,control,epoch,updates,info,schedule,report,check_cancelled):
    import torch
    root=destination/'checkpoints';root.mkdir(mode=0o700,exist_ok=True)
    # Reserve conservatively before allocating a snapshot. Native storage admission
    # separately covers the complete operation, immutable base and final exports.
    estimate=sum(v.numel()*v.element_size() for v in model.state_dict().values())+32_000_000
    estimate+=sum(p.stat().st_size for p in Path(source).iterdir() if p.is_file() and models.allowed_file(p.name) and not p.name.endswith(".safetensors"))
    estimate+=sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))
    if control.best_model:estimate+=sum(v.numel()*v.element_size() for v in control.best_model.values())
    retained=sorted(root.iterdir())
    # Never remove the newest committed boundary before its successor exists.
    while len(retained)>=max(2,c['checkpointCount']):
        shutil.rmtree(retained.pop(0))
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used+estimate>c['checkpointBytes']:raise ValueError('Completed-epoch checkpoint storage budget would be exceeded.')
    check_cancelled();target=root/f'epoch-{epoch:06}';staging=root/f'.epoch-{epoch:06}'
    _save_model(model,staging,source,c)
    _save_safe_state(staging,optimizer.state_dict(),torch.get_rng_state(),control.state(),epoch,updates,c,info,control.best_model,schedule.checkpoint() if schedule else None)
    _write_json(staging/'training.json',report)
    _write_json(staging/'result-manifest.json',dict(schemaVersion=1,files=_tree_pins(staging)))
    import os
    for path in staging.iterdir():
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
    fd=os.open(staging,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    staging.rename(target)
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    retained=sorted(p for p in root.iterdir() if not p.name.startswith('.'))
    while len(retained)>c['checkpointCount']:shutil.rmtree(retained.pop(0))


def prepared_preview(folder,training,validation,settings,check_cancelled=lambda:None):
    """Bounded explicit inspection. Full scored populations, only one displayed sample."""
    import torch
    import torch.nn.functional as F
    if __package__:from .execution_device import resolve_device
    else:from execution_device import resolve_device
    c=training_settings(settings);c['device']=resolve_device(c['device'])
    info=models.inspect_model(folder,check_cancelled=check_cancelled)
    if c['maximumTokens']>info['maximumTokens']:raise ValueError('Training context exceeds the loaded model context.')
    estimate,total,learned=_training_admission(folder,c,info)
    data_settings={**c,'maximumResidentBytes':c['maximumResidentBytes']-estimate}
    train,special=_windows(folder,training,data_settings,check_cancelled);valid,_=_windows(folder,validation,data_settings,check_cancelled)
    inventory=training_inventory(folder,check_cancelled,info)
    check_cancelled();torch.manual_seed(c['seed'])
    model=models.load_model(folder,device=c['device'],maximum_resident_bytes=c['maximumResidentBytes'],check_cancelled=check_cancelled).float()
    model,_=_configure(model,info['architecture'],c)
    if c['activationCheckpointing']:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    model.train();batch=_batch(train[:1],special['pad_token'],c['device']);labels=batch.pop('labels');logits=model(**batch,use_cache=False).logits
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
    if not torch.isfinite(loss).item():raise ValueError('Prepared sample produced a non-finite loss.')
    loss.backward();check_cancelled()
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):raise ValueError('Prepared sample did not produce finite trainable gradients.')
    del model,logits,loss,batch,gradients
    tokenizer,_=models._tokenizer(folder);first=train[0];ids=_ids(first)
    return dict(kind='fineTuningPreparation',settings=c,trainingWindows=len(train),validationWindows=len(valid),trainingScoredTokens=sum(map(_scored,train)),validationScoredTokens=sum(map(_scored,valid)),sampleIDs=ids[:512],sampleTokens=[tokenizer.id_to_token(i) for i in ids[:512]],sampleScored=(first['score'] if isinstance(first,dict) else [True]*len(ids))[:512],sampleCoverage='First window, at most 512 tokens',inventory=inventory,estimatedResidentBytes=estimate,totalParameters=total,trainableParameters=learned,gradientCheck='One training window; no optimizer update')

"""Version 1 bounded next-token inference; shared by the worker and portable replay.

No source reading, fitting, training, attention-mask invention or network authority.
Stable ID ties, temperature → top-k → nucleus, CPU multinomial RNG. Per-row seed is
(seed + row_index) modulo 2**63; results are published only after every row succeeds.
"""
import math
import time
import torch

DEFAULTS=dict(method='greedy',temperature=1.,topK=0,topP=1.,seed=0,maxNewTokens=64,
              maxPrompts=128,maxTotalTokens=8192,longPrompt='refuse',allowUnknown=False,
              stopAtEOS=True,device='cpu')
MAX_PROMPT_BYTES=1048576
MAX_REQUEST_BYTES=4194304
MAX_RESULT_BYTES=4194304


def settings(value):
    if type(value) is not dict or set(value)-set(DEFAULTS):raise ValueError('Unknown generation setting.')
    c={**DEFAULTS,**value}
    for key,lo,hi in [('topK',0,512),('seed',0,2**63-1),('maxNewTokens',1,1024),('maxPrompts',1,128),('maxTotalTokens',1,8192)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError('Generation '+key+' exceeds its integer limits.')
    for key,lo,hi in [('temperature',0,100),('topP',0,1)]:
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or not lo<c[key]<=hi:raise ValueError('Generation '+key+' must be finite and within range.')
    if c['method'] not in ('greedy','sample') or c['longPrompt'] not in ('refuse','latest') or c['device'] not in ('cpu','mps'):raise ValueError('Unsupported generation method, context policy or device.')
    if any(type(c[k]) is not bool for k in ('allowUnknown','stopAtEOS')):raise ValueError('Generation flags must be Boolean.')
    return c


def filtered_probabilities(logits,excluded,temperature,top_k,top_p):
    scores=logits.detach().cpu().to(dtype=torch.float64).flatten()
    if not torch.isfinite(scores).all().item():raise ValueError('Model logits must be finite.')
    ids=[i for i in range(len(scores)) if i not in excluded]
    ids.sort(key=lambda i:(-scores[i].item(),i))
    if top_k:ids=ids[:top_k]
    if not ids:raise ValueError('No admissible generation tokens; revise special-token policy.')
    admitted=(scores[ids]-scores[ids[0]])/temperature
    probabilities=torch.softmax(admitted,dim=0)
    # Include the token that crosses p; stable score/ID order determines equal ties.
    keep=(probabilities.cumsum(0)-probabilities)<top_p
    probabilities=probabilities*keep;probabilities/=probabilities.sum()
    if not torch.isfinite(probabilities).all().item() or probabilities.sum().item()<=0:raise ValueError("No finite admissible sampling distribution.")
    result=torch.zeros_like(scores);result[ids]=probabilities
    return result


def choose(logits,excluded,c,rng):
    if c['method']=='greedy':
        scores=logits.detach().cpu().flatten()
        if not torch.isfinite(scores).all().item():raise ValueError('Model logits must be finite.')
        ids=[i for i in range(len(scores)) if i not in excluded]
        if not ids:raise ValueError('No admissible generation tokens.')
        return min(ids,key=lambda i:(-scores[i].item(),i))
    probabilities=filtered_probabilities(logits,excluded,c['temperature'],c['topK'],c['topP'])
    return int(torch.multinomial(probabilities,1,generator=rng).item())


def admit(predictor):
    p=getattr(predictor,'plan',None)
    if type(p) is not dict or p.get('schemaVersion')!=2 or len(p.get('inputs',[]))!=1 or type(p.get('sequenceObjective')) is not dict:raise ValueError('Generate text needs a captured next-token sequence model with one named input.')
    e=p['inputs'][0];name=e['name'];shape=e['sampleShape'];meta=predictor.tokenizers.get(name)
    if e.get('dtype')!='int64' or e.get('representation')!='tokens' or len(shape)!=1 or type(shape[0]) is not int or not 1<=shape[0]<=4096 or p['sequenceObjective'].get('inputName')!=name:raise ValueError('Generation requires one fixed int64 context input.')
    if type(meta) is not dict or not 2<=len(meta.get('vocabulary',[]))<=512 or len(meta['vocabulary'])!=e['vocabularySize'] or tuple(predictor.output_shape)!=(shape[0],len(meta['vocabulary'])):raise ValueError('Captured tokenizer/context/logit vocabulary do not agree.')
    predictor.encode_tokenizer('',meta) # validates exact tokenizer authority
    v=meta['vocabulary'];tc=meta['configuration'];pad=e.get('paddingValue',e.get('paddingID'))
    if tc['pad'] and pad!=v.index('<pad>'):raise ValueError('Captured padding ID differs from the tokenizer.')
    if type(pad) is not int or not 0<=pad<len(v):raise ValueError('A bounded context requires a captured valid padding ID.')
    return name,shape[0],meta,pad


def generate(predictor,prompts,configuration,*,cancelled=lambda:False,deadline=None,progress=lambda completed,total:None):
    c=settings(configuration);name,context,meta,pad=admit(predictor);v=meta['vocabulary'];tc=meta['configuration']
    if type(prompts) is not list or not 1<=len(prompts)<=c['maxPrompts'] or len(prompts)*c['maxNewTokens']>c['maxTotalTokens']:raise ValueError('Prompt count × new-token limit exceeds the request budget. Use upstream sampling or smaller limits.')
    if any(type(s) is not str or len(s.encode('utf-8'))>MAX_PROMPT_BYTES for s in prompts) or sum(len(s.encode('utf-8')) for s in prompts)>MAX_REQUEST_BYTES:raise ValueError('Prompt bytes exceed the bounded generation allowance.')
    encoded=[]
    for text in prompts:
        ids=predictor.encode_tokenizer(text,meta)
        if tc['eos']:ids=ids[:-1] # only the synthetic training terminator
        if not ids:raise ValueError('Prompt has no tokens; enter text or use a captured beginning-of-sequence token.')
        dropped=max(0,len(ids)-context)
        if dropped and c['longPrompt']=='refuse':raise ValueError('Prompt exceeds the captured context. Choose Keep latest context explicitly.')
        encoded.append((ids[-context:],dropped))
    excluded={v.index(t) for t,enabled in [('<pad>',tc['pad']),('<bos>',tc['bos']),('<unk>',tc['unknownPolicy']=='unk' and not c['allowUnknown'])] if enabled}
    eos=v.index('<eos>') if tc['eos'] else None
    device=torch.device(c['device'])
    if device.type=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS is unavailable; choose CPU.')
    model=predictor.model
    # Forward can mutate buffers or parameters even in eval (e.g. max_norm).
    # Reset exact working state for every prompt and restore on all exit paths.
    original_device=next(iter(model.parameters()),next(iter(model.buffers()),torch.empty(0))).device
    initial={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
    model.to(device);model.eval();rows=[];evidence=[];result_bytes=0
    deadline=min(deadline if deadline is not None else math.inf,time.monotonic()+300.)
    def check():
        if cancelled():raise InterruptedError('Generation cancelled; no completed output published.')
        if time.monotonic()>=deadline:raise TimeoutError('Generation deadline exceeded; no completed output published.')
    try:
        with torch.inference_mode():
            for index,(tokens,dropped) in enumerate(encoded):
                check();model.load_state_dict(initial,strict=True)
                rng=torch.Generator(device='cpu').manual_seed((c['seed']+index)%(2**63))
                generated=[];stop='token_limit';row_bytes=0
                for _ in range(c['maxNewTokens']):
                    check();window=tokens[-context:];length=len(window)
                    x=torch.tensor([window+[pad]*(context-length)],dtype=torch.int64,device=device)
                    logits=model({name:x} if predictor.named_values else x)
                    if not isinstance(logits,torch.Tensor) or logits.dtype!=torch.float32 or tuple(logits.shape)!=(1,context,len(v)):raise ValueError('Model must return float32 [1, context, vocabulary] logits.')
                    token=choose(logits[0,length-1],excluded,c,rng);check()
                    row_bytes+=len(v[token].encode('utf-8'))+8
                    if result_bytes+row_bytes>MAX_RESULT_BYTES:raise ValueError('Generated output exceeds its byte allowance; reduce limits.')
                    generated.append(token);tokens=(window+[token])[-context:]
                    if token==eos and c['stopAtEOS']:stop='eos';break
                text=''.join(v[t] for t in generated if t!=eos)
                result_bytes+=len(text.encode('utf-8'))+8*len(generated)
                if result_bytes>MAX_RESULT_BYTES:raise ValueError('Generated output exceeds its byte allowance; reduce limits.')
                rows.append(dict(generated_text=text,generated_ids=generated,new_tokens=len(generated),stop_reason=stop))
                evidence.append(dict(droppedPromptTokens=dropped,seed=(c['seed']+index)%(2**63)))
                progress(index+1,len(prompts))
        check()
        return dict(schemaVersion=1,rows=rows,evidence=evidence,vocabulary=v,settings=c,algorithm='temperature-topk-nucleus-cpu-multinomial-v1',contextLength=context,specialTokens=dict(unknown=tc['unknownPolicy']=='unk',bos=tc['bos'],pad=tc['pad'],eos=tc['eos']))
    finally:
        model.to(original_device);model.load_state_dict(initial,strict=True)

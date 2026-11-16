"""Actual graph/optimizer MPS qualification. No silent per-operator fallback."""
import copy,io,os,time,platform,threading
import torch
if __package__:
 from .self_play_model import PolicyValueModel,joint_loss
 from .game_episodes import require,digest
else:
 from self_play_model import PolicyValueModel,joint_loss
 from game_episodes import require,digest
_cache={};_lock=threading.Lock()

def qualify_devices(graph,c,initialization=None,trainability=None):
 requested={'learning':c.device,'inference':c.compute.get('inferenceDevice','cpu')}
 if all(v!='mps' for v in requested.values()):return dict(learning='cpu',inference='cpu',reproducibility='CPU reference' if c.compute.get('activeGames',1)==1 else 'Batched CPU recovery · numerical ordering; not bitwise',reason='Explicit CPU' if all(v=='cpu' for v in requested.values()) else 'Automatic uses CPU pending matched end-to-end GPU speed qualification',qualified=True)
 if not torch.backends.mps.is_available():
  require(all(v!='mps' for v in requested.values()),'Apple GPU unavailable; explicitly choose CPU.')
  return dict(learning='cpu',inference='cpu',reproducibility='CPU reference',reason='Apple GPU unavailable',qualified=True)
 require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1','Disable silent MPS CPU fallback before qualifying this profile.')
 key=digest(['varied-probes-v2',graph,c.initializationSeed,c.optimizer,c.learningRate,c.weightDecay,c.momentum,c.beta1,c.beta2,c.epsilon,c.policyWeight,c.valueWeight,c.gradientClip,c.batch,c.compute.get('batchSize',16),str(torch.__version__),platform.mac_ver()[0],initialization,trainability])
 with _lock:
  if key not in _cache:
   cpu_rng=torch.get_rng_state();gpu_rng=torch.mps.get_rng_state()
   try:
    cpu=PolicyValueModel(graph,seed=c.initializationSeed,initialization=initialization,trainability=trainability)
    gpu=PolicyValueModel(graph,state=cpu.model.state_dict(),device='mps',trainability=trainability)
    recommended=torch.mps.recommended_max_memory();torch.mps.set_per_process_memory_fraction(min(.25,4*1024**3/max(1,recommended)))
    a=cpu.predict([0.]*__import__('math').prod(cpu.sample_shape))[0];n=len(a)
    def observations(count,phase=0):
     width=__import__('math').prod(cpu.sample_shape)
     values=torch.arange(count*width,dtype=torch.float32).reshape(count,width)
     return (((values+phase)%11)-5).div(7).reshape(count,*cpu.sample_shape)
    # Actual requested inference shape, signed observations and asymmetric targets.
    inference=observations(c.compute.get('batchSize',16)).reshape(c.compute.get('batchSize',16),-1).tolist()
    def inference_parity():
     left=cpu.predict_batch(inference);right=gpu.predict_batch(inference)
     for (p,v),(q,w) in zip(left,right):
      require(torch.allclose(torch.tensor(p),torch.tensor(q),atol=1e-4,rtol=1e-3) and abs(v-w)<=1e-4,'CPU/MPS inference parity failed.')
    inference_parity()
    mask=torch.ones((c.batch,n),dtype=torch.bool)
    if n>1:mask[torch.arange(c.batch),torch.arange(c.batch)%n]=False
    weights=torch.arange(1,n+1,dtype=torch.float32).expand(c.batch,n)*mask
    pi=weights/weights.sum(1,keepdim=True)
    z=torch.where(torch.arange(c.batch)%2==0,.75,-.5).reshape(-1,1)
    options=dict(momentum=c.momentum) if c.optimizer=='sgd' else dict(betas=(c.beta1,c.beta2),eps=c.epsilon)
    optim=lambda m:({'adam':torch.optim.Adam,'adamw':torch.optim.AdamW,'sgd':torch.optim.SGD}[c.optimizer])([p for p in m.model.parameters() if p.requires_grad],lr=c.learningRate,weight_decay=c.weightDecay,**options)
    oc,og=optim(cpu),optim(gpu)
    def step(m,o,phase=0):
     x=observations(c.batch,phase);m.model.train();o.zero_grad();out=m.forward(x.to(m.device));loss,_=joint_loss(out,mask.to(m.device),pi.to(m.device),z.to(m.device),policy_weight=c.policyWeight,value_weight=c.valueWeight);
     if __package__:from .moe import auxiliary_loss
     else:from moe import auxiliary_loss
     loss=loss+auxiliary_loss(m.model);loss.backward();torch.nn.utils.clip_grad_norm_(m.model.parameters(),c.gradientClip,error_if_nonfinite=True);o.step();return float(loss.detach())
    for phase in range(3):
     lc,lg=step(cpu,oc,phase),step(gpu,og,phase);require(abs(lc-lg)<=1e-4*max(1,abs(lc)),'CPU/MPS joint loss parity failed.')
     for (name,p),(other,q) in zip(cpu.model.named_parameters(),gpu.model.named_parameters()):
      require(name==other and ((p.grad is None and q.grad is None) or (p.grad is not None and q.grad is not None and torch.allclose(p.grad,q.grad.cpu(),atol=1e-4,rtol=1e-3))),'CPU/MPS gradient parity failed.')
      require(torch.allclose(p.detach(),q.detach().cpu(),atol=1e-4,rtol=1e-3),'CPU/MPS updated weights differ.')
     inference_parity()
    torch.mps.synchronize();saved_rng=torch.mps.get_rng_state();buf=io.BytesIO();torch.save({'model':{k:v.detach().cpu() for k,v in gpu.model.state_dict().items()},'optimizer':og.state_dict()},buf)
    loaded=torch.load(io.BytesIO(buf.getvalue()),map_location='cpu',weights_only=True)
    restored=PolicyValueModel(graph,state=loaded['model'],device='mps',trainability=trainability);orr=optim(restored);orr.load_state_dict(loaded['optimizer']);torch.mps.set_rng_state(saved_rng)
    before_rng=torch.mps.get_rng_state();live=step(gpu,og);torch.mps.set_rng_state(before_rng);again=step(restored,orr);torch.mps.synchronize()
    require(abs(live-again)<=1e-5*max(1,abs(live)),'MPS optimizer restore parity failed.')
    for k,v in gpu.model.state_dict().items():require(torch.allclose(v.cpu(),restored.model.state_dict()[k].cpu(),atol=1e-5,rtol=1e-4),'MPS restored weights differ.')
    _cache[key]=dict(profileSHA256=key,qualified=True,reproducibility='Accelerated recovery · numerical qualification; not bitwise',reason='Varied observations, partial legal masks, selected inference batch, three optimizer steps and safe restore passed',gpuBudgetBytes=min(4*1024**3,int(recommended*.25)))
   finally:torch.set_rng_state(cpu_rng);torch.mps.set_rng_state(gpu_rng)
  result=dict(_cache[key])
 # Automatic stays CPU until an end-to-end matched campaign benchmark qualifies this profile.
 for role,device in requested.items():result[role]='mps' if device=='mps' else 'cpu'
 if result['learning']=='cpu' and result['inference']=='cpu':result['reproducibility']='CPU reference' if c.compute.get('activeGames',1)==1 else 'Batched CPU recovery · numerical ordering; not bitwise'
 if 'auto' in requested.values():result['reason']+='; Automatic uses CPU pending matched end-to-end speed qualification'
 return result

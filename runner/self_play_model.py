"""Joint task adapter; parameters are shared with the bundled GraphModel interpreter."""
import copy,hashlib,math
import torch
if __package__:from .game_episodes import require,encoded
else:from game_episodes import require,encoded

class PolicyValueModel:
    def __init__(self,graph,*,seed=17,state=None,registry=None,device='cpu',input_name=None,initialization=None,trainability=None):
        if __package__:from .engine import GraphModel
        else:from engine import GraphModel
        require(device in ('cpu','mps') and (device!='mps' or torch.backends.mps.is_available()),'Requested policy/value device is unavailable.')
        outputs=[n for n in graph['nodes'] if n['type']=='output']
        require(len(outputs)==1 and outputs[0].get('config')=={'task':'gamePolicyValue','taskVersion':1},'Use the versioned Game policy/value output task.')
        names=graph.get('namedInputs',{})
        require(not names or len(names)==1,'Policy/value requires one declared named input.')
        declared=next(iter(names),'input')
        require(input_name is None or input_name==declared,'Bind exactly the declared named model input.')
        self.device=device;self.input_name=declared;self.graph=copy.deepcopy(graph)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.model=GraphModel(graph,registry,bundled_dataset=False,architecture_only=True).to(device)
            inputs=[n for n in graph['nodes'] if n['type']=='input'];require(len(inputs)==1,'Policy/value requires one numeric input.')
            c=inputs[0].get('config',{});shape=c.get('shape',[c.get('features',0)])
            require(len(shape)>=1 and math.prod(shape)<=256,'Policy/value observations exceed the admitted numeric boundary.')
            self.sample_shape=shape
            self.forward(torch.zeros((1,*shape),device=device))
            if initialization is not None:
                if __package__:from .initialization import apply_initialization
                else:from initialization import apply_initialization
                apply_initialization(self.model,initialization)
        require(sum(p.numel() for p in self.model.parameters())<=2_000_000,'Policy/value parameter limit exceeded.')
        if state is not None:
            expected=self.model.state_dict();require(isinstance(state,dict) and set(state)==set(expected),'Checkpoint parameter inventory differs.')
            for k,v in state.items():require(isinstance(v,torch.Tensor) and v.dtype==expected[k].dtype and v.shape==expected[k].shape and torch.isfinite(v).all().item(),'Checkpoint parameter contract differs.')
            self.model.load_state_dict(state,strict=True)
        if trainability is not None:
            if __package__:from .trainability import apply_trainability
            else:from trainability import apply_trainability
            apply_trainability(self.model,trainability)
        require(any(p.requires_grad for p in self.model.parameters()),'Self-play requires at least one trainable parameter.')
    def forward(self,x):return self.model({self.input_name:x} if self.model.named_inputs else x)
    def predict(self,observation):
        require(type(observation) is list and len(observation)==math.prod(self.sample_shape),'Game observation dimensions differ from the model.')
        self.model.eval()
        with torch.inference_mode():out=self.forward(torch.tensor([observation],dtype=torch.float32,device=self.device).reshape(1,*self.sample_shape))
        return out['policy'].detach().cpu()[0].tolist(),float(out['value'].detach().cpu()[0,0])
    def predict_batch(self,observations):
        require(type(observations) is list and 1<=len(observations)<=256 and all(type(x) is list and len(x)==math.prod(self.sample_shape) for x in observations),'Invalid bounded inference batch.')
        self.model.eval()
        with torch.inference_mode():
            out=self.forward(torch.tensor(observations,dtype=torch.float32,device=self.device).reshape(len(observations),*self.sample_shape))
            p=out['policy'].detach().cpu().tolist();v=out['value'].detach().cpu().flatten().tolist()
        require(all(math.isfinite(z) for row in p for z in row) and all(math.isfinite(z) and -1<=z<=1 for z in v),'Nonfinite batched policy/value.')
        return list(zip(p,v))
    def frozen(self):
        model=PolicyValueModel(self.graph,state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},device=self.device,input_name=self.input_name)
        model.model.eval().requires_grad_(False);return model
    def snapshot_digest(self):
        h=hashlib.sha256()
        for k,v in sorted(self.model.state_dict().items()):
            v=v.detach().cpu().contiguous();h.update(encoded([k,str(v.dtype),list(v.shape)]));h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

def joint_loss(outputs,mask,probabilities,outcomes,*,policy_weight=1.,value_weight=1.):
    logits=outputs['policy'];values=outputs['value']
    require(logits.dtype==torch.float32 and logits.ndim==2 and mask.dtype==torch.bool and mask.shape==logits.shape and probabilities.shape==logits.shape and values.shape==outcomes.shape==(logits.shape[0],1),'Joint loss tensor contracts differ.')
    require(torch.isfinite(logits).all().item() and torch.isfinite(values).all().item() and torch.isfinite(probabilities).all().item() and torch.isfinite(outcomes).all().item(),'Joint loss requires finite values.')
    require(mask.any(dim=1).all().item() and (probabilities>=0).all().item() and (probabilities[~mask]==0).all().item() and torch.allclose(probabilities.sum(1),torch.ones(logits.shape[0],device=logits.device),atol=1e-5) and (outcomes.abs()<=1).all().item(),'Invalid legal-mask, policy or terminal outcome targets.')
    for w in [policy_weight,value_weight]:require(type(w) in (int,float) and math.isfinite(w) and 0<=w<=100,'Invalid joint loss weight.')
    require(policy_weight+value_weight>0,'At least one joint loss must be active.')
    logp=torch.log_softmax(logits.masked_fill(~mask,float('-inf')),dim=1)
    policy=-(probabilities*logp.masked_fill(~mask,0)).sum(dim=1).mean();value=(values-outcomes).square().mean()
    loss=policy_weight*policy+value_weight*value
    return loss,dict(policyLoss=float(policy.detach()),valueLoss=float(value.detach()),loss=float(loss.detach()))

"""Independent eight-objective contracts and scalar native measurements."""
import importlib,math
import pytest
from runner.tensor_layout import BatchedLayout
IDS=('loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')
def config():return importlib.import_module('runner.loss_configuration')
def objectives():return importlib.import_module('runner.loss_objectives')
@pytest.mark.parametrize('ident',IDS)
def test_strict_canonical_recipe_and_unsupported_authorable_state(ident):
 h=config();c=h.canonical_configuration(ident,{})
 r={'schemaVersion':1,'componentID':ident,'configurationVersion':1,'config':c};assert h.validate_recipe(r)==r
 for change in ({'weights':[]},{'training':True},{'reduction':'automatic'}):
  with pytest.raises(ValueError):h.canonical_configuration(ident,change)
 for change in ({**r,'foreign':1},{**r,'schemaVersion':True},{**r,'configurationVersion':2}):
  with pytest.raises(ValueError):h.validate_recipe(change)

def test_actual_geometry_scalar_class_targets_and_explicit_sample_broadcast():
 h=config();ordinary=lambda dtype,s:{'target':{'dtype':dtype,'sampleShape':s}}
 assert h.resolve_objective('loss_cross_entropy',{},BatchedLayout((2,3,4)),ordinary('int64',[2,3]))['rawSampleShape']==[2,3]
 assert h.resolve_objective('loss_mse',{'target_broadcast':True},BatchedLayout((2,3)),ordinary('float32',[3]))['rawSampleShape']==[2,3]
 for ident,c,p,t in [('loss_cross_entropy',{'class_axis':0},(3,),ordinary('int64',[])),('loss_mse',{},(2,3),ordinary('float32',[3])),('loss_triplet_margin',{},(2,3),{}),('loss_cross_entropy',{},(3,),ordinary('float32',[]))]:
  with pytest.raises(ValueError):h.resolve_objective(ident,c,BatchedLayout(p),t)
 assert h.resolve_objective('loss_triplet_margin',{},BatchedLayout((3,2)),{})['rawSampleShape']==[]

def test_number_lists_weight_geometry_and_nonrepresentable_controls_refuse():
 h=config()
 for ident,c in [('loss_cross_entropy',{'class_weights':[True]}),('loss_cross_entropy',{'target_mode':'probabilities','ignore_index':3}),('loss_bce_with_logits',{'weight_mode':'configured','weight_shape':[2],'weight_values':[1]}),('loss_triplet_margin',{'p':0}),('loss_triplet_margin',{'p':17}),('loss_triplet_margin',{'margin':1e-46}),('loss_mse',{'target_broadcast':1})]:
  with pytest.raises(ValueError):h.canonical_configuration(ident,c)

import torch,itertools
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('ident',IDS)
def test_native_objective_is_differentiable_and_has_honest_metadata(device,ident):
 if device=='mps' and not torch.backends.mps.is_available():pytest.fail('Required MPS backend unavailable.')
 h=objectives();r={'schemaVersion':1,'componentID':ident,'configurationVersion':1,'config':{}}
 x=torch.tensor([[.2,-.5,.7],[.8,.4,-.2]],device=device,requires_grad=True)
 if ident in ('loss_nll','loss_kl_div'):x=x.log_softmax(-1)
 if ident in ('loss_cross_entropy','loss_nll'):roles={'target':torch.tensor([2,0],device=device)}
 elif ident=='loss_ctc':
  x=torch.tensor([[[.2,-.3,.7],[.6,.1,-.2],[.4,-.5,.8]],[[.7,.2,-.5],[.3,.9,-.4],[.6,-.2,.1]]],device=device,requires_grad=True).log_softmax(-1)
  roles={'target':torch.tensor([[1,2],[2,1]],device=device),'input_lengths':torch.tensor([3,3],device=device),'target_lengths':torch.tensor([2,1],device=device)}
 elif ident=='loss_triplet_margin':x=x.reshape(1,2,3).transpose(1,2);roles={}
 elif ident=='loss_kl_div':roles={'target':torch.tensor([[.2,.3,.5],[.6,.1,.3]],device=device)}
 else:roles={'target':torch.tensor([[.1,.4,.8],[.9,.3,.2]],device=device)}
 o=h.create_objective(r,device=device);result=h.evaluate_objective(o,x,roles)
 assert result['raw'].ndim==0 and result['scalar'].requires_grad
 assert result['metadata']['componentID']==ident and result['metadata']['reduction']=='mean'
 grad=torch.autograd.grad(result['scalar'],x)[0];assert torch.isfinite(grad).all()

# These expected values use plain Python arithmetic and complete CTC path
# enumeration, never a Torch objective or the production helper.
def logs(row):
 m=max(row);z=m+math.log(sum(math.exp(v-m) for v in row));return [v-z for v in row]
def scalar_rows(ident,rows,targets,c):
 scores=[];denom=0
 for x,y in zip(rows,targets):
  if ident in ('loss_cross_entropy','loss_nll'):
   lp=logs(x) if ident=='loss_cross_entropy' else x;weights=c.get('class_weights') or [1.]*len(x)
   if isinstance(y,int):
    if y==c.get('ignore_index',-100):scores.append(0.);continue
    smoothing=c.get('label_smoothing',0.);scores.append(-(1-smoothing)*weights[y]*lp[y]-smoothing*sum(w*l for w,l in zip(weights,lp))/len(x));denom+=weights[y]
   else:
    smoothing=c.get('label_smoothing',0.);scores.append(-sum(((1-smoothing)*v+smoothing/len(x))*w*l for v,w,l in zip(y,weights,lp)));denom+=1
  elif ident=='loss_bce_with_logits':
   w=c.get('_weight',[1.]*len(x));pw=c.get('_pos_weight',[1.]*len(x));scores.extend(weight*((1-v)*max(a,0)+pwv*v*max(-a,0)+((1-v)+pwv*v)*math.log1p(math.exp(-abs(a)))) for a,v,weight,pwv in zip(x,y,w,pw));denom+=len(x)
  elif ident in ('loss_mse','loss_l1'):scores.extend((a-b)**2 if ident=='loss_mse' else abs(a-b) for a,b in zip(x,y));denom+=len(x)
  elif ident=='loss_kl_div':scores.extend((math.exp(v)*(v-a) if c.get('log_target') else v*(math.log(v)-a) if v else 0.) for a,v in zip(x,y));denom+=len(x)
 if c['reduction']=='none':return scores
 if c['reduction']=='sum':return sum(scores)
 return sum(scores)/(len(rows) if c['reduction']=='batchmean' else denom)
def flat(value):
 if isinstance(value,list):return [v for x in value for v in flat(x)]
 return [value]
def triplet_oracle(rows,c):
 def dist(a,b):return sum(abs(x-y+c['eps'])**c['p'] for x,y in zip(a,b))**(1/c['p'])
 out=[]
 for a,p,n in rows:
  neg=dist(a,n)
  if c['swap']:neg=min(neg,dist(p,n))
  out.append(max(dist(a,p)-neg+c['margin'],0))
 return out if c['reduction']=='none' else sum(out)/(len(out) if c['reduction']=='mean' else 1)
def ctc_oracle(log_rows,labels,blank):
 probability=0.
 for path in itertools.product(range(len(log_rows[0])),repeat=len(log_rows)):
  merged=[v for i,v in enumerate(path) if i==0 or v!=path[i-1]];collapsed=[v for v in merged if v!=blank]
  if collapsed==labels:probability+=math.exp(sum(log_rows[t][v] for t,v in enumerate(path)))
 return -math.log(probability) if probability else math.inf

def recipe(ident,config=None):return dict(schemaVersion=1,componentID=ident,configurationVersion=1,config=config or {})
@pytest.fixture(autouse=True)
def restore_determinism_policy():
 old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 yield
 torch.use_deterministic_algorithms(old,warn_only=warn)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
@pytest.mark.parametrize('ident',IDS[:6])
def test_independent_values_all_dense_reductions_and_every_supported_float_role_gradient(device,reduction,ident):
 torch.use_deterministic_algorithms(device=='cpu')
 dtype=torch.float64 if device=='cpu' else torch.float32
 rows=[[.7,-.4,.2],[-.2,.8,.5]]
 c={'reduction':reduction};target_values=[2,0] if ident in ('loss_cross_entropy','loss_nll') else [[.2,.3,.5],[.6,.1,.3]]
 if ident=='loss_cross_entropy':c.update(class_weights=[.5,1.7,2.],label_smoothing=.3)
 if ident=='loss_nll':c.update(class_weights=[.5,1.7,2.]);rows=[logs(r) for r in rows]
 if ident=='loss_kl_div':rows=[logs(r) for r in rows]
 if ident=='loss_bce_with_logits':c.update(weight_mode='configured',weight_shape=[3],weight_values=[.5,0.,2.],pos_weight_mode='configured',pos_weight_shape=[],pos_weight_values=[1.7])
 x=torch.tensor(rows,dtype=dtype,device=device,requires_grad=True)
 integer=ident in ('loss_cross_entropy','loss_nll');y=torch.tensor(target_values,dtype=torch.int64 if integer else dtype,device=device,requires_grad=not integer)
 o=objectives().create_objective(recipe(ident,c),device,dtype);result=objectives().evaluate_objective(o,x,{'target':y},'sum')
 canonical=config().canonical_configuration(ident,c)
 if ident=='loss_bce_with_logits':canonical.update(_weight=[.5,0.,2.],_pos_weight=[1.7]*3)
 expected=scalar_rows(ident,rows,target_values,canonical)
 assert result['raw'].detach().cpu().reshape(-1).tolist()==pytest.approx(flat(expected),abs=2e-6,rel=2e-5)
 def evaluate_expected(a,b):
  value=scalar_rows(ident,a,b,canonical);return sum(value) if isinstance(value,list) else value
 native_grads=torch.autograd.grad(result['scalar'],(x,) if integer else (x,y))
 for which,table,grad in [('prediction',rows,native_grads[0])]+([] if integer else [('target',target_values,native_grads[1])]):
  expected_grad=[]
  for i,row in enumerate(table):
   for j in range(len(row)):
    plus=[list(v) for v in table];minus=[list(v) for v in table];plus[i][j]+=1e-5;minus[i][j]-=1e-5
    ep=evaluate_expected(plus,target_values) if which=='prediction' else evaluate_expected(rows,plus)
    em=evaluate_expected(minus,target_values) if which=='prediction' else evaluate_expected(rows,minus)
    expected_grad.append((ep-em)/2e-5)
  assert grad.detach().cpu().reshape(-1).tolist()==pytest.approx(expected_grad,abs=3e-6,rel=5e-5)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('reduction',['none','mean','sum','batchmean'])
@pytest.mark.parametrize('log_target',[False,True])
def test_kl_denominators_and_log_target_gradient(device,reduction,log_target):
 dtype=torch.float64 if device=='cpu' else torch.float32;rows=[logs([.5,-.2,.7]),logs([.1,.8,-.4])];targets=[[.2,.3,.5],[.6,.1,.3]]
 if log_target:targets=[[math.log(v) for v in r] for r in targets]
 c=dict(reduction=reduction,log_target=log_target);x=torch.tensor(rows,dtype=dtype,device=device,requires_grad=True);y=torch.tensor(targets,dtype=dtype,device=device,requires_grad=True)
 result=objectives().evaluate_objective(objectives().create_objective(recipe('loss_kl_div',c),device,dtype),x,{'target':y},'sum')
 expected=scalar_rows('loss_kl_div',rows,targets,c);assert result['raw'].detach().cpu().flatten().tolist()==pytest.approx(flat(expected),rel=2e-5,abs=2e-6)
 grads=torch.autograd.grad(result['scalar'],(x,y));scale=1 if reduction in ('none','sum') else 1/2 if reduction=='batchmean' else 1/6
 assert grads[0].detach().cpu().flatten().tolist()==pytest.approx([-scale*(math.exp(v) if log_target else v) for v in flat(targets)],abs=2e-6)
 assert grads[1].detach().cpu().flatten().tolist()==pytest.approx([scale*(math.exp(v)*(v-a+1) if log_target else math.log(v)-a+1) for a,v in zip(flat(rows),flat(targets))],abs=2e-6)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
@pytest.mark.parametrize('p',[.5,1.,1.5,2.,16.])
@pytest.mark.parametrize('swap',[False,True])
def test_triplet_real_p_swap_eps_all_three_gradients(device,reduction,p,swap):
 torch.use_deterministic_algorithms(device=='cpu');dtype=torch.float64 if device=='cpu' else torch.float32
 rows=[[[.2,-.5],[.7,.3],[.8,.2]],[[-.3,.4],[.2,.8],[-1.4,-.7]]];c=dict(reduction=reduction,p=p,swap=swap,eps=.02,margin=1.3)
 x=torch.tensor(rows,device=device,dtype=dtype,requires_grad=True);result=objectives().evaluate_objective(objectives().create_objective(recipe('loss_triplet_margin',c),device,dtype),x,{})
 expected=triplet_oracle(rows,c);assert result['raw'].detach().cpu().flatten().tolist()==pytest.approx(flat(expected),abs=3e-6,rel=3e-5)
 grad=torch.autograd.grad(result['scalar'],x)[0];expected_grad=[]
 for i in range(2):
  for j in range(3):
   for k in range(2):
    import copy
    plus=copy.deepcopy(rows);minus=copy.deepcopy(rows);plus[i][j][k]+=1e-5;minus[i][j][k]-=1e-5
    a=triplet_oracle(plus,c);b=triplet_oracle(minus,c)
    if isinstance(a,list):a=sum(a)/2;b=sum(b)/2
    expected_grad.append((a-b)/2e-5)
 assert grad.detach().cpu().flatten().tolist()==pytest.approx(expected_grad,abs=5e-5,rel=1e-4)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('layout',['padded','packed'])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
@pytest.mark.parametrize('blank',[0,2])
def test_ctc_complete_path_reference_lengths_empty_targets_and_logsoftmax_gradient(device,layout,reduction,blank):
 torch.use_deterministic_algorithms(device=='cpu');dtype=torch.float64 if device=='cpu' else torch.float32
 logits=[[[.3,-.2,.6],[.7,.1,-.4],[.2,.8,-.1],[.5,-.3,.2]],[[-.2,.4,.7],[.6,.3,-.4],[.1,.5,-.3],[.8,-.5,.2]]]
 labels=[1,1] if blank==0 else [0,0];targets=[labels,[999,-999]];lengths=[2,0];times=[4,3]
 c=dict(reduction=reduction,target_layout=layout,blank=blank);x=torch.tensor(logits,dtype=dtype,device=device,requires_grad=True)
 roles={'target':torch.tensor(targets,device=device),'input_lengths':torch.tensor(times,device=device),'target_lengths':torch.tensor(lengths,device=device)}
 result=objectives().evaluate_objective(objectives().create_objective(recipe('loss_ctc',c),device,dtype),x.log_softmax(-1),roles,'sum')
 def reference(rows):
  losses=[ctc_oracle([logs(r) for r in rows[i][:times[i]]],labels if i==0 else [],blank) for i in range(2)]
  return losses if reduction=='none' else sum(losses) if reduction=='sum' else sum(l/max(lengths[i],1) for i,l in enumerate(losses))/2
 expected=reference(logits);assert result['raw'].detach().cpu().flatten().tolist()==pytest.approx(flat(expected),abs=3e-6,rel=3e-5)
 grad=torch.autograd.grad(result['scalar'],x)[0];expected_grad=[]
 import copy
 for i in range(2):
  for j in range(4):
   for k in range(3):
    plus=copy.deepcopy(logits);minus=copy.deepcopy(logits);plus[i][j][k]+=1e-5;minus[i][j][k]-=1e-5;a=reference(plus);b=reference(minus)
    if isinstance(a,list):a=sum(a);b=sum(b)
    expected_grad.append((a-b)/2e-5)
 assert grad.detach().cpu().flatten().tolist()==pytest.approx(expected_grad,abs=5e-6,rel=1e-4)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('smoothing',[0.,.4,1.])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
def test_cross_entropy_probability_and_index_spatial_class_axis_ignored_weighted(device,smoothing,reduction):
 dtype=torch.float64 if device=='cpu' else torch.float32;rows=[[[.2,.8,-.3],[.5,-.1,.7]],[[.6,.1,-.4],[-.2,.9,.3]]];weights=[.5,1.4,2.]
 x=torch.tensor(rows,dtype=dtype,device=device,requires_grad=True).transpose(1,2)
 for probability in (False,True):
  yrows=[[[.2,.3,.5],[.6,.1,.3]],[[.1,.7,.2],[.3,.2,.5]]] if probability else [[2,-100],[1,0]]
  y=torch.tensor(yrows,dtype=dtype if probability else torch.int64,device=device)
  if probability:y=y.transpose(1,2)
  c=dict(target_mode='probabilities' if probability else 'class_indices',class_axis=1,class_weights=weights,label_smoothing=smoothing,reduction=reduction)
  result=objectives().evaluate_objective(objectives().create_objective(recipe('loss_cross_entropy',c),device,dtype),x,{'target':y})
  cc={**c,'ignore_index':-100};expected=scalar_rows('loss_cross_entropy',[r for batch in rows for r in batch],[r for batch in yrows for r in batch],cc)
  assert result['raw'].detach().cpu().flatten().tolist()==pytest.approx(flat(expected),abs=2e-6,rel=3e-5)
  assert result['metadata']['ignoredCount']==(0 if probability else 1)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_mapped_weight_broadcast_is_owned_by_each_observation_and_target_gradients(device):
 dtype=torch.float64 if device=='cpu' else torch.float32;c=dict(weight_mode='mapped',pos_weight_mode='mapped',reduction='none')
 x=torch.tensor([[[.2,.7],[-.3,.5]],[[.8,-.2],[.6,.1]]],device=device,dtype=dtype,requires_grad=True)
 y=torch.tensor([[[.1,.8],[.5,.3]],[[.2,.6],[.4,.9]]],device=device,dtype=dtype,requires_grad=True)
 w=torch.tensor([.5,2.],device=device,dtype=dtype);pw=torch.tensor([[1.7,.4],[.8,2.1]],device=device,dtype=dtype)
 o=objectives().create_objective(recipe('loss_bce_with_logits',c),device,dtype);result=objectives().evaluate_objective(o,x,{'target':y,'weight':w,'pos_weight':pw})
 expected=[]
 for row in range(2):
  cc={**c,'_weight':[w[row].item()]*4,'_pos_weight':pw[row].tolist()*2}
  expected+=scalar_rows('loss_bce_with_logits',[x[row].detach().cpu().flatten().tolist()],[y[row].detach().cpu().flatten().tolist()],cc)
 assert result['raw'].detach().cpu().flatten().tolist()==pytest.approx(expected,abs=2e-6)
 grads=torch.autograd.grad(result['scalar'],(x,y));assert all(torch.isfinite(g).all() for g in grads)
 with pytest.raises(ValueError):objectives().evaluate_objective(o,x,{'target':y,'weight':w.clone().requires_grad_(),'pos_weight':pw})

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('ident',['loss_mse','loss_l1'])
def test_explicit_scalar_sample_broadcast_never_mixes_observations(device,ident):
 x=torch.tensor([[.2,.8],[-.3,.5]],device=device,requires_grad=True);y=torch.tensor([.4,-.2],device=device,requires_grad=True)
 o=objectives().create_objective(recipe(ident,dict(target_broadcast=True,reduction='none')),device);r=objectives().evaluate_objective(o,x,{'target':y})
 expected=[(.2-.4)**2,(.8-.4)**2,(-.3+.2)**2,(.5+.2)**2] if ident=='loss_mse' else [.2,.4,.1,.7]
 assert r['raw'].shape==(2,2) and r['raw'].detach().cpu().flatten().tolist()==pytest.approx(expected,abs=1e-7)
 g=torch.autograd.grad(r['scalar'],y)[0];assert g.detach().cpu().tolist()==pytest.approx([-.1,-.3] if ident=='loss_mse' else [0.,0.],abs=1e-7)
 with pytest.raises(ValueError):objectives().evaluate_objective(objectives().create_objective(recipe(ident),device),x,{'target':y})

@pytest.mark.parametrize('device',['cpu','mps'])
def test_ctc_infeasible_is_explicitly_masked_or_actionably_refused(device):
 x=torch.tensor([[[.2,.5,-.3],[.7,-.4,.1]]],device=device,requires_grad=True);roles={'target':torch.tensor([[1,1]],device=device),'input_lengths':torch.tensor([2],device=device),'target_lengths':torch.tensor([2],device=device)}
 with pytest.raises(ValueError,match='nonfinite'):objectives().evaluate_objective(objectives().create_objective(recipe('loss_ctc'),device),x.log_softmax(-1),roles)
 result=objectives().evaluate_objective(objectives().create_objective(recipe('loss_ctc',{'zero_infinity':True}),device),x.log_softmax(-1),roles)
 assert result['scalar'].item()==0 and result['metadata']['maskedCount']==1 and result['metadata']['infeasibleCount']==1
 assert torch.equal(torch.autograd.grad(result['scalar'],x)[0],torch.zeros_like(x))

@pytest.mark.parametrize('ident',['loss_cross_entropy','loss_nll'])
@pytest.mark.parametrize('reduction',['none','sum','mean'])
def test_ignored_and_zero_effective_class_weight_denominators(ident,reduction):
 x=torch.tensor([[.2,.7,-.3]],requires_grad=True)
 if ident=='loss_nll':x=x.log_softmax(-1)
 for target,weights in [([-100],[]),([1],[1.,0.,2.])]:
  o=objectives().create_objective(recipe(ident,dict(reduction=reduction,class_weights=weights)))
  if reduction=='mean':
   with pytest.raises(ValueError,match='denominator'):objectives().evaluate_objective(o,x,{'target':torch.tensor(target)})
  else:assert objectives().evaluate_objective(o,x,{'target':torch.tensor(target)})['scalar'].item()==0

@pytest.mark.parametrize('ident',IDS)
def test_fixed_buffers_are_canonical_not_learned_and_foreign_state_refuses(ident):
 c=dict(class_weights=[.5,2.,1.]) if ident in ('loss_cross_entropy','loss_nll') else dict(weight_mode='configured',weight_shape=[],weight_values=[.5]) if ident=='loss_bce_with_logits' else {}
 o=objectives().create_objective(recipe(ident,c));assert list(o.parameters())==[];state=o.state_dict();o.load_state_dict(state)
 with pytest.raises(ValueError):o.load_state_dict({**state,'foreign':torch.tensor(1.)})
 if state:
  key=next(iter(state));bad={k:v.clone() for k,v in state.items()};bad[key].flatten()[0]+=.1
  with pytest.raises(ValueError):o.load_state_dict(bad)
  with pytest.raises(ValueError):o.load_state_dict({k:v.double() for k,v in state.items()})

@pytest.mark.parametrize('device',['cpu','mps'])
def test_invalid_actual_values_roles_distributions_and_float_dtypes_refuse(device):
 h=objectives();x=torch.tensor([[.2,.7,-.3]],device=device);y=torch.tensor([1],device=device)
 ce=h.create_objective(recipe('loss_cross_entropy'),device)
 for roles in ({'target':torch.tensor([9],device=device)},{'target':y,'foreign':y},{'target':y.float()},{'target':torch.tensor([True],device=device)}):
  with pytest.raises(ValueError):h.evaluate_objective(ce,x,roles)
 with pytest.raises(ValueError):h.evaluate_objective(ce,x.clone().fill_(float('nan')),{'target':y})
 with pytest.raises(ValueError):h.evaluate_objective(ce,x,{'target':torch.tensor([1,2],device=device)})
 for ident in ('loss_nll','loss_kl_div'):
  target=y if ident=='loss_nll' else torch.tensor([[.2,.3,.5]],device=device)
  with pytest.raises(ValueError,match='normalized'):h.evaluate_objective(h.create_objective(recipe(ident),device),x,{'target':target})
 with pytest.raises(ValueError):h.evaluate_objective(h.create_objective(recipe('loss_bce_with_logits'),device),x,{'target':x})
 if device=='cpu':
  with pytest.raises(ValueError):h.evaluate_objective(ce,x.double(),{'target':y})
 else:
  with pytest.raises(ValueError):h.create_objective(recipe('loss_mse'),device,torch.float64)

def test_preallocation_geometry_and_recipe_payload_budgets():
 h=config()
 with pytest.raises(ValueError):h.resolve_objective('loss_mse',{'target_broadcast':True},BatchedLayout((4096,1,4096)),{'target':dict(dtype='float32',sampleShape=[1,4096,1])})
 with pytest.raises(ValueError):h.validate_recipe(recipe('loss_bce_with_logits',dict(weight_mode='configured',weight_shape=[4096],weight_values=[1.234567891234567e20]*4096)))

def test_config_only_admission_cannot_create_oversized_recipe():
 with pytest.raises(ValueError):config().canonical_configuration('loss_bce_with_logits',dict(weight_mode='configured',weight_shape=[4096],weight_values=[1.234567891234567e20]*4096))

def test_live_fixed_buffer_tampering_is_refused_before_objective():
 o=objectives().create_objective(recipe('loss_cross_entropy',dict(class_weights=[.5,2.,1.])))
 o.class_weights[0]=.7
 with pytest.raises(ValueError):objectives().evaluate_objective(o,torch.tensor([[.2,.4,-.1]]),{'target':torch.tensor([1])})

@pytest.mark.parametrize('ident',['loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin'])
def test_aggregation_metadata_has_explicit_numerator_denominator_for_unequal_batches(ident):
 h=objectives();x=torch.tensor([[.2,.4,-.1],[.3,-.2,.7]])
 if ident in ('loss_nll','loss_kl_div'):x=x.log_softmax(-1)
 if ident in ('loss_cross_entropy','loss_nll'):roles={'target':torch.tensor([1,2])}
 elif ident=='loss_triplet_margin':x=x.reshape(1,2,3).transpose(1,2);roles={}
 elif ident=='loss_ctc':
  x=x.reshape(1,2,3).log_softmax(-1);roles={'target':torch.tensor([[1]]),'input_lengths':torch.tensor([2]),'target_lengths':torch.tensor([1])}
 else:roles={'target':torch.tensor([[.2,.3,.5],[.4,.4,.2]])}
 for reduction in ('none','mean','sum'):
  for scalarization in ('mean','sum'):
   r=h.evaluate_objective(h.create_objective(recipe(ident,dict(reduction=reduction))),x,roles,scalarization)
   m=r['metadata'];assert m['aggregationNumerator']/m['aggregationDenominator']==pytest.approx(r['scalar'].item(),abs=1e-7)
   assert m['aggregationConvention'] in ('nativeMean','nativeBatchMean','sum','rawElementMean')

def test_kl_zero_probability_admitted_but_native_nonfinite_target_gradient_diagnosed():
 x=torch.tensor([[.2,.4,-.1]],requires_grad=True).log_softmax(-1);y=torch.tensor([[0.,.3,.7]],requires_grad=True)
 r=objectives().evaluate_objective(objectives().create_objective(recipe('loss_kl_div')),x,{'target':y})
 assert torch.isfinite(r['scalar'])
 with pytest.raises(ValueError,match='gradient'):torch.autograd.grad(r['scalar'],y)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('ident',IDS)
def test_real_shared_learned_affine_parameter_gradients_and_first_update_against_scalar_reference(device,ident):
 torch.use_deterministic_algorithms(device=='cpu');dtype=torch.float64 if device=='cpu' else torch.float32
 shapes={'loss_ctc':(3,3),'loss_triplet_margin':(3,2)};s=shapes.get(ident,(3,));width=math.prod(s)
 inputs=[[.2,.7],[-.4,.3]];weights=[[.15*(j+1),-.07*(j+2)] for j in range(width)];bias=[.05*((j%3)-1) for j in range(width)]
 w=torch.nn.Parameter(torch.tensor(weights,device=device,dtype=dtype));b=torch.nn.Parameter(torch.tensor(bias,device=device,dtype=dtype));x=torch.tensor(inputs,device=device,dtype=dtype,requires_grad=True)
 pred=(x@w.T+b).reshape([2]+list(s));c={}
 if ident in ('loss_nll','loss_kl_div','loss_ctc'):pred=pred.log_softmax(-1)
 if ident in ('loss_cross_entropy','loss_nll'):roles={'target':torch.tensor([1,2],device=device)}
 elif ident=='loss_ctc':roles={'target':torch.tensor([[1],[2]],device=device),'input_lengths':torch.tensor([3,2],device=device),'target_lengths':torch.tensor([1,1],device=device)}
 elif ident=='loss_triplet_margin':roles={};c=dict(p=1.5,eps=.03,swap=True,margin=1.4)
 else:roles={'target':torch.tensor([[.2,.3,.5],[.6,.1,.3]],device=device,dtype=dtype)}
 result=objectives().evaluate_objective(objectives().create_objective(recipe(ident,c),device,dtype),pred,roles)
 def expected(ii,ww,bb):
  rows=[[sum(a*v for a,v in zip(row,weight))+bi for weight,bi in zip(ww,bb)] for row in ii]
  cc=config().canonical_configuration(ident,c)
  if ident in ('loss_nll','loss_kl_div'):rows=[logs(row) for row in rows]
  if ident=='loss_triplet_margin':return triplet_oracle([[r[j:j+2] for j in range(0,6,2)] for r in rows],cc)
  if ident=='loss_ctc':return sum(ctc_oracle([logs(r[j:j+3]) for j in range(0,9,3)][:t],[lab],0) for r,t,lab in zip(rows,[3,2],[1,2]))/2
  return scalar_rows(ident,rows,[1,2] if ident in ('loss_cross_entropy','loss_nll') else [[.2,.3,.5],[.6,.1,.3]],cc)
 assert result['scalar'].item()==pytest.approx(expected(inputs,weights,bias),rel=3e-5,abs=3e-6)
 grads=torch.autograd.grad(result['scalar'],(x,w,b));expected_grad=[]
 import copy
 for kind,table in [('input',inputs),('weight',weights),('bias',[bias])]:
  values=[]
  for i,row in enumerate(table):
   for j in range(len(row)):
    plus=copy.deepcopy(table);minus=copy.deepcopy(table);plus[i][j]+=1e-5;minus[i][j]-=1e-5
    ep=expected(plus,weights,bias) if kind=='input' else expected(inputs,plus,bias) if kind=='weight' else expected(inputs,weights,plus[0])
    em=expected(minus,weights,bias) if kind=='input' else expected(inputs,minus,bias) if kind=='weight' else expected(inputs,weights,minus[0])
    values.append((ep-em)/2e-5)
  expected_grad.append(values)
 for measured,ref in zip(grads,expected_grad):assert measured.detach().cpu().flatten().tolist()==pytest.approx(ref,abs=3e-6,rel=2e-4)
 optimizer=torch.optim.SGD([w,b],lr=.03);w.grad=grads[1];b.grad=grads[2];optimizer.step()
 assert w.detach().cpu().flatten().tolist()==pytest.approx([v-.03*g for v,g in zip(flat(weights),expected_grad[1])],abs=3e-6)
 assert b.detach().cpu().flatten().tolist()==pytest.approx([v-.03*g for v,g in zip(bias,expected_grad[2])],abs=3e-6)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_exact_l1_zero_error_and_triplet_equal_distance_subgradients(device):
 x=torch.tensor([[.2,.7]],device=device,requires_grad=True);r=objectives().evaluate_objective(objectives().create_objective(recipe('loss_l1'),device),x,{'target':x.detach().clone()})
 assert torch.equal(torch.autograd.grad(r['scalar'],x)[0],torch.zeros_like(x))
 t=torch.tensor([[[.2,.7],[.2,.7],[.2,.7]]],device=device,requires_grad=True)
 r=objectives().evaluate_objective(objectives().create_objective(recipe('loss_triplet_margin',dict(eps=0.,p=2.)),device),t,{})
 assert r['scalar'].item()==1 and torch.equal(torch.autograd.grad(r['scalar'],t)[0],torch.zeros_like(t))

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('smoothing',[0.,.4,1.])
def test_probability_class_targets_receive_their_native_weighted_gradient(device,smoothing):
 dtype=torch.float64 if device=='cpu' else torch.float32;rows=[[.2,.8,-.3],[.6,-.1,.7]];weights=[.5,1.4,2.];targets=[[.2,.3,.5],[.6,.1,.3]]
 x=torch.tensor(rows,dtype=dtype,device=device,requires_grad=True);y=torch.tensor(targets,dtype=dtype,device=device,requires_grad=True)
 c=dict(target_mode='probabilities',class_weights=weights,label_smoothing=smoothing)
 r=objectives().evaluate_objective(objectives().create_objective(recipe('loss_cross_entropy',c),device,dtype),x,{'target':y})
 g=torch.autograd.grad(r['scalar'],y)[0];expected=[-(1-smoothing)*w*v/2 for row in rows for w,v in zip(weights,logs(row))]
 assert g.detach().cpu().flatten().tolist()==pytest.approx(expected,abs=1e-6)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_configured_grid_weights_and_independent_positive_term_geometry(device):
 rows=[[.2,.7,-.3,.5],[.8,-.2,.6,.1]];targets=[[.1,.8,.5,.3],[.2,.6,.4,.9]];weights=[.5,0.,1.4,2.];positive=[1.7,1.7,.4,.4]
 x=torch.tensor(rows,device=device).reshape(2,2,2);y=torch.tensor(targets,device=device).reshape(2,2,2)
 c=dict(reduction='sum',weight_mode='configured',weight_shape=[2,2],weight_values=weights,pos_weight_mode='configured',pos_weight_shape=[2,1],pos_weight_values=[1.7,.4])
 r=objectives().evaluate_objective(objectives().create_objective(recipe('loss_bce_with_logits',c),device),x,{'target':y})
 expected=scalar_rows('loss_bce_with_logits',rows,targets,{**c,'_weight':weights,'_pos_weight':positive})
 assert r['scalar'].item()==pytest.approx(expected,abs=2e-6)
 with pytest.raises(ValueError):config().resolve_objective('loss_bce_with_logits',dict(weight_mode='configured',weight_shape=[2,2],weight_values=weights),BatchedLayout((2,)),{'target':dict(dtype='float32',sampleShape=[2])})

@pytest.mark.parametrize('device',['cpu','mps'])
def test_zero_probability_and_zero_class_weights_forward_are_not_repaired(device):
 x=torch.tensor([[.2,.7,-.1]],device=device,requires_grad=True);y=torch.tensor([[0.,.3,.7]],device=device)
 r=objectives().evaluate_objective(objectives().create_objective(recipe('loss_cross_entropy',dict(target_mode='probabilities',class_weights=[0.,0.,0.])),device),x,{'target':y})
 assert r['scalar'].item()==0 and torch.equal(torch.autograd.grad(r['scalar'],x)[0],torch.zeros_like(x))
 r=objectives().evaluate_objective(objectives().create_objective(recipe('loss_kl_div',dict(reduction='sum')),device),x.log_softmax(-1),{'target':y})
 assert torch.isfinite(torch.autograd.grad(r['scalar'],x)[0]).all()

def test_canonical_configuration_identity_cannot_be_mutated_with_live_buffer():
 o=objectives().create_objective(recipe('loss_cross_entropy',dict(class_weights=[.5,2.,1.])))
 o.recipe['config']['class_weights'][0]=.7;o.class_weights[0]=.7
 with pytest.raises(ValueError):objectives().evaluate_objective(o,torch.tensor([[.2,.4,-.1]]),{'target':torch.tensor([1])})
 with pytest.raises(ValueError):o.load_state_dict(o.state_dict())

@pytest.mark.parametrize('device',['cpu','mps'])
def test_zero_infinity_does_not_silently_mask_numeric_overflow_of_feasible_alignment(device):
 x=torch.tensor([[[0.,-3e38,-3e38]]*3],device=device)
 roles={'target':torch.tensor([[1,1]],device=device),'input_lengths':torch.tensor([3],device=device),'target_lengths':torch.tensor([2],device=device)}
 with pytest.raises(ValueError,match='overflow|nonfinite'):objectives().evaluate_objective(objectives().create_objective(recipe('loss_ctc',dict(zero_infinity=True)),device),x,roles)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('ident',['loss_cross_entropy','loss_nll'])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
def test_effective_class_weight_overflow_never_emits_nonfinite_metadata(device,ident,reduction):
 import json
 x=torch.tensor([[10.,0.],[0.,10.]],device=device,requires_grad=True)
 if ident=='loss_nll':x=x.log_softmax(-1)
 o=objectives().create_objective(recipe(ident,dict(reduction=reduction,class_weights=[3e38,3e38])),device)
 roles={'target':torch.tensor([0,1],device=device)}
 if reduction=='mean':
  with pytest.raises(ValueError,match='denominator|weight.*overflow'):objectives().evaluate_objective(o,x,roles)
 else:
  result=objectives().evaluate_objective(o,x,roles)
  assert torch.isfinite(result['raw']).all() and result['scalar'].item()>0
  assert result['metadata']['denominatorConvention']=='diagnosticActualWeightFloat64Total'
  assert result['metadata']['denominator']==pytest.approx(float(o.class_weights[0].item())*2)
  json.dumps(result['metadata'],allow_nan=False)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('ident',IDS)
def test_successful_objective_metadata_is_strict_finite_json(device,ident):
 import json
 o=objectives().create_objective(recipe(ident),device);x=torch.tensor([[.2,.4,-.1],[.3,-.2,.7]],device=device)
 if ident in ('loss_nll','loss_kl_div'):x=x.log_softmax(-1)
 if ident in ('loss_cross_entropy','loss_nll'):roles={'target':torch.tensor([1,2],device=device)}
 elif ident=='loss_ctc':
  x=x.reshape(1,2,3).log_softmax(-1);roles={'target':torch.tensor([[1]],device=device),'input_lengths':torch.tensor([2],device=device),'target_lengths':torch.tensor([1],device=device)}
 elif ident=='loss_triplet_margin':x=x.reshape(1,2,3).transpose(1,2);roles={}
 else:roles={'target':torch.tensor([[.2,.3,.5],[.4,.4,.2]],device=device)}
 m=objectives().evaluate_objective(o,x,roles)['metadata'];assert json.loads(json.dumps(m,allow_nan=False))==m

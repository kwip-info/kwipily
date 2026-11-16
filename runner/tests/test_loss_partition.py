"""Independent partition aggregation and before-mutation mean admission."""
import importlib
import pytest,torch
from runner.loss_configuration import validate_recipe
from runner.loss_objectives import create_objective,evaluate_objective
from runner.tensor_layout import BatchedLayout

def helper():return importlib.import_module('runner.loss_partition')
def recipe(op,c=None):return validate_recipe(dict(schemaVersion=1,componentID=op,configurationVersion=1,config=c or {}))
@pytest.mark.parametrize('device',['cpu','mps'])
def test_ignored_chunk_does_not_invent_mean_and_weighted_global_mean_matches_native(device):
 r=recipe('loss_cross_entropy',dict(class_weights=[.5,2.,1.],label_smoothing=.3));o=create_objective(r,device)
 x=torch.tensor([[.2,.7,-.1],[.5,-.3,.8],[-.2,.6,.4]],device=device,requires_grad=True);y=torch.tensor([-100,-100,1],device=device)
 with pytest.raises(ValueError,match='denominator'):evaluate_objective(o,x[:2],{'target':y[:2]})
 h=helper();parts=[h.objective_partition_contribution(o,x[:2],{'target':y[:2]}),h.objective_partition_contribution(o,x[2:],{'target':y[2:]})]
 assert parts[0]['denominator']==0 and parts[0]['numerator'].item()==0 and parts[0]['recipe']==r
 result=h.finish_objective_partition(parts,r);native=evaluate_objective(o,x,{'target':y})
 assert result['scalar'].item()==pytest.approx(native['scalar'].item(),abs=2e-7)
 assert torch.allclose(torch.autograd.grad(result['scalar'],x,retain_graph=True)[0],torch.autograd.grad(native['scalar'],x)[0],atol=2e-7)
 assert result['metadata']['ignoredCount']==2 and result['metadata']['classification']['eligibleCount']==1

def test_global_undefined_mean_and_training_preflight_refuse_before_caller_state_changes():
 h=helper();r=recipe('loss_nll',dict(class_weights=[0.,1.,2.]));o=create_objective(r)
 x=torch.tensor([[.2,.4,.7],[.5,.1,-.2]]).log_softmax(-1);y=torch.tensor([0,-100])
 parts=[h.objective_partition_contribution(o,x,{'target':y})]
 with pytest.raises(ValueError,match='denominator'):h.finish_objective_partition(parts,r)
 state={'updates':0}
 with pytest.raises(ValueError,match='batch'):
  h.preflight_objective_training_batches(r,{'target':y},1,BatchedLayout((3,)))
  state['updates']+=1
 assert state['updates']==0

from runner.tests.test_loss_objectives import logs,scalar_rows,triplet_oracle,ctc_oracle
IDS=('loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')
@pytest.fixture(autouse=True)
def preserve_device_determinism():
 old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 yield
 torch.use_deterministic_algorithms(old,warn_only=warn)

def case(op,device):
 rows=[[.2,.7,-.1],[.5,-.3,.8],[-.2,.6,.4],[.3,.1,.9],[.7,-.4,.2]];c={};targets=[1,2,-100,0,1] if op in ('loss_cross_entropy','loss_nll') else [[.2,.3,.5],[.6,.1,.3],[.4,.4,.2],[.1,.3,.6],[.3,.2,.5]]
 if op in ('loss_cross_entropy','loss_nll'):c['class_weights']=[.5,2.,1.]
 if op=='loss_cross_entropy':c['label_smoothing']=.3
 if op in ('loss_nll','loss_kl_div'):rows=[logs(row) for row in rows]
 if op=='loss_bce_with_logits':c=dict(weight_mode='configured',weight_shape=[3],weight_values=[.5,0.,2.],pos_weight_mode='configured',pos_weight_shape=[],pos_weight_values=[1.7])
 if op=='loss_triplet_margin':
  rows=[[[.2+i*.1,-.5],[.7,.3+i*.05],[.8+i*.03,.2]] for i in range(5)];targets=None;c=dict(p=1.5,eps=.02,margin=1.3,swap=True)
 if op=='loss_ctc':
  rows=[[[.2+i*.02,.5,-.3],[.7,-.4,.1],[.1,.3,.8],[.6,.2,-.5]] for i in range(5)];rows=[[logs(r) for r in row] for row in rows];targets=[[1,1],[2,888],[999,-999],[1,2],[2,2]]
 x=torch.tensor(rows,device=device,requires_grad=True);roles={}
 if targets is not None:roles['target']=torch.tensor(targets,device=device,dtype=torch.int64 if op in ('loss_cross_entropy','loss_nll','loss_ctc') else torch.float32)
 if op=='loss_ctc':roles.update(input_lengths=torch.tensor([4,3,2,3,4],device=device),target_lengths=torch.tensor([2,1,0,2,2],device=device))
 return rows,targets,c,x,roles

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',IDS)
@pytest.mark.parametrize('reduction',['none','mean','sum'])
@pytest.mark.parametrize('scalarization',['mean','sum'])
def test_every_objective_chunked_partition_raw_values_influence_and_independent_reference(device,op,reduction,scalarization):
 torch.use_deterministic_algorithms(device=='cpu');rows,targets,c,x,roles=case(op,device);c['reduction']=reduction;r=recipe(op,c);o=create_objective(r,device);h=helper()
 parts=[h.objective_partition_contribution(o,x[i:i+2],{k:v[i:i+2] for k,v in roles.items()},scalarization) for i in range(0,5,2)]
 result=h.finish_objective_partition(parts,r,scalarization);native=evaluate_objective(o,x,roles,scalarization)
 assert result['raw'].shape==native['raw'].shape
 assert torch.allclose(result['raw'],native['raw'],atol=2e-6,rtol=3e-5)
 assert torch.allclose(torch.autograd.grad(result['scalar'],x,retain_graph=True)[0],torch.autograd.grad(native['scalar'],x)[0],atol=2e-6,rtol=3e-5)
 canonical={**r['config']}
 if op=='loss_triplet_margin':expected=triplet_oracle(rows,canonical)
 elif op=='loss_ctc':
  values=[ctc_oracle(row[:t],target[:s],0) for row,target,t,s in zip(rows,targets,[4,3,2,3,4],[2,1,0,2,2])]
  expected=values if reduction=='none' else sum(values) if reduction=='sum' else sum(v/max(s,1) for v,s in zip(values,[2,1,0,2,2]))/5
 else:
  if op=='loss_bce_with_logits':canonical.update(_weight=[.5,0.,2.],_pos_weight=[1.7]*3)
  expected=scalar_rows(op,rows,targets,canonical)
 from runner.tests.test_loss_objectives import flat
 assert result['raw'].detach().cpu().flatten().tolist()==pytest.approx(flat(expected),rel=3e-5,abs=2e-6)
 expected_scalar=(sum(flat(expected))/len(flat(expected)) if scalarization=='mean' else sum(flat(expected))) if reduction=='none' else expected
 assert result['scalar'].item()==pytest.approx(expected_scalar,rel=3e-5,abs=2e-6)
 assert result['metadata']['observations']==5 and o.recipe==r
 assert ('classification' in result['metadata'])==(op in ('loss_cross_entropy','loss_nll'))

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('log_target',[False,True])
def test_kl_batchmean_denominator_is_total_observations(device,log_target):
 rows,targets,c,x,roles=case('loss_kl_div',device)
 if log_target:roles['target']=roles['target'].log();targets=[[__import__('math').log(v) for v in row] for row in targets]
 r=recipe('loss_kl_div',dict(reduction='batchmean',log_target=log_target));o=create_objective(r,device);h=helper()
 result=h.finish_objective_partition([h.objective_partition_contribution(o,x[:4],{'target':roles['target'][:4]}),h.objective_partition_contribution(o,x[4:],{'target':roles['target'][4:]})],r)
 assert result['metadata']['aggregationDenominator']==5
 assert result['scalar'].item()==pytest.approx(scalar_rows('loss_kl_div',rows,targets,r['config']),abs=2e-6)

def test_training_preflight_checks_short_final_batch_not_just_whole_partition():
 h=helper();r=recipe('loss_cross_entropy');roles={'target':torch.tensor([0,1,-100])}
 with pytest.raises(ValueError,match='batch 2'):h.preflight_objective_training_batches(r,roles,2,BatchedLayout((3,)))
 assert h.preflight_objective_training_batches(r,roles,3,BatchedLayout((3,)))['batches']==1
 assert h.preflight_objective_training_batches(recipe('loss_cross_entropy',dict(reduction='sum')),roles,2,BatchedLayout((3,)))['batches']==2
 assert h.preflight_objective_training_batches(recipe('loss_triplet_margin'),{},2,BatchedLayout((3,2)),training_rows=3)['lastBatchSize']==1
 with pytest.raises(ValueError):h.preflight_objective_training_batches(recipe('loss_triplet_margin'),{},2,BatchedLayout((3,2)))

def test_partition_refuses_mixed_identity_scalarization_or_geometry_and_live_buffer_tamper():
 h=helper();r=recipe('loss_mse',dict(reduction='none'));o=create_objective(r);x=torch.tensor([[.2,.7]],requires_grad=True);roles={'target':torch.tensor([[.3,.4]])};p=h.objective_partition_contribution(o,x,roles)
 for expected,scalarization in [(recipe('loss_l1',dict(reduction='none')),'mean'),(r,'sum')]:
  with pytest.raises(ValueError):h.finish_objective_partition([p],expected,scalarization)
 with pytest.raises(ValueError):h.finish_objective_partition([],r)
 with pytest.raises(ValueError):h.finish_objective_partition([{'recipe':r}],r)
 other=h.objective_partition_contribution(o,torch.tensor([[.2,.7,.1]]),{'target':torch.tensor([[.3,.4,.2]])})
 with pytest.raises(ValueError):h.finish_objective_partition([p,other],r)
 weighted=create_objective(recipe('loss_cross_entropy',dict(class_weights=[.5,2.,1.])));weighted.class_weights[0]=.7
 with pytest.raises(ValueError):h.objective_partition_contribution(weighted,torch.tensor([[.2,.7,.1]]),{'target':torch.tensor([1])})

@pytest.mark.parametrize('device',['cpu','mps'])
def test_probability_ce_does_not_invent_classification_accuracy(device):
 x=torch.tensor([[.2,.7,-.1]],device=device);y=torch.tensor([[.2,.3,.5]],device=device);r=recipe('loss_cross_entropy',dict(target_mode='probabilities'));h=helper()
 result=h.finish_objective_partition([h.objective_partition_contribution(create_objective(r,device),x,{'target':y})],r)
 assert 'classification' not in result['metadata']

@pytest.mark.parametrize('op,c,shape,roles',[
 ('loss_cross_entropy',{'reduction':'sum'},(3,),{'target':torch.tensor([-1])}),
 ('loss_cross_entropy',{'target_mode':'probabilities'},(3,),{'target':torch.tensor([[.2,.3,.8]])}),
 ('loss_bce_with_logits',{},(2,),{'target':torch.tensor([[.2,1.1]])}),
 ('loss_bce_with_logits',{'weight_mode':'mapped'},(2,),{'target':torch.tensor([[.2,.7]]),'weight':torch.tensor([-1.])}),
 ('loss_kl_div',{},(3,),{'target':torch.tensor([[.2,.3,.8]])}),
 ('loss_ctc',{},(3,3),{'target':torch.tensor([[0]]),'input_lengths':torch.tensor([3]),'target_lengths':torch.tensor([1])}),
 ('loss_ctc',{},(3,3),{'target':torch.tensor([[1]]),'input_lengths':torch.tensor([4]),'target_lengths':torch.tensor([1])})])
def test_preflight_invalid_actual_target_values_refuses_before_caller_forward(op,c,shape,roles):
 state={'forwards':0}
 with pytest.raises(ValueError):
  helper().preflight_objective_training_batches(recipe(op,c),roles,2,BatchedLayout(shape))
  state['forwards']+=1
 assert state['forwards']==0

@pytest.mark.parametrize('device',['cpu','mps'])
def test_partition_mean_refuses_combined_native_denominator_overflow_but_none_sum_survive(device):
 x=torch.tensor([[10.,0.],[0.,10.]],device=device);y=torch.tensor([0,1],device=device);h=helper()
 for reduction in ('mean','none','sum'):
  r=recipe('loss_cross_entropy',dict(reduction=reduction,class_weights=[3e38,3e38]));o=create_objective(r,device)
  parts=[h.objective_partition_contribution(o,x[i:i+1],{'target':y[i:i+1]}) for i in range(2)]
  if reduction=='mean':
   with pytest.raises(ValueError,match='denominator|overflow'):h.finish_objective_partition(parts,r)
   with pytest.raises(ValueError,match='denominator|overflow'):h.preflight_objective_training_batches(r,{'target':y},2,BatchedLayout((2,)))
  else:
   result=h.finish_objective_partition(parts,r);assert torch.isfinite(result['raw']).all() and result['scalar'].item()>0

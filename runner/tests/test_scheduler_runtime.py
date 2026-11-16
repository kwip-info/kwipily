"""Independent scheduler controls, scalar rates, timing and native lifecycle proof."""
import copy,importlib,math
import pytest,torch
IDS=['scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr']
EXPECTED={
 'scheduler_step_lr':dict(step_size=2,gamma=.1),
 'scheduler_cosine_annealing_lr':dict(T_max=10,eta_min=0.),
 'scheduler_reduce_on_plateau':dict(mode='min',factor=.1,patience=10,threshold=.0001,threshold_mode='rel',cooldown=0,min_lr=0.,eps=1e-8,metric='validation_loss'),
 'scheduler_one_cycle_lr':dict(max_lr=.1,planned_total_batches=10,pct_start=.3,anneal_strategy='cos',cycle_momentum=True,base_momentum=.85,max_momentum=.95,div_factor=25.,final_div_factor=10000.,three_phase=False),
 'scheduler_linear_lr':dict(start_factor=1/3,end_factor=1.,total_iters=5)}
def helper():return importlib.import_module('runner.scheduler_configuration')
def recipe(op,config=None):return dict(schemaVersion=1,componentID=op,configurationVersion=1,config=EXPECTED[op] if config is None else config)
@pytest.mark.parametrize('op',IDS)
def test_canonical_scheduler_defaults_and_exact_portable_envelope(op):
 h=helper();assert h.canonical_configuration(op,{})==EXPECTED[op];assert h.validate_recipe(recipe(op))==recipe(op)
 for field in ['schemaVersion','configurationVersion']:
  bad=recipe(op);bad[field]=True
  with pytest.raises(ValueError):h.validate_recipe(bad)
 bad=recipe(op);bad['foreign']=0
 with pytest.raises(ValueError):h.validate_recipe(bad)
@pytest.mark.parametrize('op,key,value',[
 ('scheduler_step_lr','step_size',True),('scheduler_step_lr','step_size',0),('scheduler_step_lr','step_size',10001),('scheduler_step_lr','gamma',-1),('scheduler_step_lr','gamma',float('inf')),
 ('scheduler_cosine_annealing_lr','T_max',1.5),('scheduler_cosine_annealing_lr','eta_min',-1),
 ('scheduler_linear_lr','start_factor',0),('scheduler_linear_lr','end_factor',1.001),('scheduler_linear_lr','total_iters',0),
 ('scheduler_reduce_on_plateau','factor',1),('scheduler_reduce_on_plateau','patience',-1),('scheduler_reduce_on_plateau','cooldown',10001),('scheduler_reduce_on_plateau','mode','auto'),('scheduler_reduce_on_plateau','metric','training_loss'),('scheduler_reduce_on_plateau','threshold_mode','relative'),
 ('scheduler_one_cycle_lr','planned_total_batches',2),('scheduler_one_cycle_lr','planned_total_batches',10001),('scheduler_one_cycle_lr','pct_start',0),('scheduler_one_cycle_lr','pct_start',1),('scheduler_one_cycle_lr','div_factor',0),('scheduler_one_cycle_lr','final_div_factor',0),('scheduler_one_cycle_lr','cycle_momentum',1),('scheduler_one_cycle_lr','anneal_strategy','quadratic'),('scheduler_one_cycle_lr','base_momentum',-1),('scheduler_one_cycle_lr','max_momentum',1.01),
])
def test_invalid_scheduler_controls_refuse(op,key,value):
 with pytest.raises(ValueError):helper().canonical_configuration(op,{key:value})
@pytest.mark.parametrize('op',IDS)
def test_scheduler_unknown_controls_refuse_without_mutation(op):
 supplied={'last_epoch':1};saved=copy.deepcopy(supplied)
 with pytest.raises(ValueError):helper().canonical_configuration(op,supplied)
 assert supplied==saved
@pytest.mark.parametrize('count,pct,three',[(3,1/3,False),(10,.1,False),(6,.6,True)])
def test_one_cycle_refuses_actual_zero_or_negative_phase_spans(count,pct,three):
 with pytest.raises(ValueError):helper().canonical_configuration('scheduler_one_cycle_lr',dict(planned_total_batches=count,pct_start=pct,three_phase=three))

def scalar_rates(op,c,base,ticks):
 result=[];lr=base
 for t in range(ticks+1):
  if op=='scheduler_step_lr':lr=base*c['gamma']**(t//c['step_size'])
  elif op=='scheduler_cosine_annealing_lr':lr=c['eta_min']+(base-c['eta_min'])*(1+math.cos(math.pi*t/c['T_max']))/2
  elif op=='scheduler_linear_lr':lr=base*(c['start_factor']+(c['end_factor']-c['start_factor'])*min(t,c['total_iters'])/c['total_iters'])
  elif op=='scheduler_one_cycle_lr':
   n=c['planned_total_batches'];initial=c['max_lr']/c['div_factor'];minimum=initial/c['final_div_factor'];ends=[c['pct_start']*n-1,n-1];pairs=[(initial,c['max_lr']),(c['max_lr'],minimum)]
   if c['three_phase']:ends=[ends[0],2*c['pct_start']*n-2,n-1];pairs=[(initial,c['max_lr']),(c['max_lr'],initial),(initial,minimum)]
   start=0
   for i,end in enumerate(ends):
    if t<=end or i==len(ends)-1:
     f=(t-start)/(end-start);a,b=pairs[i];lr=b+(a-b)*(1+math.cos(math.pi*f))/2 if c['anneal_strategy']=='cos' else a+(b-a)*f;break
    start=end
  result.append(lr)
 return result
@pytest.mark.parametrize('op', ['scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_linear_lr','scheduler_one_cycle_lr'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_native_scheduler_rates_and_scalar_learned_updates(op,device):
 from runner.optimizer_configuration import canonical_configuration
 h=helper();r=recipe(op);opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5}));p=torch.nn.Parameter(torch.tensor([1.,-2.],device=device));o=torch.optim.SGD([p],lr=.1,momentum=.5,foreach=False,differentiable=False,fused=False);s=h.create_scheduler(o,r,opt);c=r['config'];rates=scalar_rates(op,c,.1,5);wanted=[1.,-2.];memory=None
 for tick in range(5):
  assert o.param_groups[0]['lr']==pytest.approx(rates[tick],rel=1e-12,abs=1e-14)
  gradient=[.2*(tick+1),-.15*(tick+1)];momentum=o.param_groups[0]['momentum'];memory=gradient if memory is None else [momentum*m+g for m,g in zip(memory,gradient)];wanted=[v-rates[tick]*m for v,m in zip(wanted,memory)];p.grad=torch.tensor(gradient,device=device);o.step();s.step();assert o.param_groups[0]['lr']==pytest.approx(rates[tick+1],rel=1e-12,abs=1e-14);torch.testing.assert_close(p.detach().cpu(),torch.tensor(wanted),atol=2e-6,rtol=2e-6)
@pytest.mark.parametrize('op',IDS)
def test_scheduler_requires_explicit_optimizer_and_one_group(op):
 h=helper();p=torch.nn.Parameter(torch.ones(2));o=torch.optim.SGD([p],lr=.1)
 with pytest.raises(ValueError):h.create_scheduler(o,recipe(op),None)
@pytest.mark.parametrize('op',IDS)
def test_real_graph_scheduler_trace_and_state_checkpoint(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 from runner.optimizer_configuration import canonical_configuration
 opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5}))
 run(graph(),tmp_path/'run',epochs=3,device='cpu',sample_limit=32,batch_size=10,optimizer_recipe=opt,scheduler_recipe=recipe(op))
 import json
 m=json.loads((tmp_path/'run/manifest.json').read_text());p=torch.load(tmp_path/'run/checkpoint.pt',weights_only=True);trace=json.loads((tmp_path/'run/rate-trace.json').read_text());assert m['settings']['schedulerRecipe']==recipe(op);assert p['scheduler']['rateTrace']==trace;assert len(trace['records'])==(9 if op=='scheduler_one_cycle_lr' else 3)
@pytest.mark.parametrize('op',IDS)
def test_actual_cpu_resume_exact_and_finetune_resets_scheduler(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 from runner.optimizer_configuration import canonical_configuration
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5}));kw=dict(device='cpu',sample_limit=32,batch_size=10,optimizer_recipe=opt,scheduler_recipe=recipe(op));g=graph()
 run(g,tmp_path/'full',epochs=3,**kw);run(g,tmp_path/'first',epochs=1,**kw);run(g,tmp_path/'resumed',epochs=3,resume_checkpoint=tmp_path/'first/checkpoint.pt',**kw);run(g,tmp_path/'fine',epochs=3,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**kw)
 a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
 for key in ['modelState','optimizerState','scheduler','optimizerUpdates','rngState']:assert_nested_equal(a[key],b[key])
 import json
 assert json.loads((tmp_path/'full/metrics.json').read_text())==json.loads((tmp_path/'resumed/metrics.json').read_text());fine=json.loads((tmp_path/'fine/manifest.json').read_text());assert 'scheduler' in fine['lineage']['resetState'];assert fine['settings']['initialEpoch']==0
@pytest.mark.parametrize('op',IDS)
def test_rehashed_scheduler_state_and_trace_tampering_refuses_before_model_use(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 from runner.optimizer_configuration import canonical_configuration
 import json,hashlib
 opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5}));kw=dict(device='cpu',sample_limit=32,batch_size=10,optimizer_recipe=opt,scheduler_recipe=recipe(op));run(graph(),tmp_path/'first',epochs=1,**kw);path=tmp_path/'first/checkpoint.pt';state=torch.load(path,weights_only=True);state['scheduler']['state']['foreign']=True;torch.save(state,path);m=json.loads((path.parent/'manifest.json').read_text());m['checkpoint']['bytes']=path.stat().st_size;m['checkpoint']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(path.parent/'manifest.json').write_text(json.dumps(m))
 with pytest.raises(ValueError,match='scheduler state'):run(graph(),tmp_path/'refused',epochs=3,resume_checkpoint=path,**kw)
@pytest.mark.parametrize('op',IDS)
def test_standalone_exported_fresh_training_sequence_and_timing(op,tmp_path):
 from runner.optimizer_configuration import canonical_configuration,training_recipe_code
 opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5}));r=recipe(op);code=training_recipe_code(opt,r);assert len(code.encode())<=8192;scope={};exec(code,scope);model=torch.nn.Linear(2,1,bias=False);model.weight.data.fill_(1);o=scope['create_optimizer'](model);s=scope['create_scheduler'](o);rates=scalar_rates(op,r['config'],.1,5) if op!='scheduler_reduce_on_plateau' else [.1]*6;wanted=[1.,1.];memory=None
 for tick in range(5):
  assert o.param_groups[0]['lr']==pytest.approx(rates[tick],rel=1e-12);momentum=o.param_groups[0]['momentum'];grad=[.1*(tick+1),-.2*(tick+1)];memory=grad if memory is None else [momentum*m+g for m,g in zip(memory,grad)];wanted=[v-rates[tick]*m for v,m in zip(wanted,memory)];loss=(model.weight*torch.tensor([grad])).sum();scope['training_step'](loss,o,s)
  if op!='scheduler_one_cycle_lr':scope['completed_epoch_step'](s,validation_loss=1.,validation_accuracy=.5)
  torch.testing.assert_close(model.weight.flatten(),torch.tensor(wanted),rtol=2e-6,atol=2e-6);assert o.param_groups[0]['lr']==pytest.approx(rates[tick+1],rel=1e-12)
 if op=='scheduler_reduce_on_plateau':
  with pytest.raises(ValueError):scope['completed_epoch_step'](s)
@pytest.mark.parametrize('mode,metric,threshold_mode', [('min','validation_loss','rel'),('min','validation_accuracy','abs'),('max','validation_accuracy','rel'),('max','validation_loss','abs')])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_plateau_actual_metric_improvement_cooldown_floor_epsilon_state(mode,metric,threshold_mode,device):
 from runner.optimizer_configuration import canonical_configuration,create_optimizer
 h=helper();c={**EXPECTED['scheduler_reduce_on_plateau'],'mode':mode,'metric':metric,'threshold_mode':threshold_mode,'factor':.5,'patience':0,'cooldown':1,'min_lr':.02,'eps':.005,'threshold':.1};r=recipe('scheduler_reduce_on_plateau',c);opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}));p=torch.nn.Parameter(torch.ones(2,device=device));o=create_optimizer([p],opt);s=h.SchedulerController(o,r,opt,1);values=[1.,1.,1.,1.,1.,.5,.5,.5,.5] if mode=='min' else [1.,1.,1.,1.,1.,2.,2.,2.,2.];rate=.1;best=None;bad=0;cooldown=0
 for epoch,value in enumerate(values,1):
  old=rate;s.begin_epoch();p.grad=torch.ones_like(p);s.before_update([0]);o.step()
  good=best is None or (value<best*(1-.1) if mode=='min' and threshold_mode=='rel' else value>best*(1+.1) if threshold_mode=='rel' else value<best-.1 if mode=='min' else value>best+.1)
  if good:best=value;bad=0
  else:bad+=1
  if cooldown:cooldown-=1;bad=0
  if bad>0:
   next_rate=max(rate*.5,.02)
   if rate-next_rate>.005:rate=next_rate
   cooldown=1;bad=0
  row=s.tick(epoch,metric_value=value);assert row['usedRate']==old and row['nextRate']==rate;assert s.native.best==best and s.native.num_bad_epochs==bad and s.native.cooldown_counter==cooldown
 for missing in [None,True,float('nan'),float('inf')]:
  s.begin_epoch()
  with pytest.raises(ValueError):s.tick(10,metric_value=missing)
@pytest.mark.parametrize('strategy',['cos','linear'])
@pytest.mark.parametrize('three',[False,True])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_one_cycle_full_horizon_both_curves_phases_and_unused_terminal(strategy,three,device):
 from runner.optimizer_configuration import canonical_configuration,create_optimizer
 h=helper();c={**EXPECTED['scheduler_one_cycle_lr'],'anneal_strategy':strategy,'three_phase':three,'planned_total_batches':10,'pct_start':.3,'cycle_momentum':False};r=recipe('scheduler_one_cycle_lr',c);opt=dict(schemaVersion=1,componentID='optimizer_adagrad',configurationVersion=1,config=canonical_configuration('optimizer_adagrad',{'lr':.1}));p=torch.nn.Parameter(torch.ones(2,device=device));o=create_optimizer([p],opt);s=h.SchedulerController(o,r,opt,1);rates=scalar_rates(r['componentID'],c,.1,10)
 for tick in range(10):s.before_update([0]);p.grad=torch.ones_like(p);o.step();row=s.tick(1,tick+1,[0]);assert row['usedRate']==pytest.approx(rates[tick],abs=1e-14);assert row['nextRate']==pytest.approx(rates[tick+1],abs=1e-14);assert row['nextRateUnused']==(tick==9)
 if strategy=='linear':assert row['nextRate']<0
 with pytest.raises(ValueError,match='exhausted'):s.before_update([0])
 fresh=h.SchedulerController(create_optimizer([torch.nn.Parameter(torch.ones(2))],opt),r,opt,1);fresh.restore(s.checkpoint(),[10]);assert fresh.checkpoint()==s.checkpoint()
@pytest.mark.parametrize('op',IDS)
def test_actual_mps_graph_training_finetune_and_exact_resume_refusal(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 from runner.optimizer_configuration import canonical_configuration
 opt=dict(schemaVersion=1,componentID='optimizer_adamw',configurationVersion=1,config=canonical_configuration('optimizer_adamw',{'lr':.02}));kw=dict(device='mps',sample_limit=32,batch_size=10,optimizer_recipe=opt,scheduler_recipe=recipe(op));run(graph(),tmp_path/'first',epochs=1,**kw)
 with pytest.raises(ValueError,match='CPU'):run(graph(),tmp_path/'refused',epochs=2,resume_checkpoint=tmp_path/'first/checkpoint.pt',**kw)
 run(graph(),tmp_path/'fine',epochs=2,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**kw)
 import json
 m=json.loads((tmp_path/'fine/manifest.json').read_text());assert m['lineage']['guarantee']=='weightsOnly' and 'scheduler' in m['lineage']['resetState'];assert m['settings']['initialEpoch']==0

def test_one_cycle_exact_sgd_buffer_history_with_zero_momentum_and_missing_gradients():
 from runner.optimizer_configuration import canonical_configuration,create_optimizer,validate_optimizer_state,restore_optimizer
 h=helper();opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}));r=recipe('scheduler_one_cycle_lr',{**EXPECTED['scheduler_one_cycle_lr'],'anneal_strategy':'linear','base_momentum':0.,'max_momentum':.5});params=[torch.nn.Parameter(torch.ones(2)),torch.nn.Parameter(torch.ones(1))];o=create_optimizer(params,opt);s=h.SchedulerController(o,r,opt,2);counts=[0,0]
 for tick in range(4):
  o.zero_grad();indices=[1] if tick==0 else [0] if tick==2 else [];s.before_update(indices)
  for index in indices:params[index].grad=torch.ones_like(params[index]);counts[index]+=1
  o.step();validate_optimizer_state(o,o.state_dict(),opt,counts,s.context());s.tick(1,tick+1,indices)
 assert set(o.state_dict()['state'])=={1} and counts==[1,1]
 other=create_optimizer([torch.nn.Parameter(torch.ones(2)),torch.nn.Parameter(torch.ones(1))],opt);restored=h.SchedulerController(other,r,opt,2);restored.restore(s.checkpoint(),counts);restore_optimizer(other,o.state_dict(),opt,counts,restored.context());assert set(other.state_dict()['state'])=={1}
 missing=copy.deepcopy(o.state_dict());missing['state']={}
 with pytest.raises(ValueError):validate_optimizer_state(other,missing,opt,counts,restored.context())
@pytest.mark.parametrize('op',IDS)
@pytest.mark.parametrize('field',['lr','weight_decay'])
def test_scheduler_context_never_permits_undeclared_optimizer_control_changes(op,field):
 from runner.optimizer_configuration import canonical_configuration,create_optimizer,validate_optimizer_state
 h=helper();opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5}));o=create_optimizer([torch.nn.Parameter(torch.ones(2))],opt);s=h.SchedulerController(o,recipe(op),opt,1);o.param_groups[0][field]=.123
 with pytest.raises(ValueError):validate_optimizer_state(o,o.state_dict(),opt,[0],s.context())
@pytest.mark.parametrize('op,values',[('scheduler_step_lr',{'gamma':0.,'step_size':1}),('scheduler_step_lr',{'gamma':1.,'step_size':3}),('scheduler_linear_lr',{'start_factor':1.,'end_factor':.1,'total_iters':1}),('scheduler_linear_lr',{'start_factor':.5,'end_factor':.5,'total_iters':2}),('scheduler_cosine_annealing_lr',{'T_max':1,'eta_min':.02})])
def test_epoch_curve_numeric_boundaries_and_direction(op,values):
 from runner.optimizer_configuration import canonical_configuration,create_optimizer
 h=helper();r=recipe(op,{**EXPECTED[op],**values});opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}));o=create_optimizer([torch.nn.Parameter(torch.ones(2))],opt);s=h.SchedulerController(o,r,opt,1);ticks=1 if op=='scheduler_cosine_annealing_lr' else 4;rates=scalar_rates(op,r['config'],.1,ticks)
 for tick in range(ticks):s.begin_epoch();s.before_update([]);o.step();row=s.tick(tick+1);assert row['usedRate']==pytest.approx(rates[tick],abs=1e-15) and row['nextRate']==pytest.approx(rates[tick+1],abs=1e-15)

def test_horizon_compatibility_and_trace_allocation_guards_before_training(tmp_path):
 from runner.optimizer_configuration import canonical_configuration
 h=helper();opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}))
 for op,epochs,batches,count in [('scheduler_one_cycle_lr',4,3,1),('scheduler_cosine_annealing_lr',11,1,1),('scheduler_linear_lr',10001,1,1),('scheduler_one_cycle_lr',1,3,1000000)]:
  with pytest.raises(ValueError):h.preflight(recipe(op),opt,epochs,batches,count)
 h.preflight(recipe('scheduler_linear_lr'),opt,10000,1,1)
 for op,c in [('scheduler_step_lr',{'gamma':3.4028234663852886e38,'step_size':1}),('scheduler_cosine_annealing_lr',{'eta_min':.2})]:
  with pytest.raises(ValueError):h.preflight(recipe(op,{**EXPECTED[op],**c}),opt,3,1,1)
 for id,controls in [('optimizer_adagrad',{}),('optimizer_adam',{})]:
  o=dict(schemaVersion=1,componentID=id,configurationVersion=1,config=canonical_configuration(id,controls));r=recipe('scheduler_one_cycle_lr',{**EXPECTED['scheduler_one_cycle_lr'],'max_momentum':1.})
  with pytest.raises(ValueError):h.validate_compatibility(r,o)
@pytest.mark.parametrize('op',IDS)
@pytest.mark.parametrize('optimizer_id',['optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_all_scheduler_optimizer_pairs_weighted_updates_and_exact_fixed_option_context(op,optimizer_id,device):
 from runner.optimizer_configuration import canonical_configuration,create_optimizer,validate_optimizer_state
 from runner.tests.test_optimizer_runtime import scalar_update
 h=helper();controls={'lr':.1};controls.update({'momentum':.5,'nesterov':True} if optimizer_id=='optimizer_sgd' else {'momentum':.5,'centered':True} if optimizer_id=='optimizer_rmsprop' else {'amsgrad':True,'weight_decay':.02} if optimizer_id in ('optimizer_adam','optimizer_adamw') else {'initial_accumulator_value':.2,'lr_decay':.1});opt=dict(schemaVersion=1,componentID=optimizer_id,configurationVersion=1,config=canonical_configuration(optimizer_id,controls));c=dict(EXPECTED[op])
 if op=='scheduler_one_cycle_lr' and optimizer_id=='optimizer_adagrad':c['cycle_momentum']=False
 r=recipe(op,c);p=torch.nn.Parameter(torch.tensor([1.,-2.],device=device));o=create_optimizer([p],opt);s=h.SchedulerController(o,r,opt,1);wanted=[1.,-2.];memory={};count=0
 assert s.trace()['initialization']['tick']==0
 for tick in range(4):
  gradient=[.1*(tick+1),-.3/(tick+1)];effective=dict(opt['config']);effective['lr']=o.param_groups[0]['lr']
  if op=='scheduler_one_cycle_lr' and c['cycle_momentum']:
   if optimizer_id in ('optimizer_adam','optimizer_adamw'):effective['beta1']=o.param_groups[0]['betas'][0]
   else:effective['momentum']=o.param_groups[0]['momentum']
  wanted,memory=scalar_update(optimizer_id,effective,wanted,gradient,memory);validate_optimizer_state(o,o.state_dict(),opt,[count],s.context());s.begin_epoch();s.before_update([0]);p.grad=torch.tensor(gradient,device=device);o.step();count+=1;validate_optimizer_state(o,o.state_dict(),opt,[count],s.context());row=s.tick(tick+1,1,[0],1.);torch.testing.assert_close(p.detach().cpu(),torch.tensor(wanted),atol=3e-5,rtol=3e-5)
 saved=s.checkpoint();fresh=create_optimizer([torch.nn.Parameter(torch.ones(2))],opt);controller=h.SchedulerController(fresh,r,opt,1);controller.restore(saved,[count]);assert controller.checkpoint()==saved
@pytest.mark.parametrize('op',IDS)
def test_actual_authored_generated_and_worker_scheduler_descriptors(op):
 import json,sys
 from runner.engine import REGISTRY,capabilities
 h=helper();value=json.loads(REGISTRY.read_text());row=next(v for v in value['components'] if v['id']==op)
 from pathlib import Path
 authored=json.loads((Path(__file__).resolve().parents[2]/'components/schedulers/bundle.json').read_text());descriptor=next(v['runtimeContract'] for v in authored['components'] if v['definition']['id']==op);assert descriptor==h.capability(op)
 # Shipped generated tables deliberately use the standalone worker import path.
 from pathlib import Path
 runner_path=str(Path(__file__).resolve().parents[1]);sys.path.insert(0,runner_path)
 try:
  table=importlib.import_module('runner.scheduler_adapters_generated');assert table.CAPABILITIES[op]==h.capability(op);assert table.NORMALIZERS[op]({})==EXPECTED[op];assert not hasattr(table,'RESOLVERS')
 finally:sys.path.remove(runner_path)
 assert capabilities()['componentCapabilities'][op]==h.capability(op)
@pytest.mark.parametrize('mutation',['unknown','tick_boolean','rate','initialization','foreign_group','missing_buffer','ownership_counts','epoch','batch'])
def test_strict_one_cycle_state_trace_options_and_ownership_tamper_before_restore(mutation):
 from runner.optimizer_configuration import canonical_configuration,create_optimizer
 h=helper();opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}));r=recipe('scheduler_one_cycle_lr');p=torch.nn.Parameter(torch.ones(2));o=create_optimizer([p],opt);s=h.SchedulerController(o,r,opt,1)
 for tick in range(3):s.before_update([0]);p.grad=torch.ones_like(p);o.step();s.tick(1,tick+1,[0])
 payload=copy.deepcopy(s.checkpoint());counts=[3]
 if mutation=='unknown':payload['state']['unknown']=0
 elif mutation=='tick_boolean':payload['rateTrace']['records'][0]['tick']=True
 elif mutation=='rate':payload['rateTrace']['records'][0]['nextRate']+=.01
 elif mutation=='initialization':payload['rateTrace']['initialization']['tick']=True
 elif mutation=='ownership_counts':counts=[2]
 elif mutation in ('epoch','batch'):payload['rateTrace']['records'][0][mutation]=False
 elif mutation=='foreign_group':payload['state']['base_lrs']=[.04,.04]
 else:payload['rateTrace']['records'][0]['updatedParameterIndices']=[]
 fresh=create_optimizer([torch.nn.Parameter(torch.ones(2))],opt);controller=h.SchedulerController(fresh,r,opt,1);before=copy.deepcopy(fresh.state_dict())
 with pytest.raises(ValueError):controller.restore(payload,counts)
 assert before==fresh.state_dict() and controller.ticks==0
@pytest.mark.parametrize('op',IDS)
def test_missing_scheduler_rate_artifact_refuses_actual_resume_before_output(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 from runner.optimizer_configuration import canonical_configuration
 opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}));kw=dict(device='cpu',sample_limit=32,batch_size=10,optimizer_recipe=opt,scheduler_recipe=recipe(op));run(graph(),tmp_path/'first',epochs=1,**kw);(tmp_path/'first/rate-trace.json').write_text('{}')
 with pytest.raises(ValueError,match='trace integrity'):run(graph(),tmp_path/'refused',epochs=3,resume_checkpoint=tmp_path/'first/checkpoint.pt',**kw)
 assert not (tmp_path/'refused').exists()

def test_no_scheduler_old_recipe_bytes_and_optional_contract_omissions(tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 from runner.optimizer_configuration import canonical_configuration,training_recipe_code
 import json
 opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}));run(graph(),tmp_path/'old',epochs=1,device='cpu',sample_limit=32,optimizer_recipe=opt);m=json.loads((tmp_path/'old/manifest.json').read_text());p=torch.load(tmp_path/'old/checkpoint.pt',weights_only=True)
 assert 'scheduler' not in p and 'scheduler' not in p['contract'] and 'schedulerRecipe' not in m['settings'] and 'scheduler' not in m['provenance'] and 'rateTrace' not in m;assert not (tmp_path/'old/rate-trace.json').exists();assert (tmp_path/'old/training_recipe.py').read_text()==training_recipe_code(opt);assert 'SCHEDULER_RECIPE' not in training_recipe_code(opt)
@pytest.mark.parametrize('field,value',[('schemaVersion',True),('schemaVersion',1.),('records',True),('records',1.)])
def test_scheduler_manifest_rate_metadata_requires_exact_integer_types(field,value,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 from runner.optimizer_configuration import canonical_configuration
 import json
 opt=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=canonical_configuration('optimizer_sgd',{'lr':.1}));kw=dict(device='cpu',sample_limit=32,batch_size=10,optimizer_recipe=opt,scheduler_recipe=recipe('scheduler_step_lr'));run(graph(),tmp_path/'first',epochs=1,**kw);p=tmp_path/'first/manifest.json';m=json.loads(p.read_text());m['rateTrace'][field]=value;p.write_text(json.dumps(m))
 with pytest.raises(ValueError):run(graph(),tmp_path/'refused',epochs=3,resume_checkpoint=tmp_path/'first/checkpoint.pt',**kw)
 assert not (tmp_path/'refused').exists()

"""Real prepared training and hierarchy checkpoint continuity; no synthetic provenance."""
import copy,json,hashlib
import pytest,torch
from runner import engine,initialization,trainability,requested_gradients
from runner.tests.test_loss_engine import prepared
from runner.tests.test_gradient_engine import optimizer,policy,saved,oracle
from runner.tests.test_trainability import recipe as flags,owner
from runner.tests.test_initialization import recipe,rule
from runner.tests.test_checkpoint_lifecycle import assert_nested_equal,rewrite_checkpoint
from runner.tests.test_module_composition_execution import cid

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])

def composed(graph,kind='sequential'):
    graph=copy.deepcopy(graph);nodes=[n for n in graph['nodes'] if n['type'] not in ('input','index_input','output')]
    inputs=[dict(id='values',targets=[dict(node=nodes[0]['id'],port='in')])];outputs=[dict(id='result',source=dict(node=nodes[-1]['id'],port='out'))]
    block=dict(id=cid(['block']),kind=kind,path=['block'],children=[dict(kind='node',id=n['id'],key='stage_'+str(i) if kind=='namedMap' else str(i)) for i,n in enumerate(nodes)],inputs=inputs,outputs=outputs)
    root=copy.deepcopy(block);root.update(id='root',kind='graph',path=[],children=[dict(kind='container',id=block['id'],key='0')]);root['inputs'][0]['id']='input';root['outputs'][0]['id']='output'
    graph['compositionPlan']=dict(schemaVersion=1,root='root',containers=[root,block]);return graph

def setup(tmp,kind='sequential',device='cpu'):
    graph,r,m,data=prepared(tmp/'source','loss_mse')
    return composed(graph,kind),dict(seed=41,batch_size=5,device=device,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy())

def pins(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}

@pytest.fixture(autouse=True)
def preserve_policy():
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled,warn_only=warn)

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind',['graph','sequential','orderedList','namedMap'])
def test_actual_window_training_native_owner_math_and_contract(tmp_path,kind,device):
    graph,kw=setup(tmp_path,kind,device);torch.manual_seed(41)
    model=engine.GraphModel(graph,bundled_dataset=False);initial={'layers.learn.'+m['parameter']:p.detach().clone() for m,p in __import__('runner.native_ownership',fromlist=['native_owners']).native_owners(model)}
    engine.run(graph,tmp_path/'run',epochs=2,**kw)
    actual=saved(tmp_path/'run');identity=actual['contract']['composition']
    assert identity['schemaVersion']==1 and identity['plan']==graph['compositionPlan']
    assert [m['name'] for m in identity['parameterOwnership']]==list(actual['modelState'])
    assert identity['stateOwnership']==[dict(m,kind='parameter') for m in identity['parameterOwnership']]
    expected=oracle(initial,kw['prepared'][0][0].tolist(),kw['prepared'][0][1]['target'].tolist(),5,2,2)
    state=list(actual['modelState'].values())
    assert state[0].flatten().tolist()==pytest.approx([v for row in expected[0] for v in row],abs=3e-6,rel=3e-5)
    assert state[1].tolist()==pytest.approx(expected[1],abs=3e-6,rel=3e-5)

@pytest.mark.parametrize('implicit',[False,True])
def test_exact_cpu_all_state_resume_plan_identity_and_parent_immutable(tmp_path,implicit):
    graph,kw=setup(tmp_path)
    if implicit:kw.pop('optimizer_recipe')
    kw['initialization_recipe']=recipe(rule('learn',algorithm='normal',mean=.1,std=.2));kw['trainability']=flags(owner('learn','bias'))
    engine.run(graph,tmp_path/'full',epochs=4,**kw);engine.run(graph,tmp_path/'parent',epochs=2,**kw);before=pins(tmp_path/'parent')
    engine.run(graph,tmp_path/'resume',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)
    a,b=saved(tmp_path/'full'),saved(tmp_path/'resume')
    for key in ('modelState','optimizerState','rngState','updateSchedule','objectiveState'):assert_nested_equal(a[key],b[key])
    assert json.loads((tmp_path/'full/metrics.json').read_text())==json.loads((tmp_path/'resume/metrics.json').read_text())
    assert before==pins(tmp_path/'parent')
    changed=composed({k:v for k,v in graph.items() if k!='compositionPlan'},'orderedList')
    with pytest.raises(ValueError,match='architecture|composition|matching'):engine.run(changed,tmp_path/'wrong',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)

@pytest.mark.parametrize('device',DEVICES)
def test_finetune_inherits_parameters_resets_optimizer_child_flags_and_mps_refusal(tmp_path,device):
    graph,kw=setup(tmp_path,device=device);kw['trainability']=flags(owner('learn','weight'))
    engine.run(graph,tmp_path/'parent',epochs=2,**kw);parent=saved(tmp_path/'parent');before=pins(tmp_path/'parent');observed=[]
    engine.run(graph,tmp_path/'child',epochs=1,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**{**kw,'trainability':None,'gradient_observer':observed.append})
    for metadata in observed[0]['owners']:
        assert metadata['parameterValues']==parent['modelState'][metadata['name']].flatten().tolist()
        assert metadata['optimizerStateKeys']==[] and metadata['requiresGrad']
    assert before==pins(tmp_path/'parent')
    if device=='mps':
        with pytest.raises(ValueError,match='CPU'):engine.run(graph,tmp_path/'refused',epochs=3,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)

@pytest.mark.parametrize('mutation',['plan','owners','state','missing','boolVersion','shapeNumber'])
def test_checkpoint_composition_identity_tamper_refuses(tmp_path,mutation):
    graph,kw=setup(tmp_path);engine.run(graph,tmp_path/'parent',epochs=1,**kw)
    def alter(value,manifest):
        c=value['contract']['composition']
        if mutation=='plan':c['plan']['containers'][1]['kind']='orderedList'
        elif mutation=='owners':c['parameterOwnership'].reverse()
        elif mutation=='state':c['stateOwnership'][0]['name']='layers.foreign.weight'
        elif mutation=='boolVersion':c['plan']['schemaVersion']=True
        elif mutation=='shapeNumber':c['parameterOwnership'][0]['shape'][0]=float(c['parameterOwnership'][0]['shape'][0])
        else:value['contract'].pop('composition')
        manifest['checkpoint']['contract']=copy.deepcopy(value['contract'])
    rewrite_checkpoint(tmp_path/'parent',alter)
    with pytest.raises(ValueError,match='composition|Composition|architecture|matching'):engine.run(graph,tmp_path/'child',epochs=1,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)

@pytest.mark.parametrize('device',DEVICES)
def test_initializer_trainability_requested_analysis_same_owner_and_grad_isolation(device):
    from runner.tests.test_requested_gradients import request
    graph=composed(__import__('runner.tests.test_loss_engine',fromlist=['model']).model('loss_mse'))
    model=engine.GraphModel(graph,bundled_dataset=False).to(device)
    initialized=initialization.apply_initialization(model,recipe(rule('learn',algorithm='normal',mean=.25,std=0)))
    assert initialized[0]['name']==model.layer_keys['learn']+'.weight'
    changed=trainability.apply_trainability(model,flags(owner('learn','weight')));assert not changed[0]['requiresGrad']
    q=request('sum',output=dict(node='output',port='out'))
    result=requested_gradients.analyze(model,torch.tensor([[1.,2.],[-1.,3.]],device=device),q,device=device)
    assert result['targets'][0]['values']==pytest.approx([.5,.5]*2)
    assert all(p.grad is None for p in model.parameters())


def test_cpu_scheduler_exact_resume_trace_and_buffers(tmp_path):
    from runner.scheduler_configuration import canonical_configuration
    graph,kw=setup(tmp_path);kw['scheduler_recipe']=dict(schemaVersion=1,configurationVersion=1,componentID='scheduler_step_lr',config=canonical_configuration('scheduler_step_lr',dict(step_size=1,gamma=.5)))
    # A genuine tracked module contributes both floating buffers and int64 counter.
    graph['nodes'].insert(-1,dict(id='stats',type='batch_norm1d',config={'num_features':2}))
    graph['edges'][-1]['target']='stats';graph['edges'].append(dict(id='stats-output',source='stats',sourcePort='out',target='output',targetPort='in'))
    graph=composed({k:v for k,v in graph.items() if k!='compositionPlan'})
    engine.run(graph,tmp_path/'full',epochs=4,**kw);engine.run(graph,tmp_path/'parent',epochs=2,**kw)
    engine.run(graph,tmp_path/'resume',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)
    a,b=saved(tmp_path/'full'),saved(tmp_path/'resume');assert_nested_equal(a,b)
    assert (tmp_path/'full/rate-trace.json').read_bytes()==(tmp_path/'resume/rate-trace.json').read_bytes()
    state=a['contract']['composition']['stateOwnership'];assert [m['kind'] for m in state].count('buffer')==3
    assert state[-1]['dtype']=='int64' and state[-1]['shape']==[]

@pytest.mark.parametrize('mutation',['dtype','order','foreign'])
def test_actual_persistent_checkpoint_state_tamper_refuses(tmp_path,mutation):
    graph,kw=setup(tmp_path);engine.run(graph,tmp_path/'parent',epochs=1,**kw)
    def change(value,manifest):
        state=value['modelState'];first=next(iter(state))
        if mutation=='dtype':state[first]=state[first].double()
        elif mutation=='order':value['modelState']=dict(reversed(list(state.items())))
        else:state['composition_root.foreign.weight']=torch.ones(1)
    rewrite_checkpoint(tmp_path/'parent',change)
    with pytest.raises(ValueError,match='composition'):engine.run(graph,tmp_path/'child',epochs=1,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**kw)

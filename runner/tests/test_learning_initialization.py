"""Real initialization lessons; no dispatcher or native-UI claims."""
import copy
import importlib
import json
import math
import subprocess
import sys
import pytest
import torch
from runner.tests.test_initialization import rule

TASKS=['initialization-starting-values','initialization-starting-spread']
WIDTHS=[(2,3,2),(3,4,2),(4,3,2),(2,5,3)]
SGD=dict(lr=.05,momentum=0.,dampening=0.,weight_decay=0.,nesterov=False,maximize=False)
def module():return importlib.import_module('runner.learning_initialization')
def config(task):
    rules=[rule('first','bias',algorithm='zeros'),rule('first',algorithm='normal',mean=.15,std=.2)] if task==TASKS[0] else [rule('first',algorithm='xavier_uniform',gain=.7),rule('second',algorithm='kaiming_uniform',a=.2,mode='fan_out',nonlinearity='leaky_relu')]
    return dict(seed=42,rules=rules)
def submission(task):
    nodes=[dict(id='start',type='initialization',name='Starting settings',x=0,y=0,config=config(task)),dict(id='sgd',type='optimizer_sgd',name='Supplied learning step',x=0,y=80,config=SGD.copy()),dict(id='train',type='train',name='Check supplied model',x=180,y=0,config={})]
    edges=[dict(id='i',source='start',sourcePort='out',target='train',targetPort='initialization'),dict(id='o',source='sgd',sourcePort='out',target='train',targetPort='optimizer')]
    return dict(schemaVersion=1,kind='initializationLearningGraph',taskID=task,graph=dict(schemaVersion=9,name='Starting settings lesson',nodes=nodes,edges=edges))
def tensor(shape,values):return dict(shape=list(shape),values=list(values))
def read_tensor(value):return value['shape'],value['values']
def fixture(index=0,variation=0):
    i,h,o=WIDTHS[index]
    def values(n,offset):return [((k*7+index*3)%17-8)/16+offset+variation for k in range(n)]
    return dict(id='fixture-'+str(index),inputs=dict(x=tensor([6,i],values(6*i,.1)),target=tensor([4,o],values(4*o,-.2)),firstWeight=tensor([h,i],values(h*i,.15)),firstBias=tensor([h],values(h,.03)),secondWeight=tensor([o,h],values(o*h,-.1)),secondBias=tensor([o],values(o,.07))))
def execute(task,s=None,fixtures=None):
    evidence={'fixtures':[],'maximumTensorElements':0}
    module().execute(s or submission(task),fixtures or [fixture(i) for i in range(4)],evidence,lambda:False,read_tensor,task=task)
    return evidence

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment','cold'])
def test_four_varied_real_fixtures_five_typed_measured_criteria(task,mode):
    fixtures=[fixture(i,variation=len(mode)/64) for i in range(4)]
    for i,f in enumerate(fixtures):f['id']=mode+'-'+str(i)
    result=execute(task,fixtures=fixtures)
    for index,entry in enumerate(result['fixtures']):
        i,h,o=WIDTHS[index];p=h*i+h+o*h+o
        assert entry['fixtureID']==fixtures[index]['id'] and entry['partitions']=={}
        expected={'startingSettings':[p],'predictions':[6,o],'firstAdjustment':[2,p],'repeatIsolation':[2*p+1],'savedReuse':[p+6*o+1]}
        assert {k:v['shape'] for k,v in entry['tensors'].items()}==expected
        for name,value in entry['tensors'].items():
            assert len(value['values'])==math.prod(value['shape'])
            assert all(math.isfinite(x) for x in value['values'])
            tolerance=0 if name in ('startingSettings','repeatIsolation') else 2e-6
            assert value['values']==pytest.approx([0]*len(value['values']),abs=tolerance)
    assert result['maximumTensorElements']>=max(len(t['values']) for e in result['fixtures'] for t in e['tensors'].values())

@pytest.mark.parametrize('task,change',[(TASKS[0],'seed'),(TASKS[0],'owner'),(TASKS[0],'mean'),(TASKS[0],'std'),(TASKS[0],'algorithm'),(TASKS[1],'gain'),(TASKS[1],'fan'),(TASKS[1],'slope')])
def test_valid_wrong_controls_execute_and_fail_independent_measured_start(task,change):
    s=submission(task);c=s['graph']['nodes'][0]['config']
    if change=='seed':c['seed']=43
    elif change=='owner':c['rules'][1]['owner']['node']='second'
    elif change=='mean':c['rules'][1]['settings']['mean']=.3
    elif change=='std':c['rules'][1]['settings']['std']=.4
    elif change=='algorithm':c['rules'][1]=rule('first',algorithm='zeros')
    elif change=='gain':c['rules'][0]['settings']['gain']=.9
    elif change=='fan':c['rules'][1]['settings']['mode']='fan_in'
    else:c['rules'][1]['settings']['a']=.7
    result=execute(task,s)
    assert all(any(abs(x)>1e-5 for x in f['tensors']['startingSettings']['values']) for f in result['fixtures'])

@pytest.mark.parametrize('change',[
 lambda s:s.update(answers={}),lambda s:s.update(taskID='foreign'),lambda s:s['graph'].update(schemaVersion=8),
 lambda s:s['graph']['nodes'][1]['config'].update(lr=.1),lambda s:s['graph']['nodes'][2]['config'].update(epochs=2),
 lambda s:s['graph']['edges'].pop(),lambda s:s['graph']['edges'][0].update(targetPort='optimizer'),
 lambda s:s['graph']['nodes'][0].update(id='sgd'),lambda s:s['graph']['nodes'][0]['config'].update(seed=42.0)])
def test_strict_actual_connected_recipe_context_refusal(change):
    s=submission(TASKS[0]);change(s)
    with pytest.raises(ValueError):execute(TASKS[0],s)

@pytest.mark.parametrize('change',[
 lambda f:f['inputs'].update(answers=tensor([1],[0])),lambda f:f['inputs']['x'].update(shape=[7,2]),
 lambda f:f['inputs']['firstWeight'].update(shape=[4,4],values=[0]*16),lambda f:f['inputs']['target']['values'].__setitem__(0,float('nan')),
 lambda f:f['inputs']['secondBias']['values'].__setitem__(0,True)])
def test_invalid_or_outside_oracle_fixture_refuses_before_native_model(monkeypatch,change):
    f=fixture();change(f)
    monkeypatch.setattr(module(),'_model',lambda *args,**kwargs:pytest.fail('invalid fixture allocated a model'))
    with pytest.raises(ValueError):execute(TASKS[0],fixtures=[f])

def test_cancellation_and_untrusted_extra_fixture_identity_refuse():
    with pytest.raises(ValueError,match='cancel'):module().execute(submission(TASKS[0]),[fixture()],{'fixtures':[]},lambda:True,read_tensor,task=TASKS[0])
    f=fixture();f['expected']={}
    with pytest.raises(ValueError):execute(TASKS[0],fixtures=[f])

def test_native_initializer_poison_is_not_copied_into_expected_answer(monkeypatch):
    original=module().initialization.apply_initialization
    def wrong(model,recipe,*args,**kwargs):
        changed=copy.deepcopy(recipe);changed['config']['seed']=173
        return original(model,changed,*args,**kwargs)
    monkeypatch.setattr(module().initialization,'apply_initialization',wrong)
    result=execute(TASKS[0])
    assert all(any(abs(x)>1e-5 for x in e['tensors']['startingSettings']['values']) for e in result['fixtures'])

@pytest.mark.parametrize('task',TASKS)
def test_actual_host_cold_process_replays_same_typed_outcomes(task):
    inputs=[fixture(i,.125) for i in range(4)];expected=execute(task,fixtures=inputs)
    code='''import json,sys
from runner.learning_initialization import execute
s,f,task=json.load(sys.stdin);e={"fixtures":[],"maximumTensorElements":0}
execute(s,f,e,lambda:False,lambda v:(v["shape"],v["values"]),task=task)
print(json.dumps(e,allow_nan=False))'''
    done=subprocess.run([sys.executable,'-c',code],input=json.dumps([submission(task),inputs,task]),capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)==expected

def test_independent_scalar_reference_matches_hand_calculated_affine_chain():
    f=fixture();v={k:(t['shape'],t['values']) for k,t in f['inputs'].items()}
    v['x']=([6,2],[1.,2.]*6);v['target']=([4,2],[0.]*8)
    names=module().PARAMETERS
    parameters={names[0]:[1.]*6,names[1]:[0.]*3,names[2]:[1.]*6,names[3]:[0.]*2}
    prediction,gradient,updated=module()._scalar(parameters,(2,3,2),v)
    assert prediction==[9.]*12
    assert gradient=={names[0]:[18.,36.]*3,names[1]:[18.]*3,names[2]:[27.]*6,names[3]:[9.]*2}
    assert updated[names[0]]==pytest.approx([.1,-.8]*3)
    assert module()._scalar(updated,(2,3,2),v)[0]==pytest.approx([.054]*12)

def test_repeat_isolation_measures_rng_after_every_real_repeated_application(monkeypatch):
    original=module().initialization.apply_initialization;calls=[]
    def consumes_later(model,recipe,*args,**kwargs):
        result=original(model,recipe,*args,**kwargs);calls.append(1)
        if len(calls)%3==2:torch.rand(1)
        return result
    monkeypatch.setattr(module().initialization,'apply_initialization',consumes_later)
    result=execute(TASKS[0])
    assert all(f['tensors']['repeatIsolation']['values'][-1]>0 for f in result['fixtures'])

def test_measured_forward_gradient_is_not_replaced_with_scalar_answer(monkeypatch):
    original=module().GraphModel.forward
    monkeypatch.setattr(module().GraphModel,'forward',lambda self,x:original(self,x)*1.1)
    result=execute(TASKS[0])
    assert all(any(abs(x)>1e-5 for x in e['tensors']['predictions']['values']) for e in result['fixtures'])
    assert all(any(abs(x)>1e-5 for x in e['tensors']['firstAdjustment']['values']) for e in result['fixtures'])

def test_saved_reuse_reads_actual_loaded_parameters_and_predictions(monkeypatch):
    original=torch.load
    def altered(*args,**kwargs):
        state=original(*args,**kwargs);state['layers.second.bias']=state['layers.second.bias']+.2;return state
    monkeypatch.setattr(module().torch,'load',altered)
    result=execute(TASKS[0])
    assert all(any(abs(x)>1e-5 for x in e['tensors']['savedReuse']['values']) for e in result['fixtures'])

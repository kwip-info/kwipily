"""Independent measured Tanh policy: no authored answers, real CPU output evidence."""
import copy
import hashlib
import json
import math
import random
import subprocess
import sys

import pytest
import torch
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_evidence import affine_request, tensor

V={'major':1,'minor':0,'patch':0}; V11={'major':1,'minor':1,'patch':0}

def inputs(seed=42):
    # Bounded varied test inputs are independent of private expected tables.
    rng=random.Random(seed)
    return [[0.,-rng.uniform(1.25,3),rng.uniform(0,.1),rng.uniform(1.25,3)],
            [0.,rng.uniform(1.25,3),-rng.uniform(1.25,3),-rng.uniform(0,.1)],
            [-8.,rng.uniform(0,.1),rng.uniform(1.25,3),8.],
            [-rng.uniform(1.25,3),-1.,1.,rng.uniform(1.25,3)]]

def request(seed=42):
    value=affine_request();value['assessmentID']='assessment.activation-tanh'
    value['runtime']['capabilities']={'linear':V11,'tanh':V11}
    value['fixtures']=[{'id':f'probe-{seed}-{i}','inputs':{'x':tensor([2,2],row)}} for i,row in enumerate(inputs(seed))]
    return value

def submission(extra_identity=False):
    specs=[('input','input',{'features':2}),('first','linear',{'in_features':2,'out_features':2,'bias':True}),('activation','tanh',{})]
    parameters={'first':{'weight':tensor([2,2],[1,0,0,1]),'bias':tensor([2],[0,0])}}
    if extra_identity:
        specs.append(('last','linear',{'in_features':2,'out_features':2,'bias':False}))
        parameters['last']={'weight':tensor([2,2],[1,0,0,1])}
    specs.append(('output','output',{}))
    graph={'schemaVersion':2,'name':'Editable Tanh construction','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],
           'edges':[{'id':str(n),'source':specs[n][0],'sourcePort':'out','target':specs[n+1][0],'targetPort':'in'} for n in range(len(specs)-1)]}
    return {'schemaVersion':2,'kind':'affineGraph','graph':graph,'parameters':parameters}

def evidence(s=None,r=None,**kwargs):
    return generate_evidence(r or request(),s or submission(),submission_digest='d'*64,**kwargs)

@pytest.mark.parametrize('seed',[0,42,991])
@pytest.mark.parametrize('extra_identity',[False,True])
def test_seeded_actual_output_matches_independent_math_tanh(seed,extra_identity):
    result=evidence(submission(extra_identity),request(seed));assert not result.get('backendError'),result
    assert result['runtime']['capabilities']=={'linear':V11,'tanh':V11}
    assert len(result['fixtures'])==4
    for measured,row in zip(result['fixtures'],inputs(seed)):
        assert measured['tensors']['output']['shape']==[2,2]
        assert measured['tensors']['output']['values']==pytest.approx([math.tanh(x) for x in row],abs=1e-6,rel=1e-6)

@pytest.mark.parametrize('replacement',['identity','relu','sigmoid','softmax','disconnected','configuration'])
def test_required_ancestry_and_other_activation_refusals(replacement):
    s=submission();g=s['graph'];activation=g['nodes'][2]
    if replacement=='identity':
        g['nodes'].pop(2);g['edges'].pop();g['edges'][-1]['target']='output'
    elif replacement=='disconnected':
        g['edges'][0]['target']='output';g['edges']=g['edges'][:1]
    elif replacement=='configuration':activation['config']={'temperature':1}
    else:
        activation['type']=replacement;activation['config']={'dim':-1} if replacement=='softmax' else {}
    result=evidence(s);assert result.get('backendError') and not result['fixtures']
    if replacement=='identity':assert 'Tanh' in result['backendError']

@pytest.mark.parametrize('mutation',['identity','assessmentVersion','capabilityVersion','extraCapability','runtime','schema'])
def test_wrong_identity_version_runtime_and_submission_refusals(mutation):
    r=request();s=submission()
    if mutation=='identity':r['assessmentID']='assessment.activation-tanh-spoof'
    elif mutation=='assessmentVersion':r['assessmentVersion']={'major':1,'minor':1,'patch':0}
    elif mutation=='capabilityVersion':r['runtime']['capabilities']['tanh']=V
    elif mutation=='extraCapability':r['runtime']['capabilities']['sigmoid']=V
    elif mutation=='runtime':r['runtime']['version']='0.0.0'
    else:s['schemaVersion']=1
    if mutation in ('identity','assessmentVersion'):
        with pytest.raises(ValueError,match='Unsupported learning assessment'):evidence(s,r)
    else:assert not evidence(s,r)['fixtures']

@pytest.mark.parametrize('mistake',['constant','doubleTanh','mixedFeatures','wrongOffset'])
def test_mistakes_execute_but_independent_values_detect_wrong_behavior(mistake):
    s=submission()
    if mistake=='constant':s['parameters']['first']['weight']['values']=[0,0,0,0]
    elif mistake=='mixedFeatures':s['parameters']['first']['weight']['values']=[1,1,0,1]
    elif mistake=='wrongOffset':s['parameters']['first']['bias']['values'][0]=.5
    else:
        g=s['graph'];node=copy.deepcopy(g['nodes'][2]);node['id']='second';g['nodes'].insert(-1,node)
        g['edges'][-1]['target']='second';g['edges'].append({'id':'extra','source':'second','sourcePort':'out','target':'output','targetPort':'in'})
    result=evidence(s);assert result['fixtures'] and not result.get('backendError')
    assert max(abs(value-math.tanh(x)) for fixture,row in zip(result['fixtures'],inputs()) for value,x in zip(fixture['tensors']['output']['values'],row))>1e-3


def test_real_cli_private_expected_values_absent_budget_and_cancellation(tmp_path):
    r=request();assert 'expected' not in json.dumps(r) and 'taskSeed' not in r
    path=tmp_path/'request.json';path.write_text(json.dumps(r));source=tmp_path/'submission.json';raw=json.dumps(submission()).encode();source.write_bytes(raw)
    result=subprocess.run([sys.executable,'runner/main.py','--learning-evidence-request',str(path),'--learning-submission',str(source),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert result.returncode==0,result.stdout
    assert len(json.loads(result.stdout)['evidence']['fixtures'])==4
    assert evidence(cancel=lambda:True)['cancelled']
    r['budget']['maximumTensorElements']=1
    assert not evidence(r=r)['fixtures']


@pytest.mark.parametrize('device',['cpu','mps'])
def test_core_seeded_scalar_domain_independent_math_tanh_parity(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    from runner.engine import GraphModel
    from runner.product_execution import ProductExecutionAdapter
    # Core's independently enumerated generation ranges plus anchors, without
    # private expected tables or task seed data on the public teaching surface.
    positive=[1.25+i/100 for i in range(176)]
    near=[i/1000 for i in range(101)]
    values=positive+[-x for x in positive]+near+[-x for x in near]+[-8.,0.,8.,0.,-1.,1.]
    model=GraphModel(submission()['graph']).to(device)
    with torch.no_grad():model.layers['first'].weight.copy_(torch.eye(2,device=device));model.layers['first'].bias.zero_()
    actual=ProductExecutionAdapter(model).forward(torch.tensor(values,dtype=torch.float32,device=device).reshape(-1,2)).detach().cpu().flatten().tolist()
    expected=[math.tanh(value) for value in values]
    assert actual==pytest.approx(expected,abs=1e-6,rel=1e-6)
    print(json.dumps({'tanhCoreDomain':True,'device':device,'scalarCases':len(values),'maximumAbsoluteError':max(abs(a-b) for a,b in zip(actual,expected)),'absoluteTolerance':1e-6,'relativeTolerance':1e-6}))

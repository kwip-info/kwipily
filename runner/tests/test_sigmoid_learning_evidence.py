"""Measured Sigmoid assessment, independent scalar oracle and honest policy refusals."""
import copy
import hashlib
import json
import math
import subprocess
import sys

import pytest
import torch
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_evidence import affine_request, tensor
from runner.tests.test_sigmoid_graph import graph

V={'major':1,'minor':0,'patch':0}; V11={'major':1,'minor':1,'patch':0}
VALUES=[[-8,-1,0,1],[2,.01,-2,-.01],[8,-3,.5,-.5]]

def request():
    value=affine_request();value['assessmentID']='assessment.activation-sigmoid'
    value['runtime']['capabilities']={'linear':V11,'sigmoid':V}
    value['fixtures']=[{'id':f'probe-{i}','inputs':{'x':tensor([2,2],row)}} for i,row in enumerate(VALUES)]
    return value

def submission(extra_identity=False):
    g=graph();parameters={'first':{'weight':tensor([2,2],[1,0,0,1]),'bias':tensor([2],[0,0])},'last':{'weight':tensor([2,2],[1,0,0,1])}}
    if not extra_identity:
        g['nodes']=[node for node in g['nodes'] if node['id']!='last']
        g['edges']=[edge for edge in g['edges'] if edge['source']!='last']
        g['edges'][-1]['target']='output';parameters.pop('last')
    return {'schemaVersion':2,'kind':'affineGraph','graph':g,'parameters':parameters}

def oracle(value):
    if value>=0:return 1/(1+math.exp(-value))
    e=math.exp(value);return e/(1+e)

def evidence(s=None,**kwargs):return generate_evidence(request(),s or submission(),submission_digest='d'*64,**kwargs)

@pytest.mark.parametrize('extra_identity',[False,True])
def test_actual_measured_equivalent_graphs_match_independent_values(extra_identity):
    result=evidence(submission(extra_identity));assert not result.get('backendError'),result
    assert result['runtime']['capabilities']=={'linear':V11,'sigmoid':V}
    assert len(result['fixtures'])==len(VALUES)
    for measured,row in zip(result['fixtures'],VALUES):
        assert measured['tensors']['output']['shape']==[2,2]
        assert measured['tensors']['output']['values']==pytest.approx([oracle(x) for x in row],abs=1e-6,rel=1e-6)

@pytest.mark.parametrize('change',['missing','replacement','disconnected','configuration','schema'])
def test_required_executed_stage_and_configuration_refusals(change):
    s=submission();g=s['graph'];activation=next(node for node in g['nodes'] if node['id']=='activation')
    if change=='missing':
        g['nodes']=[node for node in g['nodes'] if node['id']!='activation']
        g['edges']=[edge for edge in g['edges'] if edge['source']!='activation'];g['edges'][-1]['target']='output'
    elif change=='replacement':activation['type']='tanh'
    elif change=='disconnected':
        g['edges']=[edge for edge in g['edges'] if edge['source']!='activation'];g['edges'][0]['target']='output'
    elif change=='configuration':activation['config']={'dim':-1}
    else:s['schemaVersion']=1
    result=evidence(s);assert result.get('backendError') and not result['fixtures']

@pytest.mark.parametrize('field',['identity','version'])
def test_wrong_assessment_identity_or_version_refused(field):
    r=request()
    if field=='identity':r['assessmentID']='assessment.activation-sigmoid-spoof'
    else:r['assessmentVersion']={'major':1,'minor':1,'patch':0}
    with pytest.raises(ValueError,match='Unsupported learning assessment'):generate_evidence(r,submission(),submission_digest='d'*64)

@pytest.mark.parametrize('change',['missing','wrongVersion','foreignCapability','wrongRuntime'])
def test_unmatched_runtime_contract_withholds_measurements(change):
    r=request()
    if change=='missing':r['runtime']['capabilities'].pop('sigmoid')
    elif change=='wrongVersion':r['runtime']['capabilities']['sigmoid']={'major':2,'minor':0,'patch':0}
    elif change=='foreignCapability':r['runtime']['capabilities']['gelu']=V
    else:r['runtime']['version']='0.0.0'
    result=generate_evidence(r,submission(),submission_digest='d'*64)
    assert not result['fixtures'] and not result.get('backendError')

@pytest.mark.parametrize('mistake',['weight','bias','doubleActivation'])
def test_numerical_mistakes_are_real_measurements_not_false_policy_passes(mistake):
    s=submission()
    if mistake=='weight':s['parameters']['first']['weight']['values'][0]=2
    elif mistake=='bias':s['parameters']['first']['bias']['values'][0]=.5
    else:
        g=s['graph'];node=copy.deepcopy(next(n for n in g['nodes'] if n['id']=='activation'));node['id']='second';g['nodes'].insert(-1,node)
        g['edges'][-1]['target']='second';g['edges'].append({'id':'extra','source':'second','sourcePort':'out','target':'output','targetPort':'in'})
    result=evidence(s);assert result['fixtures'] and not result.get('backendError')
    assert max(abs(value-oracle(x)) for fixture,row in zip(result['fixtures'],VALUES) for value,x in zip(fixture['tensors']['output']['values'],row))>1e-3


def test_private_expectations_absent_and_real_cli_budget_cancellation(tmp_path):
    r=request();assert 'expected' not in json.dumps(r) and 'taskSeed' not in r
    path=tmp_path/'request.json';path.write_text(json.dumps(r));source=tmp_path/'submission.json';raw=json.dumps(submission()).encode();source.write_bytes(raw)
    result=subprocess.run([sys.executable,'runner/main.py','--learning-evidence-request',str(path),'--learning-submission',str(source),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert result.returncode==0,result.stdout
    assert len(json.loads(result.stdout)['evidence']['fixtures'])==len(VALUES)
    assert evidence(cancel=lambda:True)['cancelled']
    r['budget']['maximumTensorElements']=1
    assert not generate_evidence(r,submission(),submission_digest='d'*64)['fixtures']


@pytest.mark.parametrize('device',['cpu','mps'])
def test_entire_discrete_core_input_domain_uses_independent_reference(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    from runner.engine import GraphModel
    from runner.product_execution import ProductExecutionAdapter
    # Independently enumerate the public generation ranges; no task receipts or
    # private expected tables are supplied to the backend or teaching surface.
    positive=[1.25+i/100 for i in range(176)]
    near=[i/1000 for i in range(101)]
    values=positive+[-x for x in positive]+near+[-x for x in near]+[-8.,0.,8.,0.,-1.,1.]
    s=submission();model=GraphModel(s['graph']).to(device)
    with torch.no_grad():model.layers['first'].weight.copy_(torch.eye(2,device=device));model.layers['first'].bias.zero_()
    actual=ProductExecutionAdapter(model).forward(torch.tensor(values,dtype=torch.float32,device=device).reshape(-1,2)).detach().cpu().flatten().tolist()
    expected=[oracle(value) for value in values]
    assert actual==pytest.approx(expected,abs=1e-6,rel=1e-6)
    print(json.dumps({'sigmoidCoreDomain':True,'device':device,'scalarCases':len(values),'maximumAbsoluteError':max(abs(a-b) for a,b in zip(actual,expected)),'absoluteTolerance':1e-6,'relativeTolerance':1e-6}))

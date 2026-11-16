import copy
import hashlib
import json
import math
import subprocess
import sys
import pytest
import torch
from runner.learning_evidence import generate_evidence
from runner.engine import GraphModel
from runner.product_execution import ProductExecutionAdapter
from runner.tests.test_learning_evidence import affine_request,tensor
from runner.tests.test_gelu import graph

V={'major':1,'minor':0,'patch':0};V11={'major':1,'minor':1,'patch':0}
VALUES=[[-3,-1,-.25,0],[.001,.5,1,3],[-2.5,-.1,.1,2.5],[-1.25,-.75,.75,1.25]]

def request(mode):
    r=affine_request();r['assessmentID']='assessment.activation-gelu-'+('tanh' if mode else 'exact')
    r['runtime']['capabilities']={'linear':V11,'gelu':V}
    r['fixtures']=[{'id':f'heldout-{i}','inputs':{'x':tensor([2,2],values)}} for i,values in enumerate(VALUES)]
    return r

def submission(mode,extra_identity=False):
    g=graph(mode);parameters={'first':{'weight':tensor([2,2],[1,0,0,1]),'bias':tensor([2],[0,0])},'last':{'weight':tensor([2,2],[1,0,0,1])}}
    if not extra_identity:
        g['nodes']=[n for n in g['nodes'] if n['id']!='last']
        g['edges']=[e for e in g['edges'] if e['source']!='last'];g['edges'][-1]['target']='output';parameters.pop('last')
    return {'schemaVersion':2,'kind':'affineGraph','graph':g,'parameters':parameters}

def oracle(x,mode):return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3))) if mode else .5*x*(1+math.erf(x/math.sqrt(2)))
def evidence(mode,s=None,**kwargs):return generate_evidence(request(mode),s or submission(mode),submission_digest='d'*64,**kwargs)

@pytest.mark.parametrize('mode',[False,True])
@pytest.mark.parametrize('extra_identity',[False,True])
def test_valid_graphs_measured_by_cpu_match_independent_oracle(mode,extra_identity):
    result=evidence(mode,submission(mode,extra_identity));assert not result.get('backendError'),result
    assert result['runtime']['capabilities']=={'linear':V11,'gelu':V}
    for fixture,inputs in zip(result['fixtures'],VALUES):
        expected=[oracle(x,mode) for x in inputs]
        assert fixture['tensors']['output']['values']==pytest.approx(expected,abs=1e-6,rel=1e-6)

@pytest.mark.parametrize('mode',[False,True])
def test_explicit_rubric_requires_gelu_and_mode_and_allowed_types(mode):
    for mutation in ('wrongMode','missingGelu','unsupported'):
        s=submission(mode);node=next(n for n in s['graph']['nodes'] if n['type']=='gelu')
        if mutation=='wrongMode':node['config']['approximate_tanh']=not mode
        else:node['type']='relu' if mutation=='unsupported' else 'output';node['config']={}
        if mutation=='missingGelu':
            s['graph']['nodes']=[n for n in s['graph']['nodes'] if n['id']!='activation'];s['graph']['edges']=[e for e in s['graph']['edges'] if e['source']!='activation'];s['graph']['edges'][-1]['target']='output'
        result=evidence(mode,s);assert result.get('backendError') and not result['fixtures']
    bad=request(mode);bad['assessmentVersion']={'major':1,'minor':1,'patch':0}
    with pytest.raises(ValueError):generate_evidence(bad,submission(mode),submission_digest='d'*64)
    bad=request(mode);bad['runtime']['capabilities'].pop('gelu')
    assert not generate_evidence(bad,submission(mode),submission_digest='d'*64)['fixtures']
    bad=submission(mode);bad['schemaVersion']=1
    assert evidence(mode,bad).get('backendError')

@pytest.mark.parametrize('mode',[False,True])
@pytest.mark.parametrize('mistake',['weight','bias','doubleActivation'])
def test_numerical_mistakes_execute_but_disagree_with_target(mode,mistake):
    s=submission(mode)
    if mistake=='weight':s['parameters']['first']['weight']['values'][0]=2
    elif mistake=='bias':s['parameters']['first']['bias']['values'][0]=.5
    else:
        g=s['graph'];node=copy.deepcopy(next(n for n in g['nodes'] if n['id']=='activation'));node['id']='second';g['nodes'].insert(-1,node)
        g['edges'][-1]['target']='second';g['edges'].append({'id':'extra','source':'second','sourcePort':'out','target':'output','targetPort':'in'})
    result=evidence(mode,s);assert result['fixtures'] and not result.get('backendError')
    errors=[abs(value-oracle(x,mode)) for f,values in zip(result['fixtures'],VALUES) for value,x in zip(f['tensors']['output']['values'],values)]
    assert max(errors)>1e-3

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual MPS unavailable')
@pytest.mark.parametrize('mode',[False,True])
def test_actual_mps_product_tolerance_not_assessment_device_claim(mode):
    s=submission(mode);model=GraphModel(s['graph']).to('mps')
    with torch.no_grad():model.layers['first'].weight.copy_(torch.eye(2,device='mps'));model.layers['first'].bias.zero_()
    adapter=ProductExecutionAdapter(model)
    for values in VALUES:
        output=adapter.forward(torch.tensor(values,device='mps').reshape(2,2)).detach().cpu().flatten().tolist()
        assert output==pytest.approx([oracle(x,mode) for x in values],abs=1e-6,rel=1e-6)

def test_cli_no_authored_expected_values_and_budgets_cancel(tmp_path):
    r=request(False);assert 'expected' not in json.dumps(r) and 'taskSeed' not in r
    path=tmp_path/'request.json';path.write_text(json.dumps(r));source=tmp_path/'submission.json';raw=json.dumps(submission(False)).encode();source.write_bytes(raw)
    result=subprocess.run([sys.executable,'runner/main.py','--learning-evidence-request',str(path),'--learning-submission',str(source),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert result.returncode==0,result.stdout
    assert len(json.loads(result.stdout)['evidence']['fixtures'])==4
    assert evidence(False,cancel=lambda:True)['cancelled']
    r['budget']['maximumTensorElements']=1
    assert not generate_evidence(r,submission(False),submission_digest='d'*64)['fixtures']

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('mode',[False,True])
def test_entire_discrete_core_fixture_domain_tolerance(device,mode):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    # Every possible scalar generated by the Core proposal, independently enumerated.
    positive=[1.25+i/100 for i in range(176)]+[.001+i/1000 for i in range(100)]
    values=[0.,0.]+positive+[-x for x in positive]
    model=GraphModel(submission(mode)['graph']).to(device)
    with torch.no_grad():model.layers['first'].weight.copy_(torch.eye(2,device=device));model.layers['first'].bias.zero_()
    actual=ProductExecutionAdapter(model).forward(torch.tensor(values,dtype=torch.float32,device=device).reshape(-1,2)).detach().cpu().flatten().tolist()
    expected=[oracle(x,mode) for x in values]
    maximum=max(abs(a-b) for a,b in zip(actual,expected))
    print(json.dumps({'geluLessonDomain':True,'device':device,'approximateTanh':mode,'scalarCases':len(values),'maximumAbsoluteError':maximum,'absoluteTolerance':1e-6,'relativeTolerance':1e-6}))
    assert actual==pytest.approx(expected,abs=1e-6,rel=1e-6)

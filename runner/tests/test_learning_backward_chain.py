"""Independent backward-chain mathematical oracle and shared worker dispatch evidence."""
import hashlib
import json
import subprocess
import sys
import pytest
import torch
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_evidence import affine_request,affine_submission,tensor

V={'major':1,'minor':0,'patch':0}
V11={'major':1,'minor':1,'patch':0}
W1=torch.tensor([[2.,-1.],[.5,3.]],dtype=torch.float64)
W2=torch.tensor([[1.,2.],[-2.,1.]],dtype=torch.float64)
B1=torch.tensor([1.,-2.],dtype=torch.float64)
B2=torch.tensor([.25,1.],dtype=torch.float64)

def request():
    r=affine_request();r['assessmentID']='assessment.affine-backward-chain';r['runtime']['capabilities']['linear']=V11
    # Cotangents vary sign, zeros and magnitude; expected answers never sent to worker.
    r['fixtures']=[{'id':f'cotangent-{i}','inputs':{'x':tensor([2,2],values)}} for i,values in enumerate(([1,0,0,1],[2,-3,-2,5],[0,0,.125,-.75],[-1.5,2.25,4,-.5]))]
    return r

def submission(two=False):
    s=affine_submission(two);s['schemaVersion']=2;s['graph']['schemaVersion']=2
    for node in s['graph']['nodes']:
        if node['type']=='linear':node['config']['bias']=False
    weights={'first':W2.T,'second':W1.T} if two else {'first':(W2@W1).T}
    s['parameters']={key:{'weight':tensor([2,2],value.flatten().tolist())} for key,value in weights.items()}
    return s

def measured(s=None,r=None,**kwargs):return generate_evidence(r or request(),s or submission(),submission_digest='a'*64,**kwargs)

def expected(r):return [torch.tensor(f['inputs']['x']['values'],dtype=torch.float64).reshape(2,2)@W2@W1 for f in r['fixtures']]
def outputs(e):return [torch.tensor(f['tensors']['output']['values'],dtype=torch.float64).reshape(2,2) for f in e['fixtures']]

@pytest.mark.parametrize('two',[False,True])
def test_equivalent_backward_maps_match_independent_target(two):
    e=measured(submission(two));assert not e.get('backendError')
    assert e['assessmentID']=='assessment.affine-backward-chain' and e['assessmentVersion']==V
    assert len(e['fixtures'])==4
    for actual,target in zip(outputs(e),expected(request())):torch.testing.assert_close(actual,target,rtol=1e-6,atol=1e-6)

def test_target_crosschecked_against_actual_autograd_and_central_differences():
    x=torch.tensor([[.2,-.7],[1.3,2.1]],dtype=torch.float64,requires_grad=True)
    def forward(value):return (value@W1.T+B1)@W2.T+B2
    epsilon=1e-5
    for fixture,target in zip(request()['fixtures'],expected(request())):
        g=torch.tensor(fixture['inputs']['x']['values'],dtype=torch.float64).reshape(2,2)
        autograd=torch.autograd.grad((forward(x)*g).sum(),x)[0]
        numeric=torch.empty_like(x)
        for row in range(2):
            for column in range(2):
                plus=x.detach().clone();minus=x.detach().clone();plus[row,column]+=epsilon;minus[row,column]-=epsilon
                numeric[row,column]=((forward(plus)*g).sum()-(forward(minus)*g).sum())/(2*epsilon)
        torch.testing.assert_close(target,autograd,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(target,numeric,rtol=1e-8,atol=1e-8)

@pytest.mark.parametrize('mistake',['wrongOrder','noTranspose','inheritedBias'])
def test_plausible_mistakes_execute_but_fail_numerically(mistake):
    s=submission(True)
    if mistake=='wrongOrder':
        s['parameters']['first']['weight']['values']=W1.T.flatten().tolist();s['parameters']['second']['weight']['values']=W2.T.flatten().tolist()
    elif mistake=='noTranspose':
        s['parameters']['first']['weight']['values']=W2.flatten().tolist();s['parameters']['second']['weight']['values']=W1.flatten().tolist()
    else:
        for n in s['graph']['nodes']:
            if n['type']=='linear':n['config']['bias']=True
        s['parameters']['first']['bias']=tensor([2],B2.tolist());s['parameters']['second']['bias']=tensor([2],B1.tolist())
    evidence=measured(s);assert not evidence.get('backendError') and evidence['fixtures']
    assert any(not torch.allclose(a,b,rtol=1e-5,atol=1e-5) for a,b in zip(outputs(evidence),expected(request())))

def test_backward_policy_strict_and_legacy_dispatch_preserved():
    r=request();r['assessmentVersion']={'major':1,'minor':1,'patch':0}
    with pytest.raises(ValueError):measured(r=r)
    r=request();r['runtime']['capabilities']['linear']=V
    mismatch=measured(r=r);assert not mismatch['fixtures'] and mismatch['runtime']['capabilities']['linear']==V11
    s=submission();s['schemaVersion']=1
    assert measured(s).get('backendError')
    s=submission();s['graph']['nodes'].insert(-1,{'id':'relu','type':'relu','name':'Relu','x':0,'y':0,'config':{}})
    s['graph']['edges'][-1]['target']='relu';s['graph']['edges'].append({'id':'last','source':'relu','sourcePort':'out','target':'output','targetPort':'in'})
    assert 'affine-preserving' in measured(s).get('backendError','')
    assert generate_evidence(affine_request(),affine_submission(),submission_digest='a'*64)['fixtures']
    r=request();r['budget']['maximumTensorElements']=1
    assert not measured(r=r)['fixtures']
    e=measured(cancel=lambda:True);assert e['cancelled'] and not e['fixtures']

def test_actual_cli_executes_without_authored_target(tmp_path):
    r=request();assert 'expected' not in json.dumps(r) and 'taskSeed' not in r
    request_file=tmp_path/'request.json';request_file.write_text(json.dumps(r))
    source=tmp_path/'submission.json';raw=json.dumps(submission()).encode();source.write_bytes(raw)
    result=subprocess.run([sys.executable,'runner/main.py','--learning-evidence-request',str(request_file),'--learning-submission',str(source),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    evidence=json.loads(result.stdout)['evidence'];assert len(evidence['fixtures'])==4
    for actual,target in zip(outputs(evidence),expected(r)):torch.testing.assert_close(actual,target,rtol=1e-6,atol=1e-6)

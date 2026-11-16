import copy
import math
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_tensor_transforms import request,submission

def fixture(dim=-1):
    r=request();r['assessmentID']='assessment.softmax-last-axis'
    r['runtime']['capabilities']={'softmax':{'major':1,'minor':0,'patch':0}}
    r['fixtures'][0]['inputs']={'x':{'shape':[1,2,3],'values':[1000,1001,1002,-1,2,0]},'upstream':{'shape':[1,2,3],'values':[1,-2,3,4,0,-1]}}
    s=submission();s['graph']['nodes'][1].update(type='softmax',config={'dim':dim})
    return r,s

def measure(r,s):return generate_evidence(r,s,submission_digest='a'*64)

def test_measured_values_and_coupled_gradient():
    r,s=fixture();result=measure(r,s)
    assert not result.get('backendError')
    tensors=result['fixtures'][0]['tensors'];values=[];gradients=[]
    for row,g in [([1000,1001,1002],[1,-2,3]),([-1,2,0],[4,0,-1])]:
        exponentials=[math.exp(x-max(row)) for x in row];p=[x/sum(exponentials) for x in exponentials]
        values+=p;dot=sum(a*b for a,b in zip(p,g));gradients += [a*(b-dot) for a,b in zip(p,g)]
    assert tensors['output']['values']==pytest.approx(values,abs=1e-12)
    assert tensors['inputGradient']['values']==pytest.approx(gradients,abs=1e-12)

def test_wrong_axis_is_measured_not_mistaken_for_success():
    r,s=fixture(1);wrong=measure(r,s)
    r,correct=fixture(-1);right=measure(r,correct)
    assert not wrong.get('backendError')
    assert wrong['fixtures'][0]['tensors']['output']!=right['fixtures'][0]['tensors']['output']
    assert wrong['fixtures'][0]['tensors']['inputGradient']!=right['fixtures'][0]['tensors']['inputGradient']

@pytest.mark.parametrize('dim',[0,-3,3])
def test_invalid_axes_have_no_measured_success(dim):
    r,s=fixture(dim);result=measure(r,s)
    assert result.get('backendError') and not result['fixtures']

def test_old_transform_lessons_remain_restricted_and_budget_cancel_hold():
    _,s=fixture()
    old=request();old['fixtures'][0]['inputs']['upstream']['shape']=[1,2,3]
    result=measure(old,s);assert result.get('backendError') and not result['fixtures']
    r,s=fixture();result=generate_evidence(r,s,submission_digest='a'*64,cancel=lambda:True)
    assert result['cancelled'] and not result['fixtures']
    r['budget']['maximumTensorElements']=1;assert not measure(r,s)['fixtures']
    r,s=fixture();s['graph']['nodes'][1].update(type='code',config={})
    assert measure(r,s).get('backendError')

import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_tensor_transforms import request, submission


def fixture():
    r=request();r['assessmentID']='assessment.embedding-gather'
    r['runtime']['capabilities']={'embedding':{'major':1,'minor':0,'patch':0}}
    r['fixtures'][0]['inputs']={'indices':{'shape':[1,4],'values':[0,2,2,1]},'table':{'shape':[4,3],'values':[3,4,0,1,-2,2,6,8,0,9,12,0]},'upstream':{'shape':[1,4,3],'values':list(range(1,13))}}
    s=submission();s['kind']='embeddingGraph'
    s['graph']['nodes'][1].update(type='embedding',config={'num_embeddings':4,'embedding_dim':3,'padding_idx':None,'max_norm':None,'norm_type':2,'scale_grad_by_freq':False,'sparse':False,'freeze':False})
    return r,s


def measure(r,s):return generate_evidence(r,s,submission_digest='a'*64)

@pytest.mark.parametrize('frequency',[False,True])
def test_real_gather_scatter_and_padding_policy(frequency):
    r,s=fixture();s['graph']['nodes'][1]['config'].update(padding_idx=0,scale_grad_by_freq=frequency)
    result=measure(r,s);assert not result.get('backendError'),result
    tensors=result['fixtures'][0]['tensors']
    assert tensors['output']['values']==[3,4,0,6,8,0,6,8,0,1,-2,2]
    assert tensors['tableGradient']['values']==[0,0,0,10,11,12,*([5.5,6.5,7.5] if frequency else [11,13,15]),0,0,0]
    assert tensors['tableAfterForward']==r['fixtures'][0]['inputs']['table']
    assert 'inputGradient' not in tensors


def test_norm_mutation_and_gradient_are_separate():
    r,s=fixture();s['graph']['nodes'][1]['config']['max_norm']=1
    result=measure(r,s);assert not result.get('backendError'),result
    tensors=result['fixtures'][0]['tensors']
    assert tensors['tableAfterForward']['values'][:3]==pytest.approx([3/(5+1e-7),4/(5+1e-7),0])
    assert tensors['tableAfterForward']['values'][-3:]==[9,12,0]
    assert tensors['tableGradient']['values']==[1,2,3,10,11,12,11,13,15,0,0,0]

@pytest.mark.parametrize('value',[-1,4,1.5,True,2**63])
def test_invalid_indices_reject_before_allocation(value,monkeypatch):
    r,s=fixture();r['fixtures'][0]['inputs']['indices']['values'][0]=value
    def forbidden(*args,**kwargs):raise AssertionError('allocated tensor')
    monkeypatch.setattr('torch.tensor',forbidden)
    # Request validation rejects booleans even before evidence creation.
    if value is True:
        with pytest.raises(ValueError):measure(r,s)
    else:
        result=measure(r,s);assert result.get('backendError') and 'allocated tensor' not in result['backendError'] and not result['fixtures']


def test_budget_cancel_and_authored_answers():
    r,s=fixture();r['budget']['maximumTensorElements']=1
    result=measure(r,s);assert not result['fixtures'] and result['maximumTensorElements']>1
    r,s=fixture();assert generate_evidence(r,s,submission_digest='a'*64,cancel=lambda:True)['cancelled']
    s['expected']={};assert measure(r,s).get('backendError')

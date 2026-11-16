import copy
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_tensor_transforms import request


def fixture():
    r=request();r['assessmentID']='assessment.matrix-contraction'
    r['runtime']['capabilities']={'matmul':{'major':1,'minor':0,'patch':0}}
    r['fixtures'][0]['inputs']={'left':{'shape':[1,2,2],'values':[1,2,3,4]},'right':{'shape':[1,2,2],'values':[5,6,7,8]},'upstream':{'shape':[1,2,2],'values':[1,2,3,4]}}
    nodes=[dict(id='left',type='input',config={'fixture':'left'}),dict(id='right',type='input',config={'fixture':'right'}),dict(id='multiply',type='matmul',config={}),dict(id='output',type='output',config={})]
    links=[('left','multiply','a'),('right','multiply','b'),('multiply','output','in')]
    s=dict(schemaVersion=1,kind='matrixGraph',graph=dict(schemaVersion=2,nodes=nodes,edges=[dict(id=str(i),source=a,sourcePort='out',target=b,targetPort=p) for i,(a,b,p) in enumerate(links)]))
    return r,s


def measure(r,s):return generate_evidence(r,s,submission_digest='a'*64)


def test_measured_product_and_both_operand_gradients():
    r,s=fixture();result=measure(r,s)
    assert not result.get('backendError'),result
    tensors=result['fixtures'][0]['tensors']
    assert tensors['output']['values']==[19,22,43,50]
    assert tensors['leftGradient']['values']==[17,23,39,53]
    assert tensors['rightGradient']['values']==[10,14,14,20]
    # Valid but reversed graph must be measured, not silently corrected.
    s['graph']['edges'][0]['targetPort']='b';s['graph']['edges'][1]['targetPort']='a'
    reversed_result=measure(r,s)
    assert not reversed_result.get('backendError')
    assert reversed_result['fixtures'][0]['tensors']['output']['values']==[23,34,31,46]


def test_broadcast_gradient_and_observation_isolation():
    r,s=fixture();r['assessmentID']='assessment.matrix-broadcast-gradients'
    r['fixtures'][0]['inputs']={'left':{'shape':[2,2,1,2],'values':[1,2,3,4,10,20,30,40]},'right':{'shape':[2,2,1],'values':[5,6,7,8]},'upstream':{'shape':[2,2,1,1],'values':[1,2,0,0]}}
    result=measure(r,s);assert not result.get('backendError'),result
    assert result['fixtures'][0]['tensors']['rightGradient']['values']==[7,10,0,0]


@pytest.mark.parametrize('change',['duplicate','cycle','unknown','answers','unused','binding'])
def test_rejects_malformed_or_authored_submissions(change):
    r,s=fixture();g=s['graph']
    if change=='duplicate':g['edges'][1]['targetPort']='a'
    elif change=='cycle':g['edges'][0]['source']='multiply'
    elif change=='unknown':g['nodes'][2]['type']='custom_code'
    elif change=='answers':s['expected']={}
    elif change=='unused':g['nodes'].append(dict(id='orphan',type='transpose',config={'dim0':1,'dim1':2}))
    elif change=='binding':g['nodes'][1]['config']['fixture']='left'
    result=measure(r,s);assert result.get('backendError') and not result['fixtures']


def test_budget_and_cancellation_stop_before_tensor_allocation(monkeypatch):
    r,s=fixture();r['budget']['maximumTensorElements']=1
    def forbidden(*args,**kwargs):raise AssertionError('allocated tensor')
    monkeypatch.setattr('torch.tensor',forbidden)
    result=measure(r,s)
    assert not result['fixtures'] and result['maximumTensorElements']>1
    assert 'allocated tensor' not in result.get('backendError','')
    r,s=fixture()
    assert generate_evidence(r,s,submission_digest='a'*64,cancel=lambda:True)['cancelled']

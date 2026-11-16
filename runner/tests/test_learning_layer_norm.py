import copy
import math
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_tensor_transforms import request, submission


def fixture(affine=False, bias=True, shape=None):
    r=request();r['assessmentID']='assessment.layer-norm-affine' if affine else 'assessment.layer-norm-rows'
    r['runtime']['capabilities']={'layer_norm':{'major':1,'minor':0,'patch':0}}
    r['fixtures'][0]['inputs']={'x':{'shape':[2,2,3],'values':[1,2,4,10,10,10,2,3,5,-1,0,1]},'upstream':{'shape':[2,2,3],'values':[1,-2,3,4,0,-1,2,1,-3,0,2,1]}}
    s=submission();node=s['graph']['nodes'][1];ident=node['id']
    node.update(type='layer_norm',config={'normalized_shape':shape or [3],'eps':0.125,'elementwise_affine':affine,'bias':bias})
    s.update(kind='layerNormGraph',parameters={},gradientNodeID=ident if affine else None)
    if affine:
        s['parameters'][ident]={'weight':{'shape':[3],'values':[0.5,-1,2]}}
        if bias:s['parameters'][ident]['bias']={'shape':[3],'values':[1,-0.5,0.25]}
    return r,s


def measure(r,s):return generate_evidence(r,s,submission_digest='a'*64)


@pytest.mark.parametrize('affine,bias',[(False,False),(False,True),(True,False),(True,True)])
def test_measures_values_input_and_shared_parameter_gradients(affine,bias):
    r,s=fixture(affine,bias);result=measure(r,s)
    assert not result.get('backendError'),result
    tensors=result['fixtures'][0]['tensors'];inputs=r['fixtures'][0]['inputs']
    output=[];dx=[];dg=[0.]*3;db=[0.]*3
    gamma=[0.5,-1,2] if affine else [1]*3
    beta=[1,-0.5,0.25] if affine and bias else [0]*3
    for start in range(0,12,3):
        x=inputs['x']['values'][start:start+3];g=inputs['upstream']['values'][start:start+3]
        mean=sum(x)/3;variance=sum((v-mean)**2 for v in x)/3;inverse=1/math.sqrt(variance+0.125)
        z=[(v-mean)*inverse for v in x];h=[g[i]*gamma[i] for i in range(3)]
        output += [z[i]*gamma[i]+beta[i] for i in range(3)]
        dx += [inverse*(h[i]-sum(h)/3-z[i]*sum(a*b for a,b in zip(h,z))/3) for i in range(3)]
        for i in range(3):dg[i]+=g[i]*z[i];db[i]+=g[i]
    assert tensors['output']['values']==pytest.approx(output,abs=1e-12)
    assert tensors['inputGradient']['values']==pytest.approx(dx,abs=1e-12)
    assert ('scaleGradient' in tensors)==affine
    assert ('biasGradient' in tensors)==(affine and bias)
    if affine:assert tensors['scaleGradient']['values']==pytest.approx(dg,abs=1e-12)
    if affine and bias:assert tensors['biasGradient']['values']==pytest.approx(db,abs=1e-12)


def test_wrong_shape_measured_or_rejected_and_old_lessons_not_widened():
    r,s=fixture();right=measure(r,s)
    s['graph']['nodes'][1]['config']['normalized_shape']=[2,3]
    wrong=measure(r,s)
    assert not wrong.get('backendError')
    assert wrong['fixtures'][0]['tensors']['output']!=right['fixtures'][0]['tensors']['output']
    s['graph']['nodes'][1]['config']['normalized_shape']=[2]
    invalid=measure(r,s);assert invalid.get('backendError') and not invalid['fixtures']
    r,s=fixture();r['assessmentID']='assessment.tensor-reshape';r['runtime']['capabilities']={'reshape':{'major':1,'minor':0,'patch':0}}
    assert measure(r,s).get('backendError')


@pytest.mark.parametrize('mutation',['missing','extra','shape','bool','inactive','target','answer','code'])
def test_rejects_malformed_parameters_and_authored_results(mutation):
    r,s=fixture(True);ident=s['gradientNodeID']
    if mutation=='missing':s['parameters'][ident].pop('weight')
    if mutation=='extra':s['parameters']['unknown']=copy.deepcopy(s['parameters'][ident])
    if mutation=='shape':s['parameters'][ident]['weight']['shape']=[1,3]
    if mutation=='bool':s['parameters'][ident]['weight']['values'][0]=True
    if mutation=='inactive':s['graph']['nodes'][1]['config']['elementwise_affine']=False
    if mutation=='target':s['gradientNodeID']='unknown'
    if mutation=='answer':s['expected']={}
    if mutation=='code':s['graph']['nodes'][1]['type']='code'
    result=measure(r,s);assert result.get('backendError') and not result['fixtures']


def test_budget_and_cancellation_emit_no_measurements():
    r,s=fixture(True)
    result=generate_evidence(r,s,submission_digest='a'*64,cancel=lambda:True)
    assert result['cancelled'] and not result['fixtures']
    r['budget']['maximumTensorElements']=1
    result=measure(r,s);assert not result['fixtures'] and result['maximumTensorElements']>1


def test_fixture_gradients_do_not_accumulate_and_parameters_change_measurement():
    r,s=fixture(True)
    r['fixtures'].append(copy.deepcopy(r['fixtures'][0]));r['fixtures'][1]['id']='second'
    result=measure(r,s)
    assert not result.get('backendError')
    assert result['fixtures'][0]['tensors']==result['fixtures'][1]['tensors']
    s['parameters'][s['gradientNodeID']]['weight']['values'][0]+=1
    changed=measure(r,s)
    assert changed['fixtures'][0]['tensors']['output']!=result['fixtures'][0]['tensors']['output']
    assert changed['fixtures'][0]['tensors']['inputGradient']!=result['fixtures'][0]['tensors']['inputGradient']


def test_budget_rejection_precedes_tensor_allocation(monkeypatch):
    r,s=fixture(True);r['budget']['maximumTensorElements']=1
    def forbidden(*args,**kwargs):raise AssertionError('Allocated tensor before budget rejection')
    monkeypatch.setattr('runner.learning_layer_norm.torch.tensor',forbidden)
    result=measure(r,s)
    assert not result.get('backendError') and not result['fixtures']
    assert result['maximumTensorElements']>1


def test_layout_composition_preserves_parameter_ownership():
    r,s=fixture(True);baseline=measure(r,s)
    nodes=s['graph']['nodes']
    nodes[2:2]=[{'id':'flat','type':'flatten','config':{'start_dim':1,'end_dim':-1}},
                {'id':'restore','type':'reshape','config':{'shape':[2,3]}}]
    s['graph']['edges']=[{'id':str(i),'source':nodes[i]['id'],'sourcePort':'out','target':nodes[i+1]['id'],'targetPort':'in'} for i in range(len(nodes)-1)]
    result=measure(r,s)
    assert not result.get('backendError'),result
    assert result['fixtures'][0]['tensors']==baseline['fixtures'][0]['tensors']

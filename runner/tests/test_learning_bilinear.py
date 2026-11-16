"""Measured bilinear learning; the oracle uses scalar sums, not F.bilinear."""
import copy
import itertools
import math
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_tensor_transforms import request


def tensor(shape, values): return {'shape':shape,'values':values}


def fixture(bias=True):
    r=request();r['assessmentID']='assessment.bilinear-'+('with-bias' if bias else 'without-bias')
    r['runtime']['capabilities']={'bilinear':{'major':1,'minor':0,'patch':0}}
    inputs={'left':tensor([2,2],[1,2,3,-1]),'right':tensor([2,2],[2,-1,1,4]),
            'weight':tensor([2,2,2],[1,2,3,4,-1,2,0,3]),'upstream':tensor([2,2],[1,2,-1,3])}
    if bias:inputs['bias']=tensor([2],[.5,-2])
    r['fixtures'][0]['inputs']=inputs
    names=['left','right','weight']+(['bias'] if bias else [])
    nodes=[dict(id=n,type='input',config={'fixture':n}) for n in names]
    nodes += [dict(id='bilinear',type='bilinear',config={'bias':bias}),dict(id='output',type='output',config={})]
    ports={'left':'a','right':'b','weight':'weight','bias':'bias'}
    links=[(n,'bilinear',ports[n]) for n in names]+[('bilinear','output','in')]
    s={'schemaVersion':1,'kind':'bilinearGraph','graph':{'schemaVersion':2,'nodes':nodes,'edges':[dict(id=str(i),source=a,sourcePort='out',target=b,targetPort=p) for i,(a,b,p) in enumerate(links)]}}
    return r,s


def measure(r,s,**kwargs):return generate_evidence(r,s,submission_digest='a'*64,**kwargs)


def oracle(inputs):
    a=inputs['left']['values'];b=inputs['right']['values'];w=inputs['weight']['values'];g=inputs['upstream']['values']
    o_count,h1,h2=inputs['weight']['shape'];rows=math.prod(inputs['left']['shape'][:-1])
    bias=inputs.get('bias',{}).get('values',[0]*o_count)
    y=[0.]*(rows*o_count);da=[0.]*len(a);db=[0.]*len(b);dw=[0.]*len(w);dc=[0.]*o_count
    for row,o,i,j in itertools.product(range(rows),range(o_count),range(h1),range(h2)):
        k=o*h1*h2+i*h2+j
        y[row*o_count+o]+=a[row*h1+i]*w[k]*b[row*h2+j]
        da[row*h1+i]+=g[row*o_count+o]*w[k]*b[row*h2+j]
        db[row*h2+j]+=g[row*o_count+o]*w[k]*a[row*h1+i]
        dw[k]+=g[row*o_count+o]*a[row*h1+i]*b[row*h2+j]
    for row,o in itertools.product(range(rows),range(o_count)):
        y[row*o_count+o]+=bias[o];dc[o]+=g[row*o_count+o]
    result={'output':y,'leftGradient':da,'rightGradient':db,'weightGradient':dw}
    if 'bias' in inputs:result['biasGradient']=dc
    return result



@pytest.mark.parametrize('bias',[False,True])
def test_independent_output_and_all_original_operand_gradients(bias):
    r,s=fixture(bias);result=measure(r,s)
    assert not result.get('backendError'),result
    actual=result['fixtures'][0]['tensors']
    assert set(actual)==set(oracle(r['fixtures'][0]['inputs']))
    for key,expected in oracle(r['fixtures'][0]['inputs']).items():assert actual[key]['values']==pytest.approx(expected)


def test_changed_valid_authored_graph_is_measured_not_repaired():
    r,s=fixture();baseline=measure(r,s)
    s['graph']['edges'][0]['targetPort']='b';s['graph']['edges'][1]['targetPort']='a'
    changed=measure(r,s);assert not changed.get('backendError'),changed
    assert changed['fixtures'][0]['tensors']['output']!=baseline['fixtures'][0]['tensors']['output']
    altered=copy.deepcopy(r['fixtures'][0]['inputs']);altered['left'],altered['right']=altered['right'],altered['left']
    assert changed['fixtures'][0]['tensors']['output']['values']==pytest.approx(oracle(altered)['output'])
    assert changed['fixtures'][0]['tensors']['leftGradient']['values']==pytest.approx(oracle(altered)['rightGradient'])


@pytest.mark.parametrize('change',['duplicate','cycle','unknown','answers','unused','binding','policy','numeric_bias','broadcast','extra_bias','missing_bias','two_bilinear'])
def test_rejects_malformed_broadcast_and_policy_mismatch(change):
    r,s=fixture();g=s['graph'];inputs=r['fixtures'][0]['inputs']
    if change=='duplicate':g['edges'][1]['targetPort']='a'
    elif change=='cycle':g['edges'][0]['source']='bilinear'
    elif change=='unknown':g['nodes'][-2]['type']='custom_code'
    elif change=='answers':s['expected']={}
    elif change=='unused':g['nodes'].append(dict(id='orphan',type='output',config={}))
    elif change=='binding':g['nodes'][1]['config']['fixture']='left'
    elif change=='policy':r['assessmentID']='assessment.bilinear-without-bias'
    elif change=='numeric_bias':g['nodes'][-2]['config']['bias']=1
    elif change=='broadcast':inputs['right']=tensor([1,2],[1,2])
    elif change=='extra_bias':r,s=fixture(False);r['fixtures'][0]['inputs']['bias']=tensor([2],[1,2])
    elif change=='missing_bias':del inputs['bias']
    elif change=='two_bilinear':g['nodes'].append(dict(id='second',type='bilinear',config={'bias':True}))
    result=measure(r,s);assert result.get('backendError') and not result['fixtures'],result


def test_budget_and_cancel_before_allocation(monkeypatch):
    r,s=fixture();r['budget']['maximumTensorElements']=1
    def forbidden(*args,**kwargs):raise AssertionError('allocated tensor')
    monkeypatch.setattr('torch.tensor',forbidden)
    result=measure(r,s);assert not result['fixtures'] and result['maximumTensorElements']>1
    assert 'allocated tensor' not in result.get('backendError','')
    r,s=fixture();assert measure(r,s,cancel=lambda:True)['cancelled']


def test_work_bound_before_allocation(monkeypatch):
    r,s=fixture(False);r['fixtures'][0]['inputs']={'left':tensor([16,16],[1]*256),'right':tensor([16,16],[1]*256),'weight':tensor([16,16,16],[1]*4096),'upstream':tensor([16,16],[1]*256)}
    # 65,536 forward terms are bounded; count forward plus derivative contraction work.
    monkeypatch.setattr('torch.tensor',lambda *a,**k: (_ for _ in ()).throw(AssertionError('allocated tensor')))
    result=measure(r,s);assert result.get('backendError') and 'allocated tensor' not in result['backendError'],result


def test_higher_rank_unequal_widths_with_independent_scalar_oracle():
    r,s=fixture()
    inputs={'left':tensor([2,3,2],[i/5-1 for i in range(12)]),
            'right':tensor([2,3,3],[i/7-.5 for i in range(18)]),
            'weight':tensor([3,2,3],[i/11-.25 for i in range(18)]),
            'bias':tensor([3],[.2,-.3,.4]),'upstream':tensor([2,3,3],[i/13-.6 for i in range(18)])}
    r['fixtures'][0]['inputs']=inputs
    result=measure(r,s);assert not result.get('backendError'),result
    for key,expected in oracle(inputs).items():
        assert result['fixtures'][0]['tensors'][key]['values']==pytest.approx(expected)
    assert result['fixtures'][0]['tensors']['output']['shape']==[2,3,3]


def test_authored_weight_transpose_preserves_gradients_to_original_fixture():
    r,s=fixture();g=s['graph']
    g['nodes'].append(dict(id='transpose',type='transpose',config={'dim0':1,'dim1':2}))
    g['edges'][2]['target']='transpose';g['edges'][2]['targetPort']='in'
    g['edges'].append(dict(id='transformed-weight',source='transpose',sourcePort='out',target='bilinear',targetPort='weight'))
    expected_inputs=copy.deepcopy(r['fixtures'][0]['inputs'])
    old=expected_inputs['weight']['values']
    permutation=[0,2,1,3,4,6,5,7]
    expected_inputs['weight']['values']=[old[i] for i in permutation]
    expected=oracle(expected_inputs);expected['weightGradient']=[expected['weightGradient'][i] for i in permutation]
    result=measure(r,s);assert not result.get('backendError'),result
    for key,values in expected.items():assert result['fixtures'][0]['tensors'][key]['values']==pytest.approx(values)

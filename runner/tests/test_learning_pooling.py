import math
from itertools import product
import pytest
from runner import learning_pooling


def submission(d,average=False):
    axis=lambda x:x if d==1 else [x]*d
    config={'kernel_size':axis(3),'stride':axis(2),'padding':axis(1),'ceil_mode':True}
    if average:config['count_include_pad']=True
    else:config.update(dilation=axis(1),return_indices=True)
    kind='avg_pool1d' if average else f'max_pool{d}d'
    return {'schemaVersion':1,'kind':'poolingGraph','graph':{'schemaVersion':2,'name':'Pool','nodes':[{'id':'x','type':'input','config':{}},{'id':'pool','type':kind,'config':config},{'id':'y','type':'output','config':{}}],'edges':[{'id':'a','source':'x','sourcePort':'out','target':'pool','targetPort':'in'},{'id':'b','source':'pool','sourcePort':'out','target':'y','targetPort':'in'}]}}


def fixture(d,average=False,ties=False):
    size=6 if average else 4;extent=4 if average else 3
    shape=[1,2]+[size]*d;out=[1,2]+[extent]*d
    values=[-2 if ties else ((i*7)%19-10)/4 for i in range(math.prod(shape))]
    return {'id':'independent','inputs':{'x':{'shape':shape,'values':values},'upstream':{'shape':out,'values':[1]*math.prod(out)}}}


def oracle(f,d,average=False):
    size=6 if average else 4;extent=4 if average else 3;x=f['inputs']['x']['values'];y=[];indices=[];dx=[0.]*len(x)
    def flat(c):
        value=0
        for v in c:value=value*size+v
        return value
    for channel in range(2):
        for p in product(range(extent),repeat=d):
            candidates=[]
            for offset in product(range(3),repeat=d):
                c=[q*2-1+k for q,k in zip(p,offset)]
                if all(0<=v<size for v in c):candidates.append(channel*size**d+flat(c))
            if average:
                start=p[0]*2-1;divisor=min(start+3,size+1)-max(start,-1)
                y.append(sum(x[i] for i in candidates)/divisor)
                for i in candidates:dx[i]+=1/divisor
            else:
                winner=candidates[0]
                for i in candidates[1:]:
                    if x[i]>x[winner]:winner=i
                y.append(x[winner]);indices.append(winner-channel*size**d);dx[winner]+=1
    return y,dx,indices


def measured(d,average=False,f=None,s=None,stopped=lambda:False):
    evidence={'fixtures':[],'maximumTensorElements':0}
    learning_pooling.execute(s or submission(d,average),[f or fixture(d,average)],evidence,stopped,lambda t:(t['shape'],t['values']),dimensions=d,average=average)
    return evidence

@pytest.mark.parametrize('d',[1,2,3])
@pytest.mark.parametrize('ties',[False,True])
def test_actual_negative_max_first_ties_indices_and_influence_independent(d,ties):
    f=fixture(d,ties=ties);expected=oracle(f,d);t=measured(d,f=f)['fixtures'][0]['tensors']
    assert t['output']['values']==expected[0]
    assert t['inputGradient']['values']==expected[1]
    assert t['indices']['values']==expected[2]
    assert t['output']['shape']==[1,2]+[3]*d


def test_average_declared_padding_partial_last_divisor_two_and_influence():
    f=fixture(1,True);f['inputs']['x']['values']=[1]*12
    t=measured(1,True,f=f)['fixtures'][0]['tensors'];expected=oracle(f,1,True)
    assert t['output']['values']==pytest.approx([2/3,1,1,.5]*2)
    assert t['inputGradient']['values']==pytest.approx(expected[1])
    assert 'indices' not in t

@pytest.mark.parametrize('d',[1,2,3])
def test_floor_missing_indices_and_wrong_identity_refuse(d):
    for key in ['ceil_mode','return_indices']:
        s=submission(d);s['graph']['nodes'][1]['config'][key]=False
        with pytest.raises(ValueError):measured(d,s=s)
    s=submission(d);s['graph']['nodes'][1]['type']='linear'
    with pytest.raises(ValueError):measured(d,s=s)


def test_average_excluded_padding_is_measured_as_different_behavior():
    f=fixture(1,True);f['inputs']['x']['values']=[1]*12
    s=submission(1,True);s['graph']['nodes'][1]['config']['count_include_pad']=False
    t=measured(1,True,f=f,s=s)['fixtures'][0]['tensors']
    assert t['output']['values']==[1]*8


def test_cancel_and_nonfinite_and_authored_answers_refuse():
    assert measured(3,stopped=lambda:True)['fixtures']==[]
    f=fixture(1);f['inputs']['x']['values'][0]=float('nan')
    with pytest.raises(ValueError):measured(1,f=f)
    s=submission(1);s['expected']={}
    with pytest.raises(ValueError):measured(1,s=s)


def test_budget_stops_before_any_tensor_allocation(monkeypatch):
    evidence={'fixtures':[],'maximumTensorElements':0}
    def forbidden(*args,**kwargs):raise AssertionError('Allocation before resource admission')
    monkeypatch.setattr(learning_pooling.torch,'tensor',forbidden)
    learning_pooling.execute(submission(3),[fixture(3)],evidence,lambda:evidence['maximumTensorElements']>1,lambda t:(t['shape'],t['values']),dimensions=3)
    assert evidence['fixtures']==[] and evidence['maximumTensorElements']>1


def test_four_private_fixtures_fit_declared_4096_budget_and_int64_positions():
    fixtures=[]
    for i in range(4):
        f=fixture(3,ties=i==0);f['id']=str(i)
        if i%2:
            for t in f['inputs'].values():t['shape'][0]=2;t['values']*=2
        fixtures.append(f)
    evidence={'fixtures':[],'maximumTensorElements':0}
    learning_pooling.execute(submission(3),fixtures,evidence,lambda:evidence['maximumTensorElements']>4096,lambda t:(t['shape'],t['values']),dimensions=3)
    assert len(evidence['fixtures'])==4 and evidence['maximumTensorElements']<=4096
    for row in evidence['fixtures']:
        assert all(type(v) is int for v in row['tensors']['indices']['values'])
        assert all(0<=v<64 for v in row['tensors']['indices']['values'])


def test_batch_channels_remain_independent():
    f=fixture(1);f['inputs']['x']={'shape':[2,2,4],'values':[-2]*4+[3]*4+[-9]*4+[11]*4}
    f['inputs']['upstream']={'shape':[2,2,3],'values':[1]*12}
    t=measured(1,f=f)['fixtures'][0]['tensors']
    assert t['output']['values']==[-2]*3+[3]*3+[-9]*3+[11]*3
    assert t['indices']['values']==[0,1,3]*4


def test_equivalent_identity_layout_is_measured_and_indices_cannot_feed_float_path():
    s=submission(2);s['graph']['nodes'].insert(2,{'id':'shape','type':'reshape','config':{'shape':[2,3,3]}})
    s['graph']['edges'][1]['target']='shape';s['graph']['edges'].append({'id':'c','source':'shape','sourcePort':'out','target':'y','targetPort':'in'})
    assert measured(2,s=s)['fixtures'][0]['tensors']==measured(2)['fixtures'][0]['tensors']
    s['graph']['edges'][1]['sourcePort']='indices'
    with pytest.raises(ValueError):measured(2,s=s)


def test_policy_is_exact_versioned_and_four_distinct_capabilities():
    from runner.learning_evidence import ASSESSMENT_POLICIES
    for d in (1,2,3):
        p=ASSESSMENT_POLICIES[(f'assessment.max-pool{d}d-windows',(1,0,0))]
        assert p['pooling']==d and not p['average'] and set(p['capabilities'])=={f'max_pool{d}d'}
    p=ASSESSMENT_POLICIES[('assessment.avg-pool1d-windows',(1,0,0))]
    assert p['pooling']==1 and p['average'] and set(p['capabilities'])=={'avg_pool1d'}

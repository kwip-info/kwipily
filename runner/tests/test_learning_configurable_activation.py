import math
import pytest
from runner import learning_configurable_activation as worker


def submission(kind):
    config={} if kind=='silu' else {'negative_slope':.2} if kind=='leaky_relu' else {'alpha':2.}
    return {'schemaVersion':1,'kind':'configurableActivationGraph','graph':{'schemaVersion':2,'name':'Signed fields','nodes':[{'id':'x','type':'input','config':{}},{'id':'a','type':kind,'config':config},{'id':'y','type':'output','config':{}}],'edges':[{'id':'one','source':'x','sourcePort':'out','target':'a','targetPort':'in'},{'id':'two','source':'a','sourcePort':'out','target':'y','targetPort':'in'}]}}


def measured(kind,values=None,s=None):
    values=values or [-1000,-1,0,1,2,1000]
    f={'id':'public-independent','inputs':{'x':{'shape':[1,2,3],'values':values},'upstream':{'shape':[1,2,3],'values':[1]*6}}}
    e={'fixtures':[],'maximumTensorElements':0}
    worker.execute(s or submission(kind),[f],e,lambda:False,lambda t:(t['shape'],t['values']),component=kind)
    return e['fixtures'][0]['tensors']

@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_real_signed_forward_influence_and_zero_convention(kind):
    values=[-1000,-1,0,1,2,1000];y=[];dx=[]
    for x in values:
        if kind=='silu':
            s=1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x));y.append(x*s);dx.append(s+x*s*(1-s))
        elif kind=='leaky_relu':y.append(x if x>0 else .2*x);dx.append(1 if x>0 else .2)
        else:y.append(x if x>0 else 2*math.expm1(x));dx.append(1 if x>0 else 2*math.exp(x))
    t=measured(kind)
    assert t['output']['values']==pytest.approx(y,abs=1e-6)
    assert t['inputGradient']['values']==pytest.approx(dx,abs=1e-6)

@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_wrong_kind_identity_unknown_fields_and_inplace_refuse(kind):
    for mutation in ('identity','wrong','inplace','answer'):
        s=submission(kind)
        if mutation=='identity':s['graph']['nodes'][1]['type']='reshape';s['graph']['nodes'][1]['config']={'shape':[2,3]}
        elif mutation=='wrong':s['graph']['nodes'][1]['type']='relu';s['graph']['nodes'][1]['config']={}
        elif mutation=='inplace':s['graph']['nodes'][1]['config']['inplace']=True
        else:s['expected']={}
        with pytest.raises(ValueError):measured(kind,s=s)

@pytest.mark.parametrize('kind',['leaky_relu','elu'])
def test_default_parameter_is_real_but_fails_requested_values(kind):
    s=submission(kind);s['graph']['nodes'][1]['config']={}
    assert measured(kind,s=s)['output']['values']!=measured(kind)['output']['values']

@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_nonfinite_rejected_before_allocation_and_budget_cancel(monkeypatch,kind):
    def forbidden(*args,**kwargs):raise AssertionError('Allocation before admission')
    monkeypatch.setattr(worker.torch,'tensor',forbidden)
    f={'id':'bad','inputs':{'x':{'shape':[1,2,3],'values':[float('nan')]*6},'upstream':{'shape':[1,2,3],'values':[1]*6}}}
    e={'fixtures':[],'maximumTensorElements':0}
    with pytest.raises(ValueError):worker.execute(submission(kind),[f],e,lambda:False,lambda t:(t['shape'],t['values']),component=kind)
    f['inputs']['x']['values']=[-1,0,1]*2
    worker.execute(submission(kind),[f],e,lambda:e['maximumTensorElements']>1,lambda t:(t['shape'],t['values']),component=kind)
    assert e['fixtures']==[]
    e['maximumTensorElements']=0
    worker.execute(submission(kind),[f],e,lambda:True,lambda t:(t['shape'],t['values']),component=kind)
    assert e['fixtures']==[] and e['maximumTensorElements']==0

@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_equivalent_reshape_and_double_transpose_measures_same_behavior(kind):
    s=submission(kind)
    s['graph']['nodes'][2:2]=[{'id':'t1','type':'transpose','config':{'dim0':1,'dim1':2}},{'id':'t2','type':'transpose','config':{'dim0':1,'dim1':2}}]
    s['graph']['edges'][1]['target']='t1'
    s['graph']['edges'] += [{'id':'three','source':'t1','sourcePort':'out','target':'t2','targetPort':'in'},{'id':'four','source':'t2','sourcePort':'out','target':'y','targetPort':'in'}]
    assert measured(kind,s=s)==measured(kind)


def test_three_policy_identities_and_discrete_fixed_versions():
    from runner.learning_evidence import ASSESSMENT_POLICIES
    for task,kind in [('silu-values','silu'),('leaky-relu-slope','leaky_relu'),('elu-alpha','elu')]:
        p=ASSESSMENT_POLICIES[(f'assessment.{task}',(1,0,0))]
        assert p['configurableActivation']==kind and p['submissionSchema']==1 and set(p['capabilities'])=={kind}

import copy
import torch
from runner.learning_evidence import generate_evidence

V={'major':1,'minor':0,'patch':0}
def request():
    return {'assessmentID':'assessment.tensor-transpose','assessmentVersion':V,'fixtureDigest':'b'*64,
            'runtime':{'backend':'pytorch','version':str(torch.__version__),'capabilities':{'transpose':V}},
            'budget':{'milliseconds':5000,'maximumTensorElements':4096},
            'fixtures':[{'id':'one','inputs':{'x':{'shape':[1,2,3],'values':[0,1,2,3,4,5]},'upstream':{'shape':[1,3,2],'values':[1,2,3,4,5,6]}}}]}
def submission():
    nodes=[{'id':'i','type':'input','config':{}},{'id':'t','type':'transpose','config':{'dim0':1,'dim1':2}},{'id':'o','type':'output','config':{}}]
    return {'schemaVersion':1,'kind':'tensorGraph','graph':{'schemaVersion':2,'nodes':nodes,'edges':[{'id':'a','source':'i','sourcePort':'out','target':'t','targetPort':'in'},{'id':'b','source':'t','sourcePort':'out','target':'o','targetPort':'in'}]}}
def measure(s=None,r=None,**kwargs):return generate_evidence(r or request(),s or submission(),submission_digest='a'*64,**kwargs)
def test_real_output_and_autograd_evidence():
    result=measure()
    assert not result.get('backendError')
    tensors=result['fixtures'][0]['tensors']
    assert tensors['output']=={'shape':[1,3,2],'values':[0,3,1,4,2,5]}
    assert tensors['inputGradient']=={'shape':[1,2,3],'values':[1,3,5,2,4,6]}
    assert result['submissionDigest']=='a'*64

def test_wrong_but_executable_construction_returns_measured_values():
    s=submission();s['graph']['nodes'][1].update(type='reshape',config={'shape':[3,2]})
    result=measure(s)
    assert not result.get('backendError')
    assert result['fixtures'][0]['tensors']['output']['values']==[0,1,2,3,4,5]

def test_invalid_graph_cancel_and_budget_do_not_emit_success():
    for mutate in [lambda s:s.update(expected={}),lambda s:s['graph']['nodes'][1].update(type='code'),lambda s:s['graph']['edges'][0].update(source='t'),lambda s:s['graph']['nodes'][1].update(config={'dim0':0,'dim1':1})]:
        s=submission();mutate(s);result=measure(s)
        assert result.get('backendError') and not result['fixtures']
    assert measure(cancel=lambda:True)['cancelled']
    r=request();r['budget']['maximumTensorElements']=1
    result=measure(r=r);assert not result['fixtures'] and result['maximumTensorElements']>1

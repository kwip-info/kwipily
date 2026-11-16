import copy
import pytest
import torch
from runner import learning_lazy_linear


def submission(bias=True, output=4):
    return {'schemaVersion':1,'kind':'lazyLinearGraph','graph':{'schemaVersion':2,'name':'My layer','nodes':[
        {'id':'x','type':'input','config':{}}, {'id':'map','type':'lazy_linear','config':{'out_features':output,'bias':bias}},
        {'id':'y','type':'output','config':{}}], 'edges':[
        {'id':'a','source':'x','sourcePort':'out','target':'map','targetPort':'in'},
        {'id':'b','source':'map','sourcePort':'out','target':'y','targetPort':'in'}]}}


def measured(graph, stopped=lambda:False):
    evidence={'fixtures':[],'maximumTensorElements':0}
    fixtures=[{'id':'small','inputs':{'x':{'shape':[2,2,3],'values':[i/4 for i in range(12)]}}}]
    learning_lazy_linear.execute(graph,fixtures,evidence,stopped,lambda t:(t['shape'],t['values']))
    return evidence

@pytest.mark.parametrize('bias',[False,True])
def test_real_lazy_shape_inventory_repeat_reload_and_rng(bias):
    evidence=measured(submission(bias))
    t=evidence['fixtures'][0]['tensors']
    assert t['output-shape']['values']==[2,2,4]
    assert t['inferred-width']['values']==[3]
    assert t['weight-shape']['values']==[4,3]
    for k in ['parameter-count','repeat-state','reload-state','rng-preserved']:assert t[k]['values']==[1]
    assert evidence['maximumTensorElements']>=16


def test_wrong_width_remains_measured_not_fabricated_expected_shape():
    assert measured(submission(output=5))['fixtures'][0]['tensors']['output-shape']['values']==[2,2,5]

@pytest.mark.parametrize('kind',['linear','sigmoid','constant'])
def test_wrong_identity_refused(kind):
    s=submission();s['graph']['nodes'][1]['type']=kind
    with pytest.raises(ValueError):measured(s)


def test_disconnected_and_authored_answers_refused():
    s=submission();s['graph']['edges'].pop()
    with pytest.raises(ValueError):measured(s)
    s=submission();s['expected']={}
    with pytest.raises(ValueError):measured(s)


def test_equivalent_identity_layout_passes_and_flatten_misconception_measured():
    for kind,config,width in [('reshape',{'shape':[2,3]},3),('transpose',{'dim0':1,'dim1':1},3),('flatten',{'start_dim':1,'end_dim':2},6)]:
        s=submission();s['graph']['nodes'].insert(1,{'id':'layout','type':kind,'config':config})
        s['graph']['edges'][0]['target']='layout'
        s['graph']['edges'].append({'id':'c','source':'layout','sourcePort':'out','target':'map','targetPort':'in'})
        t=measured(s)['fixtures'][0]['tensors']
        assert t['inferred-width']['values']==[width]
        assert t['output-shape']['values']==([2,4] if kind=='flatten' else [2,2,4])


def test_cancel_before_materialization_does_not_draw_rng():
    before=torch.get_rng_state().clone()
    assert measured(submission(),stopped=lambda:True)['fixtures']==[]
    assert torch.equal(before,torch.get_rng_state())


def test_nonfinite_fixture_refused():
    e={'fixtures':[],'maximumTensorElements':0}
    f=[{'id':'bad','inputs':{'x':{'shape':[1,2,3],'values':[float('nan')]*6}}}]
    with pytest.raises(ValueError):learning_lazy_linear.execute(submission(),f,e,lambda:False,lambda t:(t['shape'],t['values']))


def test_application_policy_dispatch_and_budget_boundary():
    from runner.learning_evidence import generate_evidence, runtime_versions
    version=runtime_versions()['pytorch']
    request={'assessmentID':'assessment.lazy-linear-inferred-width','assessmentVersion':{'major':1,'minor':0,'patch':0},
        'fixtureDigest':'a'*64,'runtime':{'backend':'pytorch','version':version,'capabilities':{'lazy_linear':{'major':1,'minor':0,'patch':0}}},
        'budget':{'milliseconds':5000,'maximumTensorElements':4096},'fixtures':[{'id':'one','inputs':{'x':{'shape':[1,2,3],'values':[1,2,3,4,5,6]}}}]}
    e=generate_evidence(request,submission(),submission_digest='b'*64)
    assert 'backendError' not in e and len(e['fixtures'])==1
    assert e['fixtures'][0]['tensors']['reload-state']['values']==[1]
    request['budget']['maximumTensorElements']=1
    before=torch.get_rng_state().clone()
    e=generate_evidence(request,submission(),submission_digest='b'*64)
    assert not e['fixtures'] and e['maximumTensorElements']>1
    assert torch.equal(before,torch.get_rng_state())


def test_changed_repeat_outputs_are_measured_as_failure(monkeypatch):
    original=torch.nn.Linear.forward
    calls=[0]
    def changed(module,x):
        calls[0]+=1
        return original(module,x)+calls[0]
    monkeypatch.setattr(torch.nn.Linear,'forward',changed)
    t=measured(submission())['fixtures'][0]['tensors']
    assert t['repeat-state']['values']==[0]
    assert t['reload-state']['values']==[0]


def test_reload_weight_corruption_is_measured_as_failure(monkeypatch):
    original=torch.nn.Module.load_state_dict
    def changed(module,state,*args,**kwargs):
        result=original(module,state,*args,**kwargs)
        with torch.no_grad():module.weight.add_(1)
        return result
    monkeypatch.setattr(torch.nn.Module,'load_state_dict',changed)
    t=measured(submission())['fixtures'][0]['tensors']
    assert t['repeat-state']['values']==[1]
    assert t['reload-state']['values']==[0]


def test_extra_random_draws_are_measured_as_failure(monkeypatch):
    original=torch.nn.Linear.forward
    def changed(module,x):
        torch.rand(1)
        return original(module,x)
    monkeypatch.setattr(torch.nn.Linear,'forward',changed)
    t=measured(submission())['fixtures'][0]['tensors']
    assert t['repeat-state']['values']==[1]
    assert t['rng-preserved']['values']==[0]

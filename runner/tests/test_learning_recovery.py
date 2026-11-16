import copy
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_evidence import affine_request,affine_submission
from runner.tests.test_learning_optional_bias import mixed,V11

V={'major':1,'minor':0,'patch':0}
def request():
    r=affine_request();r['assessmentID']='assessment.affine-from-probes';r['runtime']['capabilities']['linear']=V11;return r

def measured(s,r=None,**kwargs):return generate_evidence(r or request(),s,submission_digest='d'*64,**kwargs)

def current(s):s['schemaVersion']=2;return s

@pytest.mark.parametrize('submission',[current(affine_submission()),current(affine_submission(True)),mixed()])
def test_recovery_equivalent_graphs_have_same_measured_outputs(submission):
    e=measured(submission);old=measured(affine_submission(),affine_request())
    assert e['fixtures']==old['fixtures'] and not e.get('backendError')
    assert e['assessmentID']=='assessment.affine-from-probes' and e['assessmentVersion']==V
    assert e['runtime']['capabilities']['linear']==V11

@pytest.mark.parametrize('version',[{'major':1,'minor':1,'patch':0},{'major':True,'minor':0,'patch':0},{'major':2,'minor':0,'patch':0}])
def test_unknown_recovery_versions_rejected(version):
    r=request();r['assessmentVersion']=version
    with pytest.raises(ValueError):measured(mixed(),r)

def test_recovery_schema_and_runtime_routes_cannot_cross():
    e=measured(affine_submission());assert e.get('backendError') and not e['fixtures']
    r=request();r['runtime']['capabilities']['linear']=V
    e=measured(mixed(),r);assert not e['fixtures']
    r=request();r['assessmentID']='assessment.unknown'
    with pytest.raises(ValueError):measured(mixed(),r)

@pytest.mark.parametrize('kind',['relu','tanh'])
def test_recovery_rejects_nonlinear_components_without_changing_composition(kind):
    s=current(affine_submission());s['graph']['nodes'].insert(-1,{'id':'activation','type':kind,'name':'Activation','x':0,'y':0,'config':{}})
    s['graph']['edges'][-1]['target']='activation'
    s['graph']['edges'].append({'id':'extra','source':'activation','sourcePort':'out','target':'output','targetPort':'in'})
    e=measured(s);assert 'affine-preserving' in e.get('backendError','') and not e['fixtures']
    legacy=copy.deepcopy(s);legacy['schemaVersion']=1
    assert measured(legacy,affine_request())['fixtures']
    r=affine_request();r['assessmentVersion']=V11;r['runtime']['capabilities']['linear']=V11
    assert measured(s,r)['fixtures']

def test_recovery_wrong_values_execute_and_budget_cancel_remain_bounded():
    s=mixed();s['parameters']['second']['bias']['values'][0]+=1
    assert measured(s)['fixtures']!=measured(mixed())['fixtures']
    e=measured(mixed(),cancel=lambda:True);assert e['cancelled'] and not e['fixtures']
    r=request();r['budget']['maximumTensorElements']=1
    e=measured(mixed(),r);assert not e['fixtures'] and e['maximumTensorElements']>1
    assert 'expected' not in str(request().keys()) and 'taskSeed' not in request()

@pytest.mark.parametrize('merge',['add','concat'])
def test_recovery_affine_branch_combinations_execute_equivalently(merge):
    from runner.tests.test_learning_evidence import tensor
    s=current(affine_submission());g=s['graph'];g['schemaVersion']=2
    if merge=='add':
        specs=[('input','input',{}),('left','linear',{}),('right','linear',{}),('merge','add',{}),('output','output',{})]
        params={'left':copy.deepcopy(s['parameters']['first']),'right':{'weight':tensor([2,2],[0,0,0,0])}}
        widths={'left':(2,2,True),'right':(2,2,False)}
    else:
        specs=[('input','input',{}),('left','linear',{}),('right','linear',{}),('merge','concat',{}),('head','linear',{}),('output','output',{})]
        params={'left':{'weight':tensor([1,2],[3,5]),'bias':tensor([1],[-2.75])},'right':{'weight':tensor([1,2],[-3.5,5]),'bias':tensor([1],[-3])},'head':{'weight':tensor([2,2],[1,0,0,1])}}
        widths={'left':(2,1,True),'right':(2,1,True),'head':(2,2,False)}
    g['nodes']=[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':({'features':2} if t=='input' else {'in_features':widths[i][0],'out_features':widths[i][1],'bias':widths[i][2]} if t=='linear' else {})} for i,t,_ in specs]
    wires=[('input','left','in'),('input','right','in'),('left','merge','a'),('right','merge','b')]+([('merge','output','in')] if merge=='add' else [('merge','head','in'),('head','output','in')])
    g['edges']=[{'id':str(i),'source':a,'sourcePort':'out','target':b,'targetPort':p} for i,(a,b,p) in enumerate(wires)];s['parameters']=params
    e=measured(s);assert not e.get('backendError') and e['fixtures']==measured(current(affine_submission()))['fixtures']

def test_recovery_actual_cli_has_no_expected_outputs_in_request(tmp_path):
    import hashlib,json,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];r=tmp_path/'request.json';s=tmp_path/'submission.json'
    r.write_text(json.dumps(request()));raw=json.dumps(mixed()).encode();s.write_bytes(raw)
    result=subprocess.run([sys.executable,'-B',str(root/'runner/main.py'),'--learning-evidence-request',str(r),'--learning-submission',str(s),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    e=json.loads(result.stdout)['evidence'];assert e['assessmentID']=='assessment.affine-from-probes' and e['fixtures']

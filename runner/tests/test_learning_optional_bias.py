import copy
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_evidence import affine_request,affine_submission,tensor,V

V11={'major':1,'minor':1,'patch':0}
def request():
    r=affine_request();r['assessmentVersion']=V11;r['runtime']['capabilities']['linear']=V11;return r

def mixed():
    s=affine_submission(True);s['schemaVersion']=2;s['graph']['schemaVersion']=2
    for n in s['graph']['nodes']:
        if n['type']=='linear':n['config']['bias']=n['id']=='second'
    s['parameters']={'first':{'weight':tensor([2,2],[1,0,0,1])},'second':affine_submission()['parameters']['first']}
    return s

def evidence(s,r=None,**kwargs):return generate_evidence(r or request(),s,submission_digest='b'*64,**kwargs)

def test_mixed_optional_bias_matches_target_with_correct_version():
    e=evidence(mixed());old=evidence(affine_submission(),affine_request())
    assert not e.get('backendError')
    assert e['fixtures']==old['fixtures']
    assert e['assessmentVersion']==V11 and e['runtime']['capabilities']['linear']==V11
    assert e['maximumTensorElements']==26 # 10 active parameters + four two-wide node activations, two rows

def test_all_biasless_executes_and_differs_numerically():
    s=mixed();s['graph']['nodes'][2]['config']['bias']=False;del s['parameters']['second']['bias']
    e=evidence(s);target=evidence(affine_submission(),affine_request())
    assert not e.get('backendError') and len(e['fixtures'])==2
    assert e['fixtures']!=target['fixtures']
    assert e['maximumTensorElements']==24

@pytest.mark.parametrize('value',[None,tensor([2],[0,0]),False,0])
def test_disabled_bias_rejects_any_extra_bias(value):
    s=mixed();s['parameters']['first']['bias']=value;e=evidence(s)
    assert e.get('backendError') and not e['fixtures']

@pytest.mark.parametrize('mutation',['missing','null','extra','node','shape'])
def test_active_parameter_ownership_is_exact(mutation):
    s=mixed()
    if mutation=='missing':del s['parameters']['second']['bias']
    if mutation=='null':s['parameters']['second']['bias']=None
    if mutation=='extra':s['parameters']['second']['other']=tensor([1],[0])
    if mutation=='node':s['parameters']['unowned']=s['parameters']['first']
    if mutation=='shape':s['parameters']['first']['weight']=tensor([1],[1])
    e=evidence(s);assert e.get('backendError') and not e['fixtures']

def test_legacy_replay_and_version_routing_are_strict():
    assert evidence(affine_submission(),affine_request())['fixtures']
    current=affine_submission();current['schemaVersion']=2
    assert evidence(current)['fixtures'] # graph1 allowed, lossless replay
    assert evidence(mixed(),affine_request()).get('backendError')
    assert evidence(affine_submission()).get('backendError')
    r=request();r['runtime']['capabilities']['linear']=V
    e=evidence(mixed(),r);assert not e['fixtures']
    r=request();r['assessmentVersion']={'major':1,'minor':2,'patch':0}
    with pytest.raises(ValueError):evidence(mixed(),r)

def test_optional_bias_cancellation_and_budget():
    e=evidence(mixed(),cancel=lambda:True);assert e['cancelled'] and not e['fixtures']
    r=request();r['budget']['maximumTensorElements']=9
    e=evidence(mixed(),r);assert e['maximumTensorElements']==10 and not e['fixtures']

def test_optional_bias_actual_cli_echoes_protocol_and_raw_submission_identity(tmp_path):
    import hashlib,json,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    r=tmp_path/'request.json';s=tmp_path/'submission.json'
    r.write_text(json.dumps(request()));raw=json.dumps(mixed()).encode();s.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    result=subprocess.run([sys.executable,'-B',str(root/'runner/main.py'),'--learning-evidence-request',str(r),'--learning-submission',str(s),'--learning-submission-sha256',digest],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    event=json.loads(result.stdout)
    measured=event['evidence']
    assert measured['assessmentVersion']==V11 and measured['submissionDigest']==digest
    assert measured['fixtures'] and not measured.get('backendError')

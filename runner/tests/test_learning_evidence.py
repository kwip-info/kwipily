import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
V={'major':1,'minor':0,'patch':0}

def tensor(shape,values):return {'shape':shape,'values':values}
def affine_request():
    return {'assessmentID':'assessment.affine-composition','assessmentVersion':dict(V),'fixtureDigest':'a'*64,'runtime':{'backend':'pytorch','version':str(torch.__version__),'capabilities':{'linear':dict(V)}},'budget':{'milliseconds':5000,'maximumTensorElements':4096},'fixtures':[{'id':'unseen1','inputs':{'x':tensor([2,2],[1,0,0,1])}},{'id':'unseen2','inputs':{'x':tensor([2,2],[2,-3,-2,5])}}]}

def affine_submission(composed=False):
    specs=[('input','input',{'features':2}),('first','linear',{'in_features':2,'out_features':2})]
    if composed:specs.append(('second','linear',{'in_features':2,'out_features':2}))
    specs.append(('output','output',{}))
    graph={'schemaVersion':1,'name':'Alternate labels','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],'edges':[{'id':str(n),'source':specs[n][0],'sourcePort':'out','target':specs[n+1][0],'targetPort':'in'} for n in range(len(specs)-1)]}
    parameters={'first':{'weight':tensor([2,2],[3,5,-3.5,5]),'bias':tensor([2],[-2.75,-3])}}
    if composed:parameters={'first':{'weight':tensor([2,2],[2,-1,.5,3]),'bias':tensor([2],[1,-2])},'second':{'weight':tensor([2,2],[1,2,-2,1]),'bias':tensor([2],[.25,1])}}
    return {'schemaVersion':1,'kind':'affineGraph','graph':graph,'parameters':parameters}

def test_real_affine_alternatives_match_and_parameters_change_measured_output():
    from runner.learning_evidence import generate_evidence
    request=affine_request()
    a=generate_evidence(request,affine_submission(),submission_digest='b'*64)
    b=generate_evidence(request,affine_submission(True),submission_digest='c'*64)
    assert a['fixtures']==b['fixtures']
    assert a['fixtures'][0]['tensors']['output']['values']==[.25,-6.5,2.25,2.0]
    changed=affine_submission();changed['parameters']['first']['bias']['values'][0]=100
    c=generate_evidence(request,changed,submission_digest='d'*64)
    assert c['fixtures']!=a['fixtures']
    assert a['runtime']['version']==str(torch.__version__) and a['maximumTensorElements']>0

def test_authored_receipt_and_wrong_runtime_are_not_accepted():
    from runner.learning_evidence import generate_evidence
    submission=affine_submission();submission['fixtures']=[{'fake':'authored results'}]
    evidence=generate_evidence(affine_request(),submission,submission_digest='b'*64)
    assert evidence.get('backendError') and not evidence['fixtures']
    request=affine_request();request['runtime']['version']='0.0.0'
    evidence=generate_evidence(request,affine_submission(),submission_digest='b'*64)
    assert evidence['runtime']['version']==str(torch.__version__) and not evidence['fixtures']

def test_budget_and_cancellation_are_reported_separately():
    from runner.learning_evidence import generate_evidence
    request=affine_request();request['budget']['maximumTensorElements']=1
    evidence=generate_evidence(request,affine_submission(),submission_digest='b'*64)
    assert evidence['maximumTensorElements']>1 and not evidence.get('backendError')
    evidence=generate_evidence(affine_request(),affine_submission(),submission_digest='b'*64,cancel=lambda:True)
    assert evidence['cancelled'] and not evidence.get('backendError')

def test_actual_cli_and_bound_hash_pair(tmp_path):
    request=tmp_path/'request.json';request.write_text(json.dumps(affine_request()))
    submission=tmp_path/'submission.json';raw=json.dumps(affine_submission()).encode();submission.write_bytes(raw)
    command=[sys.executable,str(ROOT/'runner/main.py'),'--learning-evidence-request',str(request),'--learning-submission',str(submission),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()]
    good=subprocess.run(command,capture_output=True,text=True);assert good.returncode==0,good.stdout+good.stderr
    assert json.loads(good.stdout)['event']=='learningEvidence'
    conflict=subprocess.run(command+['--capabilities'],capture_output=True,text=True);assert conflict.returncode==1
    submission.write_text('{}');bad=subprocess.run(command,capture_output=True,text=True);assert bad.returncode==1 and 'digest' in bad.stdout

def test_boolean_versions_and_orphan_execution_flags_are_rejected(tmp_path):
    from runner.learning_evidence import generate_evidence
    request=affine_request();request['assessmentVersion']={'major':True,'minor':0,'patch':0}
    with pytest.raises(ValueError,match='Unsupported'):
        generate_evidence(request,affine_submission(),submission_digest='b'*64)
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--learning-runtime','--prepared-manifest-sha256','a'*64],capture_output=True,text=True)
    assert result.returncode==1

def test_bounded_submission_files_and_nonfinite_parameters(tmp_path):
    from runner.learning_evidence import bounded_read,generate_evidence
    source=tmp_path/'oversized.json';source.write_bytes(b' '*2_000_001)
    with pytest.raises(ValueError,match='bounded'):bounded_read(source)
    link=tmp_path/'link.json';link.symlink_to(source)
    with pytest.raises(ValueError,match='regular'):bounded_read(link)
    submission=affine_submission();submission['parameters']['first']['bias']['values'][0]=float('nan')
    evidence=generate_evidence(affine_request(),submission,submission_digest='b'*64)
    assert evidence.get('backendError') and not evidence['fixtures']

def test_packaged_launcher_discovers_and_executes_using_its_registry(tmp_path):
    import shutil
    bundle=tmp_path/'runtime';bundle.mkdir()
    shutil.copy2(ROOT/'scripts/runtime_launch.py',bundle/'launch.py')
    shutil.copytree(ROOT/'runner',bundle/'runner',ignore=shutil.ignore_patterns('__pycache__','tests'))
    registry=bundle/'components.json';shutil.copy2(ROOT/'Sources/KwipMLCore/Resources/components.json',registry)
    # A packaged bootstrap requires a build-time inventory; development main.py does not.
    import importlib.util
    spec=importlib.util.spec_from_file_location('learning_test_packager',ROOT/'scripts/runtime_package.py')
    packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)
    (bundle/'runtime-manifest.json').write_text('{"schemaVersion":1}')
    packager.refresh_worker_integrity(bundle)
    base=[sys.executable,'-I','-B',str(bundle/'launch.py')]
    discovery=subprocess.run(base+['--learning-runtime'],capture_output=True,text=True,cwd=tmp_path)
    assert discovery.returncode==0,discovery.stdout+discovery.stderr
    assert json.loads(discovery.stdout)['versions']['pytorch']==str(torch.__version__)
    request=tmp_path/'request.json';request.write_text(json.dumps(affine_request()))
    submission=tmp_path/'submission.json';raw=json.dumps(affine_submission()).encode();submission.write_bytes(raw)
    arguments=['--learning-evidence-request',str(request),'--learning-submission',str(submission),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()]
    good=subprocess.run(base+arguments,capture_output=True,text=True,cwd=tmp_path)
    assert good.returncode==0,good.stdout+good.stderr
    evidence=json.loads(good.stdout)['evidence'];assert evidence['fixtures'] and not evidence.get('backendError'),evidence
    changed=json.loads(registry.read_text())
    linear=next(c for c in changed['components'] if c['id']=='linear')
    linear['fields'][0]['max']=1;registry.write_text(json.dumps(changed))
    packager.refresh_worker_integrity(bundle)
    restricted=subprocess.run(base+arguments,capture_output=True,text=True,cwd=tmp_path)
    evidence=json.loads(restricted.stdout)['evidence']
    assert evidence.get('backendError') and not evidence['fixtures'],evidence

"""Scalar expectations and separately captured upstream masks, never adapter-derived answers."""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import pytest
import torch

BANK_PATH=Path(__file__).resolve().parents[1]/'fixtures/remaining_activation_dropout_masks_v1.json'
BANK=json.loads(BANK_PATH.read_text())
TASKS={'prelu-shared-slope':('prelu',{'num_parameters':1,'init':.25},[.25]),'prelu-channel-slopes':('prelu',{'num_parameters':2,'init':.25},[.1,.5]),'logsoftmax-sample-scores':('log_softmax',{'dim':1},None),'dropout-training-pattern':('dropout',{'p':.5},None),'dropout-evaluation-pattern':('dropout',{'p':.5},None)}

def inputs(rows=2,bank_index=0):
    shape=[rows,2,3]
    return {'x':{'shape':shape,'values':[(i%11-5)/2 for i in range(rows*6)]},'upstream':{'shape':shape,'values':[(i%7-3)/4 for i in range(rows*6)]},**({'maskBankIndex':{'shape':[1],'values':[bank_index]}} if bank_index is not None else {})}

def submission(task):
    kind,config,slopes=TASKS[task]
    nodes=[{'id':'x','type':'input','config':{}},{'id':'op','type':kind,'config':copy.deepcopy(config)},{'id':'out','type':'output','config':{}}]
    edges=[dict(id=str(i),source=a,sourcePort='out',target=b,targetPort='in') for i,(a,b) in enumerate([('x','op'),('op','out')])]
    return dict(schemaVersion=1,kind='remainingActivationGraph',graph=dict(schemaVersion=2,nodes=nodes,edges=edges),parameters={'op':{'weight':{'shape':[len(slopes)],'values':copy.deepcopy(slopes)}}} if slopes else {},gradientNodeID='op' if slopes else None)

def run(task,fixture=None,s=None,stopped=lambda:False):
    worker=importlib.import_module('runner.learning_remaining_activation')
    fixture=fixture or inputs(bank_index=0 if task.startswith('dropout') else None)
    evidence={'maximumTensorElements':0,'fixtures':[]}
    worker.execute(s or submission(task),[{'id':'fixture','inputs':fixture}],evidence,stopped,lambda t:(t['shape'],t['values']),task=task)
    return evidence

def oracle(task,fixture):
    x=fixture['x']['values'];g=fixture['upstream']['values'];rows=fixture['x']['shape'][0]
    if task.startswith('prelu'):
        slopes=TASKS[task][2];y=[];dx=[];dw=[0.]*len(slopes)
        for i,(a,b) in enumerate(zip(x,g)):
            channel=(i//3)%2;k=0 if len(slopes)==1 else channel;w=slopes[k]
            y.append(a if a>0 else a*w);dx.append(b*(1 if a>0 else w));dw[k]+=b*a if a<0 else 0
        return dict(output=y,inputGradient=dx,slopeGradient=dw)
    if task.startswith('logsoftmax'):
        y=[0.]*len(x);dx=[0.]*len(x)
        for n in range(rows):
            for col in range(3):
                ids=[n*6+col,n*6+3+col];m=max(x[i] for i in ids);den=sum(math.exp(x[i]-m) for i in ids);total=sum(g[i] for i in ids)
                for i in ids:y[i]=x[i]-m-math.log(den);dx[i]=g[i]-math.exp(x[i]-m)/den*total
        return dict(output=y,inputGradient=dx)
    entry=BANK['entries'][int(fixture['maskBankIndex']['values'][0])];mask=entry['mask']
    training=[a*m*2 for a,m in zip(x,mask)];gradient=[a*m*2 for a,m in zip(g,mask)]
    if task=='dropout-evaluation-pattern':return dict(output=x,inputGradient=g,trainingOutput=training,trainingInputGradient=gradient)
    return dict(output=training,inputGradient=gradient)

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('rows',[2,3,4])
def test_actual_float64_worker_values_and_weighted_influence(task,rows):
    index=rows-2;f=inputs(rows,index if task.startswith('dropout') else None)
    actual=run(task,f)['fixtures'][0]['tensors'];expected=oracle(task,f)
    assert set(actual)==set(expected)
    for key,values in expected.items():assert actual[key]['values']==pytest.approx(values,abs=1e-12,rel=1e-12)

@pytest.mark.parametrize('task',['dropout-training-pattern','dropout-evaluation-pattern'])
@pytest.mark.parametrize('index',range(12))
def test_all_independent_canonical_masks_measure_actual_native_dropout(task,index):
    f=inputs(BANK['entries'][index]['shape'][0],index);expected=oracle(task,f);actual=run(task,f)['fixtures'][0]['tensors']
    for key,values in expected.items():assert actual[key]['values']==pytest.approx(values)

def test_bank_provenance_identical_resources_and_generator_hash():
    root=BANK_PATH.parents[2];assert BANK_PATH.read_bytes()==(root/'Sources/KwipMLCore/Resources/remaining_activation_dropout_masks_v1.json').read_bytes()
    assert BANK['generatorSHA256']==hashlib.sha256((BANK_PATH.parent/'generate_remaining_activation_dropout_masks.py').read_bytes()).hexdigest()
    assert hashlib.sha256(BANK_PATH.read_bytes()).hexdigest()=='5a96044ae7f6286b354d6cb43cd6bec4e040a15b55bffc01ecaa57bbc1056538'
    assert BANK['torchVersion']=='2.14.1' and BANK['dtype']=='float64' and BANK['device']=='cpu'

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('bad',['answer','seed','mask','unknown','duplicate','cycle','unused','wrongop','gradientTarget','extraParameters','nonfinite','shape','fixtureExtra'])
def test_authored_answers_and_invalid_graph_state_fixtures_refused(task,bad):
    s=submission(task);f=inputs(bank_index=0 if task.startswith('dropout') else None)
    if bad in ('answer','seed','mask'):s[bad]=[1]
    elif bad=='unknown':s['graph']['nodes'][1]['config']['inplace']=True
    elif bad=='duplicate':s['graph']['edges'].append(copy.deepcopy(s['graph']['edges'][0]))
    elif bad=='cycle':s['graph']['edges'][0]['source']='op'
    elif bad=='unused':s['graph']['nodes'].append(dict(id='unused',type='reshape',config={'shape':[2,3]}))
    elif bad=='wrongop':s['graph']['nodes'][1]['type']='relu'
    elif bad=='gradientTarget':s['gradientNodeID']='out'
    elif bad=='extraParameters':s['parameters']['out']={'weight':{'shape':[1],'values':[.1]}}
    elif bad=='nonfinite':f['x']['values'][0]=math.nan
    elif bad=='shape':f['x']['shape']=[2,3,2]
    else:f['expectedOutput']={'shape':[1],'values':[0]}
    with pytest.raises(ValueError):run(task,f,s)

def test_wrong_valid_slopes_axis_and_probability_are_measured_not_replaced_with_answers():
    for task,key,value in [('prelu-channel-slopes','weights',[.4,.4]),('logsoftmax-sample-scores','dim',-1),('dropout-evaluation-pattern','p',0.)]:
        s=submission(task);f=inputs(bank_index=0 if task.startswith('dropout') else None)
        if key=='weights':s['parameters']['op']['weight']['values']=value
        else:s['graph']['nodes'][1]['config'][key]=value
        actual=run(task,f,s)['fixtures'][0]['tensors'];expected=oracle(task,f)
        assert any(actual[k]['values']!=pytest.approx(v) for k,v in expected.items())

def test_runtime_and_bank_corruption_refuse_stale_masks(monkeypatch,tmp_path):
    worker=importlib.import_module('runner.learning_remaining_activation')
    monkeypatch.setattr(torch,'__version__','2.99.0')
    with pytest.raises(ValueError,match='runtime'):run('dropout-training-pattern')
    monkeypatch.setattr(torch,'__version__','2.14.1')
    changed=tmp_path/'bank.json';changed.write_bytes(BANK_PATH.read_bytes()+b' ');monkeypatch.setattr(worker,'BANK_PATH',changed)
    with pytest.raises(ValueError,match='integrity'):run('dropout-training-pattern')

def test_cancel_and_resource_budget_precede_tensor_allocation(monkeypatch):
    worker=importlib.import_module('runner.learning_remaining_activation')
    assert run('prelu-shared-slope',stopped=lambda:True)['fixtures']==[]
    def forbidden(*a,**k):raise AssertionError('allocated before budget refusal')
    monkeypatch.setattr(torch,'tensor',forbidden)
    class Budget(dict):
        def __setitem__(self,k,v):
            if k=='maximumTensorElements' and v>0:raise ValueError('budget')
            super().__setitem__(k,v)
    with pytest.raises(ValueError,match='budget'):worker.execute(submission('prelu-shared-slope'),[dict(id='x',inputs=inputs(bank_index=None))],Budget(maximumTensorElements=0,fixtures=[]),lambda:False,lambda t:(t['shape'],t['values']),task='prelu-shared-slope')

@pytest.mark.parametrize('task',TASKS)
def test_labels_positions_identity_layout_and_renamed_state_do_not_change_grade_inputs(task):
    s=submission(task);f=inputs(bank_index=0 if task.startswith('dropout') else None)
    baseline=run(task,f)['fixtures'][0]['tensors']
    s['graph']['nodes'][1]['id']='my-block';s['graph']['nodes'][1].update(name='My experiment',x=-300,y=48)
    for edge in s['graph']['edges']:
        if edge['source']=='op':edge['source']='my-block'
        if edge['target']=='op':edge['target']='my-block'
    if task.startswith('prelu'):s['parameters']['my-block']=s['parameters'].pop('op');s['gradientNodeID']='my-block'
    s['graph']['nodes'].append(dict(id='layout',type='reshape',config={'shape':[2,3]}))
    s['graph']['edges'][0]['target']='layout'
    s['graph']['edges'].append(dict(id='layout-edge',source='layout',sourcePort='out',target='my-block',targetPort='in'))
    assert run(task,f,s)['fixtures'][0]['tensors']==baseline

def test_weight_map_configuration_bounds_and_unknown_target_are_not_answer_channels():
    task='prelu-channel-slopes'
    for mutate in ('nonfinite','missing','unknownWeight','badCount','boolCount','seedConfig'):
        s=submission(task)
        if mutate=='nonfinite':s['parameters']['op']['weight']['values'][0]=math.inf
        elif mutate=='missing':s['parameters']={}
        elif mutate=='unknownWeight':s['parameters']['op']['bias']={'shape':[1],'values':[1]}
        elif mutate=='badCount':s['graph']['nodes'][1]['config']['num_parameters']=3;s['parameters']['op']['weight']={'shape':[3],'values':[.1,.2,.3]}
        elif mutate=='boolCount':s['graph']['nodes'][1]['config']['num_parameters']=True
        else:s['graph']['nodes'][1]['config']['seed']=7
        with pytest.raises(ValueError):run(task,s=s)

@pytest.mark.parametrize('task',TASKS)
def test_real_assessment_dispatch_binds_reviewed_capability_and_measures_no_answers(task):
    from runner.learning_evidence import generate_evidence
    from runner.tests.test_learning_tensor_transforms import request
    r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':2,'patch':0}}
    f=inputs(bank_index=0 if task.startswith('dropout') else None);r['fixtures']=[dict(id='fixture',inputs=f)]
    result=generate_evidence(r,submission(task),submission_digest='a'*64)
    assert not result.get('backendError'),result
    assert result['submissionDigest']=='a'*64 and result['fixtureDigest']==r['fixtureDigest']
    tensors=result['fixtures'][0]['tensors']
    for key,values in oracle(task,f).items():assert tensors[key]['values']==pytest.approx(values)
    assert 'maskBankIndex' not in tensors and 'mask' not in tensors

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_four_changed_fixture_sets_bind_selected_policy_and_submission(tmp_path,task,mode):
    import subprocess,sys
    from runner.tests.test_learning_tensor_transforms import request
    r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':2,'patch':0}}
    r['fixtures']=[]
    for i in range(4):
        rows=i%3+2;f=inputs(rows,(i%4)*3+(rows-2) if task.startswith('dropout') else None)
        for name in ('x','upstream'):f[name]['values']=[v+(i+len(mode))/8 for v in f[name]['values']]
        r['fixtures'].append(dict(id=mode+'-'+str(i),inputs=f))
    req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(task),sort_keys=True).encode();sub.write_bytes(raw)
    completed=subprocess.run([sys.executable,str(BANK_PATH.parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
    evidence=json.loads(completed.stdout.strip().splitlines()[-1])['evidence']
    assert not evidence.get('backendError'),evidence
    assert evidence['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(evidence['fixtures'])==4
    for f,actual in zip(r['fixtures'],evidence['fixtures']):
        for key,values in oracle(task,f['inputs']).items():assert actual['tensors'][key]['values']==pytest.approx(values)
    assert all('maskBankIndex' not in row['tensors'] for row in evidence['fixtures'])

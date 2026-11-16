import copy,importlib,math
import pytest
TASKS={'unfold-overlapping-windows':('unfold',[2,3,4],[8,6],dict(kernel_size=[2,2],stride=[1,1],dilation=[1,1],padding=[0,0])),'unfold-padded-spaced-windows':('unfold',[2,3,4],[8,6],dict(kernel_size=[2,2],stride=[2,2],dilation=[2,1],padding=[1,1])),'fold-overlap-addition':('fold',[8,20],[2,3,4],dict(kernel_size=[2,2],stride=[1,1],dilation=[1,1],padding=[1,1],output_size=[3,4])),'fold-uncovered-positions':('fold',[8,4],[2,5,5],dict(kernel_size=[2,2],stride=[3,3],dilation=[1,1],padding=[0,0],output_size=[5,5]))}
def fixture(task,rows=2):
    _,shape,out,_=TASKS[task]
    return {'x':dict(shape=[rows,*shape],values=[(i%19-9)/4 for i in range(rows*math.prod(shape))]),'upstream':dict(shape=[rows,*out],values=[(i%11-5)/4 for i in range(rows*math.prod(out))])}
def submission(task):
    op,shape,out,config=TASKS[task]
    return dict(schemaVersion=1,kind='spatialPatchGraph',graph=dict(schemaVersion=2,nodes=[dict(id='x',type='input',config={}),dict(id='op',type=op,config=copy.deepcopy(config)),dict(id='out',type='output',config={})],edges=[dict(id='a',source='x',sourcePort='out',target='op',targetPort='in'),dict(id='b',source='op',sourcePort='out',target='out',targetPort='in')]))
def oracle(task,f):
    op,shape,out,c=TASKS[task];rows=f['x']['shape'][0];kh,kw=c['kernel_size'];sh,sw=c['stride'];dh,dw=c['dilation'];ph,pw=c['padding'];h,w=shape[1:] if op=='unfold' else c['output_size'];channels=shape[0] if op=='unfold' else shape[0]//(kh*kw)
    oh=(h+2*ph-dh*(kh-1)-1)//sh+1;ow=(w+2*pw-dw*(kw-1)-1)//sw+1;columns=oh*ow;packed=channels*kh*kw
    y=[0.]*(rows*math.prod(out));g=[0.]*len(f['x']['values'])
    for n in range(rows):
        for ch in range(channels):
            for ky in range(kh):
                for kx in range(kw):
                    for py in range(oh):
                        for px in range(ow):
                            iy=py*sh-ph+ky*dh;ix=px*sw-pw+kx*dw
                            if not (0<=iy<h and 0<=ix<w):continue
                            image=((n*channels+ch)*h+iy)*w+ix;column=((n*packed+ch*kh*kw+ky*kw+kx)*columns+py*ow+px)
                            source,target=(image,column) if op=='unfold' else (column,image)
                            y[target]+=f['x']['values'][source];g[source]+=f['upstream']['values'][target]
    return {'output':y,'inputGradient':g}
def run(task,f=None,s=None,stopped=lambda:False,evidence=None):
    worker=importlib.import_module('runner.learning_spatial_patches');e=evidence if evidence is not None else {'maximumTensorElements':0,'fixtures':[]}
    worker.execute(s or submission(task),[dict(id='fixture',inputs=f or fixture(task))],e,stopped,lambda t:(t['shape'],t['values']),task=task);return e
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('rows',[1,2,3])
def test_independent_window_coordinates_and_weighted_input_influence(task,rows):
    f=fixture(task,rows);e=run(task,f)['fixtures'][0]
    assert set(e)=={'fixtureID','tensors','partitions'} and e['fixtureID']=='fixture' and e['partitions']=={}
    assert set(e['tensors'])=={'output','inputGradient'}
    for key,values in oracle(task,f).items():assert e['tensors'][key]['values']==pytest.approx(values,abs=1e-12,rel=1e-12)
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('bad',['answers','seed','parameters','unknown','bool','fraction','scalar','length','negative','duplicate','cycle','unused','wrongop','missing','nonfinite','shape','extraFixture'])
def test_strict_submission_geometry_and_fixture_boundaries(task,bad):
    s=submission(task);f=fixture(task);c=s['graph']['nodes'][1]['config']
    if bad in ('answers','seed','parameters'):s[bad]={}
    elif bad=='unknown':c['normalize']=True
    elif bad in ('bool','fraction','scalar','length','negative'):c['kernel_size']={'bool':[True,2],'fraction':[1.5,2],'scalar':2,'length':[2],'negative':[-1,2]}[bad]
    elif bad=='duplicate':s['graph']['edges'].append(copy.deepcopy(s['graph']['edges'][0]))
    elif bad=='cycle':s['graph']['edges'][0]['source']='op'
    elif bad=='unused':s['graph']['nodes'].append(dict(id='unused',type='reshape',config={'shape':TASKS[task][1]}))
    elif bad=='wrongop':s['graph']['nodes'][1]['type']='relu'
    elif bad=='missing':s['graph']['nodes']=s['graph']['nodes'][:1]+s['graph']['nodes'][2:]
    elif bad=='nonfinite':f['x']['values'][0]=math.nan
    elif bad=='shape':f['x']['shape']=[2,1,1]
    else:f['answer']={'shape':[1],'values':[0]}
    with pytest.raises(ValueError):run(task,f,s)
def test_same_shape_ordinary_windows_are_measured_wrong_for_spaced_task():
    task='unfold-padded-spaced-windows';s=submission(task);s['graph']['nodes'][1]['config']=TASKS['unfold-overlapping-windows'][3]
    e=run(task,s=s)['fixtures'][0]['tensors'];expected=oracle(task,fixture(task))
    assert e['output']['shape']==[2,8,6] and e['output']['values']!=expected['output'] and e['inputGradient']['values']!=expected['inputGradient']
def test_holes_zero_and_padding_columns_do_not_influence_fold():
    e=run('fold-uncovered-positions')['fixtures'][0]['tensors']['output']['values']
    assert all(e[(ch*5+y)*5+x]==0 for ch in range(4) for y in range(5) for x in range(5) if y==2 or x==2)
    f=fixture('fold-overlap-addition');e=run('fold-overlap-addition',f)['fixtures'][0]['tensors'];expected=oracle('fold-overlap-addition',f)
    assert any(v==0 for v in expected['inputGradient']) and any(v!=0 for v in expected['inputGradient'])
    f['x']['values']=[v if expected['inputGradient'][i]!=0 else v+10 for i,v in enumerate(f['x']['values'])]
    # A zero upstream weight can also give zero influence at valid entries; use all-one influence to locate padding-only entries.
    f=fixture('fold-overlap-addition');f['upstream']['values']=[1.]*len(f['upstream']['values']);mapping=oracle('fold-overlap-addition',f)
    before=run('fold-overlap-addition',f)['fixtures'][0]['tensors']['output']['values']
    f['x']['values']=[v if mapping['inputGradient'][i] else v+100 for i,v in enumerate(f['x']['values'])]
    assert run('fold-overlap-addition',f)['fixtures'][0]['tensors']['output']['values']==before
@pytest.mark.parametrize('task',TASKS)
def test_user_labels_positions_and_identity_layout_preserve_measurements(task):
    s=submission(task);s['graph']['nodes'][1].update(name='My rectangle',x=-200,y=45)
    s['graph']['nodes'].append(dict(id='layout',type='reshape',config={'shape':TASKS[task][1]}));s['graph']['edges'][0]['target']='layout';s['graph']['edges'].append(dict(id='layout-edge',source='layout',sourcePort='out',target='op',targetPort='in'))
    assert run(task,s=s)['fixtures'][0]['tensors']==run(task)['fixtures'][0]['tensors']
def test_cancel_budget_and_column_mismatch_before_native_allocation(monkeypatch):
    assert run('fold-overlap-addition',stopped=lambda:True)['fixtures']==[]
    import torch
    def forbidden(*a,**k):raise AssertionError('allocated')
    monkeypatch.setattr(torch,'tensor',forbidden)
    s=submission('fold-overlap-addition');s['graph']['nodes'][1]['config']['output_size']=[5,5]
    with pytest.raises(ValueError):run('fold-overlap-addition',s=s)
    class Budget(dict):
        def __setitem__(self,k,v):
            if k=='maximumTensorElements' and v>0:raise ValueError('budget')
            super().__setitem__(k,v)
    with pytest.raises(ValueError,match='budget'):run('unfold-overlapping-windows',evidence=Budget(maximumTensorElements=0,fixtures=[]))

@pytest.mark.parametrize('task',TASKS)
def test_shared_dispatch_preserves_native_envelope_and_submission_identity(task):
    from runner.learning_evidence import generate_evidence
    from runner.tests.test_learning_tensor_transforms import request
    r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':0,'patch':0}};f=fixture(task);r['fixtures']=[dict(id='fixture',inputs=f)]
    e=generate_evidence(r,submission(task),submission_digest='a'*64)
    assert not e.get('backendError'),e
    assert e['submissionDigest']=='a'*64 and e['fixtures'][0]['fixtureID']=='fixture' and e['fixtures'][0]['partitions']=={}
    for key,values in oracle(task,f).items():assert e['fixtures'][0]['tensors'][key]['values']==pytest.approx(values)
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_four_changed_fixtures_match_independent_outputs_and_native_schema(tmp_path,task,mode):
    import subprocess,sys,json,hashlib
    from pathlib import Path
    from runner.tests.test_learning_tensor_transforms import request
    r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[]
    for i in range(4):
        f=fixture(task,i%3+1)
        for key in f:f[key]['values']=[v+(i+len(mode))/8 for v in f[key]['values']]
        r['fixtures'].append(dict(id=mode+'-'+str(i),inputs=f))
    req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(task),sort_keys=True).encode();sub.write_bytes(raw)
    done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert done.returncode==0,done.stdout+done.stderr
    e=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not e.get('backendError'),e
    assert e['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(e['fixtures'])==4
    for f,actual in zip(r['fixtures'],e['fixtures']):
        assert set(actual)=={'fixtureID','tensors','partitions'} and actual['fixtureID']==f['id'] and actual['partitions']=={}
        for key,values in oracle(task,f['inputs']).items():assert actual['tensors'][key]['values']==pytest.approx(values)

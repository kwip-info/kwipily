import copy
import importlib
import math
import pytest

TASKS={'split-size-order':('split',{'mode':'size','dim':2,'split_size':2,'sections':[1,1]}, {'x':[2,5]},[2,5]),'split-sections-order':('split',{'mode':'sections','dim':2,'split_size':2,'sections':[1,3,1]},{'x':[2,5]},[2,5]),'chunk-uneven-parts':('chunk',{'chunks':6,'dim':2},{'x':[2,13]},[2,1]),'gather-repeated-positions':('gather',{'dim':2},{'x':[2,4],'indices':[2,3]},[2,3]),'einsum-combine-tables':('einsum',{'pattern':'multiply','equation':'nab,nbc->nac'},{'left':[2,3],'right':[3,2]},[2,2]),'einsum-diagonal-view':('einsum',{'pattern':'diagonal','equation':'naa->na'},{'x':[3,3]},[3])}

def fixture(task,rows=2):
    op,config,shapes,out=TASKS[task]
    result={role:dict(shape=[rows,*shape],values=[(i%13-6)/4 for i in range(rows*math.prod(shape))]) for role,shape in shapes.items()}
    if 'indices' in result:result['indices']['values']=[([0,0,3] if (i//3)%2==0 else [2,1,2])[i%3] for i in range(rows*6)]
    result['upstream']=dict(shape=[rows,*out],values=[(i%7-3)/4 for i in range(rows*math.prod(out))])
    return result

def submission(task):
    op,config,shapes,_=TASKS[task]
    nodes=[dict(id=role,type='input',config={'fixture':role}) for role in shapes]
    nodes+=[dict(id='op',type=op,config=copy.deepcopy(config)),dict(id='out',type='output',config={})]
    if op=='gather':links=[('x','out','op','source'),('indices','out','op','indices'),('op','out','out','in')]
    elif task=='einsum-combine-tables':links=[('left','out','op','a'),('right','out','op','b'),('op','out','out','in')]
    elif op=='einsum':links=[('x','out','op','a'),('op','out','out','in')]
    elif op=='chunk':links=[('x','out','op','in'),('op','part4','out','in')]
    else:
        nodes += [dict(id='first',type='concat',config={'dim':2}),dict(id='second',type='concat',config={'dim':2})]
        first,second,third=(2,0,1) if task=='split-size-order' else (2,1,0)
        links=[('x','out','op','in'),('op',f'part{first}','first','a'),('op',f'part{second}','first','b'),('first','out','second','a'),('op',f'part{third}','second','b'),('second','out','out','in')]
    edges=[dict(id=str(i),source=a,sourcePort=ap,target=b,targetPort=bp) for i,(a,ap,b,bp) in enumerate(links)]
    return dict(schemaVersion=1,kind='tensorRoutingGraph',graph=dict(schemaVersion=2,nodes=nodes,edges=edges))

def oracle(task,f):
    y=[];gradients={role:[0.]*len(t['values']) for role,t in f.items() if role not in ('indices','upstream')};g=f['upstream']['values'];rows=f['upstream']['shape'][0]
    if task.startswith('split'):
        order=[4,0,1,2,3] if task=='split-size-order' else [4,1,2,3,0]
        for n in range(rows):
            for channel in range(2):
                for pos in order:
                    i=(n*2+channel)*5+pos;y.append(f['x']['values'][i]);gradients['x'][i]+=g[len(y)-1]
    elif task.startswith('chunk'):
        for n in range(rows):
            for channel in range(2):
                i=(n*2+channel)*13+12;y.append(f['x']['values'][i]);gradients['x'][i]+=g[len(y)-1]
    elif task.startswith('gather'):
        for i,index in enumerate(f['indices']['values']):
            j=(i//3)*4+index;y.append(f['x']['values'][j]);gradients['x'][j]+=g[i]
    elif task=='einsum-diagonal-view':
        for n in range(rows):
            for j in range(3):
                i=n*9+j*3+j;y.append(f['x']['values'][i]);gradients['x'][i]+=g[len(y)-1]
    else:
        for n in range(rows):
            for row in range(2):
                for col in range(2):
                    value=0.;weight=g[len(y)]
                    for k in range(3):
                        a=n*6+row*3+k;b=n*6+k*2+col;left=f['left']['values'][a];right=f['right']['values'][b]
                        value+=left*right;gradients['left'][a]+=weight*right;gradients['right'][b]+=weight*left
                    y.append(value)
    return {'output':y,**{('inputGradient' if role=='x' else role+'Gradient'):values for role,values in gradients.items()}}

def run(task,f=None,s=None,stopped=lambda:False,evidence=None):
    worker=importlib.import_module('runner.learning_tensor_routing');evidence=evidence if evidence is not None else {'maximumTensorElements':0,'fixtures':[]}
    worker.execute(s or submission(task),[dict(id='fixture',inputs=f or fixture(task))],evidence,stopped,lambda t:(t['shape'],t['values']),task=task)
    return evidence

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('rows',[1,2,3])
def test_real_outputs_and_independently_weighted_source_influences(task,rows):
    f=fixture(task,rows);actual=run(task,f)['fixtures'][0]['tensors'];expected=oracle(task,f)
    assert set(actual)==set(expected)
    for key,values in expected.items():assert actual[key]['values']==pytest.approx(values,abs=1e-12,rel=1e-12)
    assert 'indicesGradient' not in actual

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('bad',['answers','seed','unknown','duplicate','cycle','unused','wrongop','missingop','wrongrole','unavailable','nonfinite','shape','extraFixture'])
def test_strict_dag_ports_settings_and_fixture_refusals(task,bad):
    s=submission(task);f=fixture(task)
    if bad in ('answers','seed'):s[bad]=[1]
    elif bad=='unknown':next(n for n in s['graph']['nodes'] if n['id']=='op')['config']['inplace']=True
    elif bad=='duplicate':s['graph']['edges'].append(copy.deepcopy(s['graph']['edges'][0]))
    elif bad=='cycle':s['graph']['edges'][0]['source']='op'
    elif bad=='unused':s['graph']['nodes'].append(dict(id='unused',type='reshape',config={'shape':[2,5]}))
    elif bad=='wrongop':next(n for n in s['graph']['nodes'] if n['id']=='op')['type']='relu'
    elif bad=='missingop':s['graph']['nodes']=[n for n in s['graph']['nodes'] if n['id']!='op']
    elif bad=='wrongrole':next(n for n in s['graph']['nodes'] if n['type']=='input')['config']['fixture']='private-answer'
    elif bad=='unavailable':next(e for e in s['graph']['edges'] if e['source']=='op')['sourcePort']='part5' if task.startswith('chunk') else 'part7'
    elif bad=='nonfinite':next(v for k,v in f.items() if k not in ('indices','upstream'))['values'][0]=math.nan
    elif bad=='shape':next(v for k,v in f.items() if k not in ('indices','upstream'))['shape']=[2,1,1]
    else:f['expected']={'shape':[1],'values':[0]}
    with pytest.raises(ValueError):run(task,f,s)

@pytest.mark.parametrize('bad',['negative','tooLarge','fraction','bool','differentBatch','differentRank'])
def test_gather_selectors_stay_integer_and_in_range_without_broadcast(bad):
    f=fixture('gather-repeated-positions')
    if bad in ('negative','tooLarge','fraction','bool'):f['indices']['values'][0]={'negative':-1,'tooLarge':4,'fraction':.5,'bool':True}[bad]
    elif bad=='differentBatch':f['indices']['shape']=[1,2,3];f['indices']['values']=f['indices']['values'][:6]
    else:f['indices']['shape']=[2,6]
    with pytest.raises(ValueError):run('gather-repeated-positions',f)

def test_chunk_is_five_real_parts_not_six_placeholders_and_only_selected_part_contributes():
    actual=run('chunk-uneven-parts')['fixtures'][0]['tensors'];g=actual['inputGradient']['values']
    assert actual['output']['shape']==[2,2,1]
    assert all(v==0 for i,v in enumerate(g) if i%13!=12)

@pytest.mark.parametrize('task',TASKS)
def test_renamed_nodes_identity_layout_and_equivalent_axis_aliases(task):
    s=submission(task);rename={n['id']:'user-'+str(i) for i,n in enumerate(s['graph']['nodes'])}
    for n in s['graph']['nodes']:n.update(id=rename[n['id']],name='My label',x=-300,y=50)
    for e in s['graph']['edges']:e['source']=rename[e['source']];e['target']=rename[e['target']]
    op=next(n for n in s['graph']['nodes'] if n['type']==TASKS[task][0])
    if 'dim' in op['config']:op['config']['dim']=-1
    output=next(n for n in s['graph']['nodes'] if n['type']=='output')
    final=next(e for e in s['graph']['edges'] if e['target']==output['id'])
    s['graph']['nodes'].append(dict(id='identity-layout',type='reshape',config={'shape':TASKS[task][3]}))
    final['target']='identity-layout'
    s['graph']['edges'].append(dict(id='identity-edge',source='identity-layout',sourcePort='out',target=output['id'],targetPort='in'))
    f=fixture(task);actual=run(task,f,s)['fixtures'][0]['tensors']
    for key,values in oracle(task,f).items():assert actual[key]['values']==pytest.approx(values)

def test_valid_wrong_rejoin_and_gather_position_are_measured_not_corrected():
    s=submission('split-size-order')
    for e in s['graph']['edges']:
        if e['source']=='op':e['sourcePort']={'part2':'part0','part0':'part1','part1':'part2'}[e['sourcePort']]
    actual=run('split-size-order',s=s)['fixtures'][0]['tensors'];assert actual['output']['values']!=oracle('split-size-order',fixture('split-size-order'))['output']
    f=fixture('gather-repeated-positions');f['indices']['values'][0]=1
    assert run('gather-repeated-positions',f)['fixtures'][0]['tensors']['output']['values']!=oracle('gather-repeated-positions',fixture('gather-repeated-positions'))['output']

def test_cancel_and_budget_refuse_before_tensor_construction(monkeypatch):
    assert run('split-size-order',stopped=lambda:True)['fixtures']==[]
    import torch
    def forbidden(*a,**k):raise AssertionError('allocated')
    monkeypatch.setattr(torch,'tensor',forbidden)
    class Budget(dict):
        def __setitem__(self,k,v):
            if k=='maximumTensorElements' and v>0:raise ValueError('budget')
            super().__setitem__(k,v)
    with pytest.raises(ValueError,match='budget'):run('einsum-combine-tables',evidence=Budget(maximumTensorElements=0,fixtures=[]))

@pytest.mark.parametrize('task',TASKS)
def test_shared_assessment_policy_dispatch_binds_routing_and_not_generic_answers(task):
    from runner.learning_evidence import generate_evidence
    from runner.tests.test_learning_tensor_transforms import request
    r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':0,'patch':0}}
    f=fixture(task);r['fixtures']=[dict(id='fixture',inputs=f)]
    evidence=generate_evidence(r,submission(task),submission_digest='a'*64)
    assert not evidence.get('backendError'),evidence
    assert evidence['submissionDigest']=='a'*64
    for key,values in oracle(task,f).items():assert evidence['fixtures'][0]['tensors'][key]['values']==pytest.approx(values)

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_four_changed_fixtures_bind_task_digest_and_no_selector_gradient(tmp_path,task,mode):
    import subprocess,sys,json,hashlib
    from pathlib import Path
    from runner.tests.test_learning_tensor_transforms import request
    r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[]
    for i in range(4):
        f=fixture(task,i%3+1)
        for name in f:
            if name!='indices':f[name]['values']=[v+(i+len(mode))/8 for v in f[name]['values']]
        r['fixtures'].append(dict(id=mode+'-'+str(i),inputs=f))
    req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(task),sort_keys=True).encode();sub.write_bytes(raw)
    completed=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
    evidence=json.loads(completed.stdout.strip().splitlines()[-1])['evidence']
    assert not evidence.get('backendError'),evidence
    assert evidence['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(evidence['fixtures'])==4
    for f,actual in zip(r['fixtures'],evidence['fixtures']):
        assert set(actual)=={'fixtureID','tensors','partitions'} and actual['fixtureID']==f['id'] and actual['partitions']=={}
        expected=oracle(task,f['inputs']);assert set(actual['tensors'])==set(expected)
        for key,values in expected.items():assert actual['tensors'][key]['values']==pytest.approx(values)

@pytest.mark.parametrize('rows',[1,2,3])
def test_gather_valid_transposed_source_and_selectors_use_actual_configured_direction(rows):
    task='gather-repeated-positions';s=submission(task)
    s['graph']['nodes'] += [dict(id='source-layout',type='transpose',config={'dim0':1,'dim1':2}),dict(id='selector-layout',type='transpose',config={'dim0':1,'dim1':2}),dict(id='output-layout',type='transpose',config={'dim0':1,'dim1':2})]
    next(n for n in s['graph']['nodes'] if n['id']=='op')['config']['dim']=1
    s['graph']['edges']=[dict(id=str(i),source=a,sourcePort='out',target=b,targetPort=port) for i,(a,b,port) in enumerate([('x','source-layout','in'),('indices','selector-layout','in'),('source-layout','op','source'),('selector-layout','op','indices'),('op','output-layout','in'),('output-layout','out','in')])]
    f=fixture(task,rows);actual=run(task,f,s)['fixtures'][0]['tensors']
    expected=oracle(task,f)
    assert set(actual)==set(expected)
    for key,values in expected.items():assert actual[key]['values']==pytest.approx(values,abs=1e-12,rel=1e-12)

def test_gather_transformed_actual_source_bounds_still_refuse_before_native_allocation(monkeypatch):
    task='gather-repeated-positions';s=submission(task)
    s['graph']['nodes'] += [dict(id='source-layout',type='transpose',config={'dim0':1,'dim1':2}),dict(id='selector-layout',type='transpose',config={'dim0':1,'dim1':2}),dict(id='output-layout',type='transpose',config={'dim0':1,'dim1':2})]
    # Direction 2 now has extent 2, although the original fixture's direction 2 has extent 4.
    s['graph']['edges']=[dict(id=str(i),source=a,sourcePort='out',target=b,targetPort=port) for i,(a,b,port) in enumerate([('x','source-layout','in'),('indices','selector-layout','in'),('source-layout','op','source'),('selector-layout','op','indices'),('op','output-layout','in'),('output-layout','out','in')])]
    import torch
    def forbidden(*a,**k):raise AssertionError('allocated before actual selector bound')
    monkeypatch.setattr(torch,'tensor',forbidden)
    with pytest.raises(ValueError,match='actual connected source'):run(task,s=s)

@pytest.mark.parametrize('task',TASKS)
def test_emitted_fixture_envelope_matches_native_learning_evidence_schema_and_identity(task):
    actual=run(task)['fixtures'][0]
    assert set(actual)=={'fixtureID','tensors','partitions'}
    assert actual['fixtureID']=='fixture' and actual['partitions']=={}

def test_finite_inputs_that_overflow_do_not_emit_nonfinite_evidence():
    f=fixture('einsum-combine-tables')
    for role in ('left','right'):f[role]['values']=[1e308]*len(f[role]['values'])
    with pytest.raises(ValueError,match='Nonfinite'):run('einsum-combine-tables',f)

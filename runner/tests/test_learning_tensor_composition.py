import copy
import itertools
import math
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_tensor_transforms import request

TARGETS={'concat':('concat-sample-axis',{'dim':1},[2,3],[1,3],[3,3]),'add':('add-shared-pattern',{'alpha':-.5,'sample_broadcast':True},[2,3],[3],[2,3]),'stack':('stack-new-direction',{'dim':2},[2,3],[2,3],[2,2,3]),'permute':('permute-sample-order',{'dims':[0,2,3,1]},[2,3,4],None,[3,4,2])}

def fixture(kind,rows=2):
    raw,cfg,a,b,out=TARGETS[kind];r=request();r['assessmentID']='assessment.'+raw;r['runtime']['capabilities']={kind:{'major':1,'minor':0,'patch':0}}
    def tensor(shape,shift):return {'shape':[rows,*shape],'values':[(i%17-8+shift)/4 for i in range(rows*math.prod(shape))]}
    inputs={'left':tensor(a,0),'upstream':tensor(out,2)}
    if b is not None:inputs['right']=tensor(b,1)
    r['fixtures']=[{'id':'fixture','inputs':inputs}]
    nodes=[dict(id='left',type='input',config={'fixture':'left'})];links=[]
    if b is not None:nodes.append(dict(id='right',type='input',config={'fixture':'right'}));links=[('left','op','a'),('right','op','b')]
    else:links=[('left','op','in')]
    nodes += [dict(id='op',type=kind,config=copy.deepcopy(cfg)),dict(id='output',type='output',config={})];links.append(('op','output','in'))
    s=dict(schemaVersion=1,kind='tensorCompositionGraph',graph=dict(schemaVersion=2,nodes=nodes,edges=[dict(id=str(i),source=a,sourcePort='out',target=b,targetPort=p) for i,(a,b,p) in enumerate(links)]))
    return r,s

def oracle(kind,inputs):
    a=inputs['left'];b=inputs.get('right');g=inputs['upstream'];y=[];da=[0.]*len(a['values']);db=[0.]*len(b['values']) if b else None
    def flat(coords,shape):
        i=0
        for c,n in zip(coords,shape):i=i*n+c
        return i
    for coord in itertools.product(*(range(n) for n in g['shape'])):
        if kind=='concat':
            p=list(coord);isleft=p[1]<2;p[1]-=0 if isleft else 2;t=a if isleft else b;ia=flat(p,t['shape']);y.append(t['values'][ia]);(da if isleft else db)[ia]+=g['values'][len(y)-1]
        elif kind=='stack':
            p=list(coord);isleft=p.pop(2)==0;t=a if isleft else b;ia=flat(p,t['shape']);y.append(t['values'][ia]);(da if isleft else db)[ia]+=g['values'][len(y)-1]
        elif kind=='add':
            ia=flat(coord,a['shape']);ib=flat([coord[0],coord[2]],b['shape']);y.append(a['values'][ia]-.5*b['values'][ib]);w=g['values'][len(y)-1];da[ia]+=w;db[ib]+=-.5*w
        else:
            ia=flat([coord[0],coord[3],coord[1],coord[2]],a['shape']);y.append(a['values'][ia]);da[ia]+=g['values'][len(y)-1]
    return y,da,db

@pytest.mark.parametrize('kind',TARGETS)
@pytest.mark.parametrize('rows',[1,2,3])
def test_real_worker_outputs_both_weighted_influences_and_observations(kind,rows):
    r,s=fixture(kind,rows);expected=oracle(kind,r['fixtures'][0]['inputs']);result=generate_evidence(r,s,submission_digest='a'*64)
    assert not result.get('backendError'),result
    actual=result['fixtures'][0]['tensors'];assert actual['output']['values']==pytest.approx(expected[0]);assert actual['leftGradient']['values']==pytest.approx(expected[1])
    if expected[2] is not None:assert actual['rightGradient']['values']==pytest.approx(expected[2])
    else:assert 'rightGradient' not in actual

@pytest.mark.parametrize('kind',TARGETS)
@pytest.mark.parametrize('change',['duplicate','cycle','unknown','answers','unused','binding','wrongop','missingop','nan','extraFixture','oversize'])
def test_strict_graph_fixture_and_authored_answer_refusal(kind,change):
    r,s=fixture(kind);g=s['graph']
    if change=='duplicate':g['edges'].append(copy.deepcopy(g['edges'][0]))
    elif change=='cycle':g['edges'][0]['source']='op'
    elif change=='unknown':next(n for n in g['nodes'] if n['id']=='op')['config']['unknown']=1
    elif change=='answers':s['expected']={}
    elif change=='unused':g['nodes'].append(dict(id='orphan',type='transpose',config={'dim0':1,'dim1':2}))
    elif change=='binding':g['nodes'][0]['config']['fixture']='upstream'
    elif change=='wrongop':next(n for n in g['nodes'] if n['id']=='op')['type']='custom_code'
    elif change=='missingop':next(n for n in g['nodes'] if n['id']=='op').update(type='reshape',config={'shape':TARGETS[kind][4]})
    elif change=='nan':r['fixtures'][0]['inputs']['left']['values'][0]=float('nan')
    elif change=='extraFixture':r['fixtures'][0]['inputs']['answer']={'shape':[1],'values':[1]}
    else:r['fixtures'][0]['inputs']['left']['shape'][0]=10000
    if change in ('nan','oversize'):
        with pytest.raises(ValueError):generate_evidence(r,s,submission_digest='a'*64)
    else:
        result=generate_evidence(r,s,submission_digest='a'*64);assert result.get('backendError') and not result['fixtures']

@pytest.mark.parametrize('kind',TARGETS)
def test_budget_cancel_before_allocation(kind,monkeypatch):
    r,s=fixture(kind);r['budget']['maximumTensorElements']=1
    def forbidden(*args,**kw):raise AssertionError('allocated tensor')
    monkeypatch.setattr('torch.tensor',forbidden);result=generate_evidence(r,s,submission_digest='a'*64)
    assert not result['fixtures'];assert 'allocated tensor' not in result.get('backendError','')
    assert generate_evidence(r,s,submission_digest='a'*64,cancel=lambda:True)['cancelled']


def test_wrong_valid_add_coefficient_is_measured_not_corrected():
    r,s=fixture('add');next(n for n in s['graph']['nodes'] if n['id']=='op')['config']['alpha']=.5
    result=generate_evidence(r,s,submission_digest='a'*64);assert not result.get('backendError');assert result['fixtures'][0]['tensors']['output']['values']!=oracle('add',r['fixtures'][0]['inputs'])[0]

@pytest.mark.parametrize('kind',TARGETS)
def test_negative_alias_equivalent_and_renamed_graph(kind):
    r,s=fixture(kind);op=next(n for n in s['graph']['nodes'] if n['id']=='op')
    if kind=='concat':op['config']['dim']=-2
    elif kind=='stack':op['config']['dim']=-2
    elif kind=='permute':op['config']['dims']=[-4,-2,-1,-3]
    else:op['config']['alpha']=-.5
    # Graph IDs, titles and placement are never the answer.
    rename={node['id']:'renamed-'+str(i) for i,node in enumerate(s['graph']['nodes'])}
    for node in s['graph']['nodes']:node.update(id=rename[node['id']],name='My label',x=200,y=300)
    for edge in s['graph']['edges']:edge['source']=rename[edge['source']];edge['target']=rename[edge['target']]
    result=generate_evidence(r,s,submission_digest='c'*64)
    assert not result.get('backendError'),result;assert result['fixtures'][0]['tensors']['output']['values']==pytest.approx(oracle(kind,r['fixtures'][0]['inputs'])[0])

@pytest.mark.parametrize('kind',TARGETS)
def test_identity_layout_node_valid_equivalent_construction(kind):
    r,s=fixture(kind);g=s['graph'];shape=TARGETS[kind][2]
    g['nodes'].append(dict(id='layout',type='reshape',config={'shape':shape}))
    edge=next(edge for edge in g['edges'] if edge['source']=='left');edge['source']='layout'
    g['edges'].append(dict(id='identity',source='left',sourcePort='out',target='layout',targetPort='in'))
    result=generate_evidence(r,s,submission_digest='a'*64);assert not result.get('backendError'),result
    assert result['fixtures'][0]['tensors']['output']['values']==pytest.approx(oracle(kind,r['fixtures'][0]['inputs'])[0])

@pytest.mark.parametrize('kind',TARGETS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_worker_cli_multiple_seeded_inputs_and_digest_pin(tmp_path,kind,mode):
    import hashlib,json,subprocess,sys
    from pathlib import Path
    r,s=fixture(kind);r['fixtures']=[]
    for index in range(4):
      q,_=fixture(kind,index%3+1);f=q['fixtures'][0];f['id']=mode+'-'+str(index)
      for key,t in f['inputs'].items():t['values']=[v+(index+len(mode))/8 for v in t['values']]
      r['fixtures'].append(f)
    req=tmp_path/(mode+'-request.json');sub=tmp_path/(mode+'-submission.json');req.write_text(json.dumps(r));raw=json.dumps(s,sort_keys=True).encode();sub.write_bytes(raw)
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()]
    completed=subprocess.run(command,capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
    event=json.loads(completed.stdout.strip().splitlines()[-1]);evidence=event['evidence'];assert not evidence.get('backendError'),evidence;assert len(evidence['fixtures'])==4
    for f,row in zip(r['fixtures'],evidence['fixtures']):assert row['tensors']['output']['values']==pytest.approx(oracle(kind,f['inputs'])[0])
    assert all('expected' not in f for f in r['fixtures']);assert evidence['submissionDigest']==hashlib.sha256(raw).hexdigest()

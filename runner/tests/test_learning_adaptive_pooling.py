"""Actual float64 lesson execution checked by independent scalar fixtures."""
import copy
import itertools
import math
import pytest
from runner import learning_adaptive_pooling as worker

OPS=('avg_pool2d','adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d')


def settings(op):
    if op=='avg_pool2d':return dict(kernel_size=[3,3],stride=[2,3],padding=[1,1],ceil_mode=True,count_include_pad=True,divisor_override=None)
    if op=='adaptive_avg_pool1d':return {'output_size':3}
    return dict(output_height=3,output_width=None)|({'return_indices':True} if op=='adaptive_max_pool2d' else {})


def submission(op):
    return {'schemaVersion':1,'kind':'adaptivePoolingGraph','graph':{'schemaVersion':2,'name':'Groups','nodes':[{'id':'x','type':'input','config':{}},{'id':'pool','type':op,'config':settings(op)},{'id':'y','type':'output','config':{}}],'edges':[{'id':'a','source':'x','sourcePort':'out','target':'pool','targetPort':'in'},{'id':'b','source':'pool','sourcePort':'out','target':'y','targetPort':'in'}]}}


def fixture(op,batch=1,ties=False):
    sample=[2,6,6] if op=='avg_pool2d' else [2,5] if op.endswith('1d') else [2,5,4]
    out=[2,4,3] if op=='avg_pool2d' else [2,3] if op.endswith('1d') else [2,3,4]
    shape=[batch,*sample];gshape=[batch,*out]
    return {'id':'independent','inputs':{'x':{'shape':shape,'values':[-2.0 if ties else ((i*7)%19-10)/4 for i in range(math.prod(shape))]},'upstream':{'shape':gshape,'values':[(i%7-3)/4 for i in range(math.prod(gshape))]}}}


def oracle(f,op,c=None):
    c=c or settings(op);shape=f['inputs']['x']['shape'];x=f['inputs']['x']['values'];out=f['inputs']['upstream']['shape'];g=f['inputs']['upstream']['values'];values=[];indices=[];dx=[0.]*len(x)
    def offset(point,dims):
        result=0
        for i,size in zip(point,dims):result=result*size+i
        return result
    for n,ch,*point in itertools.product(*(range(size) for size in out)):
        ranges=[]
        for axis,(i,size,target) in enumerate(zip(point,shape[2:],out[2:])):
            if op=='avg_pool2d':start=i*c['stride'][axis]-c['padding'][axis];end=start+c['kernel_size'][axis]
            else:start=i*size//target;end=((i+1)*size+target-1)//target
            ranges.append((start,end))
        real=[]
        for p in itertools.product(*(range(start,end) for start,end in ranges)):
            if all(0<=i<size for i,size in zip(p,shape[2:])):real.append((offset((n,ch,*p),shape),offset(p,shape[2:])))
        if op=='adaptive_max_pool2d':
            winner=real[0]
            for candidate in real[1:]:
                if x[candidate[0]]>x[winner[0]]:winner=candidate
            values.append(x[winner[0]]);indices.append(winner[1]);dx[winner[0]]+=g[len(values)-1]
        else:
            if op=='avg_pool2d' and c['divisor_override'] is not None:divisor=c['divisor_override']
            elif op=='avg_pool2d' and c['count_include_pad']:divisor=math.prod(min(end,size+pad)-start for (start,end),size,pad in zip(ranges,shape[2:],c['padding']))
            else:divisor=len(real)
            values.append(math.fsum(x[i] for i,_ in real)/divisor)
            for i,_ in real:dx[i]+=g[len(values)-1]/divisor
    return values,dx,indices


def measured(op,f=None,s=None,stopped=lambda:False):
    evidence={'fixtures':[],'maximumTensorElements':0}
    worker.execute(s or submission(op),[f or fixture(op)],evidence,stopped,lambda t:(t['shape'],t['values']),operation=op)
    return evidence


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('batch',[1,2])
@pytest.mark.parametrize('ties',[False,True])
def test_actual_worker_values_gradients_first_indices(op,batch,ties):
    f=fixture(op,batch,ties);expected=oracle(f,op);row=measured(op,f=f)['fixtures'][0]['tensors']
    assert row['output']['shape']==f['inputs']['upstream']['shape']
    assert row['output']['values']==pytest.approx(expected[0],abs=1e-12)
    assert row['inputGradient']['values']==pytest.approx(expected[1],abs=1e-12)
    if op=='adaptive_max_pool2d':assert row['indices']['values']==expected[2] and all(type(i) is int for i in row['indices']['values'])
    else:assert 'indices' not in row


@pytest.mark.parametrize('op',OPS)
def test_equivalent_connected_layout_preserves_all_measured_tensors(op):
    s=submission(op);shape=fixture(op)['inputs']['upstream']['shape'][1:]
    s['graph']['nodes'].insert(2,{'id':'shape','type':'reshape','config':{'shape':shape}})
    s['graph']['edges'][1]['target']='shape';s['graph']['edges'].append({'id':'c','source':'shape','sourcePort':'out','target':'y','targetPort':'in'})
    assert measured(op,s=s)['fixtures'][0]['tensors']==measured(op)['fixtures'][0]['tensors']


@pytest.mark.parametrize('field,value',[('count_include_pad',False),('divisor_override',-2),('divisor_override',9),('ceil_mode',False)])
def test_valid_avg_misconfiguration_is_measured_or_shape_refused(field,value):
    s=submission('avg_pool2d');s['graph']['nodes'][1]['config'][field]=value
    if field=='ceil_mode':
        with pytest.raises(ValueError,match='shape'):measured('avg_pool2d',s=s)
    else:
        row=measured('avg_pool2d',s=s)['fixtures'][0]['tensors'];expected=oracle(fixture('avg_pool2d'),'avg_pool2d')
        assert row['output']['values']!=pytest.approx(expected[0])
        assert row['inputGradient']['values']!=pytest.approx(expected[1])


def test_valid_max_missing_indices_is_measured_for_criterion_failure():
    s=submission('adaptive_max_pool2d');s['graph']['nodes'][1]['config']['return_indices']=False
    row=measured('adaptive_max_pool2d',s=s)['fixtures'][0]['tensors']
    assert 'indices' not in row and row['output']['values']==oracle(fixture('adaptive_max_pool2d'),'adaptive_max_pool2d')[0]


@pytest.mark.parametrize('op',OPS)
def test_strict_submission_and_fixture_boundary(op):
    for mutation in ['answer','code','version','wrong-op','extra-setting','index-path','disconnect']:
        s=submission(op)
        if mutation=='answer':s['expected']={}
        elif mutation=='code':s['code']='print(1)'
        elif mutation=='version':s['schemaVersion']=True
        elif mutation=='wrong-op':s['graph']['nodes'][1]['type']='linear'
        elif mutation=='extra-setting':s['graph']['nodes'][1]['config']['inplace']=False
        elif mutation=='index-path':s['graph']['edges'][1]['sourcePort']='indices'
        else:s['graph']['edges'][1]['target']='missing'
        with pytest.raises(ValueError):measured(op,s=s)
    for mutation in ['nonfinite','bad-rank','bad-count','bad-batch','extra-input','bad-influence']:
        f=fixture(op)
        if mutation=='nonfinite':f['inputs']['x']['values'][0]=float('nan')
        elif mutation=='bad-rank':f['inputs']['x']['shape']=[len(f['inputs']['x']['values'])]
        elif mutation=='bad-count':f['inputs']['x']['values'].pop()
        elif mutation=='bad-batch':f['inputs']['x']['shape'][0]=3
        elif mutation=='extra-input':f['inputs']['answer']={}
        else:f['inputs']['upstream']['shape'][-1]+=1
        with pytest.raises(ValueError):measured(op,f=f)


@pytest.mark.parametrize('op',OPS)
def test_budget_admission_and_cancellation_before_allocation(op,monkeypatch):
    evidence={'fixtures':[],'maximumTensorElements':0}
    def forbidden(*a,**k):raise AssertionError('Allocated before admission')
    monkeypatch.setattr(worker.torch,'tensor',forbidden)
    worker.execute(submission(op),[fixture(op)],evidence,lambda:evidence['maximumTensorElements']>1,lambda t:(t['shape'],t['values']),operation=op)
    assert evidence['fixtures']==[] and evidence['maximumTensorElements']>1
    assert measured(op,stopped=lambda:True)['fixtures']==[]


@pytest.mark.parametrize('op',OPS)
def test_private_four_fixtures_bound_and_identity(op):
    fixtures=[fixture(op,i%2+1,i==0) for i in range(4)]
    for i,f in enumerate(fixtures):f['id']=str(i)
    evidence={'fixtures':[],'maximumTensorElements':0}
    worker.execute(submission(op),fixtures,evidence,lambda:evidence['maximumTensorElements']>4096,lambda t:(t['shape'],t['values']),operation=op)
    assert len(evidence['fixtures'])==4 and evidence['maximumTensorElements']<=4096
    assert [row['fixtureID'] for row in evidence['fixtures']]==['0','1','2','3']


@pytest.mark.parametrize('op',['adaptive_avg_pool2d','adaptive_max_pool2d'])
def test_explicit_original_width_is_valid_equivalent_configuration(op):
    s=submission(op);s['graph']['nodes'][1]['config']['output_width']=4
    assert measured(op,s=s)['fixtures'][0]['tensors']==measured(op)['fixtures'][0]['tensors']


@pytest.mark.parametrize('op',OPS)
def test_equivalent_double_transpose_and_upstream_weighted_influence(op):
    s=submission(op);nodes=s['graph']['nodes'];edges=s['graph']['edges']
    nodes.insert(2,{'id':'swap','type':'transpose','config':{'dim0':1,'dim1':2}})
    nodes.insert(3,{'id':'undo','type':'transpose','config':{'dim0':1,'dim1':2}})
    edges[1]['target']='swap';edges.extend([{'id':'c','source':'swap','sourcePort':'out','target':'undo','targetPort':'in'},{'id':'d','source':'undo','sourcePort':'out','target':'y','targetPort':'in'}])
    assert measured(op,s=s)['fixtures'][0]['tensors']==measured(op)['fixtures'][0]['tensors']


@pytest.mark.parametrize('op',OPS)
def test_authored_graph_cannot_replay_or_substitute_private_answers(op):
    for mutation in ['literal-answer','disconnected-pool','duplicate-pool','cycle','duplicate-node','foreign-operation']:
        s=submission(op)
        if mutation=='literal-answer':s['graph']['nodes'][1]['config']['output']=[]
        elif mutation=='disconnected-pool':s['graph']['edges'][0]['target']='y'
        elif mutation=='duplicate-pool':s['graph']['nodes'].append({'id':'second','type':op,'config':settings(op)})
        elif mutation=='cycle':s['graph']['edges'][1]['target']='x'
        elif mutation=='duplicate-node':s['graph']['nodes'][1]['id']='x'
        else:s['graph']['nodes'][1]['type']='adaptive_avg_pool3d'
        with pytest.raises(ValueError):measured(op,s=s)


@pytest.mark.parametrize('op',OPS)
def test_weighted_input_influence_matches_independent_finite_difference(op):
    # Distinct monotonic values keep maximum winners stable for finite differences.
    f=fixture(op);f['inputs']['x']['values']=[i/3.0-10 for i in range(len(f['inputs']['x']['values']))]
    row=measured(op,f=f)['fixtures'][0]['tensors'];gradient=row['inputGradient']['values'];upstream=f['inputs']['upstream']['values']
    for position in [0,1,len(gradient)//2,len(gradient)-1]:
        def weighted(delta):
            altered=copy.deepcopy(f);altered['inputs']['x']['values'][position]+=delta
            values=oracle(altered,op)[0]
            return math.fsum(value*weight for value,weight in zip(values,upstream))
        difference=(weighted(1e-5)-weighted(-1e-5))/(2e-5)
        assert gradient[position]==pytest.approx(difference,abs=2e-9)


def test_literal_boundary_partial_corner_is_quarter_not_ninth():
    f=fixture('avg_pool2d');f['inputs']['x']['values']=[1.0]*72
    row=measured('avg_pool2d',f=f)['fixtures'][0]['tensors']
    assert row['output']['values'][11]==0.25
    assert row['output']['values'][0]==pytest.approx(4/9)
    assert row['output']['shape']==[1,2,4,3]


TASKS={'avg_pool2d':'avg-pool2d-windows','adaptive_avg_pool1d':'adaptive-avg-pool1d-bins','adaptive_avg_pool2d':'adaptive-avg-pool2d-bins','adaptive_max_pool2d':'adaptive-max-pool2d-bins'}


@pytest.mark.parametrize('op',OPS)
def test_versioned_dispatch_policy_and_actual_generated_evidence(op):
    from runner.learning_evidence import ASSESSMENT_POLICIES,generate_evidence
    version={'major':1,'minor':0,'patch':0};identity='assessment.'+TASKS[op]
    policy=ASSESSMENT_POLICIES[(identity,(1,0,0))]
    assert policy['adaptivePooling']==op and policy['submissionSchema']==1 and set(policy['capabilities'])=={op}
    request={'assessmentID':identity,'assessmentVersion':version,'fixtureDigest':'a'*64,'runtime':{'backend':'pytorch','version':str(worker.torch.__version__),'capabilities':{op:version}},'budget':{'milliseconds':5000,'maximumTensorElements':4096},'fixtures':[fixture(op)]}
    evidence=generate_evidence(request,submission(op),submission_digest='b'*64)
    assert not evidence.get('backendError') and not evidence['cancelled'] and len(evidence['fixtures'])==1
    assert evidence['fixtures'][0]['tensors']==measured(op)['fixtures'][0]['tensors']

import math
from itertools import product
import pytest
from runner import learning_convolution


def submission(d):
    axes=lambda v:v if d==1 else [v]*d
    config={'in_channels':2,'out_channels':2,'kernel_size':axes(3),'stride':axes(2),'padding':axes(0),'output_padding':axes(1),'dilation':axes(1),'groups':1,'bias':True}
    return {'schemaVersion':1,'kind':'convolutionGraph','graph':{'schemaVersion':2,'name':'Scatter','nodes':[{'id':'a','type':'input','config':{}},{'id':'b','type':f'conv_transpose{d}d','config':config},{'id':'c','type':'output','config':{}}],'edges':[{'id':'ab','source':'a','sourcePort':'out','target':'b','targetPort':'in'},{'id':'bc','source':'b','sourcePort':'out','target':'c','targetPort':'in'}]}}


def fixture(d,signed=False):
    shapes={'x':[1,2]+[3]*d,'weight':[2,2]+[3]*d,'bias':[2],'upstream':[1,2]+[8]*d}
    inputs={k:{'shape':s,'values':[(((i*7+len(k))%17)-8)/4 if signed else 1 for i in range(math.prod(s))]} for k,s in shapes.items()}
    if not signed:inputs['bias']['values']=[1,-1]
    return {'id':'independent','inputs':inputs}


def oracle(f,d):
    x=f['inputs']['x']['values'];w=f['inputs']['weight']['values'];b=f['inputs']['bias']['values'];g=f['inputs']['upstream']['values']
    y=[b[o] for o in range(2) for _ in range(8**d)];dx=[0.]*len(x);dw=[0.]*len(w);db=[sum(g[o*8**d:(o+1)*8**d]) for o in range(2)]
    def flat(coord,size):
        index=0
        for c in coord:index=index*size+c
        return index
    for c in range(2):
        for pos in product(range(3),repeat=d):
            source=c*3**d+flat(pos,3)
            for o in range(2):
                for k in product(range(3),repeat=d):
                    out=o*8**d+flat([p*2+q for p,q in zip(pos,k)],8);weight=(c*2+o)*3**d+flat(k,3)
                    y[out]+=x[source]*w[weight];dx[source]+=g[out]*w[weight];dw[weight]+=g[out]*x[source]
    return {'output':y,'inputGradient':dx,'kernelGradient':dw,'biasGradient':db}


def measured(d,s=None,f=None,stopped=lambda:False):
    evidence={'fixtures':[],'maximumTensorElements':0}
    learning_convolution.execute_transposed(s or submission(d),[f or fixture(d)],evidence,stopped,lambda t:(t['shape'],t['values']),dimensions=d)
    return evidence

@pytest.mark.parametrize('d',[1,2,3])
@pytest.mark.parametrize('signed',[False,True])
def test_actual_scatter_overlap_extent_and_all_influence_match_scalar_oracle(d,signed):
    f=fixture(d,signed);t=measured(d,f=f)['fixtures'][0]['tensors']
    assert t['output']['shape']==[1,2]+[8]*d
    for key,value in oracle(f,d).items():assert t[key]['values']==pytest.approx(value,abs=1e-10)
    if not signed:assert t['output']['values'][8**d-1]==1 and t['output']['values'][-1]==-1

@pytest.mark.parametrize('d',[1,2,3])
def test_wrong_identity_output_padding_or_missing_operation_refuses(d):
    s=submission(d);s['graph']['nodes'][1]['type']=f'conv{d}d'
    with pytest.raises(ValueError):measured(d,s)
    s=submission(d);s['graph']['nodes'][1]['config']['output_padding']=0 if d==1 else [0]*d
    with pytest.raises(ValueError):measured(d,s)
    s=submission(d);s['graph']['edges'].pop()
    with pytest.raises(ValueError):measured(d,s)


def test_cancel_before_work_and_nonfinite_refusal():
    assert measured(3,stopped=lambda:True)['fixtures']==[]
    f=fixture(2);f['inputs']['weight']['values'][0]=float('nan')
    with pytest.raises(ValueError):measured(2,f=f)

@pytest.mark.parametrize('d',[1,2,3])
def test_distinct_second_observation_keeps_first_outputs_and_sums_shared_influence(d):
    base=measured(d)['fixtures'][0]['tensors'];f=fixture(d)
    x=f['inputs']['x'];x['shape'][0]=2;x['values']+= [0]*len(x['values'])
    g=f['inputs']['upstream'];g['shape'][0]=2;g['values']*=2
    t=measured(d,f=f)['fixtures'][0]['tensors']
    first_count=2*8**d
    assert t['output']['values'][:first_count]==base['output']['values']
    assert t['output']['values'][first_count:]==[1]*8**d+[-1]*8**d
    assert t['kernelGradient']['values']==base['kernelGradient']['values']
    assert t['biasGradient']['values']==[2*v for v in base['biasGradient']['values']]
    assert t['inputGradient']['values']==base['inputGradient']['values']*2

@pytest.mark.parametrize('d',[1,2,3])
def test_strict_policy_dispatch_and_budget_before_tensor_allocation(d,monkeypatch):
    from runner.learning_evidence import generate_evidence,runtime_versions
    request={'assessmentID':f'assessment.conv-transpose{d}d-shared-kernel','assessmentVersion':{'major':1,'minor':0,'patch':0},'fixtureDigest':'a'*64,
        'runtime':{'backend':'pytorch','version':runtime_versions()['pytorch'],'capabilities':{f'conv_transpose{d}d':{'major':1,'minor':0,'patch':0}}},
        'budget':{'milliseconds':5000,'maximumTensorElements':16384},'fixtures':[fixture(d)]}
    evidence=generate_evidence(request,submission(d),submission_digest='b'*64)
    assert 'backendError' not in evidence and len(evidence['fixtures'])==1
    request['budget']['maximumTensorElements']=1
    def forbidden(*args,**kwargs):raise AssertionError('Allocated before refusing budget')
    monkeypatch.setattr(__import__('torch'),'tensor',forbidden)
    evidence=generate_evidence(request,submission(d),submission_digest='b'*64)
    assert not evidence['fixtures'] and 'backendError' not in evidence and evidence['maximumTensorElements']>1


def test_extra_authored_results_and_large_output_extent_refuse_before_allocations(monkeypatch):
    s=submission(1);s['expected']={}
    with pytest.raises(ValueError):measured(1,s)
    s=submission(3);s['graph']['nodes'][1]['config']['stride']=[4096]*3
    def forbidden(*args,**kwargs):raise AssertionError('Allocated before validating output extent')
    monkeypatch.setattr(__import__('torch'),'tensor',forbidden)
    with pytest.raises(ValueError):measured(3,s)

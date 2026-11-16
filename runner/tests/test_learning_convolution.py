import math
import pytest
import torch
from runner import learning_convolution


def submission(d,padding_policy='explicit'):
    sequence=lambda x:x if d==1 else [x]*d
    config={'in_channels':2,'out_channels':2,'kernel_size':sequence(2),'stride':sequence(1),'dilation':sequence(1),'groups':1,'bias':True,'padding_policy':padding_policy,'padding':sequence(0),'padding_mode':'zeros'}
    return {'schemaVersion':1,'kind':'convolutionGraph','graph':{'schemaVersion':2,'name':'Window','nodes':[{'id':'a','type':'input','config':{}},{'id':'b','type':f'conv{d}d','config':config},{'id':'c','type':'output','config':{}}],'edges':[{'id':'ab','source':'a','sourcePort':'out','target':'b','targetPort':'in'},{'id':'bc','source':'b','sourcePort':'out','target':'c','targetPort':'in'}]}}


def fixture(d):
    def tensor(shape,values):return {'shape':shape,'values':values}
    x=[1,2]+[4]*d;w=[2,2]+[2]*d;y=[1,2]+[3]*d
    return {'id':'independent','inputs':{'x':tensor(x,[1]*math.prod(x)),'weight':tensor(w,[1]*math.prod(w)),'bias':tensor([2],[1,-1]),'upstream':tensor(y,[1]*math.prod(y))}}


def run(d,s=None,stop=lambda:False):
    evidence={'fixtures':[],'maximumTensorElements':0}
    learning_convolution.execute(s or submission(d),[fixture(d)],evidence,stop,lambda t:(t['shape'],t['values']),dimensions=d)
    return evidence

@pytest.mark.parametrize('d',[1,2,3])
def test_actual_kernel_outputs_and_input_parameter_influence_independent_of_torch(d):
    t=run(d)['fixtures'][0]['tensors'];n=3**d;window=2**d
    assert t['output']['shape']==[1,2]+[3]*d
    assert t['output']['values']==[2*window+1]*n+[2*window-1]*n
    assert t['kernelGradient']['values']==[n]*(4*window)
    assert t['biasGradient']['values']==[n,n]
    from itertools import product
    expected=[]
    for channel in range(2):
        for coordinate in product(range(4),repeat=d):expected.append(2*math.prod(1 if p in (0,3) else 2 for p in coordinate))
    assert t['inputGradient']['values']==expected

@pytest.mark.parametrize('d',[1,2,3])
def test_equivalent_valid_padding_and_identity_reshape(d):
    s=submission(d,'valid');s['graph']['nodes'].insert(1,{'id':'layout','type':'reshape','config':{'shape':[2]+[4]*d}})
    s['graph']['edges'][0]['target']='layout'
    s['graph']['edges'].append({'id':'lb','source':'layout','sourcePort':'out','target':'b','targetPort':'in'})
    assert run(d,s)['fixtures'][0]['tensors']==run(d)['fixtures'][0]['tensors']

@pytest.mark.parametrize('d',[1,2,3])
def test_wrong_operation_or_missing_kernel_or_disconnected_refuses(d):
    s=submission(d);s['graph']['nodes'][1]['type']='linear'
    with pytest.raises(ValueError):run(d,s)
    s=submission(d);s['graph']['edges'].pop()
    with pytest.raises(ValueError):run(d,s)
    f=fixture(d);del f['inputs']['weight']
    with pytest.raises(ValueError):learning_convolution.execute(submission(d),[f],{'fixtures':[],'maximumTensorElements':0},lambda:False,lambda t:(t['shape'],t['values']),dimensions=d)


def test_cancellation_before_work_preserves_rng_and_missing_output_has_no_success():
    before=torch.get_rng_state().clone()
    assert run(3,stop=lambda:True)['fixtures']==[]
    assert torch.equal(before,torch.get_rng_state())


def test_nonfinite_or_wrong_stride_refuses():
    f=fixture(1);f['inputs']['x']['values'][0]=float('nan')
    with pytest.raises(ValueError):learning_convolution.execute(submission(1),[f],{'fixtures':[],'maximumTensorElements':0},lambda:False,lambda t:(t['shape'],t['values']),dimensions=1)
    s=submission(1);s['graph']['nodes'][1]['config']['stride']=2
    with pytest.raises(ValueError):run(1,s)


def scalar_oracle(f,d):
    from itertools import product
    x=f['inputs']['x']['values'];w=f['inputs']['weight']['values'];b=f['inputs']['bias']['values'];g=f['inputs']['upstream']['values'];batch=f['inputs']['x']['shape'][0]
    y=[0.]*(batch*2*3**d);dx=[0.]*len(x);dw=[0.]*len(w);db=[0.,0.]
    def flat(coord,size):
        value=0
        for c in coord:value=value*size+c
        return value
    for n in range(batch):
        for o in range(2):
            for pos in product(range(3),repeat=d):
                out=(n*2+o)*3**d+flat(pos,3);y[out]=b[o];db[o]+=g[out]
                for c in range(2):
                    for offset in product(range(2),repeat=d):
                        source=(n*2+c)*4**d+flat([p+k for p,k in zip(pos,offset)],4)
                        weight=(o*2+c)*2**d+flat(offset,2)
                        y[out]+=x[source]*w[weight];dx[source]+=g[out]*w[weight];dw[weight]+=g[out]*x[source]
    return {'output':y,'inputGradient':dx,'kernelGradient':dw,'biasGradient':db}

@pytest.mark.parametrize('d',[1,2,3])
def test_signed_distinct_channels_repeated_batch_scalar_oracle(d):
    f=fixture(d)
    for key,t in f['inputs'].items():
        t['values']=[((i*7+len(key))%19-9)/4 for i in range(len(t['values']))]
    expected=scalar_oracle(f,d);e={'fixtures':[],'maximumTensorElements':0}
    learning_convolution.execute(submission(d),[f],e,lambda:False,lambda t:(t['shape'],t['values']),dimensions=d)
    for key,values in expected.items():assert e['fixtures'][0]['tensors'][key]['values']==pytest.approx(values,abs=1e-10)

@pytest.mark.parametrize('d',[1,2,3])
def test_actual_dispatch_and_budget_before_tensor_allocation(d,monkeypatch):
    from runner.learning_evidence import generate_evidence,runtime_versions
    request={'assessmentID':f'assessment.conv{d}d-shared-kernel','assessmentVersion':{'major':1,'minor':0,'patch':0},'fixtureDigest':'a'*64,
        'runtime':{'backend':'pytorch','version':runtime_versions()['pytorch'],'capabilities':{f'conv{d}d':{'major':1,'minor':0,'patch':0}}},
        'budget':{'milliseconds':5000,'maximumTensorElements':4096},'fixtures':[fixture(d)]}
    e=generate_evidence(request,submission(d),submission_digest='b'*64)
    assert 'backendError' not in e and len(e['fixtures'])==1
    request['budget']['maximumTensorElements']=1
    def forbidden(*args,**kwargs):raise AssertionError('Tensor allocated before budget refusal')
    monkeypatch.setattr(torch,'tensor',forbidden)
    e=generate_evidence(request,submission(d),submission_digest='b'*64)
    assert not e['fixtures'] and 'backendError' not in e
    assert e['maximumTensorElements']>1


def test_large_padding_shape_rejected_before_tensor_allocation(monkeypatch):
    s=submission(3);s['graph']['nodes'][1]['config']['padding']=[4096]*3
    def forbidden(*args,**kwargs):raise AssertionError('Tensor allocated before shape validation')
    monkeypatch.setattr(torch,'tensor',forbidden)
    with pytest.raises(ValueError):run(3,s)

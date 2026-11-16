"""Independent bounded geometry, values, indices and input-gradient checks."""
import itertools
import math
import pytest
import torch
from runner.tensor_layout import BatchedLayout
from runner.avgpool2d_configuration import normalize_avgpool2d, resolve_avgpool2d, capability as avg_capability
from runner.adaptive_pooling_configuration import normalize_adaptive_pooling, resolve_adaptive_pooling, torch_output_size, capability

OPS=('adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d')


def avg_config(**changes):
    return dict(kernel_size=[3,2],stride=None,padding=[1,0],ceil_mode=False,count_include_pad=True,divisor_override=None)|changes


def adaptive_config(op,**changes):
    c={'output_size':3} if op.endswith('1d') else {'output_height':2,'output_width':3}
    if op=='adaptive_max_pool2d':c['return_indices']=False
    return c|changes


@pytest.mark.parametrize('key,value', [('divisor_override',v) for v in [0,True,False,'2',[],{},float('nan'),float('inf'),2.5,2147483648,-2147483648]] + [('kernel_size',True),('kernel_size',[0,2]),('stride',[1,True]),('padding',[2,0]),('ceil_mode',1),('count_include_pad',0)])
def test_avg_rejects_invalid(key,value):
    with pytest.raises(ValueError):normalize_avgpool2d(avg_config(**{key:value}))


@pytest.mark.parametrize('value',[None,-2147483647,2147483647,-2,2.0])
def test_signed_nullable_divisor_and_exact_fields(value):
    c=avg_config(divisor_override=value)
    assert normalize_avgpool2d(c)['divisor_override']==value
    for bad in [None,{},c|{'dilation':[1,1]},c|{'return_indices':False}]:
        with pytest.raises(ValueError):normalize_avgpool2d(bad)
    assert c==avg_config(divisor_override=value),'Normalization must not mutate caller settings'


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('value',[True,False,0,-1,4097,1.5,'2',[],{},float('nan'),float('inf')])
def test_adaptive_rejects_invalid_size(op,value):
    key='output_size' if op.endswith('1d') else 'output_height'
    with pytest.raises(ValueError):normalize_adaptive_pooling(op,adaptive_config(op,**{key:value}))


@pytest.mark.parametrize('op',OPS)
def test_adaptive_exact_fields_and_nullable_axes(op):
    c=adaptive_config(op)
    for bad in [None,{},c|{'inplace':False},c|{'output_size':2}] if not op.endswith('1d') else [None,{},c|{'return_indices':False}]:
        with pytest.raises(ValueError):normalize_adaptive_pooling(op,bad)
    if op.endswith('1d'):
        with pytest.raises(ValueError):normalize_adaptive_pooling(op,{'output_size':None})
        assert torch_output_size(op,{'output_size':2.0})==2
    else:
        nullable=adaptive_config(op,output_height=None,output_width=None)
        assert torch_output_size(op,nullable)==(None,None)
        assert resolve_adaptive_pooling(op,nullable,BatchedLayout((2,5,4)))['layout'].sample==(2,5,4)
    with pytest.raises(ValueError):normalize_adaptive_pooling('adaptive_max_pool1d',c)


@pytest.mark.parametrize('value',[0,1,None,'true'])
def test_max_requires_boolean_indices(value):
    with pytest.raises(ValueError):normalize_adaptive_pooling('adaptive_max_pool2d',adaptive_config('adaptive_max_pool2d',return_indices=value))


@pytest.mark.parametrize('op',OPS)
def test_shape_bounds_and_output_budget(op):
    c=adaptive_config(op)
    shapes=[(1,), (513,4)] if op.endswith('1d') else [(1,), (1,2), (513,3,4),(1,4097,4)]
    shapes += [(1,4097)] if op.endswith('1d') else [(1,3,4097)]
    for shape in shapes:
        with pytest.raises(ValueError):resolve_adaptive_pooling(op,c,BatchedLayout(shape))
    with pytest.raises(ValueError):resolve_adaptive_pooling(op,c,(1,3,4))
    if not op.endswith('1d'):
        with pytest.raises(ValueError):resolve_adaptive_pooling(op,adaptive_config(op,output_height=4096,output_width=4096),BatchedLayout((512,2,2)))
    result=resolve_adaptive_pooling(op,c,BatchedLayout((2,5) if op.endswith('1d') else (2,5,4)))
    assert result['layout'].sample==((2,3) if op.endswith('1d') else (2,2,3))
    assert result['indicesLayout'] is None
    if op=='adaptive_max_pool2d':assert resolve_adaptive_pooling(op,c|{'return_indices':True},BatchedLayout((2,5,4)))['indicesLayout']==result['layout']


def scalar_adaptive(shape,flat,output,grad,maximum):
    """Floor/ceiling bins and first winning position; no Torch pooling calls."""
    outshape=(*shape[:2],*output);values=[];indices=[];dx=[[] for _ in flat]
    def offset(coords,dims):
        result=0
        for value,width in zip(coords,dims):result=result*width+value
        return result
    for n,ch,*point in itertools.product(*(range(d) for d in outshape)):
        positions=[]
        for spatial in itertools.product(*(range(i*length//size,((i+1)*length+size-1)//size) for i,length,size in zip(point,shape[2:],output))):
            positions.append((offset((n,ch,*spatial),shape),offset(spatial,shape[2:])))
        oi=len(values)
        if maximum:
            winner=positions[0]
            for candidate in positions[1:]:
                if flat[candidate[0]]>flat[winner[0]]:winner=candidate
            values.append(flat[winner[0]]);indices.append(winner[1]);dx[winner[0]].append(grad[oi])
        else:
            values.append(math.fsum(flat[i] for i,_ in positions)/len(positions))
            for i,_ in positions:dx[i].append(grad[oi]/len(positions))
    return outshape,values,indices,[math.fsum(items) for items in dx]


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('case',['divisible','nondivisible','upsample','keep-axis','all-ties'])
@pytest.mark.parametrize('batch',[1,3])
def test_actual_runtime_independent_values_gradients_and_indices(op,device,case,batch):
    assert device!='mps' or torch.backends.mps.is_available(),'This evidence run requires actual MPS'
    shape=(batch,2,4) if op.endswith('1d') else (batch,2,4,4)
    sizes=(2,) if op.endswith('1d') else (2,2)
    if case=='nondivisible':shape=(batch,2,5) if op.endswith('1d') else (batch,2,5,4);sizes=(3,) if op.endswith('1d') else (3,3)
    if case=='upsample':sizes=(5,) if op.endswith('1d') else (5,6)
    if case=='keep-axis':sizes=(4,) if op.endswith('1d') else (None,3)
    c={'output_size':sizes[0]} if op.endswith('1d') else {'output_height':sizes[0],'output_width':sizes[1]}
    if op=='adaptive_max_pool2d':c['return_indices']=True
    config=normalize_adaptive_pooling(op,c);layout=resolve_adaptive_pooling(op,config,BatchedLayout(shape[1:]));output=layout['layout'].sample[1:]
    flat=[-2.0 if case=='all-ties' else float((i*7)%13-11) for i in range(math.prod(shape))]
    grad=[(i%5-2)*0.25 for i in range(batch*2*math.prod(output))]
    expected_shape,values,indices,dx=scalar_adaptive(shape,flat,output,grad,op=='adaptive_max_pool2d')
    # Use strided source and verify no in-place mutation.
    storage=torch.zeros((*shape[:-1],shape[-1]*2),device=device)
    storage[...,::2]=torch.tensor(flat,device=device).reshape(shape);x=storage[...,::2].detach().requires_grad_();before=x.detach().clone()
    unsupported_average=(device=='mps' and op!='adaptive_max_pool2d' and any(a%b for a,b in zip(shape[2:],output)))
    if unsupported_average:
        # Explicit device transfer preserves differentiation and CPU bin semantics;
        # the actual graph adapter must declare and record this fallback separately.
        actual=getattr(torch.nn.functional,op)(x.to('cpu'),torch_output_size(op,config)).to(device);actual_indices=None
    elif op=='adaptive_max_pool2d':actual,actual_indices=torch.nn.functional.adaptive_max_pool2d(x,torch_output_size(op,config),return_indices=True)
    else:actual=getattr(torch.nn.functional,op)(x,torch_output_size(op,config));actual_indices=None
    assert actual.device.type==device and actual.dtype==torch.float32 and tuple(actual.shape)==expected_shape
    torch.testing.assert_close(actual.cpu(),torch.tensor(values).reshape(expected_shape),atol=2e-6,rtol=2e-6)
    actual.backward(torch.tensor(grad,device=device).reshape(expected_shape))
    torch.testing.assert_close(x.grad.cpu(),torch.tensor(dx).reshape(shape),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(x.detach(),before,atol=0,rtol=0)
    if actual_indices is not None:
        assert actual_indices.dtype==torch.int64 and actual_indices.device.type==device
        assert actual_indices.cpu().flatten().tolist()==indices


def test_capabilities_retain_per_member_identity():
    assert avg_capability()['adapterID']=='kwip.runner.graph-v2.avg_pool2d'
    assert avg_capability()['divisorOverrideMinimum']==-2147483647
    for op in OPS:
        descriptor=capability(op)
        assert descriptor['adapterID']==f'kwip.runner.graph-v2.{op}'
        assert descriptor['graphSchemaVersions']==[2] and descriptor['graphDTypes']==['float32']
        assert descriptor['preservesBatch'] is True
        assert descriptor['sampleRank']==(2 if op.endswith('1d') else 3)
    assert capability('adaptive_max_pool2d')['outputDTypes']=={'out':'float32','indices':'int64'}


def scalar_average_windows(shape,flat,c,grad=None):
    """Enumerate starts/windows; ceiling-only extra padding never changes divisor."""
    starts=[]
    for length,k,stride,pad in zip(shape[2:],c['kernel_size'],c['stride'] or c['kernel_size'],c['padding']):
        positions=[];start=-pad
        while start<length:
            if start+k>length+pad+(stride-1 if c['ceil_mode'] else 0):break
            positions.append(start);start+=stride
        if not positions:raise ValueError('No windows')
        starts.append(positions)
    outputshape=(*shape[:2],*(len(v) for v in starts));values=[];dx=[[] for _ in flat]
    for n,ch in itertools.product(range(shape[0]),range(shape[1])):
        for h,w in itertools.product(*starts):
            positions=[]
            for row,col in itertools.product(range(h,h+c['kernel_size'][0]),range(w,w+c['kernel_size'][1])):
                if 0<=row<shape[2] and 0<=col<shape[3]:positions.append(((n*shape[1]+ch)*shape[2]+row)*shape[3]+col)
            if c['divisor_override'] is not None:divisor=c['divisor_override']
            elif c['count_include_pad']:
                divisor=math.prod(min(start+k,length+pad)-start for start,k,length,pad in zip((h,w),c['kernel_size'],shape[2:],c['padding']))
            else:divisor=len(positions)
            values.append(math.fsum(flat[i] for i in positions)/divisor)
            if grad is not None:
                for i in positions:dx[i].append(grad[len(values)-1]/divisor)
    return outputshape,values,[math.fsum(v) for v in dx]


@pytest.mark.parametrize('case',['overlap','ceil','null-stride','asymmetric','large-window','keep-window'])
@pytest.mark.parametrize('divisor',[None,-2,3])
@pytest.mark.parametrize('padcount',[False,True])
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('batch',[1,3])
def test_avg_actual_independent_windows_divisors_and_gradients(case,divisor,padcount,device,batch):
    assert device!='mps' or torch.backends.mps.is_available(),'Actual MPS required for this evidence run'
    shape=(batch,2,5,4)
    c=avg_config(kernel_size=[3,3],stride=[1,2],padding=[1,1],divisor_override=divisor,count_include_pad=padcount)
    if case=='ceil':c.update(stride=[2,3],ceil_mode=True)
    if case=='null-stride':c.update(stride=None,ceil_mode=True)
    if case=='asymmetric':c.update(kernel_size=[2,3],stride=[2,1],padding=[1,0],ceil_mode=True)
    if case=='large-window':c.update(kernel_size=[6,4],stride=[2,3],padding=[3,1],ceil_mode=True)
    if case=='keep-window':c.update(kernel_size=[1,1],stride=None,padding=[0,0])
    flat=[float((i*11)%17-14)/3 for i in range(math.prod(shape))]
    outshape,_,_=scalar_average_windows(shape,flat,c)
    grad=[float(i%7-3)/4 for i in range(math.prod(outshape))]
    _,expected,dx=scalar_average_windows(shape,flat,c,grad)
    assert resolve_avgpool2d(c,BatchedLayout(shape[1:]))['layout'].sample==outshape[1:]
    c=normalize_avgpool2d(c)
    x=torch.tensor(flat,device=device).reshape(shape).requires_grad_();before=x.detach().clone()
    actual=torch.nn.functional.avg_pool2d(x,**c)
    torch.testing.assert_close(actual.cpu(),torch.tensor(expected).reshape(outshape),atol=3e-6,rtol=3e-6)
    actual.backward(torch.tensor(grad,device=device).reshape(outshape))
    torch.testing.assert_close(x.grad.cpu(),torch.tensor(dx).reshape(shape),atol=3e-6,rtol=3e-6)
    torch.testing.assert_close(x.detach(),before,atol=0,rtol=0)


def test_avg_shape_empty_and_budget_rejection_without_allocating():
    for shape in [(1,), (1,2), (513,3,4), (1,4097,4), (1,3,4097)]:
        with pytest.raises(ValueError):resolve_avgpool2d(avg_config(),BatchedLayout(shape))
    with pytest.raises(ValueError):resolve_avgpool2d(avg_config(kernel_size=[6,4],padding=[0,0]),BatchedLayout((2,5,4)))
    with pytest.raises(ValueError):resolve_avgpool2d(avg_config(kernel_size=[2,2],padding=[1,1],stride=[1,1]),BatchedLayout((512,441,441)))
    with pytest.raises(ValueError):resolve_avgpool2d(avg_config(),(1,3,4))

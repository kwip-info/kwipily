"""Independent indexed pooling windows, selected indices, and scalar input gradients."""
import itertools
import math
import json
import pytest
import torch
from runner.engine import GraphModel, export_source, run


def coords(shape):return itertools.product(*(range(n) for n in shape))
def offset(point,shape):
    result=0
    for p,n in zip(point,shape):result=result*n+p
    return result


def normalized(op,c,dim):
    result=dict(c)
    for key in ('kernel_size','stride','padding','dilation'):
        if key not in result:continue
        v=result[key]
        if dim==1 and v is not None:result[key]=[v]
    if result['stride'] is None:result['stride']=result['kernel_size']
    if op=='avg_pool1d':result['dilation']=[1]
    return result


def output_shape(spatial,c):
    shape=[]
    for n,k,s,p,d in zip(spatial,c['kernel_size'],c['stride'],c['padding'],c['dilation']):
        length=(n+2*p-d*(k-1)-1+(s-1 if c['ceil_mode'] else 0))//s+1
        if c['ceil_mode'] and (length-1)*s>=n+p:length-=1
        shape.append(length)
    return tuple(shape)


def reference(op,shape,x,c,g):
    dim=len(shape)-2;c=normalized(op,c,dim);spatial=shape[2:]
    outshape=(shape[0],shape[1],*output_shape(spatial,c));values=[];indices=[];dx=[[] for _ in x]
    for n,ch,*point in coords(outshape):
        oi=offset((n,ch,*point),outshape)
        start=[v*s-p for v,s,p in zip(point,c['stride'],c['padding'])]
        positions=[]
        for kernel in coords(c['kernel_size']):
            source=tuple(a+k*d for a,k,d in zip(start,kernel,c['dilation']))
            if all(0<=v<extent for v,extent in zip(source,spatial)):
                positions.append((offset((n,ch,*source),shape),offset(source,spatial)))
        if op=='avg_pool1d':
            # The implicit right-side ceil extension is outside declared padding.
            divisor=(min(start[0]+c['kernel_size'][0],spatial[0]+c['padding'][0])-max(start[0],-c['padding'][0])) if c['count_include_pad'] else len(positions)
            values.append(math.fsum(x[i] for i,_ in positions)/divisor)
            for i,_ in positions:dx[i].append(g[oi]/divisor)
        else:
            assert positions,'These representative windows must contain real input values'
            winner=positions[0]
            for candidate in positions[1:]:
                if x[candidate[0]]>x[winner[0]]:winner=candidate
            values.append(x[winner[0]]);indices.append(winner[1]);dx[winner[0]].append(g[oi])
    return outshape,values,indices,[math.fsum(v) for v in dx]


def configuration(op,case,flag):
    dim=int(op[-2]) if op.startswith('max') else 1
    c=dict(kernel_size=[2]*dim,stride=[1]*dim,padding=[0]*dim,dilation=[1]*dim,ceil_mode=False,return_indices=flag)
    if case=='ceil':c.update(kernel_size=[3]*dim,stride=[2]*dim,padding=[1]*dim,ceil_mode=True)
    elif case=='null-trailing':c.update(stride=None,padding=[1]*dim,ceil_mode=True)
    elif case=='dilation':c.update(dilation=[2]*dim,padding=[1]*dim)
    elif case=='anisotropic':c.update(kernel_size=[2,3,2][:dim],stride=[2,1,2][:dim],padding=[1,1,0][:dim],ceil_mode=True)
    elif case=='partial':c.update(stride=[2]*dim,ceil_mode=True)
    if op=='avg_pool1d':
        c.pop('dilation');c.pop('return_indices');c['count_include_pad']=flag
    if dim==1:
        for k in ('kernel_size','stride','padding','dilation'):
            if k in c and c[k] is not None:c[k]=c[k][0]
    return c


def graph(op,c):
    dim=int(op[-2]) if op.startswith('max') else 1;spatial=(5,4,3)[:dim]
    out=output_shape(spatial,normalized(op,c,dim))
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':2*math.prod(spatial),'bias':True}),
           ('shape','reshape',{'shape':[2,*spatial]}),('pool',op,c),('flat','flatten',{'start_dim':1,'end_dim':-1}),
           ('head','linear',{'in_features':2*math.prod(out),'out_features':2,'bias':True}),('output','output',{})]
    return {'schemaVersion':2,'nodes':[dict(id=i,type=t,name=i,config=v) for i,t,v in specs],
            'edges':[dict(id=str(i),source=a[0],sourcePort='out',target=b[0],targetPort='in') for i,(a,b) in enumerate(zip(specs,specs[1:]))]},spatial


CASES=[(op,case,flag) for op in ('max_pool1d','max_pool2d','max_pool3d')
       for case in ('overlap','ceil','null-trailing','dilation','anisotropic') for flag in (False,True)]
CASES += [('avg_pool1d',case,flag) for case in ('overlap','ceil','null-trailing','partial') for flag in (False,True)]


@pytest.mark.parametrize('op,case,flag',CASES)
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('batch',[1,3])
def test_independent_windows_indices_input_gradients_export_and_parameterless_identity(op,case,flag,device,batch):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    c=configuration(op,case,flag);authored,spatial=graph(op,c);model=GraphModel(authored).to(device)
    xshape=(batch,2,*spatial);storage=torch.empty((*xshape[:-1],xshape[-1]*2),device=device)
    # Every value is negative; repeated equal maxima check first spatial winner.
    storage[...,::2]=torch.tensor([(-5,-3,-3,-7,-1,-1,-9)[i%7] for i in range(math.prod(xshape))],dtype=torch.float32,device=device).reshape(xshape)
    storage[...,1::2]=999;x=storage[...,::2].detach().requires_grad_();assert not x.is_contiguous()
    outshape=(batch,2,*output_shape(spatial,normalized(op,c,len(spatial))))
    upstream=torch.tensor([(i%9-4)/8 for i in range(math.prod(outshape))],device=device).reshape(outshape)
    shape,y,ix,dx=reference(op,xshape,x.detach().cpu().flatten().tolist(),c,upstream.cpu().flatten().tolist())
    before=x.detach().clone();state={k:v.clone() for k,v in model.state_dict().items()}
    result=model.execute_node('pool',{'shape':x},None);actual=result['out'] if isinstance(result,dict) else result
    tol=dict(atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual,torch.tensor(y,device=device).reshape(shape),**tol)
    torch.testing.assert_close(torch.autograd.grad(actual,x,upstream)[0],torch.tensor(dx,device=device).reshape(xshape),**tol)
    if op.startswith('max') and flag:
        assert result['indices'].dtype==torch.int64
        assert torch.equal(result['indices'].cpu(),torch.tensor(ix,dtype=torch.int64).reshape(shape))
    assert torch.equal(before,x) and all(torch.equal(state[k],v) for k,v in model.state_dict().items())
    assert set(model.layers)=={'expand','head'}
    assert not any('pool' in key for key in state)
    if batch>1:
        changed=x.detach().clone();changed[0]+=13
        new=model.execute_node('pool',{'shape':changed},None);new=new['out'] if isinstance(new,dict) else new
        assert torch.equal(new[1:],actual[1:])
    changed=x.detach().clone();changed[:,0]+=13
    separated=model.execute_node('pool',{'shape':changed},None)
    separated=separated['out'] if isinstance(separated,dict) else separated
    assert torch.equal(separated[:,1:],actual[:,1:])
    ns={};exec(export_source(model),ns);exported=ns['Model']().to(device);exported.load_state_dict(state,strict=True)
    public=torch.tensor([[.25,-.5]]*batch,device=device,requires_grad=True)
    first=model(public);second=exported(public)
    torch.testing.assert_close(first,second,**tol)
    ga=torch.autograd.grad(first,(public,*model.parameters()),torch.ones_like(first))
    gb=torch.autograd.grad(second,(public,*exported.parameters()),torch.ones_like(second))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,**tol)
    model.eval();exported.eval()
    with torch.no_grad():torch.testing.assert_close(model(public),exported(public),**tol)


@pytest.mark.parametrize('op',['max_pool1d','max_pool2d','max_pool3d','avg_pool1d'])
def test_pooling_fine_tune_keeps_parameterless_node_and_resets_optimizer_epoch(tmp_path,op):
    c=configuration(op,'overlap',False);authored,_=graph(op,c)
    parent=tmp_path/'parent';child=tmp_path/'child'
    run(authored,parent,device='cpu',epochs=2,sample_limit=32,batch_size=32)
    before=(parent/'checkpoint.pt').read_bytes();events=[]
    run(authored,child,device='cpu',epochs=1,sample_limit=32,batch_size=32,seed=19,split_seed=29,
        finetune_checkpoint=parent/'checkpoint.pt',emit=events.append)
    checkpoint=torch.load(child/'checkpoint.pt',weights_only=True)
    assert checkpoint['epoch']==1 and checkpoint['seed']==19 and events[0]['initialEpoch']==0
    assert all(value['step'].item()==1 for value in checkpoint['optimizerState']['state'].values())
    assert set(checkpoint['modelState'])=={'layers.expand.weight','layers.expand.bias','layers.head.weight','layers.head.bias'}
    assert json.loads((child/'manifest.json').read_text())['lineage']['mode']=='weightsOnlyFineTune'
    assert (parent/'checkpoint.pt').read_bytes()==before

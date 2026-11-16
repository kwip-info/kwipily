"""Indexed scalar lookup/scatter and selected-row norm evidence, no Torch oracle."""
import collections
import math
import pytest
import torch
from runner.engine import GraphModel,export_source,run
from runner.embedding import CheckedEmbedding


def config(**patch):
    result=dict(num_embeddings=5,embedding_dim=3,padding_idx=None,max_norm=None,norm_type=2,
                scale_grad_by_freq=False,sparse=False,freeze=False)
    result.update(patch);return result


def graph(sample,settings):
    width=math.prod(sample)*settings['embedding_dim']
    specs=[('input','index_input',{'shape':list(sample)}),('lookup','embedding',settings),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':width,'out_features':2,'bias':True}),('output','output',{})]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(specs,specs[1:]))]}


def oracle(table,indices,upstream,settings):
    width=settings['embedding_dim'];rows=[list(table[i:i+width]) for i in range(0,len(table),width)]
    if settings['max_norm'] is not None:
        for row in set(indices):
            norm=math.fsum(abs(v)**settings['norm_type'] for v in rows[row])**(1/settings['norm_type'])
            if norm>settings['max_norm']:rows[row]=[v*settings['max_norm']/(norm+1e-7) for v in rows[row]]
    gathered=[v for row in indices for v in rows[row]]
    frequencies=collections.Counter(indices);parts=[[] for _ in table]
    padding=None if settings['padding_idx'] is None else settings['padding_idx']%settings['num_embeddings']
    for i,row in enumerate(indices):
        if row==padding:continue
        for column in range(width):parts[row*width+column].append(upstream[i*width+column]/(frequencies[row] if settings['scale_grad_by_freq'] else 1))
    return gathered,[math.fsum(v) for v in parts],[v for row in rows for v in row]


def ready(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; no device acceptance')


def exercise(device,rank,batch,settings):
    ready(device);sample=(4,)+(2,)*(rank-2);shape=(batch,*sample);out_shape=(*shape,settings['embedding_dim'])
    model=GraphModel(graph(sample,settings)).to(device);layer=model.layers['lookup']
    table=[(i%7-3)/2 for i in range(layer.weight.numel())]
    with torch.no_grad():layer.weight.copy_(torch.tensor(table,device=device).reshape(layer.weight.shape))
    namespace={};exec(export_source(model),namespace);exported=namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    original=[(1,1,0,4)[i%4] for i in range(math.prod(shape))]
    storage=torch.empty((*shape[:-1],shape[-1]*2),device=device,dtype=torch.int64)
    storage[...,::2]=torch.tensor(original,device=device).reshape(shape);storage[...,1::2]=3
    indices=storage[...,::2];assert not indices.is_contiguous()
    before_indices=indices.clone()
    gvalues=[(i%11-5)/16 for i in range(math.prod(out_shape))]
    expected,gradient,state=oracle(table,original,gvalues,settings)
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    for current in (layer,exported.layers['lookup']):
        out=current(indices)
        torch.testing.assert_close(out,torch.tensor(expected,device=device).reshape(out_shape),atol=atol,rtol=rtol)
        torch.testing.assert_close(current.weight,torch.tensor(state,device=device).reshape(current.weight.shape),atol=atol,rtol=rtol)
        assert out.dtype==torch.float32 and out.shape==out_shape and torch.isfinite(out).all()
        assert current.weight.requires_grad== (not settings['freeze'])
        assert set(current.state_dict())=={'weight'}
        assert torch.equal(indices,before_indices) and not indices.requires_grad
        assert torch.equal(current.weight[2],torch.tensor(table[6:9],device=device))
        assert torch.equal(current.weight[3],torch.tensor(table[9:12],device=device))
        if settings['freeze']:
            assert not out.requires_grad
        else:
            measured,=torch.autograd.grad(out,current.weight,torch.tensor(gvalues,device=device).reshape(out_shape))
            torch.testing.assert_close(measured,torch.tensor(gradient,device=device).reshape(current.weight.shape),atol=atol,rtol=rtol)
            pad=settings['padding_idx']
            if pad is not None:assert torch.count_nonzero(measured[pad%5])==0
            assert torch.count_nonzero(measured[2:4])==0
        # Lookup values remain row-independent; normalized shared table state was
        # already established above, so this observation makes no new norm claim.
        if batch>1:
            changed=indices.clone();changed[0]=0
            torch.testing.assert_close(current(changed)[1:],out[1:],atol=atol,rtol=rtol)
    return model,exported,indices


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank',range(2,8))
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('policy',[{}, {'padding_idx':4,'scale_grad_by_freq':True}, {'padding_idx':-1}])
def test_scalar_gather_dense_frequency_padding_scatter(device,rank,batch,policy):
    exercise(device,rank,batch,config(**policy))


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('norm',[1,2,3])
@pytest.mark.parametrize('freeze',[False,True])
def test_selected_row_norm_nondefault_orders_and_frozen_mutation(device,norm,freeze):
    exercise(device,3,3,config(max_norm=.75,norm_type=norm,freeze=freeze,padding_idx=-1))


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('maximum',[None,.75])
def test_frozen_sparse_forward_state_and_export(device,maximum):
    model,exported,x=exercise(device,3,3,config(sparse=True,freeze=True,max_norm=maximum))
    torch.testing.assert_close(model(x),exported(x))
    assert model.layers['lookup'].sparse and not model.layers['lookup'].weight.requires_grad


@pytest.mark.parametrize('device',['cpu','mps'])
def test_real_high_rank_whole_graph_export_and_parameter_gradients(device):
    model,exported,x=exercise(device,7,3,config(padding_idx=4,scale_grad_by_freq=True))
    actual=model(x);copied=exported(x)
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    torch.testing.assert_close(actual,copied,atol=atol,rtol=rtol)
    up=torch.tensor([[.5,-.25]]*3,device=device)
    ga=torch.autograd.grad(actual,tuple(model.parameters()),up);gb=torch.autograd.grad(copied,tuple(exported.parameters()),up)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


def test_sparse_frequency_trainable_adam_and_allocation_refusals(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='frequency'):GraphModel(graph((4,),config(sparse=True,scale_grad_by_freq=True)))
    with pytest.raises(ValueError,match='parameters|parameter|2 million'):GraphModel(graph((4,),config(num_embeddings=1_000_001,embedding_dim=2)))
    layer=CheckedEmbedding(config())
    for x in [torch.tensor([[.5]]),torch.tensor([[True]]),torch.tensor([[-1]]),torch.tensor([[5]])]:
        with pytest.raises(ValueError):layer(x)
    huge=torch.ones((1,1),dtype=torch.int64).expand(1,33_333_334)
    with pytest.raises(ValueError,match='allocation'):layer(huge)
    for shape in [(3,), (1,)*8, (1,0)]:
        with pytest.raises(ValueError):layer(torch.zeros(shape,dtype=torch.int64))
    # Adam refusal occurs before output directory/worker-artifact creation.
    x=torch.tensor([[1,1,0,4],[4,0,1,1]]);y=torch.tensor([0,1])
    prepared=((x,y,x,y),{'sourceHash':'sparse-refusal','splitHash':'sparse-refusal'}, {})
    with pytest.raises(ValueError,match='sparse'):
        run(graph((4,),config(sparse=True)),tmp_path/'refused',epochs=1,device='cpu',prepared=prepared)
    assert not (tmp_path/'refused').exists()

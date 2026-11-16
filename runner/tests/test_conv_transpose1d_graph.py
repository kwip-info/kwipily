"""Private transposed convolution graph; reference is explicit indexed scatter sums."""
import copy
import json
import pytest
import torch
import torch.nn.functional as F
from runner.engine import GraphModel,REGISTRY,validate,export_source
from runner.tests.test_conv_transpose1d_contract import config


@pytest.fixture
def registry(tmp_path):
    d=json.loads(REGISTRY.read_text());d['components']=[c for c in d['components'] if c['id']!='conv_transpose1d']
    defaults=config();fields=[]
    for key in ('in_channels','out_channels','kernel_size','stride','padding','output_padding','dilation','groups'):
        fields.append(dict(key=key,title=key,kind='integer',defaultValue=defaults[key],min=0 if key in ('padding','output_padding') else 1,max=512 if key in ('in_channels','out_channels','groups') else 4096))
    fields.append(dict(key='bias',title='Bias',kind='boolean',defaultValue=True))
    d['components'].append(dict(id='conv_transpose1d',title='ConvTranspose1d',category='Layers',inputs=['in'],outputs=['out'],fields=fields))
    path=tmp_path/'registry.json';path.write_text(json.dumps(d));return path


def graph(settings=None,length=5):
    c=config(**({'in_channels':2,'out_channels':4,'groups':2}|(settings or {})))
    out=(length-1)*c['stride']-2*c['padding']+c['dilation']*(c['kernel_size']-1)+c['output_padding']+1
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':c['in_channels']*length,'bias':True}),
           ('shaped','reshape',{'shape':[c['in_channels'],length]}),('convolution','conv_transpose1d',c),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':c['out_channels']*out,'out_features':2,'bias':True}),('output','output',{})]
    return dict(schemaVersion=2,nodes=[dict(id=i,type=t,config=c) for i,t,c in specs],edges=[dict(id=str(k),source=a[0],sourcePort='out',target=b[0],targetPort='in') for k,(a,b) in enumerate(zip(specs,specs[1:]))])


def indexed_reference(x,weight,bias,c):
    length=x.shape[-1];out=(length-1)*c['stride']-2*c['padding']+c['dilation']*(c['kernel_size']-1)+c['output_padding']+1
    per_in=c['in_channels']//c['groups'];per_out=c['out_channels']//c['groups'];batches=[]
    for n in range(len(x)):
        channels=[]
        for o in range(c['out_channels']):
            positions=[x[n,0,0]*0 for _ in range(out)]
            group=o//per_out
            for i in range(group*per_in,(group+1)*per_in):
                for position in range(length):
                    for k in range(c['kernel_size']):
                        destination=position*c['stride']-c['padding']+k*c['dilation']
                        if 0<=destination<out:positions[destination]=positions[destination]+x[n,i,position]*weight[i,o%per_out,k]
            channels.append(torch.stack([v+(bias[o] if bias is not None else 0) for v in positions]))
        batches.append(torch.stack(channels))
    return torch.stack(batches)


SETTINGS=[{}, {'stride':2,'padding':1,'output_padding':1}, {'stride':1,'dilation':2,'output_padding':1},
          {'groups':1,'dilation':2,'padding':1}, {'kernel_size':1,'stride':3,'padding':3}]


@pytest.mark.parametrize('settings',SETTINGS)
@pytest.mark.parametrize('bias',[False,True])
def test_scatter_oracle_all_gradients_and_export(registry,settings,bias):
    authored=graph(settings|{'bias':bias});authored['nodes'][2]['name']='tuple';original=copy.deepcopy(authored)
    torch.manual_seed(736);model=GraphModel(authored,registry).double()
    x=torch.randn(2,3,dtype=torch.float64).t().requires_grad_();assert not x.is_contiguous()
    actual,captured=model(x,inspect=True);layer=model.layers['convolution'];assert isinstance(layer,torch.nn.ConvTranspose1d)
    expected_conv=indexed_reference(captured['shaped'],layer.weight,layer.bias,authored['nodes'][3]['config'])
    torch.testing.assert_close(captured['convolution'],expected_conv)
    expected=F.linear(expected_conv.flatten(1),model.layers['head'].weight,model.layers['head'].bias)
    a=torch.autograd.grad(actual.square().sum(),(x,*model.parameters()),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(x,*model.parameters()))
    for first,second in zip(a,b):torch.testing.assert_close(first,second)
    namespace={};source=export_source(model);exec(source,namespace)
    assert 'nn.ConvTranspose1d(' in source and 'output_padding=' in source and 'runner' not in source
    exported=namespace['Model']().double();exported.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(exported(x),actual)
    gradients=torch.autograd.grad(exported(x).square().sum(),(x,*exported.parameters()))
    for first,second in zip(a,gradients):torch.testing.assert_close(first,second)
    assert authored==original


@pytest.mark.parametrize('settings',SETTINGS)
@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple GPU unavailable')
def test_actual_mps_valid_settings_match_cpu(registry,settings):
    authored=graph(settings);torch.manual_seed(839)
    cpu=GraphModel(authored,registry);gpu=GraphModel(authored,registry).to('mps');gpu.load_state_dict(cpu.state_dict())
    x=torch.randn(3,2,requires_grad=True);y=x.detach().to('mps').requires_grad_()
    a=cpu(x);b=gpu(y);torch.testing.assert_close(a,b.cpu(),atol=1e-5,rtol=1e-4)
    da=torch.autograd.grad(a.square().sum(),(x,*cpu.parameters()));db=torch.autograd.grad(b.square().sum(),(y,*gpu.parameters()))
    for first,second in zip(da,db):torch.testing.assert_close(first,second.cpu(),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('change',['rank','channels','int64','legacy','bad_output_padding','extra_policy'])
def test_graph_contract_refusals(registry,change):
    authored=graph()
    if change=='rank':authored['nodes'][2]['config']['shape']=[10]
    elif change=='channels':authored['nodes'][2]['config']['shape']=[1,10]
    elif change=='legacy':authored['schemaVersion']=1
    elif change=='bad_output_padding':authored['nodes'][3]['config']['output_padding']=1
    elif change=='extra_policy':authored['nodes'][3]['config']['padding_policy']='same'
    else:
        authored['nodes']=authored['nodes'][2:];authored['nodes'][0]=dict(id='shaped',type='index_input',config={'shape':[2,5]});authored['edges']=authored['edges'][2:]
    with pytest.raises(ValueError):validate(authored,registry)


@pytest.mark.parametrize('shape',[(2,5),(0,2,5),(3,2,6),(3,1,5),(3,2,5,1)])
def test_exact_runtime_nonempty_batched_sample_guard(registry,shape):
    model=GraphModel(graph(),registry)
    with pytest.raises(ValueError,match='(?i)(shape|batch|transpose)'):model.execute_node('convolution',{'shaped':torch.zeros(shape)},None)


def test_export_nonempty_guard(registry):
    model=GraphModel(graph(),registry);namespace={};exec(export_source(model),namespace);portable=namespace['Model']()
    with pytest.raises(ValueError,match='(?i)(shape|batch|transpose)'):portable(torch.empty(0,2))


def test_aggregate_parameter_rejection_precedes_allocation(registry,monkeypatch):
    authored=graph({'in_channels':500,'out_channels':500,'groups':1,'kernel_size':1},length=1)
    nodes=authored['nodes'];nodes[3:4]=[dict(id=f'conv{i}',type='conv_transpose1d',config=copy.deepcopy(nodes[3]['config'])) for i in range(8)]
    authored['edges']=[dict(id=str(i),source=a['id'],sourcePort='out',target=b['id'],targetPort='in') for i,(a,b) in enumerate(zip(nodes,nodes[1:]))]
    from runner import engine
    monkeypatch.setitem(engine.TRANSPOSE_CONVOLUTION_LAYERS,'conv_transpose1d',lambda *a,**k:pytest.fail('Allocated before aggregate validation'))
    with pytest.raises(ValueError,match='(?i)(parameter|2 million)'):GraphModel(authored,registry)

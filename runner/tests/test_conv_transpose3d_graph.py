"""Private transposed convolution graph; reference is explicit indexed scatter sums."""
import copy
import json
import pytest
import torch
import torch.nn.functional as F
from runner.engine import GraphModel,REGISTRY,validate,export_source
from runner.tests.test_conv_transpose3d_contract import config


@pytest.fixture
def registry(tmp_path):
    d=json.loads(REGISTRY.read_text());d['components']=[c for c in d['components'] if c['id']!='conv_transpose3d']
    defaults=config();fields=[]
    for key in ('in_channels','out_channels','groups'):
        fields.append(dict(key=key,title=key,kind='integer',defaultValue=defaults[key],min=1,max=512))
    for key in ('kernel_size','stride','padding','output_padding','dilation'):
        fields.append(dict(key=key,title=key,kind='integerList',defaultValue=defaults[key],min=0 if key in ('padding','output_padding') else 1,max=4096,minItems=3,maxItems=3))
    fields.append(dict(key='bias',title='Bias',kind='boolean',defaultValue=True))
    d['components'].append(dict(id='conv_transpose3d',title='ConvTranspose3d',category='Layers',inputs=['in'],outputs=['out'],fields=fields))
    path=tmp_path/'registry.json';path.write_text(json.dumps(d));return path


def graph(settings=None,depth=2,height=3,width=4):
    c=config(**({'in_channels':2,'out_channels':4,'groups':2}|(settings or {})))
    spatial=(depth,height,width)
    out=[(length-1)*c['stride'][i]-2*c['padding'][i]+c['dilation'][i]*(c['kernel_size'][i]-1)+c['output_padding'][i]+1 for i,length in enumerate(spatial)]
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':c['in_channels']*depth*height*width,'bias':True}),
           ('shaped','reshape',{'shape':[c['in_channels'],*spatial]}),('convolution','conv_transpose3d',c),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':c['out_channels']*out[0]*out[1]*out[2],'out_features':2,'bias':True}),('output','output',{})]
    return dict(schemaVersion=2,nodes=[dict(id=i,type=t,config=c) for i,t,c in specs],edges=[dict(id=str(k),source=a[0],sourcePort='out',target=b[0],targetPort='in') for k,(a,b) in enumerate(zip(specs,specs[1:]))])


def indexed_reference(x,weight,bias,c):
    depth,height,width=x.shape[-3:]
    out=[(length-1)*c['stride'][axis]-2*c['padding'][axis]+c['dilation'][axis]*(c['kernel_size'][axis]-1)+c['output_padding'][axis]+1 for axis,length in enumerate((depth,height,width))]
    per_in=c['in_channels']//c['groups'];per_out=c['out_channels']//c['groups'];batches=[]
    for n in range(len(x)):
        channels=[]
        for o in range(c['out_channels']):
            positions=[[[(x[n,0,0,0,0]+weight.sum())*0 for _ in range(out[2])] for _ in range(out[1])] for _ in range(out[0])]
            group=o//per_out
            for i in range(group*per_in,(group+1)*per_in):
                for d in range(depth):
                    for h in range(height):
                        for w in range(width):
                            for kd in range(c['kernel_size'][0]):
                                for kh in range(c['kernel_size'][1]):
                                    for kw in range(c['kernel_size'][2]):
                                        dd=d*c['stride'][0]-c['padding'][0]+kd*c['dilation'][0]
                                        dh=h*c['stride'][1]-c['padding'][1]+kh*c['dilation'][1]
                                        dw=w*c['stride'][2]-c['padding'][2]+kw*c['dilation'][2]
                                        if 0<=dd<out[0] and 0<=dh<out[1] and 0<=dw<out[2]:
                                            positions[dd][dh][dw]=positions[dd][dh][dw]+x[n,i,d,h,w]*weight[i,o%per_out,kd,kh,kw]
            channels.append(torch.stack([torch.stack([torch.stack([v+(bias[o] if bias is not None else 0) for v in row]) for row in plane]) for plane in positions]))
        batches.append(torch.stack(channels))
    return torch.stack(batches)


SETTINGS=[{}, {'stride':[2,1,2],'padding':[1,1,1],'output_padding':[1,0,1]},
          {'stride':[1,2,1],'dilation':[2,1,3],'output_padding':[1,1,2],'padding':[1,1,2]},
          {'groups':1,'dilation':[2,1,1],'padding':[1,1,0]},
          # Complete depth cropping intentionally tests mathematically zero parameter gradients.
          {'kernel_size':[1,1,1],'stride':[3,2,2],'padding':[1,2,2]},
          {'kernel_size':[1,1,1],'stride':[3,2,2],'padding':[0,2,2]}]


@pytest.mark.parametrize('settings',SETTINGS)
@pytest.mark.parametrize('bias',[False,True])
def test_scatter_oracle_all_gradients_and_export(registry,settings,bias):
    authored=graph(settings|{'bias':bias});authored['nodes'][2]['name']='tuple';original=copy.deepcopy(authored)
    torch.manual_seed(736);model=GraphModel(authored,registry).double()
    x=torch.randn(2,3,dtype=torch.float64).t().requires_grad_();assert not x.is_contiguous()
    actual,captured=model(x,inspect=True);layer=model.layers['convolution'];assert isinstance(layer,torch.nn.ConvTranspose3d)
    expected_conv=indexed_reference(captured['shaped'],layer.weight,layer.bias,authored['nodes'][3]['config'])
    torch.testing.assert_close(captured['convolution'],expected_conv)
    if settings.get('kernel_size')==[1,1,1] and settings.get('padding')==[1,2,2]:
        baseline=torch.zeros_like(expected_conv) if layer.bias is None else layer.bias.view(1,-1,1,1,1).expand_as(expected_conv)
        torch.testing.assert_close(expected_conv,baseline,atol=0,rtol=0)
    expected=F.linear(expected_conv.flatten(1),model.layers['head'].weight,model.layers['head'].bias)
    a=torch.autograd.grad(actual.square().sum(),(x,*model.parameters()),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(x,*model.parameters()))
    for first,second in zip(a,b):torch.testing.assert_close(first,second)
    namespace={};source=export_source(model);exec(source,namespace)
    assert 'nn.ConvTranspose3d(' in source and 'output_padding=' in source and 'runner' not in source
    exported=namespace['Model']().double();exported.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(exported(x),actual)
    gradients=torch.autograd.grad(exported(x).square().sum(),(x,*exported.parameters()))
    for first,second in zip(a,gradients):torch.testing.assert_close(first,second)
    assert authored==original


@pytest.mark.parametrize('settings',SETTINGS)
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple GPU unavailable')
def test_actual_mps_valid_settings_match_cpu(registry,settings,bias):
    authored=graph(settings|{'bias':bias});torch.manual_seed(839)
    cpu=GraphModel(authored,registry);gpu=GraphModel(authored,registry).to('mps');gpu.load_state_dict(cpu.state_dict())
    x=torch.randn(3,2,requires_grad=True);y=x.detach().to('mps').requires_grad_()
    a=cpu(x);b=gpu(y);torch.testing.assert_close(a,b.cpu(),atol=1e-5,rtol=1e-4)
    da=torch.autograd.grad(a.square().sum(),(x,*cpu.parameters()));db=torch.autograd.grad(b.square().sum(),(y,*gpu.parameters()))
    for first,second in zip(da,db):torch.testing.assert_close(first,second.cpu(),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('change',['rank','channels','int64','legacy','bad_output_padding','extra_policy'])
def test_graph_contract_refusals(registry,change):
    authored=graph()
    if change=='rank':authored['nodes'][2]['config']['shape']=[2,6,4]
    elif change=='channels':authored['nodes'][2]['config']['shape']=[1,4,3,4]
    elif change=='legacy':authored['schemaVersion']=1
    elif change=='bad_output_padding':authored['nodes'][3]['config']['output_padding']=[1,0,0]
    elif change=='extra_policy':authored['nodes'][3]['config']['padding_policy']='same'
    else:
        authored['nodes']=authored['nodes'][2:];authored['nodes'][0]=dict(id='shaped',type='index_input',config={'shape':[2,2,3,4]});authored['edges']=authored['edges'][2:]
    with pytest.raises(ValueError):validate(authored,registry)


@pytest.mark.parametrize('shape',[(2,2,3,4),(0,2,2,3,4),(3,2,2,3,5),(3,1,2,3,4),(3,2,2,3,4,1)])
def test_exact_runtime_nonempty_batched_sample_guard(registry,shape):
    model=GraphModel(graph(),registry)
    with pytest.raises(ValueError,match='ConvTranspose3d.*(?:shape|batch)'):model.execute_node('convolution',{'shaped':torch.zeros(shape)},None)


def test_export_nonempty_guard(registry):
    model=GraphModel(graph(),registry);namespace={};exec(export_source(model),namespace);portable=namespace['Model']()
    with pytest.raises(ValueError,match='ConvTranspose3d.*(?:shape|batch)'):portable(torch.empty(0,2))


def test_aggregate_parameter_rejection_precedes_allocation(registry,monkeypatch):
    authored=graph({'in_channels':500,'out_channels':500,'groups':1,'kernel_size':[1,1,1]},depth=1,height=1,width=1)
    nodes=authored['nodes'];nodes[3:4]=[dict(id=f'conv{i}',type='conv_transpose3d',config=copy.deepcopy(nodes[3]['config'])) for i in range(8)]
    authored['edges']=[dict(id=str(i),source=a['id'],sourcePort='out',target=b['id'],targetPort='in') for i,(a,b) in enumerate(zip(nodes,nodes[1:]))]
    from runner import engine
    monkeypatch.setitem(engine.TRANSPOSE_CONVOLUTION_LAYERS,'conv_transpose3d',lambda *a,**k:pytest.fail('Allocated before aggregate validation'))
    with pytest.raises(ValueError,match='(?i)(parameter|2 million)'):GraphModel(authored,registry)

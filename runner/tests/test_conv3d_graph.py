"""Private Conv3d graph acceptance using explicit indexed cross-correlation."""
import copy
import itertools
import json
import pytest
import torch
import torch.nn.functional as F
from runner.engine import GraphModel, REGISTRY, export_source, validate
from runner.tests.test_conv3d_contract import config


@pytest.fixture
def registry(tmp_path):
    descriptor=json.loads(REGISTRY.read_text());descriptor['components']=[c for c in descriptor['components'] if c['id']!='conv3d']
    defaults=config();fields=[]
    for key in ('in_channels','out_channels','groups'):
        fields.append(dict(key=key,title=key,kind='integer',defaultValue=defaults[key],min=1,max=512))
    for key in ('kernel_size','stride','dilation','padding'):
        fields.append(dict(key=key,title=key,kind='integerList',defaultValue=defaults[key],min=0 if key=='padding' else 1,max=4096,minItems=3,maxItems=3))
    fields.append(dict(key='bias',title='Bias',kind='boolean',defaultValue=True))
    for key,choices in [('padding_policy',['explicit','valid','same']),('padding_mode',['zeros','reflect','replicate','circular'])]:
        fields.append(dict(key=key,title=key,kind='choice',defaultValue=defaults[key],choices=[dict(value=v,title=v.title()) for v in choices]))
    descriptor['components'].append(dict(id='conv3d',title='Conv3d',category='Layers',inputs=['in'],outputs=['out'],fields=fields))
    path=tmp_path/'registry.json';path.write_text(json.dumps(descriptor));return path


def graph(settings=None,depth=3,height=4,width=5):
    c=config(**({'in_channels':2,'out_channels':4,'groups':2,'kernel_size':[2,2,3]} | (settings or {})))
    lengths=[]
    for axis,length in enumerate((depth,height,width)):
        effective=c['dilation'][axis]*(c['kernel_size'][axis]-1)+1
        padding=2*c['padding'][axis] if c['padding_policy']=='explicit' else effective-1 if c['padding_policy']=='same' else 0
        lengths.append((length+padding-effective)//c['stride'][axis]+1)
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':c['in_channels']*depth*height*width,'bias':True}),
           ('shaped','reshape',{'shape':[c['in_channels'],depth,height,width]}),('convolution','conv3d',c),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':c['out_channels']*lengths[0]*lengths[1]*lengths[2],'out_features':2,'bias':True}),('output','output',{})]
    return dict(schemaVersion=2,nodes=[dict(id=i,type=t,config=c) for i,t,c in specs],
        edges=[dict(id=str(k),source=a[0],sourcePort='out',target=b[0],targetPort='in') for k,(a,b) in enumerate(zip(specs,specs[1:]))])


def indexed_reference(x,weight,bias,c):
    sizes=x.shape[-3:];pads=[];lengths=[]
    for axis,size in enumerate(sizes):
        effective=c['dilation'][axis]*(c['kernel_size'][axis]-1)+1
        total=2*c['padding'][axis] if c['padding_policy']=='explicit' else effective-1 if c['padding_policy']=='same' else 0
        pads.append(total//2);lengths.append((size+total-effective)//c['stride'][axis]+1)
    def mapped(index,size):
        if 0<=index<size:return index
        if c['padding_mode']=='zeros':return None
        if c['padding_mode']=='replicate':return max(0,min(size-1,index))
        if c['padding_mode']=='circular':return index%size
        return -index if index<0 else 2*size-2-index
    per_in=c['in_channels']//c['groups'];per_out=c['out_channels']//c['groups'];batches=[]
    for n in range(len(x)):
        channels=[]
        for out in range(c['out_channels']):
            positions=[]
            for position in itertools.product(*(range(size) for size in lengths)):
                total=x[n,0,0,0,0]*0
                for channel in range(per_in):
                    for kernel in itertools.product(*(range(size) for size in c['kernel_size'])):
                        indices=[mapped(position[i]*c['stride'][i]-pads[i]+kernel[i]*c['dilation'][i],sizes[i]) for i in range(3)]
                        if all(index is not None for index in indices):
                            total=total+x[(n,(out//per_out)*per_in+channel,*indices)]*weight[(out,channel,*kernel)]
                positions.append(total+(bias[out] if bias is not None else 0))
            channels.append(torch.stack(positions).reshape(lengths))
        batches.append(torch.stack(channels))
    return torch.stack(batches)


@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
@pytest.mark.parametrize('bias',[False,True])
def test_values_gradients_and_export_against_indexed_oracle(registry,mode,bias):
    authored=graph({'padding_mode':mode,'padding_policy':'same','bias':bias});original=copy.deepcopy(authored)
    authored['nodes'][3]['name']='Spatial (depth, height, width) operator';original=copy.deepcopy(authored)
    torch.manual_seed(639);model=GraphModel(authored,registry).double()
    x=torch.randn(2,2,dtype=torch.float64).t().requires_grad_()
    assert not x.is_contiguous()
    actual,captured=model(x,inspect=True);layer=model.layers['convolution']
    assert isinstance(layer,torch.nn.Conv3d)
    conv=indexed_reference(captured['shaped'],layer.weight,layer.bias,authored['nodes'][3]['config'])
    torch.testing.assert_close(captured['convolution'],conv)
    expected=F.linear(conv.flatten(1),model.layers['head'].weight,model.layers['head'].bias)
    a=torch.autograd.grad(actual.square().sum(),(x,*model.parameters()),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(x,*model.parameters()))
    for first,second in zip(a,b):torch.testing.assert_close(first,second)
    namespace={};source=export_source(model);exec(source,namespace)
    assert 'nn.Conv3d(' in source and 'runner' not in source and 'spatial_depth_height_width_operator' in source
    exported=namespace['Model']().double();exported.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(exported(x),actual)
    portable=torch.autograd.grad(exported(x).square().sum(),(x,*exported.parameters()))
    for first,second in zip(a,portable):torch.testing.assert_close(first,second)
    assert authored==original


@pytest.mark.parametrize('changes',[
    {'stride':[2,1,2],'padding':[1,1,1]}, {'dilation':[2,1,1],'padding_policy':'valid'},
    {'groups':1,'padding':[0,1,0]}, {'groups':2,'kernel_size':[1,1,1],'bias':False},
])
def test_nonsquare_stride_dilation_groups_and_depthwise(registry,changes):
    authored=graph(changes);model=GraphModel(authored,registry).double();x=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    _,values=model(x,inspect=True);layer=model.layers['convolution']
    expected=indexed_reference(values['shaped'],layer.weight,layer.bias,authored['nodes'][3]['config'])
    torch.testing.assert_close(values['convolution'],expected)
    a=torch.autograd.grad(values['convolution'].sum(),(x,layer.weight),retain_graph=True)
    b=torch.autograd.grad(expected.sum(),(x,layer.weight))
    for first,second in zip(a,b):torch.testing.assert_close(first,second)


@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
@pytest.mark.parametrize('policy',['explicit','valid','same'])
@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple GPU unavailable')
def test_actual_mps_all_padding_contracts(registry,mode,policy):
    authored=graph({'padding_mode':mode,'padding_policy':policy,'padding':[1,1,1] if policy=='explicit' else [0,0,0]})
    torch.manual_seed(732);cpu=GraphModel(authored,registry);gpu=GraphModel(authored,registry).to('mps');gpu.load_state_dict(cpu.state_dict())
    x=torch.randn(2,2,requires_grad=True);y=x.detach().to('mps').requires_grad_()
    a=cpu(x);b=gpu(y);torch.testing.assert_close(a,b.cpu(),atol=1e-5,rtol=1e-4)
    first=torch.autograd.grad(a.square().sum(),(x,*cpu.parameters()));second=torch.autograd.grad(b.square().sum(),(y,*gpu.parameters()))
    for p,q in zip(first,second):torch.testing.assert_close(p,q.cpu(),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('change',['rank','channels','int64','legacy','scalar_kernel','bad_choice'])
def test_graph_contract_rejections(registry,change):
    authored=graph()
    if change=='rank':authored['nodes'][2]['config']['shape']=[2,12,5]
    elif change=='channels':authored['nodes'][2]['config']['shape']=[1,6,4,5]
    elif change=='legacy':authored['schemaVersion']=1
    elif change=='scalar_kernel':authored['nodes'][3]['config']['kernel_size']=2
    elif change=='bad_choice':authored['nodes'][3]['config']['padding_mode']='constant'
    else:
        authored['nodes']=authored['nodes'][2:];authored['nodes'][0]=dict(id='shaped',type='index_input',config={'shape':[2,3,4,5]});authored['edges']=authored['edges'][2:]
    with pytest.raises(ValueError):validate(authored,registry)


@pytest.mark.parametrize('shape',[(2,3,4,5),(0,2,3,4,5),(3,2,4,3,5),(3,1,3,4,5),(3,2,3,4,5,1)])
def test_exact_runtime_batched_shape_guard(registry,shape):
    model=GraphModel(graph(),registry)
    with pytest.raises(ValueError,match='(?i)(shape|batch|Conv3d)'):model.execute_node('convolution',{'shaped':torch.zeros(shape)},None)


def test_export_nonempty_batch_guard(registry):
    model=GraphModel(graph(),registry);namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    with pytest.raises(ValueError,match='(?i)(shape|batch|Conv3d)'):exported(torch.empty(0,2))


def test_aggregate_parameters_rejected_before_allocating_conv3d(registry,monkeypatch):
    authored=graph({'in_channels':500,'out_channels':500,'groups':1,'kernel_size':[1,1,1]},depth=1,height=1,width=1)
    nodes=authored['nodes'];nodes[3:4]=[dict(id=f'conv{i}',type='conv3d',config=copy.deepcopy(nodes[3]['config'])) for i in range(8)]
    authored['edges']=[dict(id=str(i),source=a['id'],sourcePort='out',target=b['id'],targetPort='in') for i,(a,b) in enumerate(zip(nodes,nodes[1:]))]
    from runner import engine
    monkeypatch.setitem(engine.CONVOLUTION_LAYERS,'conv3d',lambda *a,**k:pytest.fail('Allocated before aggregate validation'))
    with pytest.raises(ValueError,match='(?i)(parameter|2 million)'):GraphModel(authored,registry)



def test_export_label_tuple_cannot_shadow_shape_guard_builtin(registry):
    authored=graph();authored['nodes'][2]['name']='tuple'
    model=GraphModel(authored,registry)
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    x=torch.randn(2,2)
    torch.testing.assert_close(exported(x),model(x))

"""Private registry Conv1d graph execution, independent indexed cross-correlation."""
import copy
import json
import pytest
import torch
import torch.nn.functional as F
from runner.engine import GraphModel, REGISTRY, export_source, validate
from runner.tests.test_conv1d_contract import config


@pytest.fixture
def registry(tmp_path):
    descriptor=json.loads(REGISTRY.read_text())
    descriptor['components']=[c for c in descriptor['components'] if c['id']!='conv1d']
    fields=[]
    defaults=config()
    for key in ('in_channels','out_channels','kernel_size','stride','dilation','groups','padding'):
        fields.append(dict(key=key,title=key,kind='integer',defaultValue=defaults[key],min=0 if key=='padding' else 1,max=512 if key in ('in_channels','out_channels','groups') else 4096))
    fields.append(dict(key='bias',title='Bias',kind='boolean',defaultValue=True))
    for key,values in [('padding_policy',['explicit','valid','same']),('padding_mode',['zeros','reflect','replicate','circular'])]:
        fields.append(dict(key=key,title=key,kind='choice',defaultValue=defaults[key],choices=[dict(value=v,title=v.title()) for v in values]))
    descriptor['components'].append(dict(id='conv1d',title='Conv1d',category='Layers',inputs=['in'],outputs=['out'],fields=fields))
    path=tmp_path/'registry.json';path.write_text(json.dumps(descriptor));return path


def graph(settings=None,length=7):
    c=config(**({'in_channels':2,'out_channels':4,'groups':2} | (settings or {})))
    effective=c['dilation']*(c['kernel_size']-1)+1
    padding=2*c['padding'] if c['padding_policy']=='explicit' else effective-1 if c['padding_policy']=='same' else 0
    out=(length+padding-effective)//c['stride']+1
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':c['in_channels']*length,'bias':True}),
           ('shaped','reshape',{'shape':[c['in_channels'],length]}),('convolution','conv1d',c),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':c['out_channels']*out,'out_features':2,'bias':True}),('output','output',{})]
    return dict(schemaVersion=2,nodes=[dict(id=i,type=t,config=c) for i,t,c in specs],
                edges=[dict(id=str(k),source=a[0],sourcePort='out',target=b[0],targetPort='in') for k,(a,b) in enumerate(zip(specs,specs[1:]))])


def indexed_reference(x,weight,bias,c):
    length=x.shape[-1];effective=c['dilation']*(c['kernel_size']-1)+1
    left=c['padding'] if c['padding_policy']=='explicit' else (effective-1)//2 if c['padding_policy']=='same' else 0
    right=c['padding'] if c['padding_policy']=='explicit' else effective-1-left if c['padding_policy']=='same' else 0
    output_length=(length+left+right-effective)//c['stride']+1
    per_in=c['in_channels']//c['groups'];per_out=c['out_channels']//c['groups']
    batches=[]
    for n in range(len(x)):
        channels=[]
        for output in range(c['out_channels']):
            positions=[];group=output//per_out
            for position in range(output_length):
                total=x[n,0,0]*0
                for channel in range(per_in):
                    for k in range(c['kernel_size']):
                        index=position*c['stride']-left+k*c['dilation']
                        if not 0<=index<length:
                            if c['padding_mode']=='zeros':continue
                            if c['padding_mode']=='replicate':index=max(0,min(length-1,index))
                            elif c['padding_mode']=='circular':index%=length
                            elif c['padding_mode']=='reflect':index=-index if index<0 else 2*length-2-index
                        total=total+x[n,group*per_in+channel,index]*weight[output,channel,k]
                positions.append(total+(bias[output] if bias is not None else 0))
            channels.append(torch.stack(positions))
        batches.append(torch.stack(channels))
    return torch.stack(batches)


@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
@pytest.mark.parametrize('bias',[False,True])
def test_outputs_all_gradients_and_portable_state_against_index_oracle(registry,mode,bias):
    c={'padding_mode':mode,'padding_policy':'same','kernel_size':4,'bias':bias}
    authored=graph(c);original=copy.deepcopy(authored)
    torch.manual_seed(947);model=GraphModel(authored,registry).double()
    x=torch.randn(2,3,dtype=torch.float64).t().requires_grad_()
    assert not x.is_contiguous()
    actual,captured=model(x,inspect=True);layer=model.layers['convolution']
    assert isinstance(layer,torch.nn.Conv1d)
    expected_conv=indexed_reference(captured['shaped'],layer.weight,layer.bias,authored['nodes'][3]['config'])
    torch.testing.assert_close(captured['convolution'],expected_conv)
    expected=F.linear(expected_conv.flatten(1),model.layers['head'].weight,model.layers['head'].bias)
    measured=torch.autograd.grad(actual.square().sum(),(x,*model.parameters()),retain_graph=True)
    independent=torch.autograd.grad(expected.square().sum(),(x,*model.parameters()))
    for a,b in zip(measured,independent):torch.testing.assert_close(a,b)
    namespace={};source=export_source(model);exec(source,namespace)
    assert 'nn.Conv1d(' in source and 'runner' not in source
    exported=namespace['Model']().double();exported.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(exported(x),actual)
    portable=torch.autograd.grad(exported(x).square().sum(),(x,*exported.parameters()))
    for a,b in zip(portable,measured):torch.testing.assert_close(a,b)
    assert authored==original


@pytest.mark.parametrize('changes',[{'stride':2,'padding':2},{'dilation':2,'padding_policy':'valid'},{'groups':1,'padding':1},{'kernel_size':1,'groups':2,'bias':False}])
def test_stride_dilation_group_and_depthwise_shapes(registry,changes):
    authored=graph(changes);model=GraphModel(authored,registry).double()
    x=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    _,captured=model(x,inspect=True);layer=model.layers['convolution']
    expected=indexed_reference(captured['shaped'],layer.weight,layer.bias,authored['nodes'][3]['config'])
    torch.testing.assert_close(captured['convolution'],expected)
    a=torch.autograd.grad(captured['convolution'].sum(),(x,layer.weight),retain_graph=True)
    b=torch.autograd.grad(expected.sum(),(x,layer.weight))
    for first,second in zip(a,b):torch.testing.assert_close(first,second)


@pytest.mark.parametrize('change',['rank','int64','channels','legacy'])
def test_invalid_graph_layout_dtype_and_version(registry,change):
    authored=graph()
    if change=='rank':authored['nodes'][2]['config']['shape']=[14]
    elif change=='channels':authored['nodes'][2]['config']['shape']=[1,14]
    elif change=='legacy':authored['schemaVersion']=1
    else:
        authored['nodes']=authored['nodes'][2:]
        authored['nodes'][0]=dict(id='shaped',type='index_input',config={'shape':[2,7]})
        authored['edges']=authored['edges'][2:]
    with pytest.raises(ValueError):validate(authored,registry)


def test_aggregate_parameter_cap_before_module_allocation(registry,monkeypatch):
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':500,'bias':True}),('shaped','reshape',{'shape':[500,1]})]
    specs += [(f'conv{i}','conv1d',config(in_channels=500,out_channels=500,kernel_size=1,groups=1)) for i in range(8)]
    specs += [('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':500,'out_features':2,'bias':True}),('output','output',{})]
    authored=dict(schemaVersion=2,nodes=[dict(id=i,type=t,config=c) for i,t,c in specs],edges=[dict(id=str(k),source=a[0],sourcePort='out',target=b[0],targetPort='in') for k,(a,b) in enumerate(zip(specs,specs[1:]))])
    from runner import engine
    monkeypatch.setitem(engine.CONVOLUTION_LAYERS,'conv1d',lambda *a,**k:pytest.fail('Allocated before aggregate validation'))
    with pytest.raises(ValueError,match='(?i)(2 million|parameter)'):GraphModel(authored,registry)


@pytest.mark.parametrize('change',['restricted','duplicate','empty','blank_title','numeric_value','numeric_bounds','nullable'])
def test_categorical_registry_contract_is_enforced(registry,change):
    descriptor=json.loads(registry.read_text());field=next(f for f in descriptor['components'][-1]['fields'] if f['key']=='padding_mode')
    if change=='restricted':field['choices']=[dict(value='zeros',title='Zeros')]
    elif change=='duplicate':field['choices'].append(field['choices'][0].copy())
    elif change=='empty':field['choices']=[]
    elif change=='blank_title':field['choices'][0]['title']=' '
    elif change=='numeric_value':field['choices'][0]['value']=0
    elif change=='numeric_bounds':field['min']=0
    else:field['allowsNull']=True
    registry.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError):validate(graph({'padding_mode':'reflect','padding':1}),registry)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple GPU unavailable')
@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
@pytest.mark.parametrize('policy',['explicit','valid','same'])
def test_actual_mps_output_and_gradients_match_cpu(registry,mode,policy):
    authored=graph({'padding_mode':mode,'padding_policy':policy,'padding':1 if policy=='explicit' else 0,'stride':1 if policy=='same' else 2,'kernel_size':4})
    torch.manual_seed(437);cpu=GraphModel(authored,registry);gpu=GraphModel(authored,registry).to('mps');gpu.load_state_dict(cpu.state_dict())
    x=torch.randn(4,2,requires_grad=True);y=x.detach().to('mps').requires_grad_()
    a=cpu(x);b=gpu(y);torch.testing.assert_close(a,b.cpu(),atol=1e-5,rtol=1e-4)
    first=torch.autograd.grad(a.square().sum(),(x,*cpu.parameters()));second=torch.autograd.grad(b.square().sum(),(y,*gpu.parameters()))
    for p,q in zip(first,second):torch.testing.assert_close(p,q.cpu(),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('shape',[(2,7),(0,2,7),(3,2,8),(3,1,7),(3,2,1,7)])
def test_runtime_boundary_refuses_shape_budget_bypass(registry,shape):
    model=GraphModel(graph(),registry)
    with pytest.raises(ValueError,match='(?i)(shape|batch|Conv1d)'):
        model.execute_node('convolution',{'shaped':torch.zeros(shape)},None)


def test_export_retains_exact_batched_conv_boundary(registry):
    model=GraphModel(graph(),registry)
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    # The preceding projection/reshape permits an empty leading axis unless the
    # explicit Conv1d boundary checks it before invoking the layer.
    with pytest.raises(ValueError,match='(?i)(shape|batch|Conv1d)'):
        exported(torch.empty(0,2))


def test_export_user_label_tuple_does_not_shadow_shape_check(registry):
    authored=graph()
    authored['nodes'][0]['name']='tuple'
    model=GraphModel(authored,registry)
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    x=torch.tensor([[.2,.4]])
    torch.testing.assert_close(exported(x),model(x))

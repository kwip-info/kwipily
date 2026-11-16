"""Real production graph/export checks with independent scalar bin expectations."""
import copy
import math
import warnings
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source, validate

CONFIGS={
 'avg_pool2d':dict(kernel_size=[3,2],stride=[2,2],padding=[1,1],ceil_mode=True,count_include_pad=True,divisor_override=-3),
 'adaptive_avg_pool1d':dict(output_size=3),
 'adaptive_avg_pool2d':dict(output_height=3,output_width=None),
 'adaptive_max_pool2d':dict(output_height=3,output_width=2,return_indices=True),
}
SHAPES={kind:([2,5] if kind.endswith('1d') else [2,5,4]) for kind in CONFIGS}

def graph(kind,config=None):
    config=copy.deepcopy(CONFIGS[kind] if config is None else config);shape=SHAPES[kind]
    if kind=='avg_pool2d':out=[2,3,3]
    elif kind.endswith('1d'):out=[2,config['output_size']]
    else:out=[2,config['output_height'] or 5,config['output_width'] or 4]
    rows=[('input','input',{'features':2}),('project','linear',{'in_features':2,'out_features':math.prod(shape),'bias':True}),('shaped','reshape',{'shape':shape}),('pool',kind,config),('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':math.prod(out),'out_features':2,'bias':True}),('output','output',{})]
    return {'schemaVersion':2,'name':'Pool independent fixture','nodes':[{'id':ident,'type':op,'name':ident,'config':cfg} for ident,op,cfg in rows], 'edges':[{'id':str(i),'source':rows[i][0],'sourcePort':'out','target':rows[i+1][0],'targetPort':'in'} for i in range(len(rows)-1)]}

def scalar_reference(x,kind,cfg):
    sample=list(x.shape[2:]);one=kind.endswith('1d');avg='avg' in kind
    if kind=='avg_pool2d':output=[3,3]
    else:output=[cfg['output_size']] if one else [cfg['output_height'] or sample[0],cfg['output_width'] or sample[1]]
    y=[];indices=[];contributors=[];divisors=[]
    for n in range(x.shape[0]):
      for c in range(x.shape[1]):
       for oi in range(math.prod(output)):
        pos=[oi] if one else [oi//output[1],oi%output[1]]
        ranges=[];padded_counts=[]
        for axis,(v,lin,lout) in enumerate(zip(pos,sample,output)):
          if kind=='avg_pool2d':
            start=v*cfg['stride'][axis]-cfg['padding'][axis];end=start+cfg['kernel_size'][axis]
            padded_counts.append(max(0,min(end,lin+cfg['padding'][axis])-max(start,-cfg['padding'][axis])))
            ranges.append(range(max(0,start),min(lin,end)))
          else:ranges.append(range(v*lin//lout,((v+1)*lin+lout-1)//lout))
        candidates=[(n,c,p) for p in ranges[0]] if one else [(n,c,h,w) for h in ranges[0] for w in ranges[1]]
        vals=[float(x[p]) for p in candidates]
        divisor=cfg['divisor_override'] if kind=='avg_pool2d' and cfg['divisor_override'] is not None else (math.prod(padded_counts) if kind=='avg_pool2d' and cfg['count_include_pad'] else len(candidates))
        winner=max(range(len(vals)),key=lambda i:vals[i])
        y.append(sum(vals)/divisor if avg else vals[winner]);contributors.append(candidates if avg else [candidates[winner]]);divisors.append(divisor if avg else 1)
        p=candidates[winner];indices.append(p[-1] if one else p[-2]*sample[1]+p[-1])
    return torch.tensor(y).reshape(x.shape[0],x.shape[1],*output),torch.tensor(indices,dtype=torch.int64).reshape(x.shape[0],x.shape[1],*output),contributors,divisors

@pytest.mark.parametrize('kind',CONFIGS)
def test_actual_outputs_weighted_gradients_and_standalone_export(kind):
    model=GraphModel(graph(kind),REGISTRY)
    sample=SHAPES[kind];x=(torch.arange(2*math.prod(sample),dtype=torch.float32)%11-9).reshape(2,*sample).requires_grad_()
    result=model.execute_node('pool',{'shaped':x},None);actual=result['out'] if isinstance(result,dict) else result
    expected,locations,contributors,divisors=scalar_reference(x.detach(),kind,CONFIGS[kind])
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
    weights=(torch.arange(actual.numel(),dtype=torch.float32)%7-3).reshape(actual.shape)
    wanted=torch.zeros_like(x)
    for weight,positions,divisor in zip(weights.flatten(),contributors,divisors):
      for position in positions:wanted[position]+=weight/divisor
    torch.testing.assert_close(torch.autograd.grad((actual*weights).sum(),x)[0],wanted,rtol=2e-6,atol=2e-6)
    if isinstance(result,dict):assert result['indices'].dtype==torch.int64;torch.testing.assert_close(result['indices'],locations,rtol=0,atol=0)
    source=export_source(model);assert 'runner.' not in source
    namespace={};exec(compile(source,'standalone_model.py','exec'),namespace);exported=namespace['Model']();exported.load_state_dict(model.state_dict(),strict=True)
    values=torch.randn(3,2,requires_grad=True);other=values.detach().clone().requires_grad_()
    torch.testing.assert_close(model(values),exported(other),rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(model(values).sum(),values)[0],torch.autograd.grad(exported(other).sum(),other)[0],rtol=0,atol=0)
    assert set(model.state_dict())==set(exported.state_dict());assert not any('pool' in key for key in model.state_dict())

@pytest.mark.parametrize('kind',CONFIGS)
@pytest.mark.parametrize('problem',['dtype','sparse','empty','shape'])
def test_execution_refuses_invalid_actual_tensor(kind,problem):
    model=GraphModel(graph(kind),REGISTRY);x=torch.ones(1,*SHAPES[kind])
    if problem=='dtype':x=x.double()
    elif problem=='sparse':x=x.to_sparse()
    elif problem=='empty':x=x[:0]
    else:x=x[...,0]
    with pytest.raises(ValueError):model.execute_node('pool',{'shaped':x},None)

@pytest.mark.parametrize('kind',CONFIGS)
def test_legacy_schema_refused(kind):
    g=graph(kind);g['schemaVersion']=1
    with pytest.raises(ValueError):validate(g,REGISTRY)

@pytest.mark.parametrize('kind',['adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d'])
def test_allocation_budget_refused_before_pool_function(kind,monkeypatch):
    model=GraphModel(graph(kind),REGISTRY)
    # A stride-zero view has a huge logical batch without allocating that batch.
    sample=SHAPES[kind];x=torch.zeros(1,*sample).expand(100_000_001,*sample)
    called=[];monkeypatch.setattr(torch.nn.functional,kind,lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='budget'):model.execute_node('pool',{'shaped':x},None)
    assert called==[]

@pytest.mark.parametrize('kind',CONFIGS)
def test_mps_execution_export_and_visible_shape_fallback(kind):
    if not torch.backends.mps.is_available():pytest.skip('MPS is unavailable on this host')
    a=GraphModel(graph(kind),REGISTRY);b=GraphModel(graph(kind),REGISTRY).to('mps');b.load_state_dict(a.state_dict())
    namespace={};exec(export_source(a),namespace);exported=namespace['Model']().to('mps');exported.load_state_dict(a.state_dict())
    x=torch.randn(2,2,requires_grad=True);y=x.detach().to('mps').requires_grad_()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always');actual=b(y);other=exported(y)
    torch.testing.assert_close(actual.cpu(),a(x),rtol=2e-5,atol=2e-6);torch.testing.assert_close(actual,other,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),y)[0].cpu(),torch.autograd.grad(a(x).sum(),x)[0],rtol=2e-5,atol=2e-6)
    assert any('CPU fallback' in str(item.message) for item in caught)==(kind.startswith('adaptive_avg'))

ADAPTIVE_VARIANTS=[
 ('adaptive_avg_pool1d',{'output_size':1}),('adaptive_avg_pool1d',{'output_size':7}),
 ('adaptive_avg_pool2d',{'output_height':None,'output_width':2}),
 ('adaptive_avg_pool2d',{'output_height':None,'output_width':None}),
 ('adaptive_avg_pool2d',{'output_height':7,'output_width':6}),
 ('adaptive_max_pool2d',{'output_height':None,'output_width':None,'return_indices':True}),
 ('adaptive_max_pool2d',{'output_height':7,'output_width':6,'return_indices':True}),
 ('adaptive_max_pool2d',{'output_height':1,'output_width':1,'return_indices':False}),
]

@pytest.mark.parametrize('kind,cfg',ADAPTIVE_VARIANTS)
def test_kept_axes_global_upsampling_ties_and_port_availability(kind,cfg):
    model=GraphModel(graph(kind,cfg),REGISTRY);x=torch.full((2,*SHAPES[kind]),-2.0,requires_grad=True)
    result=model.execute_node('pool',{'shaped':x},None);actual=result['out'] if isinstance(result,dict) else result
    expected,indices,contributors,divisors=scalar_reference(x.detach(),kind,cfg)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    wanted=torch.zeros_like(x)
    for positions,divisor in zip(contributors,divisors):
      for position in positions:wanted[position]+=1/divisor
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),x)[0],wanted)
    if isinstance(result,dict):torch.testing.assert_close(result['indices'],indices,rtol=0,atol=0)
    else:assert cfg.get('return_indices',False) is False
    namespace={};exec(export_source(model),namespace);exported=namespace['Model']();exported.load_state_dict(model.state_dict(),strict=True)
    values=torch.randn(1,2);torch.testing.assert_close(model(values),exported(values),rtol=0,atol=0)

@pytest.mark.parametrize('kind',CONFIGS)
@pytest.mark.parametrize('bad',[True,None,0,2.5,'3',float('inf')])
def test_bad_configuration_refused(kind,bad):
    cfg=copy.deepcopy(CONFIGS[kind]);key='kernel_size' if kind=='avg_pool2d' else 'output_size' if kind.endswith('1d') else 'output_height'
    if bad is None and key=='output_height':cfg['unknown']=None
    else:cfg[key]=bad
    g=graph(kind);next(node for node in g['nodes'] if node['id']=='pool')['config']=cfg
    with pytest.raises(ValueError):GraphModel(g,REGISTRY)

@pytest.mark.parametrize('kind',CONFIGS)
def test_nonfinite_and_standalone_dtype_refusal(kind):
    model=GraphModel(graph(kind),REGISTRY);x=torch.ones(1,*SHAPES[kind]);x.flatten()[0]=float('nan')
    with pytest.raises(ValueError,match='finite'):model.execute_node('pool',{'shaped':x},None)
    namespace={};exec(export_source(model),namespace);exported=namespace['Model']()
    # Invoking the portable adapter directly checks its boundary before other layers.
    from runner.engine import _pooling_kwargs
    sample=SHAPES[kind];output_sample=(2,3,3) if kind=='avg_pool2d' else (2,3) if kind.endswith('1d') else (2,3,4) if kind=='adaptive_avg_pool2d' else (2,3,2)
    with pytest.raises(ValueError,match='float32'):namespace['_execute_reviewed_pooling'](torch.ones(1,*sample,dtype=torch.float64),kind,_pooling_kwargs(kind,CONFIGS[kind]),sample,output_sample)

@pytest.mark.parametrize('kind,cfg',[('adaptive_avg_pool1d',{'output_size':1}),('adaptive_avg_pool2d',{'output_height':None,'output_width':2})])
def test_divisible_mps_bins_run_without_fallback(kind,cfg):
    if not torch.backends.mps.is_available():pytest.skip('MPS is unavailable on this host')
    model=GraphModel(graph(kind,cfg),REGISTRY).to('mps');x=torch.ones(1,*SHAPES[kind],device='mps',requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always');result=model.execute_node('pool',{'shaped':x},None);torch.autograd.grad(result.sum(),x)
    assert not any('CPU fallback' in str(item.message) for item in caught)


def test_named_indices_contract_and_disabled_edge_refusal():
    g=graph('adaptive_max_pool2d');_,_,layouts,types=validate(g,REGISTRY,include_types=True)
    assert types['pool']=={'out':'float32','indices':'int64'};assert layouts['pool']['out']==layouts['pool']['indices']
    cfg=dict(CONFIGS['adaptive_max_pool2d'],return_indices=False);g=graph('adaptive_max_pool2d',cfg)
    next(edge for edge in g['edges'] if edge['source']=='pool')['sourcePort']='indices'
    with pytest.raises(ValueError):validate(g,REGISTRY)


def test_divisor_zero_is_not_hidden_by_other_configuration():
    cfg=dict(CONFIGS['avg_pool2d'],divisor_override=0);g=graph('avg_pool2d',cfg)
    with pytest.raises(ValueError,match='nonzero'):validate(g,REGISTRY)

@pytest.mark.parametrize('kind,cfg,device,fallback',[
 ('adaptive_avg_pool1d',CONFIGS['adaptive_avg_pool1d'],'mps',True),
 ('adaptive_avg_pool2d',CONFIGS['adaptive_avg_pool2d'],'mps',True),
 ('adaptive_avg_pool1d',{'output_size':1},'mps',False),
 ('adaptive_avg_pool2d',{'output_height':None,'output_width':2},'mps',False),
 *[(kind,CONFIGS[kind],'cpu',False) for kind in CONFIGS],
])
def test_actual_training_persists_bounded_device_policy_and_warning(tmp_path,kind,cfg,device,fallback):
    import json
    from runner.engine import run
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS is unavailable on this host')
    events=[]
    with warnings.catch_warnings():
      warnings.simplefilter('ignore',RuntimeWarning)
      result=run(graph(kind,cfg),tmp_path/'run',epochs=1,sample_limit=32,batch_size=8,device=device,registry=REGISTRY,emit=events.append)
    manifest=json.loads((tmp_path/'run'/'manifest.json').read_text());started=next(event for event in events if event['event']=='started');completed=next(event for event in events if event['event']=='completed')
    emitted=[event for event in events if event['event']=='warning' and event.get('code')=='pooling.cpuFallback']
    if fallback:
      expected=[{'nodeID':'pool','operator':kind,'computeDevice':'cpu','returnDevice':'mps','reason':'nondivisibleAdaptiveAverageBins'}]
      assert manifest['runtime']['operationDevices']==started['operationDevices']==completed['operationDevices']==result['operationDevices']==expected
      assert len(emitted)==1;assert emitted[0]['nodeID']=='pool';assert emitted[0]['runID']==manifest['runID'];assert emitted[0]['schemaVersion']==1;assert 'CPU fallback' in emitted[0]['message']
    else:
      assert 'operationDevices' not in manifest['runtime'];assert 'operationDevices' not in started;assert 'operationDevices' not in completed;assert emitted==[]

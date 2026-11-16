"""Actual portable bag graph math; no runtime module imports in generated source."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import pytest
import torch
from runner import engine
from runner.bag_export import standalone_bag_helpers
from runner.embedding_bag_packet import EmbeddingBagPacket,collate_bags
from runner.tests.test_bag_source_graph import graph,packet,registry

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
TABLE=[[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]]
UPSTREAM=[[2.,-1.],[7.,11.],[-3.,5.],[4.,6.]]
OUTPUTS={'sum':[[5.,3.],[0.,0.],[-2.,4.],[3.,4.]],'mean':[[5/3,1.],[0.,0.],[-2.,4.],[3.,4.]],'max':[[3.,2.],[0.,0.],[-2.,4.],[3.,4.]]}
GRADS={'sum':[[0.,0.],[4.,-2.],[2.,-1.],[-3.,5.],[4.,6.]],'mean':[[0.,0.],[4/3,-2/3],[2/3,-1/3],[-3.,5.],[4.,6.]],'max':[[0.,0.],[0.,-1.],[2.,0.],[-3.,5.],[4.,6.]]}


def exported_model(tmp_path,model):
    source=engine.export_source(model)
    assert 'from runner' not in source and 'import runner' not in source
    path=tmp_path/(uuid.uuid4().hex+'.py');path.write_text(source)
    name='bag_portable_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    exported=module.Model();exported.load_state_dict(model.state_dict(),strict=True)
    return module,exported,source

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('layout',['fixed','packed'])
@pytest.mark.parametrize('mode,weighted',[('sum',False),('mean',False),('max',False),('sum',True)])
def test_independent_portable_outputs_and_all_derivatives(registry,tmp_path,device,layout,mode,weighted):
    value=graph(layout,weighted);value['nodes'][1]['config']['mode']=mode
    original=engine.GraphModel(value,registry,bundled_dataset=False)
    with torch.no_grad():original.layers['bag'].weight.copy_(torch.tensor(TABLE))
    module,exported,source=exported_model(tmp_path,original);exported.to(device)
    assert list(exported.state_dict())==['layers.bag.weight']
    item=packet(device,layout,weighted)
    output,observed=exported(item.indices,item.offsets,item.per_sample_weights,inspect=True)
    expected=[[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]] if weighted else OUTPUTS[mode]
    torch.testing.assert_close(output,torch.tensor(expected,device=device),rtol=2e-6,atol=2e-6)
    assert observed['source']['indices'] is item.indices
    if weighted:assert observed['source']['per_sample_weights'] is item.per_sample_weights
    (output*torch.tensor(UPSTREAM,device=device)).sum().backward()
    expected_grad=[[0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]] if weighted else GRADS[mode]
    torch.testing.assert_close(exported.layers['bag'].weight.grad,torch.tensor(expected_grad,device=device),rtol=2e-6,atol=2e-6)
    if weighted:
        desired=[0.,7.,0.,0.,26.,36.,0.] if layout=='packed' else [[0.,7.,0.],[0.,0.,0.],[0.,26.,0.],[36.,0.,0.]]
        torch.testing.assert_close(item.per_sample_weights.grad,torch.tensor(desired,device=device))
    typed=module.EmbeddingBagPacket(layout,item.indices,item.offsets,item.per_sample_weights,True)
    torch.testing.assert_close(exported(typed),output)
    with pytest.raises(ValueError):exported(typed,per_sample_weights=torch.ones_like(item.indices,dtype=torch.float32))
    chunks=[]
    for i in range(4):
        micro,_=collate_bags(item,[i],num_embeddings=5,embedding_dim=2,mode=mode)
        chunks.append(exported(micro.indices,micro.offsets,micro.per_sample_weights))
    torch.testing.assert_close(torch.cat(chunks),output)

@pytest.mark.parametrize('initial,actual',[(False,False),(True,True)])
def test_captured_trainability_overrides_initial_freeze(registry,tmp_path,initial,actual):
    value=graph();value['nodes'][1]['config']['freeze']=initial
    # Keep a real trainable downstream head when initial table is frozen.
    value['nodes'].insert(2,dict(id='head',type='linear',config={'in_features':2,'out_features':2,'bias':False}))
    value['edges'][-1].update(target='head');value['edges'].append(dict(id='headout',source='head',sourcePort='out',target='result',targetPort='in'))
    original=engine.GraphModel(value,registry,bundled_dataset=False);original.layers['bag'].weight.requires_grad_(actual)
    _,exported,_=exported_model(tmp_path,original)
    assert exported.layers['bag'].weight.requires_grad is actual
    source=packet('cpu');out=exported(source.indices,source.offsets,source.per_sample_weights);out.sum().backward()
    assert (exported.layers['bag'].weight.grad is not None) is actual
    assert source.per_sample_weights.grad is not None

@pytest.mark.parametrize('bad',['indices','weights','terminal','capacity','dtype','native_controls','nonfinite'])
def test_export_preserves_strict_admission(registry,tmp_path,bad):
    original=engine.GraphModel(graph(),registry,bundled_dataset=False)
    module,exported,_=exported_model(tmp_path,original);item=packet('cpu')
    if bad=='indices':item.indices[0]=5
    elif bad=='weights':item=EmbeddingBagPacket('packed',item.indices,item.offsets,None,True)
    elif bad=='terminal':item=EmbeddingBagPacket('packed',item.indices,torch.tensor([0,3,3,5,6]),item.per_sample_weights,True)
    elif bad=='capacity':item=EmbeddingBagPacket('packed',item.indices,torch.tensor([0,4,4,5,7]),item.per_sample_weights,True)
    elif bad=='dtype':item=EmbeddingBagPacket('packed',item.indices.float(),item.offsets,item.per_sample_weights,True)
    elif bad=='native_controls':exported.layers['bag'].mode='max'
    else:
        with torch.no_grad():exported.layers['bag'].weight[1,0]=float('nan')
    with pytest.raises(ValueError):exported(item.indices,item.offsets,item.per_sample_weights)

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
@pytest.mark.parametrize('setting',['max_norm','frequency'])
def test_export_preserves_real_mps_limitations(registry,tmp_path,setting):
    value=graph();value['nodes'][1]['config'].update({'max_norm':2.} if setting=='max_norm' else {'scale_grad_by_freq':True})
    original=engine.GraphModel(value,registry,bundled_dataset=False)
    _,exported,_=exported_model(tmp_path,original);exported.to('mps');item=packet('mps')
    with pytest.raises(ValueError,match='CPU'):exported(item.indices,item.offsets,item.per_sample_weights)


def test_standalone_helpers_are_bounded_and_import_only_stdlib_torch():
    source=standalone_bag_helpers()
    assert 0<len(source.encode())<=131072
    assert 'class EmbeddingBagPacket' in source and 'class CheckedEmbeddingBag' in source
    assert 'BAG_HELPER_SHA256' in source
    assert 'from runner' not in source and 'import runner' not in source


def test_isolated_saved_state_reload_outside_checkout(registry,tmp_path):
    original=engine.GraphModel(graph(),registry,bundled_dataset=False)
    with torch.no_grad():original.layers['bag'].weight.copy_(torch.tensor(TABLE))
    source=engine.export_source(original)
    isolated=tmp_path/'isolated';isolated.mkdir()
    (isolated/'model.py').write_text(source);torch.save(original.state_dict(),isolated/'weights.pt')
    code="""
import importlib.util,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('portable',root/'model.py')
m=importlib.util.module_from_spec(spec);sys.modules['portable']=m;spec.loader.exec_module(m)
model=m.Model();model.load_state_dict(torch.load(root/'weights.pt',weights_only=True),strict=True)
x=torch.tensor([1,2,1,0,3,4,0]);o=torch.tensor([0,3,3,5,7]);w=torch.tensor([.5,2.,-1.,99.,3.,.25,88.],requires_grad=True)
y=model(x,o,w)
assert y.tolist()==[[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]]
assert not any(k=='runner' or k.startswith('runner.') for k in sys.modules)
print('ISOLATED_BAG_EXPORT_PASS')
"""
    result=subprocess.run([sys.executable,'-I','-c',code,str(isolated)],cwd=isolated,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='ISOLATED_BAG_EXPORT_PASS'


@pytest.mark.parametrize('layout',['fixed','packed'])
def test_false_terminal_flag_is_retained(registry,tmp_path,layout):
    original=engine.GraphModel(graph(layout,False,False),registry,bundled_dataset=False)
    with torch.no_grad():original.layers['bag'].weight.copy_(torch.tensor(TABLE))
    _,exported,_=exported_model(tmp_path,original)
    item=packet('cpu',layout,False,False)
    torch.testing.assert_close(exported(item.indices,item.offsets),torch.tensor(OUTPUTS['sum']))


@pytest.mark.parametrize('control',['frequency','sparse','max_norm'])
def test_real_cpu_nondefault_controls_and_native_state(registry,tmp_path,control):
    value=graph('packed',False,False)
    value['nodes'][1]['config'].update({'scale_grad_by_freq':True} if control=='frequency' else {'sparse':True} if control=='sparse' else {'max_norm':1.,'norm_type':2.})
    original=engine.GraphModel(value,registry,bundled_dataset=False)
    with torch.no_grad():original.layers['bag'].weight.copy_(torch.tensor(TABLE))
    _,exported,_=exported_model(tmp_path,original)
    indices=torch.tensor([1,1,1]);offsets=torch.tensor([0])
    output=exported(indices,offsets)
    row=[1.,2.] if control!='max_norm' else [1/(5**.5),2/(5**.5)]
    torch.testing.assert_close(output,torch.tensor([[3*row[0],3*row[1]]]))
    (output*torch.tensor([[2.,-1.]])).sum().backward()
    grad=exported.layers['bag'].weight.grad
    assert grad.is_sparse is (control=='sparse')
    if grad.is_sparse:grad=grad.to_dense()
    expected=torch.zeros(5,2);expected[1]=torch.tensor([2.,-1.] if control=='frequency' else [6.,-3.])
    torch.testing.assert_close(grad,expected)
    if control=='max_norm':torch.testing.assert_close(exported.layers['bag'].weight[1],torch.tensor(row))

@pytest.mark.parametrize('label',['inspect','offsets','per_sample_weights'])
def test_bag_signature_names_are_reserved_tensor_aliases(registry,tmp_path,label):
    value=graph();value['nodes'][1]['name']=label
    original=engine.GraphModel(value,registry,bundled_dataset=False)
    _,exported,_=exported_model(tmp_path,original)
    item=packet('cpu');actual,observed=exported(item.indices,item.offsets,item.per_sample_weights,inspect=True)
    torch.testing.assert_close(actual,original(item))
    assert 'bag' in observed

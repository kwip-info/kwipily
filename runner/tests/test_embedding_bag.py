"""Actual native adapter; expected arrays below are independent hand arithmetic."""
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from runner.embedding_bag import CheckedEmbeddingBag
from runner.embedding_bag_configuration import DEFAULTS
from runner.embedding_bag_packet import EmbeddingBagPacket

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
TABLE = [[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]]
UPSTREAM = [[2.,-1.],[7.,11.],[-3.,5.],[4.,6.]]
OUTPUTS = {'sum':[[5.,3.],[0.,0.],[-2.,4.],[3.,4.]], 'mean':[[5/3,1.],[0.,0.],[-2.,4.],[3.,4.]], 'max':[[3.,2.],[0.,0.],[-2.,4.],[3.,4.]]}
GRADS = {'sum':[[0.,0.],[4.,-2.],[2.,-1.],[-3.,5.],[4.,6.]], 'mean':[[0.,0.],[4/3,-2/3],[2/3,-1/3],[-3.,5.],[4.,6.]], 'max':[[0.,0.],[0.,-1.],[2.,0.],[-3.,5.],[4.,6.]]}


def config(**changes):return {**DEFAULTS,'num_embeddings':5,'embedding_dim':2,'padding_idx':0,**changes}
def adapter(device, **changes):
    layer = CheckedEmbeddingBag(config(**changes),device=device)
    with torch.no_grad():layer.weight.copy_(torch.tensor(TABLE,device=device))
    return layer

def packet(device,layout='packed',terminal=True,weighted=False):
    if layout=='fixed':
        ids=[[1,2,1],[0,0,0],[0,3,0],[4,0,0]]
        weights=[[.5,2.,-1.],[0.,0.,0.],[99.,3.,0.],[.25,88.,0.]]
        offsets=None
    else:
        ids=[1,2,1,0,3,4,0];weights=[.5,2.,-1.,99.,3.,.25,88.]
        offsets=torch.tensor([0,3,3,5,7] if terminal else [0,3,3,5],device=device)
    weights=torch.tensor(weights,device=device,requires_grad=True) if weighted else None
    return EmbeddingBagPacket(layout,torch.tensor(ids,device=device),offsets,weights,terminal)

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('layout,terminal',[('fixed',False),('fixed',True),('packed',False),('packed',True)])
@pytest.mark.parametrize('mode',['sum','mean','max'])
def test_literal_reductions_and_table_gradients(device,layout,terminal,mode):
    layer=adapter(device,mode=mode,layout=layout,include_last_offset=terminal)
    source=packet(device,layout,terminal);ids=source.indices.clone()
    out=layer(source)
    torch.testing.assert_close(out,torch.tensor(OUTPUTS[mode],device=device),rtol=2e-6,atol=2e-6)
    (out*torch.tensor(UPSTREAM,device=device)).sum().backward()
    torch.testing.assert_close(layer.weight.grad,torch.tensor(GRADS[mode],device=device),rtol=2e-6,atol=2e-6)
    assert torch.equal(ids,source.indices)
    assert list(layer.state_dict())==['weight']

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('layout',['fixed','packed'])
@pytest.mark.parametrize('frozen',[False,True])
def test_weighted_sum_preserves_actual_input_gradient_even_frozen_table(device,layout,frozen):
    layer=adapter(device,mode='sum',layout=layout,include_last_offset=True,per_sample_weights=True,freeze=frozen)
    source=packet(device,layout,True,True);out=layer(source)
    torch.testing.assert_close(out,torch.tensor([[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]],device=device))
    (out*torch.tensor(UPSTREAM,device=device)).sum().backward()
    expected=[0.,7.,0.,0.,26.,36.,0.] if layout=='packed' else [[0.,7.,0.],[0.,0.,0.],[0.,26.,0.],[36.,0.,0.]]
    torch.testing.assert_close(source.per_sample_weights.grad,torch.tensor(expected,device=device))
    assert layer.weight.requires_grad is not frozen
    if frozen:assert layer.weight.grad is None
    else:torch.testing.assert_close(layer.weight.grad,torch.tensor([[0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]],device=device))

@pytest.mark.parametrize('device',DEVICES)
def test_simple_frequency_scaling_and_first_maximum_tie(device):
    if device == 'mps':
        with pytest.raises(ValueError, match='CPU'):
            adapter(device,mode='sum',layout='packed',scale_grad_by_freq=True)
    else:
        _check_frequency(device)
    _check_ties(device)

def _check_frequency(device):
    layer=adapter(device,mode='sum',layout='packed',scale_grad_by_freq=True)
    source=EmbeddingBagPacket('packed',torch.tensor([1,1,1],device=device),torch.tensor([0],device=device))
    (layer(source)*torch.tensor([[2.,-1.]],device=device)).sum().backward()
    torch.testing.assert_close(layer.weight.grad[1],torch.tensor([2.,-1.],device=device))

def _check_ties(device):
    for ids,expected in [([2,4],[[0.,0.],[0.,0.],[2.,0.],[0.,0.],[0.,6.]]),([4,2],[[0.,0.],[0.,0.],[0.,0.],[0.,0.],[2.,6.]])]:
        layer=adapter(device,mode='max',layout='packed')
        source=EmbeddingBagPacket('packed',torch.tensor(ids,device=device),torch.tensor([0],device=device))
        (layer(source)*torch.tensor([[2.,6.]],device=device)).sum().backward()
        torch.testing.assert_close(layer.weight.grad,torch.tensor(expected,device=device))

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('frozen',[False,True])
def test_max_norm_mutates_only_selected_rows_even_frozen(device,frozen):
    if device == 'mps':
        with pytest.raises(ValueError, match='CPU'):
            adapter(device,mode='sum',layout='packed',padding_idx=None,max_norm=2.,norm_type=2.,freeze=frozen)
        return
    layer=adapter(device,mode='sum',layout='packed',padding_idx=None,max_norm=2.,norm_type=2.,freeze=frozen)
    before=layer.weight.detach().clone()
    source=EmbeddingBagPacket('packed',torch.tensor([4],device=device),torch.tensor([0],device=device))
    output=layer(source)
    expected=torch.tensor([[6/(5+1e-7),8/(5+1e-7)]],device=device)
    torch.testing.assert_close(output,expected,rtol=2e-6,atol=2e-6)
    assert torch.equal(layer.weight[:4],before[:4])
    assert layer.weight.requires_grad is not frozen

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('weighted',[False,True])
def test_sparse_actual_gradient_and_empty_packed_bags(device,weighted):
    layer=adapter(device,mode='sum',layout='packed',include_last_offset=True,sparse=True,per_sample_weights=weighted)
    source=packet(device,weighted=weighted);out=layer(source)
    (out*torch.tensor(UPSTREAM,device=device)).sum().backward()
    assert layer.weight.grad.layout==torch.sparse_coo
    expected=GRADS['sum'] if not weighted else [[0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]]
    torch.testing.assert_close(layer.weight.grad.to_dense(),torch.tensor(expected,device=device))
    empty=EmbeddingBagPacket('packed',torch.empty(0,dtype=torch.int64,device=device),torch.tensor([0,0,0],device=device),torch.empty(0,device=device) if weighted else None,True)
    torch.testing.assert_close(layer(empty),torch.zeros(2,2,device=device))

@pytest.mark.parametrize('device',DEVICES)
def test_independent_plain_sgd_step_and_state_reload(device):
    layer=adapter(device,mode='sum',layout='packed',include_last_offset=True)
    out=layer(packet(device));(out*torch.tensor(UPSTREAM,device=device)).sum().backward()
    torch.optim.SGD(layer.parameters(),lr=.1).step()
    expected=[[9.,9.],[.6,2.2],[2.8,-.9],[-1.7,3.5],[2.6,3.4]]
    torch.testing.assert_close(layer.weight,torch.tensor(expected,device=device))
    child=adapter(device,mode='sum',layout='packed',include_last_offset=True);child.load_state_dict(layer.state_dict(),strict=True)
    torch.testing.assert_close(child(packet(device)),layer(packet(device)),rtol=0,atol=0)

@pytest.mark.parametrize('change',[{'layout':'fixed'},{'include_last_offset':False},{'per_sample_weights':True}])
def test_packet_configuration_identity_refuses(change):
    layer=adapter('cpu',mode='sum',layout='packed',include_last_offset=True,**({} if not change else {}))
    layer.config.update(change)
    with pytest.raises(ValueError):layer(packet('cpu'))

@pytest.mark.parametrize('mutation',['badindex','nonfiniteweight','badtable','wrongdtype','nativecontrol'])
def test_invalid_values_or_mutated_table_refuse(mutation):
    layer=adapter('cpu',mode='sum',layout='packed',include_last_offset=True)
    source=packet('cpu')
    if mutation=='badindex':source.indices[0]=5
    elif mutation=='nonfiniteweight':source=replace(source,per_sample_weights=torch.full((7,),float('nan')));layer.config['per_sample_weights']=True
    elif mutation=='badtable':
        with torch.no_grad():layer.weight[1,0]=float('nan')
    elif mutation=='wrongdtype':layer.double()
    else:layer.mode='max'
    with pytest.raises(ValueError):layer(source)


def test_preallocation_parameter_limit_and_device_refusal():
    with patch('torch.nn.EmbeddingBag.__init__',side_effect=AssertionError('must refuse before native allocation')):
        with pytest.raises(ValueError):CheckedEmbeddingBag(config(num_embeddings=2_000_001,embedding_dim=1))
        with pytest.raises(ValueError):CheckedEmbeddingBag(config(),device='cuda')

@pytest.mark.parametrize('device',DEVICES)
def test_frozen_frequency_outputs_and_float_input_gradients(device):
    layer=adapter(device,mode='sum',layout='packed',include_last_offset=True,
                  scale_grad_by_freq=True,per_sample_weights=True,freeze=True)
    source=packet(device,weighted=True)
    output=layer(source)
    torch.testing.assert_close(output,torch.tensor([[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]],device=device))
    (output*torch.tensor(UPSTREAM,device=device)).sum().backward()
    torch.testing.assert_close(source.per_sample_weights.grad,torch.tensor([0.,7.,0.,0.,26.,36.,0.],device=device))
    assert layer.weight.grad is None

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
@pytest.mark.parametrize('changes',[{'max_norm':2.},{'scale_grad_by_freq':True}])
def test_mps_unsupported_controls_refuse_before_allocation(changes):
    with patch('torch.nn.EmbeddingBag.__init__',side_effect=AssertionError('native allocation forbidden')):
        with pytest.raises(ValueError,match='CPU'):
            CheckedEmbeddingBag(config(mode='sum',**changes),device='mps')

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_moved_model_and_changed_trainability_cannot_bypass_mps_refusal():
    for changes in [{'max_norm':2.},{'scale_grad_by_freq':True}]:
        layer=adapter('cpu',mode='sum',layout='packed',include_last_offset=True,**changes).to('mps')
        with pytest.raises(ValueError,match='CPU'):layer(packet('mps'))
    layer=adapter('mps',mode='sum',layout='packed',include_last_offset=True,freeze=True,scale_grad_by_freq=True)
    layer.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='CPU'):layer(packet('mps'))


def test_nondefault_norm_order_and_nonfinite_output_guard():
    layer=adapter('cpu',mode='sum',layout='packed',padding_idx=None,max_norm=2.,norm_type=1.)
    source=EmbeddingBagPacket('packed',torch.tensor([4]),torch.tensor([0]))
    torch.testing.assert_close(layer(source),torch.tensor([[6/(7+1e-7),8/(7+1e-7)]]),rtol=2e-6,atol=2e-6)
    layer=adapter('cpu',mode='sum',layout='packed',include_last_offset=True,per_sample_weights=True)
    source=replace(packet('cpu'),per_sample_weights=torch.full((7,),3e38))
    with pytest.raises(ValueError,match='nonfinite'):layer(source)

@pytest.mark.parametrize('initial_freeze',[False,True])
def test_cpu_separate_trainability_policy_overrides_initial_freeze(initial_freeze):
    layer=adapter('cpu',mode='sum',layout='packed',include_last_offset=True,
                  per_sample_weights=True,freeze=initial_freeze)
    layer.weight.requires_grad_(initial_freeze)
    source=packet('cpu',weighted=True)
    output=layer(source)
    torch.testing.assert_close(output,torch.tensor([[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]]))
    (output*torch.tensor(UPSTREAM)).sum().backward()
    torch.testing.assert_close(source.per_sample_weights.grad,torch.tensor([0.,7.,0.,0.,26.,36.,0.]))
    if initial_freeze:
        torch.testing.assert_close(layer.weight.grad,torch.tensor([[0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]]))
    else:assert layer.weight.grad is None


def test_copied_loose_worker_import_and_real_forward(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    runner=Path(__file__).resolve().parents[1]
    for name in ['embedding_bag.py','embedding_bag_configuration.py','embedding_bag_packet.py']:
        shutil.copyfile(runner/name,tmp_path/name)
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import torch
import embedding_bag
from embedding_bag_configuration import DEFAULTS
from embedding_bag_packet import EmbeddingBagPacket
assert embedding_bag.__package__ == ''
layer=embedding_bag.CheckedEmbeddingBag({**DEFAULTS,'num_embeddings':3,'embedding_dim':2,'mode':'sum'})
with torch.no_grad():layer.weight.copy_(torch.tensor([[0.,0.],[1.,2.],[3.,4.]]))
result=layer(EmbeddingBagPacket('fixed',torch.tensor([[1,2]])))
assert result.tolist()==[[4.,6.]]
assert 'runner.engine' not in sys.modules
"""
    result=subprocess.run([sys.executable,'-I','-c',code,str(tmp_path)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stderr

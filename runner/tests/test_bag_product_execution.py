"""Actual product adapter packet/stage boundaries; fixture registry remains unpublished."""
import copy
from dataclasses import replace

import pytest
import torch

from runner import engine
from runner.product_execution import ProductExecutionAdapter
from runner.embedding_bag_packet import EmbeddingBagPacket
from runner.tests.test_bag_source_graph import registry, graph, packet
from runner.tests.test_learning_evidence import affine_submission

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


def model_for(registry,device='cpu',layout='packed',weighted=True,terminal=True):
    model=engine.GraphModel(graph(layout,weighted,terminal),registry,bundled_dataset=False).to(device)
    with torch.no_grad():model.layers['bag'].weight.copy_(torch.tensor([[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]],device=device))
    return model


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('layout',['fixed','packed'])
@pytest.mark.parametrize('weighted',[False,True])
def test_actual_product_forward_inspection_empty_bags_and_live_item_gradients(registry,device,layout,weighted):
    model=model_for(registry,device,layout,weighted)
    adapter=ProductExecutionAdapter(model);source=packet(device,layout,weighted)
    assert adapter.input_shape is None and adapter.input_dtype is None and adapter.features is None
    contract=adapter.input_contract
    assert contract['kind']=='bagPacket' and contract['schemaVersion']==1 and contract['sourceID']=='source'
    assert contract['configuration']==model.nodes['source']['config']
    assert contract['ports']=={port:{'role':port,'dtype':'float32' if port=='per_sample_weights' else 'int64'} for port in model.layouts['source']}
    assert not any('sampleShape' in value for value in contract['ports'].values())
    assert adapter.validate_input(source) is source and adapter.observation_count(source)==4
    output,observed=adapter.forward(source,inspect=True)
    expected=[[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]] if weighted else [[5.,3.],[0.,0.],[-2.,4.],[3.,4.]]
    torch.testing.assert_close(output,torch.tensor(expected,device=device))
    assert observed['source']['indices'] is source.indices
    if weighted:assert observed['source']['per_sample_weights'] is source.per_sample_weights
    assert adapter.validate_stage(observed['source'],'source',rows=4) is observed['source']
    output.mul(torch.tensor([[2.,-1.],[7.,11.],[-3.,5.],[4.,6.]],device=device)).sum().backward()
    expected_table=[[0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]] if weighted else [[0.,0.],[4.,-2.],[2.,-1.],[-3.,5.],[4.,6.]]
    torch.testing.assert_close(model.layers['bag'].weight.grad,torch.tensor(expected_table,device=device))
    if weighted:
        expected_weights=[0.,7.,0.,0.,26.,36.,0.] if layout=='packed' else [[0.,7.,0.],[0.,0.,0.],[0.,26.,0.],[36.,0.,0.]]
        torch.testing.assert_close(source.per_sample_weights.grad,torch.tensor(expected_weights,device=device))


@pytest.mark.parametrize('mutation',['raw','dtype','offsets','weights','weight-shape','terminal','source-limit','source-config','source-role','source-dtype','source-foreign'])
def test_corrupt_packet_and_captured_contract_refuse_before_native_forward(registry,mutation):
    model=model_for(registry);adapter=ProductExecutionAdapter(model);source=packet('cpu')
    if mutation=='raw':source=source.indices
    elif mutation=='dtype':source=replace(source,indices=source.indices.to(torch.int32))
    elif mutation=='offsets':source=replace(source,offsets=torch.tensor([0,3,2,5,7]))
    elif mutation=='weights':source.per_sample_weights.data[0]=float('nan')
    elif mutation=='weight-shape':source=replace(source,per_sample_weights=source.per_sample_weights[:-1])
    elif mutation=='terminal':source=replace(source,include_last_offset=False)
    elif mutation=='source-limit':source=replace(source,offsets=torch.tensor([0,4,4,5,7]))
    elif mutation=='source-config':model.nodes['source']['config']['maximum_items_per_bag']=4
    elif mutation=='source-role':model.layouts['source']['offsets']=replace(model.layouts['source']['offsets'],role='indices')
    elif mutation=='source-dtype':model.dtypes['source']['offsets']='float32'
    else:model.layouts['source']['offsets']=replace(model.layouts['source']['offsets'],source_id='foreign')
    calls=[]
    handle=model.register_forward_pre_hook(lambda *args:calls.append(True))
    try:
        with pytest.raises(ValueError):adapter.forward(source)
    finally:handle.remove()
    assert calls==[]


@pytest.mark.parametrize('mutation',['missing','extra','geometry','rows','dtype'])
def test_captured_source_named_ports_validate_packet_domain_without_dense_sample_assumptions(registry,mutation):
    adapter=ProductExecutionAdapter(model_for(registry));source=packet('cpu')
    values={'indices':source.indices,'offsets':source.offsets,'per_sample_weights':source.per_sample_weights}
    rows=4
    if mutation=='missing':del values['offsets']
    elif mutation=='extra':values['other']=source.indices
    elif mutation=='geometry':values['offsets']=torch.tensor([0,3,3,5,8])
    elif mutation=='rows':rows=7
    else:values['offsets']=source.offsets.float()
    with pytest.raises(ValueError):adapter.validate_stage(values,'source',rows=rows)


@pytest.mark.parametrize('device',DEVICES)
def test_all_empty_packed_observations_are_not_zero_batch_or_item_count(registry,device):
    adapter=ProductExecutionAdapter(model_for(registry,device,weighted=False))
    source=EmbeddingBagPacket('packed',torch.empty(0,dtype=torch.int64,device=device),torch.tensor([0,0,0,0],device=device),include_last_offset=True)
    assert adapter.observation_count(source)==3
    output,stages=adapter.forward(source,inspect=True)
    torch.testing.assert_close(output,torch.zeros(3,2,device=device))
    assert stages['source']['indices'].numel()==0


def test_explicit_cpu_partition_validation_before_mps_movement(registry):
    if not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    adapter=ProductExecutionAdapter(model_for(registry,'mps'));source=packet('cpu')
    assert adapter.validate_input(source,device='cpu',context='Prepared training') is source
    with pytest.raises(ValueError,match='device'):adapter.forward(source)
    observed={'indices':source.indices,'offsets':source.offsets,'per_sample_weights':source.per_sample_weights}
    assert adapter.validate_stage(observed,'source',rows=4,device='cpu') is observed
    with pytest.raises(ValueError,match='device'):adapter.validate_stage(observed,'source',rows=4)


def test_trusted_path_does_not_skip_finite_packet_weights_and_metadata_is_not_authority(registry):
    model=model_for(registry);adapter=ProductExecutionAdapter(model);source=packet('cpu')
    info=adapter.input_contract;info['configuration']['maximum_items_per_bag']=128
    assert adapter.input_contract['configuration']['maximum_items_per_bag']==3
    source.per_sample_weights.data[0]=float('inf')
    with pytest.raises(ValueError):adapter._forward_trusted(source)


def test_ordinary_product_metadata_and_validation_remain_exact():
    model=engine.GraphModel(affine_submission()['graph']);adapter=ProductExecutionAdapter(model)
    assert adapter.input_contract=={'dtype':'float32','sampleShape':[2]}
    assert adapter.input_shape==(2,) and adapter.input_dtype==torch.float32 and adapter.features==2
    x=torch.tensor([[1.,2.],[3.,4.]])
    assert adapter.observation_count(x)==2
    torch.testing.assert_close(adapter.forward(x),model(x))
    with pytest.raises(ValueError):adapter.validate_input(torch.tensor([[1,2]]))

import pytest
import torch
from runner.product_execution import ProductExecutionAdapter
from runner.tests.test_product_execution import model
from runner.tensor_layout import BatchedLayout


def adapter():
    m=model()
    m.layouts['pool']={'out':BatchedLayout((2,3)),'indices':BatchedLayout((2,3))}
    m.dtypes['pool']={'out':'float32','indices':'int64'}
    return ProductExecutionAdapter(m)


def values():
    return {'out':torch.ones(4,2,3,requires_grad=True),'indices':torch.zeros(4,2,3,dtype=torch.int64)}


def test_validates_each_output_without_copying_or_detaching():
    a=adapter();v=values()
    assert a.validate_stage(v,'pool',rows=4) is v
    v['out'].sum().backward()
    assert torch.equal(v['out'].grad,torch.ones_like(v['out']))


@pytest.mark.parametrize('case',['missing','extra','tuple','float_indices','wrong_shape','wrong_rows','nan','sparse','nested_mapping'])
def test_refuses_partial_mistyped_or_malformed_output_sets(case):
    a=adapter();v=values()
    if case=='missing':del v['indices']
    elif case=='extra':v['other']=v['out']
    elif case=='tuple':v=tuple(v.values())
    elif case=='float_indices':v['indices']=v['indices'].float()
    elif case=='wrong_shape':v['indices']=torch.zeros(4,3,2,dtype=torch.int64)
    elif case=='wrong_rows':v['indices']=torch.zeros(3,2,3,dtype=torch.int64)
    elif case=='nan':v['out']=torch.full((4,2,3),float('nan'))
    elif case=='sparse':v['indices']=v['indices'].to_sparse()
    elif case=='nested_mapping':v['indices']={'nested':v['indices']}
    with pytest.raises(ValueError,match='Captured component pool'):
        a.validate_stage(v,'pool',rows=4)


def test_contract_port_sets_must_agree():
    a=adapter();del a.model.dtypes['pool']['indices']
    with pytest.raises(ValueError,match='contract'):a.validate_stage(values(),'pool')

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset, Subset, TensorDataset
from runner.dataset_sources import make_source
from runner.tests.test_dataset_loader_configuration import recipe

@pytest.mark.parametrize('kind,cls',[('indexed',Dataset),('boundedIterable',IterableDataset),('tensor',TensorDataset),('subset',Subset)])
def test_real_native_sources(kind,cls):
    x=torch.arange(10,dtype=torch.float32).reshape(5,2);y=torch.arange(5)
    source=make_source(x,{'target':y},['a','b','c','d','e'],recipe(kind=kind,indices=[4,1] if kind=='subset' else []))
    assert isinstance(source.native,cls)
    assert source.positions==((4,1) if kind=='subset' else tuple(range(5)))
    if kind!='boundedIterable':
        row=source.native[0];assert row.position==source.positions[0]
        torch.testing.assert_close(row.input,x[row.position:row.position+1])
        assert row.roles['target'].item()==row.position
    else:
        assert [r.position for r in source.native]==list(range(5))
        assert [r.position for r in source.native]==list(range(5))

def test_iterable_limit_is_explicit_and_training_only():
    source=make_source(torch.ones(5,2),{},list(range(5)),recipe(kind='boundedIterable',max_rows=3))
    assert source.positions==(0,1,2)

@pytest.mark.parametrize('bad',['roles','rows','duplicate','subset','finite','empty'])
def test_source_value_admission(bad):
    x=torch.ones(5,2);roles={'target':torch.arange(5)};ids=list(range(5));config=recipe()
    if bad=='roles':roles['target']=torch.arange(4)
    if bad=='rows':ids.pop()
    if bad=='duplicate':ids[1]=ids[0]
    if bad=='subset':config=recipe(kind='subset',indices=[5])
    if bad=='finite':x[0,0]=float('nan')
    if bad=='empty':x=x[:0];roles={};ids=[]
    with pytest.raises(ValueError):make_source(x,roles,ids,config)

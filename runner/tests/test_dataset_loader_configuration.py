import copy
import pytest
from runner.dataset_loader_configuration import validate_recipe, DATASET_DEFAULTS, LOADER_DEFAULTS

def recipe(component='dataset_source', **changes):
    config=copy.deepcopy(DATASET_DEFAULTS if component=='dataset_source' else LOADER_DEFAULTS)
    config.update(changes)
    return dict(schemaVersion=1,configurationVersion=1,componentID=component,config=config)

@pytest.mark.parametrize('kind',['indexed','boundedIterable','tensor','subset'])
def test_modes_and_canonical_copy(kind):
    value=recipe(kind=kind,indices=[4,1] if kind=='subset' else [])
    out=validate_recipe(value);assert out==value and out is not value
    assert out['config'] is not value['config']

@pytest.mark.parametrize('change',[{'kind':'foreign'},{'indices':[0]},{'max_rows':0},{'max_rows':True},{'max_rows':99999},{'extra':1}])
def test_dataset_invalid(change):
    with pytest.raises(ValueError):validate_recipe(recipe(**change))

@pytest.mark.parametrize('indices',[[],[1,1],[-1],[True],[1.0],list(range(100001))])
def test_subset_invalid(indices):
    with pytest.raises(ValueError):validate_recipe(recipe(kind='subset',indices=indices))

@pytest.mark.parametrize('change',[{'batch_size':0},{'batch_size':1025},{'batch_size':True},{'shuffle':1},{'drop_last':None},{'seed':42.0},{'seed':-1},{'seed':2**63},{'num_workers':3},{'num_workers':True},{'prefetch_factor':1},{'timeout':.1},{'collation':'foreign'},{'persistent_workers':True}])
def test_loader_invalid(change):
    with pytest.raises(ValueError):validate_recipe(recipe('data_loader',**change))

@pytest.mark.parametrize('change',[{'prefetch_factor':None,'timeout':1},{'prefetch_factor':3,'timeout':1},{'prefetch_factor':1,'timeout':0},{'prefetch_factor':1,'timeout':float('nan')},{'prefetch_factor':1,'timeout':31}])
def test_worker_conditional_invalid(change):
    with pytest.raises(ValueError):validate_recipe(recipe('data_loader',num_workers=1,**change))

@pytest.mark.parametrize('kind',['auto','fixed_stack','padded_sequence','whole_bags'])
def test_loader_valid(kind):
    value=recipe('data_loader',num_workers=2,prefetch_factor=2,timeout=5,collation=kind)
    assert validate_recipe(value)['config']['timeout']==5.0

@pytest.mark.parametrize('key',['schemaVersion','configurationVersion','componentID','config'])
def test_exact_envelope(key):
    value=recipe();del value[key]
    with pytest.raises(ValueError):validate_recipe(value)

def test_wire_budget_independent_of_row_limit():
    with pytest.raises(ValueError,match='32768'):validate_recipe(recipe(kind='subset',indices=list(range(10000))))

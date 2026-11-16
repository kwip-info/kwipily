import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_output_configuration import validate_model_output_config, check_model_output, MODEL_OUTPUT_CODE_SOURCE

class Tensor:
    def __init__(self, shape, dtype='torch.float32'): self.shape=shape; self.dtype=dtype

CONFIG={'checkExpectedOutput':True,'expectedDType':'float32','expectedSampleRank':'exact','expectedSampleDimensions':[None,3]}

def test_legacy_and_disabled_are_identity():
    value=Tensor((5,2,3))
    assert check_model_output(value,{},'old-output') is value
    assert check_model_output(value,dict(CONFIG,checkExpectedOutput=False),'off') is value

def test_actual_shape_variable_batch_and_exact_rank():
    for batch in (1,7,32):
        value=Tensor((batch,2,3)); assert check_model_output(value,CONFIG,'head') is value
    for shape in ((5,3),(5,2,4),(5,2,3,1)):
        with pytest.raises(ValueError,match='head.*Model Output'):
            check_model_output(Tensor(shape),CONFIG,'head')

def test_any_rank_dtype_only_and_no_mutation():
    config=dict(CONFIG,expectedSampleRank='any',expectedDType='int64')
    value=Tensor((3,9,1,2),'torch.int64'); assert check_model_output(value,config,'head') is value
    with pytest.raises(ValueError,match='expected dtype int64; captured float32'):
        check_model_output(Tensor((3,9)),config,'head')

def test_exact_scalar_sample_and_wildcard_dimension():
    config=dict(CONFIG,expectedSampleDimensions=[])
    assert check_model_output(Tensor((8,)),config,'head').shape==(8,)
    with pytest.raises(ValueError):check_model_output(Tensor(()),config,'head')
    assert check_model_output(Tensor((2,99,3)),CONFIG,'head').shape==(2,99,3)

@pytest.mark.parametrize('change',[{'checkExpectedOutput':1},{'expectedDType':'float16'},{'expectedSampleRank':'guess'},{'expectedSampleDimensions':[True]},{'expectedSampleDimensions':[0]},{'expectedSampleDimensions':[-1]},{'expectedSampleDimensions':[1]*8},{'surprise':1}])
def test_invalid_configuration_refused(change):
    with pytest.raises(ValueError,match='Model Output'):validate_model_output_config(dict(CONFIG,**change))

def test_standalone_code_enforces_same_contract():
    scope={};exec(MODEL_OUTPUT_CODE_SOURCE,scope)
    value=Tensor((2,2,3)); assert scope['check_model_output'](value,CONFIG,'export-head') is value
    with pytest.raises(ValueError,match='export-head.*Model Output'):
        scope['check_model_output'](Tensor((2,2,4)),CONFIG,'export-head')

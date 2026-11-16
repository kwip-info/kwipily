import json
import pytest
from runner.worker_authorization import validate_phase_arguments

POLICY={'schemaVersion':1,'activationValues':False,'predictionRows':False}

def test_training_phase_accepts_explicit_capture_but_preparation_refuses():
    args=['--sample-capture-json',json.dumps(POLICY)]
    assert validate_phase_arguments('training',args)['--sample-capture-json']==json.dumps(POLICY)
    with pytest.raises(ValueError):validate_phase_arguments('preparation',args)

@pytest.mark.parametrize('value',['null','{}',json.dumps({**POLICY,'activationValues':0}), ' '*1025])
def test_phase_refuses_invalid_capture(value):
    with pytest.raises(ValueError):validate_phase_arguments('training',['--sample-capture-json',value])

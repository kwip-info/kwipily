"""Schedulers are bounded, train-only retained authority, never unpinned options."""
import json
import pytest
from runner.tests.test_model_training_operation import training
from runner.tests.test_preview_operation_authorization import preview_operation
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_optimizer_operation_contract import recipe as optimizer
from runner.scheduler_configuration import canonical_configuration

IDS=['scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr']
def recipe(kind=IDS[0]):
    return dict(schemaVersion=1,componentID=kind,configurationVersion=1,config=canonical_configuration(kind,{}))
@pytest.mark.parametrize('kind',IDS)
def test_scheduler_authority_admits_then_rechecks_exact_bytes(training,kind):
    f=training;f[1]['arguments']+=['--optimizer-recipe-json',json.dumps(optimizer()),'--scheduler-recipe-json',json.dumps(recipe(kind))]
    publish(f)
    with claim(f) as authority:
        authority.check()
        f[1]['arguments'][-1]=json.dumps(recipe(IDS[1] if kind!=IDS[1] else IDS[0]));publish(f)
        with pytest.raises(ValueError):authority.check()
@pytest.mark.parametrize('bad',[None,{},dict(schemaVersion=2,componentID=IDS[0],configurationVersion=1,config={}),dict(schemaVersion=1,componentID=IDS[0],configurationVersion=1,config={'step_size':True}),dict(schemaVersion=1,componentID=IDS[0],configurationVersion=1,config={},foreign=True)])
def test_bad_scheduler_refuses_before_claim(training,bad):
    f=training;f[1]['arguments']+=['--optimizer-recipe-json',json.dumps(optimizer()),'--scheduler-recipe-json',json.dumps(bad)];publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()
def test_scheduler_without_explicit_optimizer_refuses(training):
    f=training;f[1]['arguments']+=['--scheduler-recipe-json',json.dumps(recipe())];publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()

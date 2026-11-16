"""An explicit optimizer recipe is part of retained launch authority."""
import json
import pytest
from runner.tests.test_model_training_operation import training
from runner.tests.test_preview_operation_authorization import preview_operation
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.optimizer_configuration import canonical_configuration


def recipe(kind='optimizer_sgd'):
    return dict(schemaVersion=1,componentID=kind,configurationVersion=1,config=canonical_configuration(kind,{'lr':0,'momentum':.8} if kind=='optimizer_sgd' else {'lr':.05}))

@pytest.mark.parametrize('kind',['optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad'])
def test_explicit_recipe_is_admitted_and_authority_rechecks_exact_bytes(training,kind):
    f=training
    text=json.dumps(recipe(kind),sort_keys=True,separators=(',',':'))
    f[1]['arguments']+=['--optimizer-recipe-json',text]
    publish(f)
    with claim(f) as authority:
        authority.check()
        f[1]['arguments'][-1]=json.dumps(recipe('optimizer_adamw'))
        publish(f)
        with pytest.raises(ValueError):authority.check()

@pytest.mark.parametrize('value',[None,{},dict(schemaVersion=2,componentID='optimizer_sgd',configurationVersion=1,config={}),dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config={'lr':True}),dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config={},foreign=True)])
def test_malformed_recipe_refuses_before_claim(training,value):
    f=training;f[1]['arguments']+=['--optimizer-recipe-json',json.dumps(value)];publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()

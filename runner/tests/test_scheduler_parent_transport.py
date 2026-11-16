"""Operation admission pins the optional parent trace before loading any Torch state."""
import hashlib,json
import pytest
from runner.tests.test_model_training_operation import training
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_preview_operation_authorization import preview_operation,inventory
from runner.model_training_operation_contract import verify_inputs
from runner.scheduler_configuration import validate_recipe


def parent_files(f,scheduled=True,trace=True):
    root,value=f[0],f[1];directory=root/'input/continuation-parent';directory.mkdir()
    checkpoint=b'opaque application-owned checkpoint; admission must not deserialize'
    metrics=b'[]'
    recipe=validate_recipe({'schemaVersion':1,'componentID':'scheduler_step_lr','configurationVersion':1,'config':{}})
    rate=json.dumps({'schemaVersion':1,'recipe':recipe,'timing':'afterCompletedEpochValidation','initialization':{'tick':0,'rate':.03},'records':[]}).encode()
    manifest={'settings':{},'checkpoint':{'contract':{}}}
    if scheduled:
        manifest['settings']['schedulerRecipe']=recipe
        manifest['checkpoint']['contract']['scheduler']={'schemaVersion':1,'recipe':recipe}
        manifest['rateTrace']={'schemaVersion':1,'path':'rate-trace.json','bytes':len(rate),'sha256':hashlib.sha256(rate).hexdigest(),'records':0,'timing':'afterCompletedEpochValidation'}
    for name,raw in [('checkpoint.pt',checkpoint),('metrics.json',metrics),('manifest.json',json.dumps(manifest).encode())]:
        (directory/name).write_bytes(raw)
    if trace:(directory/'rate-trace.json').write_bytes(rate)
    value['arguments']+=['--resume-checkpoint',str(directory/'checkpoint.pt')]
    inventory(f)
    return directory


def test_scheduled_parent_four_pinned_files_admitted_before_deserialization(training):
    parent_files(training)
    verify_inputs(training[1],training[0],{})


def test_legacy_parent_retains_exact_three_file_inventory(training):
    parent_files(training,scheduled=False,trace=False)
    verify_inputs(training[1],training[0],{})


@pytest.mark.parametrize('case',['missing','foreign','repinned-trace','repinned-size'])
def test_parent_trace_presence_and_manifest_pin_cannot_be_forged(training,case):
    directory=parent_files(training,scheduled=case!='foreign',trace=case!='missing')
    if case in ('repinned-trace','repinned-size'):
        path=directory/'rate-trace.json'
        raw=path.read_bytes()+b' ' if case=='repinned-size' else path.read_bytes().replace(b'afterCompletedEpochValidation',b'afterOptimizerBatch')
        path.write_bytes(raw)
        item=next(x for x in training[1]['inputFiles'] if x['path'].endswith('/rate-trace.json'))
        item.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises((ValueError,OSError)):
        verify_inputs(training[1],training[0],{})

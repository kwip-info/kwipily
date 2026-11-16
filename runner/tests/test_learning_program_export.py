"""Real isolated executable-program lesson observations, independent affine oracle."""
import copy
import pytest
from runner.learning_program_export import execute
from runner.program_export_configuration import DEFAULTS


def tensor(shape, values): return dict(shape=shape, values=values)
def submission(bounded=False, **changes):
    config = dict(DEFAULTS, **changes)
    if bounded: config.update(batch_mode='bounded', minimum_batch_size=1, maximum_batch_size=4)
    return dict(schemaVersion=1, kind='programExportLearningAction', taskID='program-export-size-range' if bounded else 'program-export-computation', recipe=dict(schemaVersion=1,componentID='program_export',configurationVersion=1,config=config))
def fixture(count=2):
    rows = [value for i in range(count) for value in (i+1, 2-i)]
    return dict(id='owned-private-fixture', inputs={'sourceRows':tensor([count,2],rows),'parameter.weight':tensor([2,2],[2,-1,.5,3]),'parameter.bias':tensor([2],[.25,-.5])})
def measure(s,f,stop=lambda:False):
    evidence=dict(maximumTensorElements=0,fixtures=[])
    execute(s,[f],evidence,stop,lambda x:(x['shape'],x['values']))
    return evidence

@pytest.mark.parametrize('bounded,count',[(False,2),(True,3)])
def test_actual_isolated_program_reload_and_bounds_against_hand_affine(bounded,count):
    observed=measure(submission(bounded),fixture(count))['fixtures'][0]['tensors']
    expected=[]
    for i in range(count):
        x,y=i+1,2-i;expected.extend([2*x-y+.25,.5*x+3*y-.5])
    assert observed['reloadedOutput']==dict(shape=[count,2],values=pytest.approx(expected))
    assert observed['constraintChecks']['values']==[1]*4
    assert observed['artifactChecks']['values']==[1]*3
    assert observed['parentByteDifference']['values']==[0]
    assert set(observed)=={'reloadedOutput','constraintChecks','artifactChecks','parentByteDifference'}

@pytest.mark.parametrize('mutation',[lambda s:s.update(expected={}),lambda s:s['recipe'].update(componentID='checkpoint'),lambda s:s['recipe']['config'].update(strict=1),lambda s:s.update(taskID='foreign')])
def test_injected_answers_foreign_recipe_and_controls_refused_before_export(mutation,monkeypatch):
    s=submission();mutation(s)
    monkeypatch.setattr('runner.learning_program_export.build_package',lambda *a,**k:pytest.fail('exported before admission'))
    with pytest.raises(ValueError):measure(s,fixture())

@pytest.mark.parametrize('role,value',[('sourceRows',float('nan')),('parameter.weight',float('inf')),('parameter.bias',True)])
def test_invalid_private_cells_reject_before_model_allocation(role,value,monkeypatch):
    f=fixture();f['inputs'][role]['values'][-1]=value
    monkeypatch.setattr('torch.tensor',lambda *a,**k:pytest.fail('allocated before fixture validation'))
    with pytest.raises(ValueError):measure(submission(),f)

def test_no_isolation_success_fabricated_when_portable_process_fails(monkeypatch):
    class Failed:
        returncode=1;stdout='';stderr='owned portable refusal'
    monkeypatch.setattr('runner.learning_program_export.subprocess.run',lambda *a,**k:Failed())
    with pytest.raises(ValueError,match='portable|isolated'):measure(submission(),fixture())

def test_stopped_before_execution_and_measured_budget():
    assert not measure(submission(),fixture(),lambda:True)['fixtures']

def test_valid_but_wrong_requested_capture_mode_is_measured_as_incomplete():
    s=submission(True);s['taskID']='program-export-computation'
    observed=measure(s,fixture())['fixtures'][0]['tensors']
    assert observed['constraintChecks']['values'] != [1]*4

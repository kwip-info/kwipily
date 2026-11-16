"""Versioned experiment intents preserve one captured operation allowance."""
from pathlib import Path
import pytest
from runner.tests.test_worker_authorization import launch_fixture,claim,encoded,digest,uid
from runner.tests.test_worker_bootstrap import control


def budget(intent):
    intent.update(schemaVersion=2,executionBudget=dict(schemaVersion=1,requestedSeconds=10,
        maximumRunSeconds=10,source='project'),remainingSeconds=8.5)


def publish(f):
    folder,intent,_,_=f
    raw=encoded(intent)
    (folder/f"launch-{intent['nonce']}.json").write_bytes(raw)
    (folder/f"phase-{intent['runID']}-{intent['phase']}.json").write_bytes(raw)


def test_budgeted_preparation_retains_claim(launch_fixture):
    budget(launch_fixture[1]);publish(launch_fixture)
    with claim(launch_fixture) as authority:
        authority.check();assert authority.intent['remainingSeconds']==8.5


@pytest.mark.parametrize('change',['missing-budget','missing-remaining','schema1-with-budget','schema2-without-budget',
    'zero','bool','nonfinite','exceeds','policy-forged','policy-null','policy-unknown','unknown-intent'])
def test_budgeted_intent_strict_before_claim(launch_fixture,change):
    f=launch_fixture;value=f[1];budget(value)
    if change=='missing-budget':del value['executionBudget']
    elif change=='missing-remaining':del value['remainingSeconds']
    elif change=='schema1-with-budget':value['schemaVersion']=1
    elif change=='schema2-without-budget':del value['executionBudget'];del value['remainingSeconds']
    elif change in ('zero','bool','nonfinite','exceeds'):
        value['remainingSeconds']={'zero':0,'bool':True,'nonfinite':float('inf'),'exceeds':11}[change]
    elif change=='policy-forged':value['executionBudget']['maximumRunSeconds']=9
    elif change=='policy-null':value['executionBudget']['capabilityMaximumSeconds']=None
    elif change=='policy-unknown':value['executionBudget']['other']=1
    else:value['other']=1
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/f"claim-{value['nonce']}.json").exists()


@pytest.mark.parametrize('change',['decrease','equal','increase','policy-change','downgrade','upgrade'])
def test_training_preserves_preparation_budget(launch_fixture,change):
    f=launch_fixture;value=f[1]
    if change!='upgrade':budget(value)
    publish(f)
    with claim(f):pass
    phase=Path(value['arguments'][1]).parent.with_name('training');phase.mkdir()
    (phase/'request.json').write_text('{}')
    value.update(nonce=uid(),phase='training',arguments=['--graph',str(phase/'request.json'),'--output',str(phase/'artifacts')])
    if change=='decrease':value['remainingSeconds']=4.5
    elif change=='increase':value['remainingSeconds']=9
    elif change=='policy-change':value['executionBudget']['source']='application'
    elif change=='downgrade':value['schemaVersion']=1;del value['executionBudget'];del value['remainingSeconds']
    elif change=='upgrade':budget(value)
    publish(f)
    if change in ('decrease','equal'):
        with claim(f) as authority:authority.check()
    else:
        with pytest.raises((ValueError,OSError)):claim(f)
        assert not (f[0]/f"claim-{value['nonce']}.json").exists()


@pytest.mark.parametrize('bounded',[True,False])
def test_bootstrap_passes_captured_remaining_budget(launch_fixture,tmp_path,monkeypatch,bounded):
    from runner import worker_bootstrap,worker_supervisor
    f=launch_fixture
    if bounded:budget(f[1]);publish(f)
    calls=[]
    monkeypatch.setattr(worker_supervisor,'supervise',lambda command,**kwargs:calls.append(kwargs) or 124)
    assert worker_bootstrap.prepare(control(f),worker_bootstrap.RuntimeContext(tmp_path/'entry',tmp_path/'guard',None))==124
    assert calls[0]['max_wall_seconds']==(8.5 if bounded else None)

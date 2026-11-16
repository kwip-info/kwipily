"""Evaluation/export launch allowances are pinned after immutable staging."""
import json
import sys
import pytest
from runner.tests.test_evaluation_authorization import evaluation_authority,encoded,digest
from runner.tests.test_portable_export_worker import export_authority
from runner.tests.test_frozen_evaluation_cli import evaluation_fixture
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries

POLICY=dict(schemaVersion=1,requestedSeconds=10,maximumRunSeconds=10,source='project')


@pytest.fixture(params=['evaluation','export'])
def bounded_authority(request):
    f=request.getfixturevalue(request.param+'_authority')
    f[1].update(schemaVersion=2,executionBudget=dict(POLICY))
    f[2]['requestSHA256']=digest(encoded(f[1]))
    (f[0]/'request.json').write_bytes(encoded(f[1]));(f[0]/'ownership.json').write_bytes(encoded(f[2]))
    launch=dict(schemaVersion=1,requestSHA256=f[2]['requestSHA256'],remainingSeconds=7.5)
    (f[0]/'launch-budget.json').write_bytes(encoded(launch))
    return request.param,f,launch


def claim(fixture,pin=True):
    from runner.evaluation_authorization import claim_evaluation
    from runner.portable_export_authorization import claim_export
    kind,f,launch=fixture
    return (claim_evaluation if kind=='evaluation' else claim_export)(str(f[0]),digest(encoded(f[1])),
        digest(encoded(f[2])),launch_budget_sha256=digest(encoded(launch)) if pin else None)


def control(fixture):
    kind,f,launch=fixture
    return [('--kwip-evaluate-supervise' if kind=='evaluation' else '--kwip-export-supervise'),
        str(f[0]),digest(encoded(f[1])),digest(encoded(f[2])),'--kwip-launch-budget-sha256',digest(encoded(launch))]


def test_pinned_late_budget_bound_into_claim_and_rechecked(bounded_authority):
    _,f,launch=bounded_authority
    with claim(bounded_authority) as authority:
        assert authority.remaining_seconds==7.5
        assert json.loads((f[0]/'claim.json').read_bytes())['launchBudgetSHA256']==digest(encoded(launch))
        (f[0]/'launch-budget.json').write_bytes(encoded({**launch,'remainingSeconds':8}))
        with pytest.raises((ValueError,OSError)):authority.check()


@pytest.mark.parametrize('mutation',['no-pin','missing','wrong-request','extra','zero','bool','exceeds','null','symlink','legacy','policy-forged','request-extra'])
def test_budget_invalid_before_claim(bounded_authority,mutation):
    _,f,launch=bounded_authority;path=f[0]/'launch-budget.json'
    if mutation=='wrong-request':launch['requestSHA256']='0'*64
    elif mutation=='extra':launch['other']=1
    elif mutation in ('zero','bool','exceeds','null'):launch['remainingSeconds']={'zero':0,'bool':True,'exceeds':11,'null':None}[mutation]
    elif mutation=='legacy':f[1]['schemaVersion']=1;del f[1]['executionBudget']
    elif mutation=='policy-forged':f[1]['executionBudget']['maximumRunSeconds']=9
    elif mutation=='request-extra':f[1]['other']=1
    f[2]['requestSHA256']=digest(encoded(f[1]));(f[0]/'request.json').write_bytes(encoded(f[1]));(f[0]/'ownership.json').write_bytes(encoded(f[2]))
    if mutation!='wrong-request':launch['requestSHA256']=f[2]['requestSHA256']
    path.write_bytes(encoded(launch))
    if mutation=='missing':path.unlink()
    elif mutation=='symlink':path.rename(f[0]/'other.json');path.symlink_to('other.json')
    with pytest.raises((ValueError,OSError)):claim(bounded_authority,pin=mutation!='no-pin')
    assert not (f[0]/'claim.json').exists()


def test_bootstrap_forwards_late_budget_to_worker_and_supervisor(bounded_authority,tmp_path,monkeypatch):
    from runner import worker_bootstrap,worker_supervisor
    calls=[]
    monkeypatch.setattr(worker_bootstrap,'_parent_revoked',lambda:False)
    monkeypatch.setattr(worker_supervisor,'supervise',lambda command,**kwargs:calls.append((command,kwargs)) or 124)
    assert worker_bootstrap.prepare(control(bounded_authority),worker_bootstrap.RuntimeContext(tmp_path/'entry',tmp_path/'guard',None))==124
    assert calls[0][1]['max_wall_seconds']==7.5
    assert calls[0][0][-2:]==control(bounded_authority)[-2:]


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
def test_budgeted_authority_real_inherited_reexec(bounded_authority,tmp_path,libraries):
    guard,_=libraries
    code,events,error=run_supervisor(entry(tmp_path,guard),control(bounded_authority))
    assert code==0,(events,error)
    assert events[-1]['event']=='engineMarker'


def test_actual_budgeted_evaluation_manifest_exports(evaluation_fixture,tmp_path):
    from runner.frozen_evaluation_cli import execute_request
    from runner.portable_evaluation_export import build_package
    path,_=evaluation_fixture
    value=json.loads(path.read_bytes());value.update(schemaVersion=2,executionBudget=POLICY)
    path.write_bytes(encoded(value));pin=digest(encoded(value))
    manifest=execute_request(path,pin,lambda _:None)
    assert manifest['requestSHA256']==pin
    result=build_package(path.parent,tmp_path/'package',digest((path.parent/'artifacts/manifest.json').read_bytes()))
    assert result['kind']=='portableFrozenEvaluation'

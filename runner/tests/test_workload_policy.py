import json
from pathlib import Path
import pytest
from runner.workload_policy import WorkloadPolicy, LEGACY, current_policy, workload_scope


def test_shared_spec_matches_native_resource_and_preserves_legacy_limits():
    root=Path(__file__).resolve().parents[2]
    assert (root/'runner/execution-workload-policy.json').read_bytes()==(root/'Sources/KwipMLCore/Resources/execution-workload-policy.json').read_bytes()
    assert LEGACY.limit('scratchBytes')==100_000_000
    assert LEGACY.limit('retainedBytes')==200_000_000
    assert LEGACY.limit('parallelTasks')==1


@pytest.mark.parametrize('preset',['legacy','light','balanced','throughput'])
def test_preset_roundtrip_and_independent_custom_capture(preset):
    policy=WorkloadPolicy(preset)
    assert WorkloadPolicy.from_document(policy.to_document())==policy
    custom=policy.overriding({'scratchBytes':3_000_000_000})
    assert custom.preset=='custom' and custom.limit('scratchBytes')==3_000_000_000
    assert policy.limit('scratchBytes')!=custom.limit('scratchBytes')
    with pytest.raises(TypeError):custom.limits['scratchBytes']=1
    document=custom.to_document();document['limits']['scratchBytes']=1
    assert custom.limit('scratchBytes')==3_000_000_000


def test_scoped_policy_restores_on_failure_without_cross_run_mutation():
    assert current_policy()==LEGACY
    with workload_scope(WorkloadPolicy('balanced')) as outer:
        with pytest.raises(RuntimeError):
            with workload_scope(WorkloadPolicy('light')):
                assert current_policy().preset=='light'
                raise RuntimeError()
        assert current_policy()==outer
    assert current_policy()==LEGACY


@pytest.mark.parametrize('change',['bool-version','unknown-key','missing-key','bool-limit','zero','float','overflow','forged-preset','unknown-preset'])
def test_untrusted_or_forged_capture_is_rejected(change):
    document=WorkloadPolicy('balanced').to_document()
    if change=='bool-version':document['schemaVersion']=True
    elif change=='unknown-key':document['limits']['privatePath']='/private'
    elif change=='missing-key':document['limits'].pop('tableColumns')
    elif change=='bool-limit':document['limits']['tableColumns']=True
    elif change=='zero':document['limits']['tableColumns']=0
    elif change=='float':document['limits']['tableColumns']=128.0
    elif change=='overflow':document['limits']['tableColumns']=2**63
    elif change=='forged-preset':document['limits']['scratchBytes']=1
    elif change=='unknown-preset':document['preset']='auto-unlimited'
    with pytest.raises(ValueError):WorkloadPolicy.from_document(document)


def test_scratch_uses_captured_policy_and_preserves_legacy_refusal():
    from runner.dataflow_disk import ScratchBudget, disk_scope
    with pytest.raises(ValueError):ScratchBudget(100_000_001)
    with workload_scope(WorkloadPolicy('balanced')):
        with disk_scope() as budget:
            budget.resize(0, 300_000_000)
            assert budget.used==300_000_000 and budget.peak==300_000_000
            with pytest.raises(ValueError):budget.resize(0, 2_000_000_000)
            assert budget.used==300_000_000
            budget.resize(300_000_000,0)
            assert budget.used==0 and budget.peak==300_000_000
    assert current_policy()==LEGACY


def test_versioned_resource_envelope_carries_policy_without_changing_legacy():
    from runner.execution_budget import validate_resources
    legacy={'schemaVersion':1,'cpuThreads':1}
    validate_resources(legacy)
    captured={'schemaVersion':2,'cpuThreads':2,'workloadPolicy':WorkloadPolicy('balanced').to_document()}
    validate_resources(captured)
    malformed=json.loads(json.dumps(captured));malformed['workloadPolicy']['limits']['scratchBytes']=1
    with pytest.raises(ValueError):validate_resources(malformed)
    assert legacy=={'schemaVersion':1,'cpuThreads':1}


def test_bootstrap_installs_exact_policy_before_numerical_imports(tmp_path):
    import subprocess,sys
    root=Path(__file__).resolve().parents[2]
    envelope={'executionBudget':{'schemaVersion':2,'requestedSeconds':60,'maximumRunSeconds':60,'source':'application',
        'resources':{'schemaVersion':2,'cpuThreads':2,'workloadPolicy':WorkloadPolicy('balanced').to_document()}}}
    code="""import sys,json
sys.path.insert(0,sys.argv[1])
from runner.worker_resources import apply_verified_resources
from runner.workload_policy import current_policy
from runner.dataflow_disk import ScratchBudget
apply_verified_resources(json.loads(sys.argv[2]))
assert current_policy().preset=='balanced'
assert ScratchBudget().limit==2147483648
apply_verified_resources({})
assert current_policy().preset=='legacy'
assert ScratchBudget().limit==100000000
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(root),json.dumps(envelope)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr


def test_real_archive_binds_explicit_policy_and_portable_plan_restores_it(tmp_path):
    from runner.dataflow_cli import run_dataflow
    from runner.preparation_plan import PreparationPlan, execute_plan
    source=tmp_path/'source.csv';source.write_text('value\n1\n2\n')
    graph=tmp_path/'graph.json';graph.write_text(json.dumps({'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}],'edges':[]}))
    policy=WorkloadPolicy('balanced')
    output=tmp_path/'archive'
    run_dataflow(graph,{'source':str(source)},output,lambda _:None,workload_policy=policy.to_document())
    manifest=json.loads((output/'manifest.json').read_bytes())
    from runner.dataflow_storage import completed_inventory
    verified,_,_=completed_inventory(output/'manifest.json')
    assert verified==manifest
    assert manifest['workloadPolicy']==policy.to_document()
    assert manifest['limits']['maxScratchBytes']==policy.limit('scratchBytes')
    plan=PreparationPlan.from_json((output/'preparation-plan.json').read_text())
    assert plan.to_document()['schemaVersion']==3
    observed=[]
    execute_plan(plan,{'source':str(source)},on_node=lambda *_:observed.append(current_policy().to_document()))
    assert observed==[policy.to_document()]
    assert current_policy()==LEGACY
    changed=plan.to_document();changed['workloadPolicy']['limits']['scratchBytes']=1
    with pytest.raises(ValueError):PreparationPlan.from_json(json.dumps(changed))

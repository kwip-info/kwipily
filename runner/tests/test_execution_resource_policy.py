"""Pinned CPU and sampled-resident-memory policy preserves historical budget semantics."""
import copy
import pytest
from runner.execution_budget import validate_budget,validate_remaining_budget

LEGACY=dict(schemaVersion=1,requestedSeconds=300,maximumRunSeconds=60,source='project',capabilityMaximumSeconds=60)


def policy(**resources):
    return {**LEGACY,'schemaVersion':2,'resources':dict(schemaVersion=1,cpuThreads=4,**resources)}


def test_versioned_resources_preserve_time_and_never_rewrite_legacy():
    before=copy.deepcopy(LEGACY)
    validate_budget(LEGACY)
    assert LEGACY==before and 'resources' not in LEGACY
    for threads in (1,8,64):
        value=policy(maximumResidentBytes=536870912)
        value['resources']['cpuThreads']=threads
        validate_budget(value);validate_remaining_budget(value,12.5)
        assert value['resources']['cpuThreads']==threads
    for limit in (134217728,1099511627776):validate_budget(policy(maximumResidentBytes=limit))
    validate_budget(policy())


@pytest.mark.parametrize('resources',[
    {},dict(schemaVersion=2,cpuThreads=4),dict(schemaVersion=1,cpuThreads=True),
    dict(schemaVersion=1,cpuThreads=1.5),dict(schemaVersion=1,cpuThreads='4'),
    dict(schemaVersion=1,cpuThreads=0),dict(schemaVersion=1,cpuThreads=65),
    dict(schemaVersion=1,cpuThreads=4,maximumResidentBytes=None),
    dict(schemaVersion=1,cpuThreads=4,maximumResidentBytes=134217727),
    dict(schemaVersion=1,cpuThreads=4,maximumResidentBytes=1099511627777),
    dict(schemaVersion=1,cpuThreads=4,gpuMemoryBytes=123),
])
def test_invalid_resource_bounds_types_and_fields_rejected(resources):
    with pytest.raises(ValueError):validate_budget({**LEGACY,'schemaVersion':2,'resources':resources})


@pytest.mark.parametrize('change',[
    dict(schemaVersion=2),dict(resources=dict(schemaVersion=1,cpuThreads=4)),
    dict(schemaVersion=2,resources=None),dict(schemaVersion=3,resources=dict(schemaVersion=1,cpuThreads=4)),
    dict(schemaVersion=2,resources=dict(schemaVersion=1,cpuThreads=4),maximumRunSeconds=61),
])
def test_budget_version_exact_resource_fields(change):
    with pytest.raises(ValueError):validate_budget({**LEGACY,**change})


def captured_policy(**provenance):
    return {**policy(), 'schemaVersion': 3, 'provenance': dict(schemaVersion=1,
        storageScope='workspace', storageSource='application', cpuThreadsSource='project',
        maximumResidentBytesSource='application', **provenance)}


def test_native_storage_provenance_is_retained_and_legacy_is_not_rewritten():
    for value in (captured_policy(), captured_policy(managedStorageLimitBytes=2_000_000_000),
                  captured_policy(managedStorageLimitBytes=2**63-1)):
        before = copy.deepcopy(value)
        validate_budget(value)
        validate_remaining_budget(value, 12)
        assert value == before


@pytest.mark.parametrize('key,value', [('storageScope','project'), ('storageSource','project'),
    ('cpuThreadsSource','unknown'), ('maximumResidentBytesSource',None), ('extra',True),
    ('managedStorageLimitBytes',0), ('managedStorageLimitBytes',True), ('managedStorageLimitBytes',1.5),
    ('managedStorageLimitBytes',None), ('managedStorageLimitBytes',2**63)])
def test_native_storage_provenance_is_strict(key,value):
    budget = captured_policy()
    budget['provenance'][key] = value
    with pytest.raises(ValueError): validate_budget(budget)


def test_captured_workload_policy_is_independent_of_later_preference_documents():
    from runner.workload_policy import WorkloadPolicy
    selected=WorkloadPolicy('balanced').to_document()
    value=policy();value['resources']={'schemaVersion':2,'cpuThreads':4,'workloadPolicy':selected}
    captured=copy.deepcopy(value)
    selected['preset']='custom';selected['limits']['scratchBytes']=1
    validate_budget(captured)
    assert captured['resources']['workloadPolicy']['limits']['scratchBytes']==2147483648

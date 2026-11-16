"""Dependency-free validation shared by retained worker launch envelopes."""
import math

BUDGET_FIELDS = {'schemaVersion','requestedSeconds','maximumRunSeconds','source'}


def _require(condition,message):
    if not condition:raise ValueError(message)


def validate_resources(value):
    _require(type(value) is dict and type(value.get('schemaVersion')) is int
             and value['schemaVersion'] in (1,2), 'Unsupported execution resources')
    required={'schemaVersion','cpuThreads'} | ({'workloadPolicy'} if value['schemaVersion']==2 else set())
    _require(required<=set(value) and set(value)<=required|{'maximumResidentBytes'}, 'Invalid execution resource fields')
    _require(type(value['cpuThreads']) is int and 1<=value['cpuThreads']<=64, 'Invalid CPU thread count')
    if 'maximumResidentBytes' in value:
        _require(type(value['maximumResidentBytes']) is int and 134217728<=value['maximumResidentBytes']<=1099511627776,
                 'Invalid sampled resident-memory threshold')
    if value['schemaVersion']==2:
        if __package__:from .workload_policy import WorkloadPolicy
        else:from workload_policy import WorkloadPolicy
        WorkloadPolicy.from_document(value['workloadPolicy'])


def validate_provenance(value):
    required = {'schemaVersion','storageScope','storageSource','cpuThreadsSource','maximumResidentBytesSource'}
    _require(isinstance(value,dict) and required <= set(value)
             and set(value) <= required | {'managedStorageLimitBytes'}, 'Invalid resource provenance fields')
    _require(type(value['schemaVersion']) is int and value['schemaVersion']==1
             and value['storageScope']=='workspace' and value['storageSource']=='application',
             'Unsupported resource provenance scope')
    _require(value['cpuThreadsSource'] in ('application','project')
             and value['maximumResidentBytesSource'] in ('application','project'), 'Invalid resource provenance source')
    if 'managedStorageLimitBytes' in value:
        _require(type(value['managedStorageLimitBytes']) is int and 0 < value['managedStorageLimitBytes'] <= 2**63-1,
                 'Invalid captured native storage allowance')


def validate_budget(value):
    _require(isinstance(value,dict) and type(value.get('schemaVersion')) is int
            and value['schemaVersion'] in (1,2,3), 'Unsupported execution budget')
    required=BUDGET_FIELDS | ({'resources'} if value['schemaVersion']>=2 else set()) | ({'provenance'} if value['schemaVersion']==3 else set())
    _require(required <= set(value) and set(value) <= required | {'capabilityMaximumSeconds'}, 'Invalid execution budget fields')
    if value['schemaVersion']>=2:validate_resources(value['resources'])
    if value['schemaVersion']==3:validate_provenance(value['provenance'])
    _require(value['source'] in ('application','project'), 'Invalid execution budget source')
    for key in ('requestedSeconds','maximumRunSeconds','capabilityMaximumSeconds'):
        if key in value:
            _require(type(value[key]) is int and 1<=value[key]<=86400, 'Invalid execution budget seconds')
    _require(value['maximumRunSeconds']==min(value['requestedSeconds'],
            value.get('capabilityMaximumSeconds',value['requestedSeconds'])), 'Forged effective execution budget')


def validate_remaining_budget(policy,remaining):
    validate_budget(policy)
    # Compare the bounded range before math.isfinite so arbitrarily large JSON
    # integers reject predictably instead of overflowing a float conversion.
    _require(type(remaining) in (int,float) and 0<remaining<=policy['maximumRunSeconds']
             and math.isfinite(remaining), 'Invalid remaining execution budget')


def validate_versioned_request(value,legacy_fields):
    _require(isinstance(value,dict) and type(value.get('schemaVersion')) is int, 'Invalid budget request schema')
    version=value['schemaVersion']
    _require(version in (1,2) and set(value)==(legacy_fields if version==1 else legacy_fields|{'executionBudget'}),
             'Invalid budget request fields')
    if version==2:validate_budget(value['executionBudget'])

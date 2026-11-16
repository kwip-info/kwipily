"""Process resource policy derived only from a verified launch envelope.

This module imports no numerical runtime. Bootstrap installs thread environment
before importing one; engine adapters then apply the same captured PyTorch limit.
Ordinary CLI/portable execution has the historical single-thread default.
"""
import os
import sys
if __package__:
    from .execution_budget import validate_budget
else:
    from execution_budget import validate_budget

_CPU_THREADS=1
_POLARS_THREADS=1
_PARALLEL_TASKS=None
_RESIDENT_BYTES=None
THREAD_ENVIRONMENT=('OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS','GOTO_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','BLIS_NUM_THREADS',
                    'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','POLARS_MAX_THREADS')


def _resources(envelope):
    policy=envelope.get('executionBudget')
    if policy is None:return {}
    validate_budget(policy)
    return policy.get('resources',{})


def resident_limit(envelope):
    return _resources(envelope).get('maximumResidentBytes')


def apply_verified_resources(envelope):
    global _CPU_THREADS,_POLARS_THREADS,_PARALLEL_TASKS,_RESIDENT_BYTES
    resources=_resources(envelope)
    _RESIDENT_BYTES=resources.get('maximumResidentBytes')
    count=resources.get('cpuThreads',1)
    # Polars owns one process-wide native pool initialized on first import/use.
    # An environment change afterward cannot enforce a captured launch budget.
    if 'polars' in sys.modules or any(name.startswith('_polars_runtime') for name in sys.modules):
        raise ValueError('Install captured CPU limits before importing the Polars native runtime.')
    # Measured lessons intentionally retain their stricter reproducible recipe.
    if envelope.get('toolMode') in ('learningRuntime','learningAssessment'):count=1
    if __package__:from .workload_policy import WorkloadPolicy,LEGACY,install_verified_policy
    else:from workload_policy import WorkloadPolicy,LEGACY,install_verified_policy
    workload=resources.get('workloadPolicy')
    policy=WorkloadPolicy.from_document(workload) if workload is not None else LEGACY
    install_verified_policy(policy)
    _CPU_THREADS=count
    _PARALLEL_TASKS=min(count,policy.limit('parallelTasks'))
    # With a synchronous native call, at least one task controller waits for the
    # shared pool. Reserve the other controllers inside the same compute budget;
    # hashing/SQLite may release the GIL and compute outside the native pool.
    _POLARS_THREADS=max(1,count-_PARALLEL_TASKS+1)
    for key in THREAD_ENVIRONMENT:os.environ[key]=str(_POLARS_THREADS if key=='POLARS_MAX_THREADS' else count)
    os.environ['OMP_DYNAMIC']='FALSE'
    os.environ['MKL_DYNAMIC']='FALSE'


def captured_resident_limit():return _RESIDENT_BYTES


def cpu_threads():
    return _CPU_THREADS


def polars_threads():return _POLARS_THREADS
def parallel_tasks():return _PARALLEL_TASKS


def configure_torch(torch_module):
    torch_module.set_num_threads(cpu_threads())

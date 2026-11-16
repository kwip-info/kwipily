"""Captured thread limits precede imports and remain part of exact continuation."""
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish
from runner.tests.test_preview_operation_authorization import preview_operation,inventory
from runner.tests.test_model_training_operation import training
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


def budget(threads):
    return dict(schemaVersion=2,requestedSeconds=30,maximumRunSeconds=30,source='application',resources=dict(schemaVersion=1,cpuThreads=threads))


@pytest.mark.parametrize('threads',[1,2,8])
def test_parallel_controller_slots_and_shared_native_pool_fit_one_capture(threads):
    from runner.workload_policy import WorkloadPolicy
    envelope={'executionBudget':budget(threads)}
    envelope['executionBudget']['resources']={'schemaVersion':2,'cpuThreads':threads,'workloadPolicy':WorkloadPolicy('balanced').to_document()}
    script='''import sys,json
sys.path.insert(0,sys.argv[1])
from worker_resources import apply_verified_resources,cpu_threads,polars_threads
apply_verified_resources(json.loads(sys.argv[2]))
from preparation_scheduler import maximum_native_tasks
import polars
tasks=maximum_native_tasks();pool=polars.thread_pool_size()
assert tasks==min(cpu_threads(),4)
assert pool==polars_threads()==cpu_threads()-tasks+1
assert pool+tasks-1<=cpu_threads()
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[1]),json.dumps(envelope)],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr[-2000:]


def test_captured_workload_cannot_be_increased_by_a_nested_scope():
    script='''import sys
sys.path.insert(0,sys.argv[1])
from worker_resources import apply_verified_resources
from workload_policy import WorkloadPolicy,workload_scope,current_policy
p=WorkloadPolicy('balanced')
apply_verified_resources({'executionBudget':{'schemaVersion':2,'requestedSeconds':30,'maximumRunSeconds':30,'source':'application','resources':{'schemaVersion':2,'cpuThreads':8,'workloadPolicy':p.to_document()}}})
try:
 with workload_scope(WorkloadPolicy('throughput')):pass
except ValueError:pass
else:raise AssertionError('Nested scope exceeded captured admission')
with workload_scope(WorkloadPolicy('light')):assert current_policy().preset=='light'
assert current_policy()==p
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[1])],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr[-2000:]


def test_resource_helper_overrides_host_environment_before_numerical_imports(tmp_path):
    script='''import sys,os,json
sys.path.insert(0,sys.argv[1])
from worker_resources import apply_verified_resources,cpu_threads
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
apply_verified_resources(json.loads(sys.argv[2]))
assert cpu_threads()==2
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):assert os.environ[key]=='2'
assert os.environ['OMP_DYNAMIC']=='FALSE' and os.environ['MKL_DYNAMIC']=='FALSE'
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[1]),json.dumps({'executionBudget':budget(2)})],env={**os.environ,'OMP_NUM_THREADS':'64','OPENBLAS_NUM_THREADS':'64'},capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('mode',[None,'learningRuntime','learningAssessment'])
def test_default_or_learning_resources_remain_one_in_fresh_process(mode):
    envelope={} if mode is None else dict(toolMode=mode,executionBudget=budget(8))
    script="import sys,json;sys.path.insert(0,sys.argv[1]);from worker_resources import apply_verified_resources,cpu_threads;apply_verified_resources(json.loads(sys.argv[2]));assert cpu_threads()==1"
    result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[1]),json.dumps(envelope)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr


def guarded(f,tmp_path,guard,threads,tail=''):
    if threads is not None:f[1]['executionBudget']=budget(threads)
    f[1]['remainingSeconds']=8.5;publish(f)
    tmp_path.mkdir(parents=True,exist_ok=True);path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    replacement='    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    status=main(check_cancelled=result.authority.check)\n    import torch,os\n    print(json.dumps({"event":"threadEvidence","actual":torch.get_num_threads(),"omp":os.environ.get("OMP_NUM_THREADS")}),flush=True)\n'+tail+'    sys.exit(status)'
    text=path.read_text();assert marker in text;path.write_text(text.replace(marker,replacement))
    return run_supervisor(path,control(f))


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('kind',['preview','training-legacy'])
def test_guarded_preview_threads_and_legacy_training_contract(preview_operation,tmp_path,libraries,kind):
    f=preview_operation;threads=2
    if kind=='training-legacy':
        f[1]['toolMode']='modelTraining';f[1]['arguments']=f[1]['arguments'][:-3]+['--epochs','1','--batch-size','32','--learning-rate','0.03'];threads=None
    code,events,error=guarded(f,tmp_path/'worker',libraries[0],threads)
    assert code==0,(events,error)
    assert events[-1]==dict(event='threadEvidence',actual=threads or 1,omp=str(threads or 1))
    if kind=='training-legacy':
        manifest=json.loads((f[0]/'artifacts/manifest.json').read_bytes())
        assert manifest['checkpoint']['contract']['schedule']['numThreads']==1


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('mode,threads,success',[('resume',2,True),('resume',1,False),('finetune',1,True)])
def test_guarded_training_captured_threads_control_checkpoint_continuation(training,tmp_path,libraries,mode,threads,success):
    first=training;guard=libraries[0]
    code,events,error=guarded(first,tmp_path/'first-worker',guard,2)
    assert code==0,(events,error)
    assert events[-1]['actual']==2
    parent=first[0]/'artifacts';manifest=json.loads((parent/'manifest.json').read_bytes())
    assert manifest['checkpoint']['contract']['schedule']['numThreads']==2
    generator=operation_authority.__wrapped__(tmp_path/'child')
    second=next(generator)
    try:
        value=second[1]
        for key in ('runID','directoryID','phase'):del value[key]
        folder=second[0]/'input';folder.mkdir()
        shutil.copyfile(first[0]/'input/graph.json',folder/'graph.json')
        continuation=folder/'continuation-parent';continuation.mkdir()
        for name in ('manifest.json','metrics.json','checkpoint.pt'):shutil.copyfile(parent/name,continuation/name)
        value.update(schemaVersion=2,toolMode='modelTraining',arguments=[arg.replace(str(first[0]),str(second[0])) for arg in first[1]['arguments']])
        value['arguments'][value['arguments'].index('--epochs')+1]='2'
        value['arguments']+=['--resume-checkpoint' if mode=='resume' else '--finetune-checkpoint',str(continuation/'checkpoint.pt')]
        inventory(second)
        code,events,error=guarded(second,tmp_path/'second-worker',guard,threads)
        assert (code==0)==success,(events,error)
        completed=second[0]/'artifacts/manifest.json'
        assert completed.exists()==success
        if success:
            result=json.loads(completed.read_bytes())
            assert result['checkpoint']['contract']['schedule']['numThreads']==threads
            assert result['lineage']['mode']==('exactResume' if mode=='resume' else 'weightsOnlyFineTune')
        else:assert any('matching runtime' in event.get('message','') for event in events)
    finally:
        try:next(generator)
        except StopIteration:pass


@pytest.mark.parametrize('mode,module,claim_name',[('--kwip-operation-supervise','operation_authorization','claim_operation'),('--kwip-evaluate-supervise','evaluation_authorization','claim_evaluation'),('--kwip-export-supervise','portable_export_authorization','claim_export'),('--kwip-supervise','worker_authorization','claim_launch')])
@pytest.mark.parametrize('bounded',[False,True])
def test_all_supervisor_routes_receive_only_captured_resident_limit(tmp_path,monkeypatch,mode,module,claim_name,bounded):
    import importlib
    from types import SimpleNamespace
    from runner import worker_bootstrap,worker_supervisor
    policy=budget(2)
    if bounded:policy['resources']['maximumResidentBytes']=268435456
    envelope=dict(arguments=['--capabilities'],executionBudget=policy,remainingSeconds=7.5)
    class Authority:
        request=envelope;intent=envelope;activity_fd=99;remaining_seconds=7.5
        def __enter__(self):return self
        def __exit__(self,*_):pass
    monkeypatch.setattr(importlib.import_module('runner.'+module),claim_name,lambda *a,**k:Authority())
    monkeypatch.setattr(worker_bootstrap,'_parent_revoked',lambda:False)
    calls=[]
    monkeypatch.setattr(worker_supervisor,'supervise',lambda *a,**k:calls.append(k) or 0)
    args=[mode,str(tmp_path),'a'*64,'b'*64]
    if mode=='--kwip-supervise':args.insert(2,'nonce')
    assert worker_bootstrap.prepare(args,worker_bootstrap.RuntimeContext(tmp_path/'entry',tmp_path/'guard',None))==0
    assert len(calls)==1 and calls[0]['max_resident_bytes']==(268435456 if bounded else None)


@pytest.mark.parametrize('threads',[1,2,4])
def test_polars_native_pool_honors_verified_budget_before_first_import(threads):
    script='''import sys,os,json
sys.path.insert(0,sys.argv[1])
from worker_resources import apply_verified_resources,cpu_threads
assert 'polars' not in sys.modules
apply_verified_resources(json.loads(sys.argv[2]))
import polars as pl
count=cpu_threads()
assert os.environ['POLARS_MAX_THREADS']==str(count)
assert pl.thread_pool_size()==count,(pl.thread_pool_size(),count)
frame=pl.DataFrame({'x':range(10000)}).lazy().select(pl.col('x').sum()).collect(engine='streaming')
assert frame.item()==49995000
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[1]),json.dumps({'executionBudget':budget(threads)})],
        env={**os.environ,'POLARS_MAX_THREADS':'64'},capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr


def test_polars_policy_refuses_late_import_or_invalid_budget_without_environment_mutation():
    script='''import sys,os,json
sys.path.insert(0,sys.argv[1])
from worker_resources import apply_verified_resources
os.environ['POLARS_MAX_THREADS']='7'
try:apply_verified_resources({'executionBudget':json.loads(sys.argv[2])})
except ValueError:pass
else:raise AssertionError('Invalid captured policy accepted')
assert os.environ['POLARS_MAX_THREADS']=='7'
import polars
assert polars.thread_pool_size()==7
try:apply_verified_resources({'executionBudget':json.loads(sys.argv[3])})
except ValueError:pass
else:raise AssertionError('Policy applied after native pool initialization')
assert os.environ['POLARS_MAX_THREADS']=='7'
'''
    bad=budget(65)
    result=subprocess.run([sys.executable,'-I','-c',script,str(Path(__file__).resolve().parents[1]),json.dumps(bad),json.dumps(budget(2))],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr

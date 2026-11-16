"""Bounded representative-operation observations; independent clocks/RSS and actual runs."""
import hashlib
import json
import pytest
import torch
from runner.capture_policy import validate_training_sample_capture_policy, resolve_training_sample_capture_policy
from runner.engine import run, GraphModel
from runner.tests.test_runner import graph

POLICY = {'schemaVersion': 2, 'activationValues': False, 'predictionRows': False, 'performanceMeasurements': True}

def collector(nodes=('a', 'b'), device='cpu', **kwargs):
    from runner.performance_capture import PerformanceCapture
    return PerformanceCapture(nodes, device, **kwargs)

def test_policy_preserves_v1_and_absence_while_v2_is_strict():
    old = {'schemaVersion': 1, 'activationValues': False, 'predictionRows': True}
    assert validate_training_sample_capture_policy(old) == old
    assert resolve_training_sample_capture_policy() == {'schemaVersion': 1, 'activationValues': True, 'predictionRows': True}
    assert validate_training_sample_capture_policy(POLICY) == POLICY
    for bad in (None, {**POLICY, 'extra': 1}, {**POLICY, 'performanceMeasurements': 1}, {k:v for k,v in POLICY.items() if k != 'performanceMeasurements'}, {**POLICY, 'schemaVersion': True}, {**old, 'performanceMeasurements': True}):
        with pytest.raises(ValueError): validate_training_sample_capture_policy(bad)

def test_independent_clock_and_resident_samples_have_exact_units_and_order():
    clock = iter((100, 150, 200, 280)); rss = iter((1024, 2048, 4096, 3072))
    value = collector(clock=lambda: next(clock), memory_reader=lambda: next(rss))
    for node in ('a', 'b'): value.observe('started', node); value.observe('completed', node)
    artifact = value.artifact('run', 'a'*64, POLICY)
    assert artifact['records'] == [
        dict(nodeID='a', durationNanoseconds=50, timingUnavailableReason=None, residentBytesBefore=1024, residentBytesAfter=2048, memoryUnavailableReason=None),
        dict(nodeID='b', durationNanoseconds=80, timingUnavailableReason=None, residentBytesBefore=4096, residentBytesAfter=3072, memoryUnavailableReason=None)]
    assert artifact['timingMethod'] == 'hostWall'
    assert artifact['memoryMethod'] == 'processResidentBytes'
    assert artifact['maximumRows'] == 8
    assert len(json.dumps(artifact).encode()) <= 131072

def test_synchronized_device_wall_and_failure_are_explicit():
    calls=[]; ticks=iter((50, 75))
    value=collector(nodes=('a',),device='mps',clock=lambda: next(ticks), memory_reader=lambda: 500, synchronize=lambda: calls.append('sync'))
    value.observe('started','a');value.observe('completed','a')
    assert calls == ['sync','sync']
    assert value.artifact('run','a'*64,POLICY)['records'][0]['durationNanoseconds'] == 25
    def fail(): raise RuntimeError('private driver text')
    value=collector(nodes=('a',),device='mps',memory_reader=fail,synchronize=fail)
    value.observe('started','a');value.observe('completed','a')
    record=value.artifact('run','a'*64,POLICY)['records'][0]
    assert record['durationNanoseconds'] is None
    assert record['timingUnavailableReason'] == 'deviceSynchronizationUnavailable'
    assert record['residentBytesBefore'] is None and record['residentBytesAfter'] is None
    assert record['memoryUnavailableReason'] == 'residentMemoryUnavailable'
    assert 'private' not in json.dumps(record)

@pytest.mark.parametrize('ticks', [(10,9),(0,86400*10**9+1),(True,100)])
def test_invalid_clock_never_becomes_measured_duration(ticks):
    iterator=iter(ticks)
    value=collector(nodes=('a',),clock=lambda:next(iterator),memory_reader=lambda:10)
    value.observe('started','a');value.observe('completed','a')
    record=value.artifact('run','a'*64,POLICY)['records'][0]
    assert record['durationNanoseconds'] is None and record['timingUnavailableReason']=='clockUnavailable'

def test_partial_duplicate_and_out_of_order_cannot_publish():
    value=collector(); value.observe('started','a')
    with pytest.raises(ValueError): value.artifact('run','a'*64,POLICY)
    with pytest.raises(ValueError): value.observe('completed','b')
    value.observe('completed','a')
    with pytest.raises(ValueError): value.observe('started','a')
    with pytest.raises(ValueError): collector(nodes=tuple(str(i) for i in range(129)))

def test_actual_run_capture_preserves_training_state_and_saved_values(tmp_path):
    outputs=[]
    for enabled in (False, True):
        path=tmp_path/str(enabled);events=[]
        result=run(graph(),path,epochs=2,seed=13,device='cpu',sample_limit=40,emit=events.append,sample_capture={**POLICY,'performanceMeasurements':enabled})
        outputs.append((path,result,events))
    old,new=outputs
    assert json.loads((old[0]/'metrics.json').read_text()) == json.loads((new[0]/'metrics.json').read_text())
    a=torch.load(old[0]/'checkpoint.pt',weights_only=False); b=torch.load(new[0]/'checkpoint.pt',weights_only=False)
    assert all(torch.equal(a['modelState'][key],b['modelState'][key]) for key in a['modelState'])
    assert torch.equal(a['rngState'],b['rngState'])
    assert not (old[0]/'performance.json').exists() and 'performanceCapture' not in old[1]
    raw=(new[0]/'performance.json').read_bytes(); artifact=json.loads(raw)
    manifest=json.loads((new[0]/'manifest.json').read_text())
    expected=dict(schemaVersion=1,file='performance.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert new[1]['performanceCapture'] == manifest['performanceCapture'] == expected
    assert artifact['runID'] == new[1]['runID'] and artifact['graphHash']==manifest['graphHash']
    assert artifact['policy'] == POLICY and artifact['device']=='cpu'
    assert [r['nodeID'] for r in artifact['records']] == GraphModel(graph()).order
    assert all(type(r['durationNanoseconds']) is int and r['durationNanoseconds']>=0 for r in artifact['records'])
    capture=lambda events:[{k:v for k,v in e.items() if k!='runID'} for e in events if e['event'].startswith('model')]
    assert capture(old[2]) == capture(new[2])
    assert not any('performance' in e['event'].lower() for e in new[2])
    assert all(not set(r)&{'values','pid','path','rows'} for r in artifact['records'])

def test_invalid_policy_is_rejected_before_any_output(tmp_path):
    output=tmp_path/'invalid'
    with pytest.raises(ValueError): run(graph(),output,epochs=1,device='cpu',sample_capture={**POLICY,'performanceMeasurements':'yes'})
    assert not output.exists()

def test_cancelled_capture_never_publishes_completion_or_profile(tmp_path):
    events=[]; cancelled=False
    def emit(event):
        nonlocal cancelled
        events.append(event)
        if event['event']=='modelNodeStarted': cancelled=True
    def check():
        if cancelled: raise KeyboardInterrupt('cancelled')
    output=tmp_path/'cancelled'
    with pytest.raises(KeyboardInterrupt): run(graph(),output,epochs=1,device='cpu',sample_limit=40,sample_capture=POLICY,emit=emit,check_cancelled=check)
    assert not (output/'performance.json').exists()
    assert not (output/'manifest.json').exists()
    assert not any(e['event']=='completed' for e in events)

def test_original_operation_error_survives_telemetry_failure(tmp_path,monkeypatch):
    from runner import performance_capture
    original=GraphModel.execute_node
    def execute(self,node,values,x):
        if getattr(self,'_capture_operation_observer',None) is not None: raise RuntimeError('original operation error')
        return original(self,node,values,x)
    # Fail only when final capture has started; ordinary training remains real.
    active=False
    def emit(event):
        nonlocal active
        if event['event']=='modelCaptureStarted': active=True
    def execute(self,node,values,x):
        if active: raise RuntimeError('original operation error')
        return original(self,node,values,x)
    monkeypatch.setattr(GraphModel,'execute_node',execute)
    monkeypatch.setattr(performance_capture,'resident_bytes',lambda: (_ for _ in ()).throw(RuntimeError('telemetry detail')))
    output=tmp_path/'failure'
    with pytest.raises(RuntimeError,match='original operation error'):run(graph(),output,epochs=1,device='cpu',sample_limit=40,sample_capture=POLICY,emit=emit)
    assert not (output/'performance.json').exists() and not (output/'manifest.json').exists()

def test_actual_nested_container_capture_follows_dependency_valid_native_schedule(tmp_path):
    from runner.tests.test_module_composition_execution import fixture
    nested=fixture(nested=True)
    output=tmp_path/'nested'; events=[]
    run(nested,output,epochs=1,device='cpu',sample_limit=40,sample_capture=POLICY,emit=events.append)
    records=json.loads((output/'performance.json').read_text())['records']
    actual=[r['nodeID'] for r in records]
    assert actual == ['input','la','lb','ra','rb','sum','output']
    assert actual != GraphModel(nested,bundled_dataset=False).order
    assert actual == [e['nodeID'] for e in events if e['event']=='modelNodeCompleted']
    completed=set()
    for node in actual:
        assert all(edge['source'] in completed for edge in nested['edges'] if edge['target']==node)
        completed.add(node)
    assert completed == {n['id'] for n in nested['nodes']}


def test_dependency_valid_alternative_order_is_accepted_but_early_consumer_is_not():
    deps={'input':set(),'left':{'input'},'right':{'input'},'output':{'left','right'}}
    value=collector(nodes=tuple(deps),dependencies=deps,clock=lambda:100,memory_reader=lambda:10)
    for node in ('input','right','left','output'):
        value.observe('started',node);value.observe('completed',node)
    assert [r['nodeID'] for r in value.artifact('run','a'*64,POLICY)['records']] == ['input','right','left','output']
    value=collector(nodes=tuple(deps),dependencies=deps)
    with pytest.raises(ValueError): value.observe('started','output')
    with pytest.raises(ValueError): value.observe('started','foreign')

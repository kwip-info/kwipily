"""Standalone model training retains every selected input through completion."""
import json
import os
from pathlib import Path
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_preview_operation_authorization import preview_operation,source,inventory
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


@pytest.fixture
def training(preview_operation):
    f=preview_operation;v=f[1];v['toolMode']='modelTraining'
    # Positive cold-interpreter training/export is not the short deadline fixture.
    # Operation-level deadline/cancellation tests retain their own small budgets.
    v.update(executionBudget=dict(schemaVersion=1,requestedSeconds=30,
        maximumRunSeconds=30,source='application'),remainingSeconds=30)
    v['arguments']=v['arguments'][:-3]+['--epochs','1','--batch-size','32','--learning-rate','0.03']
    publish(f);return f


@pytest.mark.parametrize('kind',['bundled','csv','prepared'])
def test_training_captures_data_and_rechecks_pins(training,tmp_path,kind):
    f=training;source(f,kind,tmp_path);inventory(f)
    f[1].update(executionBudget=dict(schemaVersion=1,requestedSeconds=86400,maximumRunSeconds=86400,source='application'),remainingSeconds=86400);publish(f)
    with claim(f) as authority:
        target=next((f[0]/item['path'] for item in f[1]['inputFiles'] if item['path']!='input/graph.json'),f[0]/'input/graph.json')
        before=target.stat();raw=target.read_bytes();target.write_bytes(b'x'+raw[1:]);os.utime(target,ns=(before.st_atime_ns,before.st_mtime_ns))
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['epochs','batch','rate','seed','device','output','registry','preview','extra-input','orphan-gradient','orphan-composite','missing-parent','mixed-parent'])
def test_invalid_training_never_claims(training,change):
    f=training;args=f[1]['arguments']
    settings={'epochs':('--epochs','0'),'batch':('--batch-size','100001'),'rate':('--learning-rate','nan'),'seed':('--seed','-1'),'device':('--device','cuda'),'output':('--output','/tmp/out')}
    if change in settings:
        key,value=settings[change];args[args.index(key)+1]=value
    elif change=='registry':args+=['--registry','/tmp/registry']
    elif change=='preview':args+=['--preview','--stop-at','output']
    elif change=='extra-input':f[1]['inputFiles'].append(dict(path='input/extra',bytes=0,sha256='0'*64))
    elif change=='orphan-gradient':args+=['--gradient-max-captures','1']
    elif change=='orphan-composite':args+=['--composite-source',str(f[0]/'input/composite/source.json')]
    elif change=='missing-parent':args+=['--resume-checkpoint',str(f[0]/'input/continuation-parent/checkpoint.pt')]
    else:args+=['--resume-checkpoint',str(f[0]/'input/continuation-parent/checkpoint.pt'),'--finetune-checkpoint',str(f[0]/'input/continuation-parent/checkpoint.pt')]
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('kind',['bundled','csv','prepared'])
def test_actual_guarded_training_publishes_owned_checkpoint_and_export(training,tmp_path,libraries,kind):
    f=training;source(f,kind,tmp_path);inventory(f)
    guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    path.write_text(text.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
    code,events,error=run_supervisor(path,control(f),timeout=40)
    assert code==0,(events,error)
    assert events[0]['toolMode']=='modelTraining' and events[-1]['event']=='completed'
    output=f[0]/'artifacts';assert events[-1]['outputDir']==str(output)
    manifest=json.loads((output/'manifest.json').read_bytes())
    assert manifest['status']=='completed' and manifest['checkpoint']['completedEpoch']==1
    assert (output/'checkpoint.pt').stat().st_size>0 and (output/'model.py').stat().st_size>0
    if kind=='csv':assert manifest['provenance']['sourceName']=='original-data.csv'
    if kind=='prepared':assert manifest['settings']['partitionMode']=='prepared'

@pytest.mark.parametrize('policy',[{'schemaVersion':1,'activationValues':False,'predictionRows':False},None])
def test_training_capture_policy_is_validated_before_claim(training,policy):
    f=training;f[1]['arguments']+=['--sample-capture-json',json.dumps(policy)];publish(f)
    if policy is None:
        with pytest.raises(ValueError):claim(f)
        assert not (f[0]/'claim.json').exists()
    else:
        with claim(f) as authority:authority.check()

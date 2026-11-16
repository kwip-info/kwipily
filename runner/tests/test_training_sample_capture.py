"""Optional observations must not change training or leak into emitted samples."""
import json
import pytest
from runner.engine import run
from runner.tests.test_runner import graph
from runner.tests.test_checkpoint_lifecycle import state, assert_nested_equal

@pytest.mark.parametrize('activation_values,prediction_rows', [(False,False),(False,True),(True,False)])
def test_optional_samples_preserve_metrics_and_state(tmp_path,activation_values,prediction_rows):
    policy={'schemaVersion':1,'activationValues':activation_values,'predictionRows':prediction_rows}
    baseline=[]; events=[]
    run(graph(),tmp_path/'baseline',epochs=1,device='cpu',emit=baseline.append)
    run(graph(),tmp_path/'private',epochs=1,device='cpu',emit=events.append,sample_capture=policy)
    assert_nested_equal(state(tmp_path/'baseline'),state(tmp_path/'private'))
    assert (tmp_path/'baseline/metrics.json').read_bytes()==(tmp_path/'private/metrics.json').read_bytes()
    result=events[-1];original=baseline[-1]
    assert result['accuracy']==original['accuracy'] and result['evaluationCount']==original['evaluationCount']
    assert result['sampleCapture']==policy
    assert result['predictionRows']==(original['predictionRows'] if prediction_rows else [])
    assert result['predictions']==(original['predictions'] if prediction_rows else [])
    for key,preview in result['previews'].items():
        assert preview['shape']==original['previews'][key]['shape']
        assert preview['dtype']==original['previews'][key]['dtype']
        assert preview['values']==(original['previews'][key]['values'] if activation_values else [])
        assert preview['captureStatus']==('captured' if activation_values else 'omittedByPolicy')
    artifact=json.loads((tmp_path/'private/predictions.json').read_text())
    assert artifact['schemaVersion']==2
    assert artifact['captureStatus']==('captured' if prediction_rows else 'omittedByPolicy')
    assert artifact['predictionRows']==result['predictionRows']
    manifest=json.loads((tmp_path/'private/manifest.json').read_text())
    assert manifest['sampleCapture']==policy
    assert manifest['predictionPreview']['schemaVersion']==2
    # No earlier event may contain either optional projection.
    assert all(not any(key in event for key in ('previews','predictions','predictionRows')) for event in events[:-1])

def test_invalid_sample_policy_refuses_before_writes(tmp_path):
    with pytest.raises(ValueError):
        run(graph(),tmp_path/'invalid',epochs=1,sample_capture={'schemaVersion':1,'activationValues':0,'predictionRows':True})
    assert not (tmp_path/'invalid').exists()

def test_cli_pins_policy_and_refuses_unrelated_or_null_capture(tmp_path):
    import subprocess
    import sys
    source=tmp_path/'graph.json';source.write_text(json.dumps(graph()))
    policy={'schemaVersion':1,'activationValues':False,'predictionRows':False}
    base=[sys.executable,'runner/main.py']
    result=subprocess.run(base+['--graph',str(source),'--output',str(tmp_path/'cli'),'--epochs','1','--device','cpu','--sample-capture-json',json.dumps(policy)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout
    completed=json.loads(result.stdout.splitlines()[-1])
    assert completed['sampleCapture']==policy and completed['predictionRows']==[]
    assert all(p['values']==[] for p in completed['previews'].values())
    for extra in (['--capabilities','--sample-capture-json',json.dumps(policy)],['--graph',str(source),'--output',str(tmp_path/'bad'),'--sample-capture-json','null']):
        failure=subprocess.run(base+extra,capture_output=True,text=True)
        assert failure.returncode==1 and json.loads(failure.stdout)['event']=='error'
    assert not (tmp_path/'bad').exists()

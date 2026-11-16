import json
import subprocess
import sys
from pathlib import Path
import pytest
from runner.tests.test_runner import graph
from runner.engine import run

ROOT=Path(__file__).resolve().parents[2]

def test_dataflow_cancellation_preserves_another_writers_manifest(tmp_path):
    source=tmp_path/'source.csv';source.write_text('amount,other,category,group\n'+''.join(f'{i},{i*2},a,{i%2}\n' for i in range(100000)))
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}));output=tmp_path/'output'
    p=subprocess.Popen([sys.executable,str(ROOT/'runner/main.py'),'--execute-dataflow',str(ROOT/'runner/dataflow_examples/preparation.json'),'--source-bindings',str(bindings),'--output',str(output),'--dataflow-cache-policy','materialize'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    assert json.loads(p.stdout.readline())['event']=='dataflowStarted'
    output.mkdir();marker=output/'manifest.json';marker.write_text('{"owner":"another writer"}')
    p.terminate();p.communicate(timeout=15)
    assert p.returncode==130
    assert marker.read_text()=='{"owner":"another writer"}'
    assert not list(tmp_path.glob('.dataflow-*'))

def test_training_completion_cancellation_invalidates_own_manifest(tmp_path):
    def cancel(event):
        if event['event']=='completed':raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):run(graph(),tmp_path/'run',epochs=1,device='cpu',emit=cancel)
    assert not (tmp_path/'run/manifest.json').exists()

def test_training_completion_preserves_replacement_manifest(tmp_path):
    output=tmp_path/'run'
    def replace_then_cancel(event):
        if event['event']=='completed':
            replacement=output/'other.json';replacement.write_text('{"owner":"another writer"}');replacement.replace(output/'manifest.json')
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):run(graph(),output,epochs=1,device='cpu',emit=replace_then_cancel)
    assert (output/'manifest.json').read_text()=='{"owner":"another writer"}'

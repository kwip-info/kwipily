import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]

def invoke(*args):
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),*map(str,args)],capture_output=True,text=True)
    return result,[json.loads(line) for line in result.stdout.splitlines()]

def test_shared_registry_is_machine_readable():
    result,events=invoke('--dataflow-registry')
    assert result.returncode==0
    registry=events[0]['registry'];assert registry['schemaVersion']==1
    operators={op['id']:op for op in registry['operators']}
    current={'sourceCSV','sampleRows','join','aggregate','outputDelimited','view','sourceFiles','columns','filterRows','partition','formula','preprocess','union','sort','unique','reshapeTable','splitText'}
    assert set(operators)=={'kwip.data.'+name for name in current}|{'kwip.data.capturedTable','kwip.data.textEncode','kwip.data.sequenceWindows','kwip.data.prepareMedia','kwip.data.extractText','kwip.data.transcribeAudio','kwip.data.classifyAudio','kwip.data.gameRules','kwip.data.decide'}
    assert operators['kwip.data.capturedTable']['inputs']==[]
    assert len(current)==17
    assert [(p['id'],p['kind']) for p in operators['kwip.data.view']['inputs']]==[('table','table')]
    assert [(p['id'],p['kind']) for p in operators['kwip.data.view']['outputs']]==[('table','table')]
    assert operators['kwip.data.sourceFiles']['inputs']==[]
    assert [(p['id'],p['kind']) for p in operators['kwip.data.sourceFiles']['outputs']]==[('table','table')]
    assert operators['kwip.data.sampleRows']['configSchema']['required']==['count','seed']
    assert operators['kwip.data.partition']['configSchema']['properties']['seed']['type']=='integer'


def test_cli_artifacts_and_reproducible_preparation_export(tmp_path):
    bindings=tmp_path/'private.json';bindings.write_text(json.dumps({'source':str(ROOT/'examples/dataflow/observations.csv')}))
    output=tmp_path/'result'
    process,events=invoke('--execute-dataflow',ROOT/'examples/dataflow/scaler.json','--source-bindings',bindings,'--output',output)
    assert process.returncode==0,events
    assert events[0]['event']=='dataflowStarted' and events[-1]['event']=='dataflowCompleted'
    previews=[e for e in events if e['event']=='dataflowPreview'];assert len(previews)==8
    starts=[e for e in events if e['event']=='dataflowNodeStarted']
    assert [e['nodeID'] for e in starts] == [e['nodeID'] for e in previews]
    lifecycle=[e['event'] for e in events if e['event'] in ('dataflowNodeStarted','dataflowPreview')]
    assert lifecycle == ['dataflowNodeStarted','dataflowPreview']*8
    for event in previews:
        for port in event['outputs'].values():
            if port['kind']=='table':assert len(port['preview']['rows'])<=8 and 'observationIDs' not in port['lineage']
    manifest=json.loads((output/'manifest.json').read_text());assert manifest['status']=='completed'
    assert str(ROOT) not in (output/'prep.py').read_text()
    reproduced=tmp_path/'reproduced'
    p=subprocess.run([sys.executable,str(output/'prep.py'),'--source-bindings',str(bindings),'--output',str(reproduced)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr+p.stdout
    again=json.loads((reproduced/'manifest.json').read_text())
    assert manifest['graphDigest']==again['graphDigest']
    assert manifest['artifacts']==again['artifacts']


def test_failed_or_cancelled_adapter_never_publishes_manifest(tmp_path):
    from runner.dataflow_cli import run_dataflow
    graph=ROOT/'examples/dataflow/scaler.json';bindings={'source':str(ROOT/'examples/dataflow/observations.csv')}
    output=tmp_path/'cancelled'
    def stop(event):
        if event['event']=='dataflowPreview':raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):run_dataflow(graph,bindings,output,stop)
    assert not output.exists()
    marker=tmp_path/'keep';marker.mkdir();(marker/'keep.txt').write_text('untouched')
    with pytest.raises(ValueError):run_dataflow(graph,bindings,marker,lambda _:None)
    assert (marker/'keep.txt').read_text()=='untouched'


def test_real_worker_cancellation_and_invalid_graph_are_errors(tmp_path):
    source=tmp_path/'large.csv';source.write_text('measurement,second,group\n'+''.join(f'{i},{i*2},{i%2}\n' for i in range(100000)))
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}))
    output=tmp_path/'interrupted'
    process=subprocess.Popen([sys.executable,str(ROOT/'runner/main.py'),'--execute-dataflow',str(ROOT/'examples/dataflow/scaler.json'),'--source-bindings',str(bindings),'--output',str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    first=json.loads(process.stdout.readline());assert first['event']=='dataflowStarted'
    process.terminate();stdout,stderr=process.communicate(timeout=10)
    assert process.returncode==130
    assert not any(json.loads(line)['event']=='dataflowCompleted' for line in stdout.splitlines())
    assert not (output/'manifest.json').exists()
    assert not list(tmp_path.glob('.dataflow-*'))
    graph=tmp_path/'bad.json';graph.write_text('{"schemaVersion":2}')
    p,events=invoke('--execute-dataflow',graph,'--source-bindings',bindings,'--output',tmp_path/'invalid')
    assert p.returncode==1 and events[-1]['event']=='error'
    assert not (tmp_path/'invalid').exists()


def test_registry_enforces_conditional_missing_and_bounded_types():
    from runner.dataflow import DEFINITIONS,schema_validate
    schema=DEFINITIONS['missing']['configSchema']
    for config in [{'columns':['x'],'policy':'fill'}, {'columns':['x'],'policy':'error','value':1}]:
        with pytest.raises(ValueError):schema_validate(config,schema)
    with pytest.raises(ValueError):schema_validate({'seed':True,'validationPercent':20},DEFINITIONS['split']['configSchema'])

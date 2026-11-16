import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runner/dataflow_examples'

def command(tmp_path,output,*flags,source=None):
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source or BASE/'observations.csv')}))
    return [sys.executable,str(ROOT/'runner/main.py'),'--execute-dataflow',str(BASE/'preparation.json'),'--source-bindings',str(bindings),'--output',str(output),*map(str,flags)]

def call(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    return p,[json.loads(line) for line in p.stdout.splitlines()]

def test_actual_cli_materializes_then_reuses_verified_nodes(tmp_path):
    first=tmp_path/'first'
    p,events=call(command(tmp_path,first,'--dataflow-cache-policy','materialize'))
    assert p.returncode==0,events
    assert events[0]['cachePolicy']=='materialize'
    manifest=first/'manifest.json';pin=hashlib.sha256(manifest.read_bytes()).hexdigest()
    second=tmp_path/'second'
    p,events=call(command(tmp_path,second,'--dataflow-cache-policy','reuse','--dataflow-reuse-manifest',manifest,'--dataflow-reuse-manifest-sha256',pin))
    assert p.returncode==0,events
    assert events[-1]['cache']['executedNodeIDs']==['source']
    expected_reused={'columns','partition','prepare','training','validation','test'}
    assert set(events[-1]['cache']['reusedNodeIDs'])==expected_reused
    assert len(events[-1]['cache']['reusedNodeIDs'])==len(expected_reused)
    assert {e['nodeID'] for e in events if e.get('execution',{}).get('status')=='reused'}==expected_reused
    assert sum(e.get('execution',{}).get('status')=='reused' for e in events)==len(expected_reused)
    starts=[e['nodeID'] for e in events if e['event']=='dataflowNodeStarted']
    finishes=[e['nodeID'] for e in events if e['event']=='dataflowPreview']
    assert starts==finishes and len(starts)==7
    assert set(starts)==expected_reused|{'source'}
    assert json.loads(manifest.read_text())['artifacts']==json.loads((second/'manifest.json').read_text())['artifacts']
    p,events=call(command(tmp_path,tmp_path/'badpin','--dataflow-cache-policy','reuse','--dataflow-reuse-manifest',manifest,'--dataflow-reuse-manifest-sha256','0'*64))
    assert p.returncode==1 and events[-1]['event']=='error'
    assert not (tmp_path/'badpin').exists()

@pytest.mark.parametrize('flags',[
    ['--dataflow-cache-policy','reuse'],
    ['--dataflow-cache-policy','reuse','--dataflow-reuse-manifest','missing'],
    ['--dataflow-cache-policy','materialize','--dataflow-reuse-manifest','missing','--dataflow-reuse-manifest-sha256','0'*64],
    ['--dataflow-reuse-manifest-sha256','0'*64],
])
def test_cli_rejects_incomplete_or_conflicting_cache_selection(tmp_path,flags):
    p,events=call(command(tmp_path,tmp_path/'out',*flags))
    assert p.returncode==1 and events[-1]['event']=='error'
    assert 'unrecognized arguments' not in events[-1]['message']
    assert 'cache' in events[-1]['message'].lower() or 'reuse' in events[-1]['message'].lower()
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('flags',[['--dataflow-cache-policy','off'],['--dataflow-reuse-manifest','missing']])
def test_cache_flags_rejected_outside_dataflow(flags):
    p,events=call([sys.executable,str(ROOT/'runner/main.py'),'--capabilities',*flags])
    assert p.returncode==1 and 'execute-dataflow' in events[-1]['message']

def test_real_materialize_cancel_cannot_publish_cache(tmp_path):
    source=tmp_path/'large.csv'
    source.write_text('amount,other,category,group\n'+''.join(f'{i},{i*2},a,{i%2}\n' for i in range(100000)))
    output=tmp_path/'cancelled'
    p=subprocess.Popen(command(tmp_path,output,'--dataflow-cache-policy','materialize',source=source),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    first=json.loads(p.stdout.readline());assert first['event']=='dataflowStarted',first
    p.terminate();out,err=p.communicate(timeout=15)
    assert p.returncode==130,err
    assert not any(json.loads(line)['event']=='dataflowCompleted' for line in out.splitlines())
    assert not (output/'manifest.json').exists()
    assert not list(tmp_path.glob('.dataflow-*'))

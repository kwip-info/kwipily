import hashlib
import json
import subprocess
import sys
import pytest
from runner.tests.test_prepared_data import prepared,load,ROOT
from runner.tests.test_runner import graph
from runner.preview import preview
from runner.prepared_data import load_prepared
from runner.engine import run


def test_frozen_validation_stage_and_resume_preserve_rows_and_outputs(tmp_path):
    path=prepared(tmp_path);frozen=load(path);g=graph()
    first=preview(g,tmp_path/'stage',device='cpu',prepared=frozen,stop_at='merge')
    assert first['inputProvenance']['partition']=='validation'
    assert first['inputProvenance']['rowIDs']==frozen[1]['validationRowIDs'][:8]
    assert first['inputProvenance']['count']==8
    assert first['previews']['input']['values']==frozen[0][2][:8].flatten()[:8].tolist()
    resumed=preview(g,tmp_path/'resumed',device='cpu',prepared=frozen,resume_from='merge',cache=first['cachePath'])
    full=preview(g,tmp_path/'full',device='cpu',prepared=frozen)
    assert resumed['previews']['output']==full['previews']['output']
    assert resumed['reusedNodeIDs'] and resumed['inputProvenance']==full['inputProvenance']
    assert 'accuracy' not in resumed and 'loss' not in resumed
    changed=load_prepared(path/'manifest.json','training','table','heldout','table',['second','feature'],'group')
    with pytest.raises(ValueError,match='stale'):
        preview(g,tmp_path/'mapping',device='cpu',prepared=changed,resume_from='merge',cache=first['cachePath'])
    run(g,tmp_path/'trained',epochs=2,device='cpu',prepared=frozen)
    with pytest.raises(ValueError,match='stale'):
        preview(g,tmp_path/'weights',device='cpu',prepared=frozen,resume_from='merge',cache=first['cachePath'],checkpoint=tmp_path/'trained/checkpoint.pt')


def test_prepared_preview_cli_revalidates_pins_and_rejects_split_overrides(tmp_path):
    path=prepared(tmp_path);manifest=(path/'manifest.json').read_bytes();g=tmp_path/'graph.json';g.write_text(json.dumps(graph()))
    args=[sys.executable,str(ROOT/'runner/main.py'),'--graph',str(g),'--output',str(tmp_path/'out'),'--device','cpu','--preview','--stop-at','merge','--prepared-dataflow',str(path/'manifest.json'),'--prepared-manifest-sha256',hashlib.sha256(manifest).hexdigest(),'--prepared-graph-digest',json.loads(manifest)['graphDigest'],'--prepared-training-node','training','--prepared-training-port','table','--prepared-validation-node','heldout','--prepared-validation-port','table','--feature-columns-json','["feature","second"]','--target-column','group']
    result=subprocess.run(args,capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['inputProvenance']['partition']=='validation'
    for flag in ('--sample-limit','--split-seed','--validation-percent'):
        result=subprocess.run(args+[flag,'1'],capture_output=True,text=True);assert result.returncode==1 and 'partitioned' in result.stdout
    descriptor=next(a for a in json.loads(manifest)['artifacts'] if a.get('nodeID')=='heldout')
    (path/descriptor['path']).write_text('{}')
    result=subprocess.run(args,capture_output=True,text=True);assert result.returncode==1 and 'hash' in result.stdout
    (path/'manifest.json').write_text('{}')
    result=subprocess.run(args,capture_output=True,text=True);assert result.returncode==1 and 'manifest' in result.stdout

def test_prepared_checkpoint_preview_pins_weights_and_rejects_nonregular_inputs(tmp_path):
    import torch
    frozen=load(prepared(tmp_path));g=graph();trained=tmp_path/'trained'
    run(g,trained,epochs=2,device='cpu',prepared=frozen)
    checkpoint=trained/'checkpoint.pt'
    first=preview(g,tmp_path/'first',device='cpu',prepared=frozen,checkpoint=checkpoint)
    resumed=preview(g,tmp_path/'resumed',device='cpu',prepared=frozen,checkpoint=checkpoint,resume_from='merge',cache=first['cachePath'])
    assert first['weights']=='checkpoint' and first['previews']['output']==resumed['previews']['output']
    state=torch.load(checkpoint,weights_only=True);key=next(iter(state['modelState']));state['modelState'][key]+=1;torch.save(state,checkpoint)
    with pytest.raises(ValueError,match='stale'):
        preview(g,tmp_path/'changed',device='cpu',prepared=frozen,checkpoint=checkpoint,resume_from='merge',cache=first['cachePath'])
    symlink=tmp_path/'linked.pt';symlink.symlink_to(checkpoint)
    with pytest.raises(ValueError,match='regular'):
        preview(g,tmp_path/'linked',device='cpu',prepared=frozen,checkpoint=symlink)

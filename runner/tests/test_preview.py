import json
import torch
import pytest
from runner.tests.test_runner import graph,csv_fixture
from runner.preview import preview
from runner.engine import run

def test_branch_preview_resume_actual_execution(tmp_path):
    g=graph(); calls=[]
    first=preview(g,tmp_path/'first',device='cpu',on_execute=calls.append)
    assert calls==first['executedNodeIDs'] and not first['reusedNodeIDs']
    assert first['weights']=='initialized'
    calls=[]
    second=preview(g,tmp_path/'second',device='cpu',resume_from='merge',cache=first['cachePath'],on_execute=calls.append)
    assert calls==['merge','head','output']
    assert second['executedNodeIDs']==calls
    assert set(second['reusedNodeIDs'])=={'input','left','right','relu','tanh'}
    assert second['previews']['output']==first['previews']['output']

def test_stop_and_layout_change(tmp_path):
    g=graph(); a=preview(g,tmp_path/'a',device='cpu',stop_at='merge')
    assert 'output' not in a['previews']
    g['name']='renamed'; g['nodes'][0]['name']='Features'; g['nodes'][0]['x']=100
    b=preview(g,tmp_path/'b',device='cpu',resume_from='merge',cache=a['cachePath'])
    assert 'output' in b['executedNodeIDs']

@pytest.mark.parametrize('change',['config','seed','source','weights'])
def test_stale_cache_rejected(tmp_path,change):
    g=graph(); p=csv_fixture(tmp_path/'source.csv'); args={'dataset_csv':p,'feature_columns':['first','second'],'target_column':'label'}
    a=preview(g,tmp_path/'a',device='cpu',**args)
    kwargs={}
    if change=='config':
        g['nodes'][1]['config']['out_features']=15; g['nodes'][6]['config']['in_features']=31
    if change=='seed': kwargs['seed']=43
    if change=='source': p.write_text(p.read_text().replace('sample 0','edited 0'))
    if change=='weights':
        run(g,tmp_path/'trained',epochs=2,device='cpu'); kwargs['checkpoint']=tmp_path/'trained/checkpoint.pt'
    with pytest.raises(ValueError,match='stale|match'):
        preview(g,tmp_path/'b',device='cpu',resume_from='merge',cache=a['cachePath'],**args,**kwargs)

def test_cache_integrity_and_checkpoint_graph(tmp_path):
    a=preview(graph(),tmp_path/'a',device='cpu')
    with open(a['cachePath'],'ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):
        preview(graph(),tmp_path/'b',device='cpu',resume_from='merge',cache=a['cachePath'])
    run(graph(),tmp_path/'trained',epochs=2,device='cpu')
    bad=graph(); bad['nodes'][3]['type']='tanh'
    with pytest.raises(ValueError,match='graph'):
        preview(bad,tmp_path/'c',device='cpu',checkpoint=tmp_path/'trained/checkpoint.pt')

def test_checkpoint_preview_parity_and_no_grad(tmp_path):
    run(graph(),tmp_path/'trained',epochs=2,device='cpu')
    a=preview(graph(),tmp_path/'a',device='cpu',checkpoint=tmp_path/'trained/checkpoint.pt')
    assert a['weights']=='checkpoint'
    values=torch.load(a['cachePath'],weights_only=True)
    assert all(not v.requires_grad and v.grad_fn is None for v in values.values())
    b=preview(graph(),tmp_path/'b',device='cpu',checkpoint=tmp_path/'trained/checkpoint.pt',resume_from='head',cache=a['cachePath'])
    assert a['previews']['output']==b['previews']['output']

def test_incomplete_branch_frontier_rejected(tmp_path):
    a=preview(graph(),tmp_path/'a',device='cpu',stop_at='left')
    with pytest.raises(ValueError,match='incomplete'):
        preview(graph(),tmp_path/'b',device='cpu',resume_from='merge',cache=a['cachePath'])

def test_cache_load_never_unpickles_arbitrary_objects(tmp_path):
    from datetime import datetime
    from pathlib import Path
    from runner.preview import file_hash
    a=preview(graph(),tmp_path/'a',device='cpu'); path=Path(a['cachePath'])
    torch.save({'input':datetime.now()},path)
    meta=json.loads(path.with_suffix('.json').read_text()); meta['fileHash']=file_hash(path,32_000_000)
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(Exception,match='Weights only load failed'):
        preview(graph(),tmp_path/'b',device='cpu',resume_from='merge',cache=path)

def test_cli_preview_resume_auto_device(tmp_path):
    import subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    g=tmp_path/'graph.json'; g.write_text(json.dumps(graph()))
    args=[sys.executable,str(root/'runner/main.py'),'--graph',str(g),'--device','auto']
    a=subprocess.run([*args,'--output',str(tmp_path/'a'),'--preview','--stop-at','merge'],capture_output=True,text=True)
    assert a.returncode==0,a.stdout+a.stderr
    events=[json.loads(line) for line in a.stdout.splitlines()]; assert [e['event'] for e in events]==['previewStarted','previewCompleted']
    b=subprocess.run([*args,'--output',str(tmp_path/'b'),'--resume-from','merge','--cache',events[-1]['cachePath']],capture_output=True,text=True)
    assert b.returncode==0,b.stdout+b.stderr
    result=json.loads(b.stdout.splitlines()[-1]); assert result['executedNodeIDs']==['merge','head','output']
    assert result['device']==('mps' if torch.backends.mps.is_available() else 'cpu')

def test_preview_reports_loaded_source_hash(tmp_path):
    import hashlib
    p=csv_fixture(tmp_path/'source.csv')
    result=preview(graph(),tmp_path/'preview',device='cpu',dataset_csv=p,feature_columns=['first','second'],target_column='label')
    assert result['sourceHash']==hashlib.sha256(p.read_bytes()).hexdigest()

def test_resume_loads_the_same_cache_bytes_that_were_verified(tmp_path,monkeypatch):
    g=graph();first=preview(g,tmp_path/'first',device='cpu')
    original=torch.load
    cached=original(first['cachePath'],weights_only=True)
    changed={key:torch.full_like(value,17) for key,value in cached.items()}
    def replace_before_load(*args,**kwargs):
        torch.save(changed,first['cachePath'])
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',replace_before_load)
    resumed=preview(g,tmp_path/'resumed',device='cpu',resume_from='merge',cache=first['cachePath'])
    assert resumed['previews']['output']==first['previews']['output']


def test_preview_cache_and_sidecar_refuse_symlinks_and_oversize(tmp_path):
    from pathlib import Path
    g=graph();first=preview(g,tmp_path/'first',device='cpu');cache=Path(first['cachePath']);sidecar=cache.with_suffix('.json')
    saved=sidecar.read_bytes();sidecar.write_bytes(b' '*100001)
    with pytest.raises(ValueError,match='bound|large'):
        preview(g,tmp_path/'large',device='cpu',resume_from='merge',cache=cache)
    sidecar.write_bytes(saved)
    moved=cache.with_name('saved.pt');cache.rename(moved);cache.symlink_to(moved)
    with pytest.raises(ValueError,match='regular'):
        preview(g,tmp_path/'linked-cache',device='cpu',resume_from='merge',cache=cache)
    cache.unlink();moved.rename(cache)
    moved_sidecar=sidecar.with_name('saved.json');sidecar.rename(moved_sidecar);sidecar.symlink_to(moved_sidecar)
    with pytest.raises(ValueError,match='regular'):
        preview(g,tmp_path/'linked-sidecar',device='cpu',resume_from='merge',cache=cache)

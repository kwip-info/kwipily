import importlib.util
import json
from pathlib import Path
import subprocess
import signal
from contextlib import contextmanager
import sys
import torch
import pytest
from runner.engine import GraphModel, validate, dataset, run

ROOT = Path(__file__).resolve().parents[2]

@contextmanager
def preserved_cli_signal_handlers():
    """Only direct embedded CLI calls borrow the test host's signal ownership.

    A real worker owns these handlers until process exit. Embedding its one-shot
    main in pytest must not replace pytest's interruption policy afterward.
    """
    previous={number:signal.getsignal(number) for number in (signal.SIGINT,signal.SIGTERM)}
    try:
        yield previous
    finally:
        for number,handler in previous.items():signal.signal(number,handler)


def graph():
    specs = [('input','input',{'features':2}), ('left','linear',{'in_features':2,'out_features':16}), ('right','linear',{'in_features':2,'out_features':16}), ('relu','relu',{}), ('tanh','tanh',{}), ('merge','concat',{}), ('head','linear',{'in_features':32,'out_features':2}), ('output','output',{})]
    wires = [('input','left','in'),('input','right','in'),('left','relu','in'),('right','tanh','in'),('relu','merge','a'),('tanh','merge','b'),('merge','head','in'),('head','output','in')]
    return {'schemaVersion':1,'name':'Test classifier','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs], 'edges':[{'id':f'e{n}','source':s,'sourcePort':'out','target':t,'targetPort':p} for n,(s,t,p) in enumerate(wires)]}

def test_branch_output_and_gradients():
    m = GraphModel(graph()); x=torch.randn(5,2,requires_grad=True)
    actual=m(x)
    expected=m.layers['head'](torch.cat((torch.relu(m.layers['left'](x)),torch.tanh(m.layers['right'](x))),dim=1))
    assert torch.allclose(actual,expected)
    ga=torch.autograd.grad(actual.sum(), (x,*m.parameters()),retain_graph=True)
    gb=torch.autograd.grad(expected.sum(), (x,*m.parameters()))
    assert all(torch.allclose(a,b) for a,b in zip(ga,gb))

def test_split_reproducible_and_held_out():
    a=dataset(42); b=dataset(42); c=dataset(43)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert not torch.equal(a[0],c[0])
    assert set(map(tuple,a[0].tolist())).isdisjoint(set(map(tuple,a[2].tolist())))
    assert len(a[0])==800 and len(a[2])==200

@pytest.mark.parametrize('mutation', ['unknown','shape','cycle','missing','duplicate','disconnected','port','huge','fractional','schema'])
def test_validation(mutation):
    g=graph()
    if mutation=='unknown': g['nodes'][1]['type']='code'
    elif mutation=='shape': g['nodes'][1]['config']['in_features']=3
    elif mutation=='cycle': g['edges'][0]['source']='relu'
    elif mutation=='missing': g['edges'].pop()
    elif mutation=='duplicate': g['nodes'][1]['id']='input'
    elif mutation=='disconnected': g['nodes'].append({'id':'loose','type':'relu','config':{}})
    elif mutation=='port': g['edges'][0]['sourcePort']='bogus'
    elif mutation=='huge': g['nodes'][1]['config']['out_features']=513
    elif mutation=='fractional': g['nodes'][1]['config']['out_features']=1.5
    else: g['schemaVersion']=2
    with pytest.raises(ValueError): validate(g)

def test_training_artifacts_export(tmp_path):
    events=[]; result=run(graph(),tmp_path,epochs=100,seed=42,emit=events.append,device='cpu')
    assert result['accuracy']>=.94
    epochs=[e for e in events if e['event']=='epoch']
    assert epochs[-1]['trainLoss']<epochs[0]['trainLoss']*.4
    assert events[-1]['event']=='completed'
    assert all(len(p['values'])<=8 for p in result['previews'].values())
    assert len(result['predictions'])==200
    assert all((tmp_path/n).exists() for n in ['manifest.json','graph.json','checkpoint.pt','model.py','metrics.json'])
    manifest=json.loads((tmp_path/'manifest.json').read_text()); assert manifest['status']=='completed'
    spec=importlib.util.spec_from_file_location('exported',tmp_path/'model.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    exported=mod.Model(); checkpoint=torch.load(tmp_path/'checkpoint.pt',weights_only=True); exported.load_state_dict(checkpoint['modelState'])
    live=GraphModel(graph()); live.load_state_dict(checkpoint['modelState']); x=dataset(42)[2]
    assert torch.equal(live(x),exported(x))

def test_cli_errors(tmp_path):
    p=tmp_path/'bad.json'; p.write_text('{')
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(p),'--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode!=0
    assert json.loads(result.stdout)['event']=='error'
    assert not (tmp_path/'out/manifest.json').exists()

def test_cli_cancel(tmp_path):
    p=tmp_path/'graph.json'; p.write_text(json.dumps(graph()))
    proc=subprocess.Popen([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(p),'--output',str(tmp_path/'out'),'--epochs','10000'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    line=proc.stdout.readline(); assert json.loads(line)['event']=='started'
    proc.terminate(); stdout,stderr=proc.communicate(timeout=15)
    assert proc.returncode!=0
    assert not any(json.loads(l)['event']=='completed' for l in stdout.splitlines())
    assert not (tmp_path/'out/manifest.json').exists()

def test_existing_run_cannot_be_overwritten(tmp_path):
    marker=tmp_path/'manifest.json'; marker.write_text('{"keep":true}')
    with pytest.raises(ValueError,match='empty'):
        run(graph(),tmp_path)
    assert marker.read_text()=='{"keep":true}'


def test_cli_success_protocol(tmp_path):
    p=tmp_path/'graph.json'; p.write_text(json.dumps(graph()))
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(p),'--output',str(tmp_path/'out'),'--epochs','2'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines()]
    lifecycle=[e['event'] for e in events if e['event'] in ('started','epoch','completed')]
    assert lifecycle==['started','epoch','epoch','completed']
    progress=[e for e in events if e['event']=='trainingProgress']
    assert progress and [e['sequence'] for e in progress]==list(range(1,len(progress)+1))
    assert all(e['schemaVersion']==1 and e['runID']==events[0]['runID'] for e in progress)
    # Progress has its own sequence; retain the exact legacy capture protocol.
    legacy=[e for e in events if e['event']!='trainingProgress']
    capture=legacy[3:-1]
    assert capture[0]['event']=='modelCaptureStarted'
    assert capture[-1]['event']=='modelCaptureCompleted'
    assert [e['sequence'] for e in capture]==list(range(1,len(capture)+1))
    assert all(e['stage']=='validationCapture' and e['runID']==events[0]['runID'] for e in capture)
    nodes=capture[1:-1]
    assert len(nodes)==2*len(graph()['nodes'])
    for start,end in zip(nodes[::2],nodes[1::2]):
        assert start['event']=='modelNodeStarted' and end['event']=='modelNodeCompleted'
        assert start['nodeID']==end['nodeID']
    assert {e['nodeID'] for e in nodes}=={n['id'] for n in graph()['nodes']}


def test_trainable_component_required_before_run(tmp_path):
    g=graph(); g['nodes']=[g['nodes'][0],g['nodes'][-1]]
    g['edges']=[{'id':'direct','source':'input','sourcePort':'out','target':'output','targetPort':'in'}]
    with pytest.raises(ValueError,match='trainable parameters'):
        run(g,tmp_path/'artifacts')
    assert not (tmp_path/'artifacts').exists()

def test_cancel_does_not_delete_existing_manifest(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'runner'))
    import main
    import engine
    marker=tmp_path/'manifest.json'; marker.write_text('{"keep":true}')
    g=tmp_path/'request.json'; g.write_text(json.dumps(graph()))
    monkeypatch.setattr(sys,'argv',['main.py','--graph',str(g),'--output',str(tmp_path)])
    def cancel(*args,**kwargs): raise main.Cancelled('Run cancelled.')
    monkeypatch.setattr(engine,'run',cancel)
    with preserved_cli_signal_handlers() as previous:
        assert main.main()==130
        assert all(signal.getsignal(number) is main.cancel for number in previous)
    assert all(signal.getsignal(number)==handler for number,handler in previous.items())
    assert marker.read_text()=='{"keep":true}'

@pytest.mark.parametrize('mutation',['schemaBool','unknownConfig','missingConfig','badPosition'])
def test_strict_contract(mutation):
    g=graph()
    if mutation=='schemaBool': g['schemaVersion']=True
    if mutation=='unknownConfig': g['nodes'][0]['config']['unexpected']=2
    if mutation=='missingConfig': g['nodes'][0]['config']={}
    if mutation=='badPosition': g['nodes'][0]['x']=float('nan')
    with pytest.raises(ValueError): validate(g)

def test_cli_argument_errors_are_json(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph','unused','--output',str(tmp_path),'--epochs','wrong'],capture_output=True,text=True)
    assert result.returncode!=0
    assert json.loads(result.stdout)['event']=='error'

def test_node_ids_can_match_pytorch_attribute_names(tmp_path):
    g=graph()
    for n in g['nodes']:
        if n['id']=='left': n['id']='training'
    for e in g['edges']:
        if e['source']=='left': e['source']='training'
        if e['target']=='left': e['target']='training'
    result=run(g,tmp_path,epochs=2)
    assert result['event']=='completed'
    spec=importlib.util.spec_from_file_location('reserved_export',tmp_path/'model.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    exported=mod.Model(); ckpt=torch.load(tmp_path/'checkpoint.pt',weights_only=True); exported.load_state_dict(ckpt['modelState'])
    live=GraphModel(g); live.load_state_dict(ckpt['modelState'])
    assert torch.equal(exported(dataset(42)[2]),live(dataset(42)[2]))

def test_device_resolution_and_cpu_metadata(tmp_path):
    from runner.engine import resolve_device, capabilities
    assert resolve_device('cpu')=='cpu'
    assert capabilities()['defaultDevice']==('mps' if torch.backends.mps.is_available() else 'cpu')
    with pytest.raises(ValueError): resolve_device('cuda')
    events=[]; result=run(graph(),tmp_path,epochs=2,device='cpu',emit=events.append)
    assert events[0]['device']=='cpu' and result['device']=='cpu'
    assert json.loads((tmp_path/'manifest.json').read_text())['runtime']['device']=='cpu'

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS hardware unavailable')
def test_actual_mps_run(tmp_path):
    result=run(graph(),tmp_path,epochs=5,device='mps')
    assert result['device']=='mps'
    checkpoint=torch.load(tmp_path/'checkpoint.pt',weights_only=True,map_location='cpu')
    assert all(v.device.type=='cpu' for v in checkpoint['modelState'].values())

def csv_fixture(path):
    path.write_text('first,second,label,unused\n'+'\n'.join(f'{i/10},{i%3},{i%2},sample {i}' for i in range(40)))
    return path

def test_csv_preview_split_provenance(tmp_path):
    from runner.engine import inspect_data, csv_dataset
    p=csv_fixture(tmp_path/'source.csv'); preview=inspect_data(p)
    assert preview['columns']==['first','second','label','unused']
    assert preview['rowCount']==40 and len(preview['rows'])==8
    a,provenance=csv_dataset(p,['first','second'],'label',42)
    b,_=csv_dataset(p,['first','second'],'label',42)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert len(a[0])==32 and len(a[2])==8
    assert set(a[1].tolist())=={0,1} and set(a[3].tolist())=={0,1}
    assert set(map(tuple,a[0].tolist())).isdisjoint(set(map(tuple,a[2].tolist())))
    result=run(graph(),tmp_path/'run',epochs=2,device='cpu',dataset_csv=p,feature_columns=['first','second'],target_column='label')
    manifest=json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['provenance']['sourceHash']==provenance['sourceHash']
    assert manifest['evaluation']['count']==8 and len(result['predictions'])==8

@pytest.mark.parametrize('mode',['targetFeature','missing','nonfinite','label','smallClass','duplicateHeader','ragged'])
def test_csv_invalid(tmp_path,mode):
    from runner.engine import csv_dataset
    p=csv_fixture(tmp_path/'source.csv'); features=['first','second']; target='label'
    if mode=='targetFeature': features=['first','label']
    if mode=='missing': features=['absent','second']
    if mode=='nonfinite': p.write_text(p.read_text().replace('0.0,0,0','nan,0,0'))
    if mode=='label': p.write_text(p.read_text().replace('0.0,0,0','0.0,0,2'))
    if mode=='smallClass': p.write_text('first,second,label\n0,0,0\n1,1,1')
    if mode=='duplicateHeader': p.write_text(p.read_text().replace('first,second','first,first'))
    if mode=='ragged': p.write_text(p.read_text()+'\n1,2')
    with pytest.raises(ValueError): csv_dataset(p,features,target,42)

def test_unavailable_mps_is_explicit(monkeypatch):
    from runner.engine import resolve_device
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    assert resolve_device('auto')=='cpu'
    with pytest.raises(ValueError,match='unavailable'): resolve_device('mps')

def test_csv_duplicate_leakage_and_size_limit(tmp_path,monkeypatch):
    from runner import data
    p=csv_fixture(tmp_path/'source.csv')
    p.write_text(p.read_text()+'\n0.0,0,0,duplicate')
    with pytest.raises(ValueError,match='duplicate'): data.csv_dataset(p,['first','second'],'label',42)
    monkeypatch.setattr(data,'MAX_BYTES',10)
    with pytest.raises(ValueError,match='limit'): data.inspect_data(p)

def test_cli_data_and_capabilities(tmp_path):
    p=csv_fixture(tmp_path/'source.csv')
    for args,event in [(['--inspect-data',str(p)],'dataPreview'),(['--capabilities'],'capabilities')]:
        result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),*args],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['event']==event
    g=tmp_path/'graph.json'; g.write_text(json.dumps(graph()))
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(g),'--output',str(tmp_path/'run'),'--device','auto','--dataset-csv',str(p),'--feature-columns','first,second','--target-column','label','--epochs','3'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]['event']=='completed'
    assert events[-1]['device']==('mps' if torch.backends.mps.is_available() else 'cpu')
    assert events[-1]['evaluationCount']==8

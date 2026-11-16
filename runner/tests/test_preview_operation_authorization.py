"""Retained preview contract covers every current source and inference input."""
import json
import os
from pathlib import Path
import shutil
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


def inventory(f):
    f[1]['inputFiles']=[dict(path=str(path.relative_to(f[0])),bytes=path.stat().st_size,sha256=digest(path.read_bytes()))
        for path in sorted((f[0]/'input').rglob('*')) if path.is_file()]
    publish(f)


@pytest.fixture
def preview_operation(operation_authority):
    from runner.tests.test_runner import graph
    f=operation_authority;value=f[1]
    for key in ('runID','directoryID','phase'):del value[key]
    inputs=f[0]/'input';inputs.mkdir(mode=0o700);(inputs/'graph.json').write_bytes(encoded(graph()))
    value.update(schemaVersion=2,toolMode='modelPreview',arguments=['--graph',str(inputs/'graph.json'),
        '--output',str(f[0]/'artifacts'),'--device','cpu','--seed','42','--sample-limit','0',
        '--split-seed','42','--validation-percent','20','--preview','--stop-at','output'])
    inventory(f);return f


def source(f,kind,tmp_path):
    if kind=='csv':
        from runner.tests.test_runner import csv_fixture
        folder=f[0]/'input/data';folder.mkdir()
        path=csv_fixture(folder/'original-data.csv')
        f[1]['arguments']+=['--dataset-csv',str(path),'--feature-columns-json','["first","second"]','--target-column','label']
        return dict(dataset_csv=path,feature_columns=['first','second'],target_column='label')
    if kind=='prepared':
        from runner.tests.test_prepared_data import prepared,load
        folder=f[0]/'input/preparation';shutil.copytree(prepared(tmp_path),folder)
        raw=(folder/'manifest.json').read_bytes();manifest=json.loads(raw)
        args=f[1]['arguments'];start=args.index('--sample-limit');del args[start:start+6]
        args+=['--prepared-dataflow',str(folder/'manifest.json'),'--prepared-manifest-sha256',digest(raw),
            '--prepared-graph-digest',manifest['graphDigest'],'--prepared-training-node','training','--prepared-training-port','table',
            '--prepared-validation-node','heldout','--prepared-validation-port','table','--feature-columns-json','["feature","second"]','--target-column','group']
        return dict(prepared=load(folder))
    return {}


def test_preview_authority_rechecks_graph_and_allows_captured_long_budget(preview_operation):
    f=preview_operation;f[1].update(executionBudget=dict(schemaVersion=1,requestedSeconds=86400,maximumRunSeconds=86400,source='application'),remainingSeconds=86400);publish(f)
    with claim(f) as authority:
        authority.check();(f[0]/'input/graph.json').write_bytes(b'{}')
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['missing-device','duplicate','wrong-output','registry','training','mixed-preview-resume','missing-stop','wrong-input','extra-input','bad-size','bool-size','missing-weight-graph','missing-cache-sidecar','bad-source-path','oversized-total','unsafe-link'])
def test_invalid_preview_contract_never_claims(preview_operation,change):
    f=preview_operation;v=f[1];args=v['arguments']
    if change=='missing-device':i=args.index('--device');del args[i:i+2]
    elif change=='duplicate':args+=['--seed','43']
    elif change=='wrong-output':args[args.index('--output')+1]='/tmp/escape'
    elif change=='registry':args+=['--registry','/tmp/registry']
    elif change=='training':args+=['--epochs','10']
    elif change=='mixed-preview-resume':args+=['--resume-from','output','--cache',str(f[0]/'input/cache/activation-cache.pt')]
    elif change=='missing-stop':del args[-2:]
    elif change=='wrong-input':args[1]='/tmp/graph.json'
    elif change=='extra-input':v['inputFiles'].append(dict(path='input/other.json',bytes=0,sha256=digest(b'')))
    elif change=='bad-size':v['inputFiles'][0]['bytes']=2000001
    elif change=='bool-size':v['inputFiles'][0]['bytes']=True
    elif change=='missing-weight-graph':args+=['--checkpoint',str(f[0]/'input/weights/checkpoint.pt')]
    elif change=='missing-cache-sidecar':
        del args[-3:];args+=['--resume-from','output','--cache',str(f[0]/'input/cache/activation-cache.pt')]
    elif change=='bad-source-path':args+=['--dataset-csv',str(f[0]/'input/data/../source.csv'),'--feature-columns-json','["a","b"]','--target-column','c']
    elif change=='oversized-total':v['inputFiles']=[dict(path='input/preparation/'+str(i),bytes=50000000,sha256='0'*64) for i in range(8)]
    else:
        p=f[0]/'input/graph.json';p.rename(p.with_name('old'));p.symlink_to('old')
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.parametrize('change',['none','missing-artifact','manifest-extra','graph-pin','source-mix'])
def test_prepared_inventory_must_match_pinned_manifest(preview_operation,tmp_path,change):
    f=preview_operation;source(f,'prepared',tmp_path);inventory(f)
    if change=='missing-artifact':f[1]['inputFiles'].pop()
    elif change=='manifest-extra':
        path=f[0]/'input/preparation/manifest.json';v=json.loads(path.read_bytes());v['artifacts'].append(dict(path='other',bytes=0,sha256=digest(b'')));path.write_bytes(encoded(v))
        f[1]['arguments'][f[1]['arguments'].index('--prepared-manifest-sha256')+1]=digest(path.read_bytes());inventory(f)
    elif change=='graph-pin':f[1]['arguments'][f[1]['arguments'].index('--prepared-graph-digest')+1]='0'*64
    elif change=='source-mix':f[1]['arguments']+=['--sample-limit','0']
    publish(f)
    if change=='none':
        with claim(f) as authority:authority.check()
    else:
        with pytest.raises((ValueError,OSError)):claim(f)


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('kind',['bundled','csv','prepared'])
@pytest.mark.parametrize('resume',[False,True])
def test_real_supervised_preview_sources_and_resume(preview_operation,tmp_path,libraries,kind,resume):
    from runner.tests.test_runner import graph
    from runner.preview import preview
    f=preview_operation;kwargs=source(f,kind,tmp_path)
    if resume:
        baseline=preview(graph(),tmp_path/'prior',device='cpu',**kwargs)
        folder=f[0]/'input/cache';folder.mkdir()
        for name in ('activation-cache.pt','activation-cache.json'):shutil.copyfile(tmp_path/'prior'/name,folder/name)
        args=f[1]['arguments'];i=args.index('--preview');del args[i:i+3]
        args+=['--resume-from','merge','--cache',str(folder/'activation-cache.pt')]
    inventory(f)
    guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    path.write_text(text.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,(events,error)
    assert events[0]['toolMode']=='modelPreview'
    result=events[-1];assert result['event']=='previewCompleted' and Path(result['cachePath']).is_file()
    if resume:
        assert result['reusedNodeIDs'] and result['previews']['output']==baseline['previews']['output']
    if kind=='csv':assert result['sourceHash']
    if kind=='prepared':assert result['inputProvenance']['preparation']


@pytest.mark.parametrize('change',['restored-mtime','replaced'])
def test_retained_preview_file_signature_never_hides_substitution(preview_operation,change):
    f=preview_operation;path=f[0]/'input/graph.json'
    with claim(f) as authority:
        authority.check();before=path.stat();raw=path.read_bytes()
        if change=='replaced':
            other=path.with_name('replacement');other.write_bytes(raw);os.replace(other,path)
        else:
            path.write_bytes(b' '+raw[1:]);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('variant',['backslash-basename','long-csv-column','whitespace-node'])
def test_existing_preview_names_remain_accepted(preview_operation,tmp_path,variant):
    f=preview_operation
    if variant=='whitespace-node':f[1]['arguments'][-1]=' '
    else:
        source(f,'csv',tmp_path)
        args=f[1]['arguments']
        if variant=='backslash-basename':
            index=args.index('--dataset-csv')+1;path=Path(args[index]);renamed=path.with_name('source'+chr(92)+'name.csv');path.rename(renamed);args[index]=str(renamed)
        else:
            args[args.index('--feature-columns-json')+1]=json.dumps(['first'*50,'second'])
    inventory(f)
    with claim(f) as authority:authority.check()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('resume',[False,True])
def test_real_supervised_checkpoint_preview_and_resume(preview_operation,tmp_path,libraries,resume):
    from runner.tests.test_runner import graph
    from runner.engine import run
    from runner.preview import preview
    f=preview_operation
    trained=tmp_path/'trained';run(graph(),trained,epochs=2,device='cpu')
    weights=f[0]/'input/weights';weights.mkdir()
    for name in ('checkpoint.pt','graph.json'):shutil.copyfile(trained/name,weights/name)
    checkpoint=weights/'checkpoint.pt'
    baseline=preview(graph(),tmp_path/'baseline',device='cpu',checkpoint=checkpoint)
    f[1]['arguments']+=['--checkpoint',str(checkpoint)]
    if resume:
        cache=f[0]/'input/cache';cache.mkdir()
        for name in ('activation-cache.pt','activation-cache.json'):shutil.copyfile(tmp_path/'baseline'/name,cache/name)
        args=f[1]['arguments'];index=args.index('--preview');del args[index:index+3]
        args+=['--resume-from','merge','--cache',str(cache/'activation-cache.pt')]
    inventory(f)
    guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    path.write_text(text.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,(events,error)
    result=events[-1]
    assert result['event']=='previewCompleted' and result['weights']=='checkpoint'
    assert result['previews']['output']==baseline['previews']['output']
    assert bool(result['reusedNodeIDs'])==resume

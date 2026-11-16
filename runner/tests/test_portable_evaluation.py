"""Real-weight portable evaluation, privacy and relocated standalone execution."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from runner.tests.test_frozen_evaluation_cli import evaluation_fixture
from runner.frozen_evaluation_cli import execute_request
from runner.portable_evaluation_export import build_package
from runner.portable_evaluation import evaluate_package, validate_package


def sha(raw): return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope='module')
def completed_snapshot(tmp_path_factory):
    # Package tests consume frozen bytes, not a live trained model. Build one
    # identical capture and retain immutable bytes only; every consumer gets a
    # private filesystem copy so tampering, deletion and relocation stay isolated.
    request,pin=evaluation_fixture.__wrapped__(tmp_path_factory.mktemp('portable-evaluation-capture'))
    execute_request(request,pin,lambda _:None)
    root=request.parent
    files=tuple((str(p.relative_to(root)),p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file())
    return files,sha((root/'artifacts/manifest.json').read_bytes()),str(root)


@pytest.fixture
def completed(completed_snapshot,tmp_path):
    files,pin,_=completed_snapshot
    root=tmp_path/'evaluation'
    for name,raw in files:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return root,pin


def test_default_package_minimizes_private_evidence_and_requires_external_input(completed, completed_snapshot, tmp_path):
    root, pin = completed
    destination = tmp_path / 'portable'
    receipt = build_package(root, destination, pin)
    assert receipt['kind'] == 'portableFrozenEvaluation' and receipt['schemaVersion'] == 2
    assert not receipt['includesTestRows']
    assert not (destination / 'test-input.json').exists()
    assert not (destination / 'saved-predictions.jsonl').exists()
    summary = json.loads((destination / 'saved-summary.json').read_bytes())
    assert 'preview' not in summary and 'predictions' not in summary
    weights = torch.load(destination / 'weights.pt', weights_only=True)
    assert weights and all(isinstance(v, torch.Tensor) for v in weights.values())
    private = json.loads((root / 'input/training/artifacts/manifest.json').read_bytes())['provenance']
    exported = b''.join(p.read_bytes() for p in destination.iterdir() if p.suffix != '.pt')
    for ident in private['trainingRowIDs'] + private['validationRowIDs']:
        assert ident.encode() not in exported
    assert str(root).encode() not in exported
    assert completed_snapshot[2].encode() not in exported
    assert b'fitRowIDs' not in exported and b'optimizerState' not in exported
    validate_package(destination)
    with pytest.raises(ValueError, match='(?i)bundled|replay'):
        evaluate_package(destination, replay=True)


@pytest.mark.parametrize('include_rows',[False,True])
def test_segmented_frozen_evaluation_exports_and_replays_without_live_sources(tmp_path,include_rows):
    request,pin=evaluation_fixture.__wrapped__(tmp_path/'capture',segmented=True)
    execute_request(request,pin,lambda _:None)
    root=request.parent;manifest=root/'artifacts/manifest.json';destination=tmp_path/'package'
    build_package(root,destination,sha(manifest.read_bytes()),include_test_rows=include_rows)
    validate_package(destination)
    saved=json.loads((destination/'saved-summary.json').read_bytes())
    private=json.loads((root/'input/training/artifacts/manifest.json').read_bytes())['provenance']
    exported=b''.join(p.read_bytes() for p in destination.iterdir() if p.suffix!='.pt')
    # source:4 is a prefix of the explicitly exported Test identity source:45.
    # Check complete encoded identifiers, not substrings of other identities.
    for ident in private['trainingRowIDs']+private['validationRowIDs']:assert json.dumps(ident).encode() not in exported
    assert (destination/'test-input.json').exists() is include_rows
    shutil.rmtree(root)
    if include_rows:
        result=evaluate_package(destination,replay=True,batch_size=3,device='cpu')
        assert result['population']=='originalProtectedTest'
        assert result['evaluatedCount']==saved['evaluatedCount']==14
        assert result['lossMean']==pytest.approx(saved['lossMean'],abs=1e-6)
        assert result['confusionMatrix']==saved['confusionMatrix']
    else:
        with pytest.raises(ValueError,match='(?i)bundled|replay'):evaluate_package(destination,replay=True)


def test_image_schema_is_refused_before_publishing_a_scalar_portable_package(completed,tmp_path,monkeypatch):
    from runner import portable_evaluation_export as exporter
    original=exporter.object_json
    def captured_image_schema(raw):
        value=original(raw)
        if value.get('kind')=='table' and 'columns' in value:
            value['columns'].append({'name':'image_content','dtype':'image','nullable':True})
        return value
    # Exercise the actual builder's capture-to-package seam. The scalar runtime
    # must not publish a dependency-incomplete package for an image-bearing schema.
    monkeypatch.setattr(exporter,'object_json',captured_image_schema)
    root,pin=completed;destination=tmp_path/'image-package'
    with pytest.raises(ValueError,match='(?i)image.*export'):build_package(root,destination,pin)
    assert not destination.exists()


def test_optional_rows_relocate_and_replay_real_weights_uneven_batches(completed, tmp_path):
    root, pin = completed
    destination = tmp_path / 'portable'
    build_package(root, destination, pin, include_test_rows=True)
    saved = json.loads((destination / 'saved-summary.json').read_bytes())
    relocated = tmp_path / 'elsewhere' / 'package'
    relocated.parent.mkdir()
    shutil.move(destination, relocated)
    shutil.rmtree(root)
    result = evaluate_package(relocated, replay=True, batch_size=3, device='cpu')
    assert result['population'] == 'originalProtectedTest'
    assert result['evaluatedCount'] == saved['evaluatedCount'] == 14
    assert result['lossMean'] == pytest.approx(saved['lossMean'], abs=1e-6)
    assert result['confusionMatrix'] == saved['confusionMatrix']
    assert result['perClass'] == saved['perClass']
    run = subprocess.run([sys.executable, '-I', str(relocated / 'portable_evaluation.py'), '--replay', '--batch-size', '3', '--device', 'cpu'], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)['population'] == 'originalProtectedTest'


def test_explicit_external_table_is_not_original_test_claim(completed, tmp_path):
    root, pin = completed; destination = tmp_path / 'portable'
    build_package(root, destination, pin, include_test_rows=True)
    external = json.loads((destination / 'test-input.json').read_bytes())
    external['rows'] = external['rows'][:3]
    external['observationIDs'] = ['external-1', 'external-2', 'external-3']
    path = tmp_path / 'external.json'; path.write_text(json.dumps(external))
    result = evaluate_package(destination, input_path=path, batch_size=2, device='cpu')
    assert result['population'] == 'externalInput' and result['evaluatedCount'] == 3
    assert [p['observationID'] for p in result['predictions']] == external['observationIDs']


@pytest.mark.parametrize('file', ['weights.pt', 'model.py', 'fitted-parameters.json', 'test-input.json', 'saved-summary.json'])
def test_inventory_tampering_rejected_before_inference(completed, tmp_path, file):
    root, pin = completed; destination = tmp_path / 'portable'
    build_package(root, destination, pin, include_test_rows=True)
    with (destination / file).open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError): validate_package(destination)


def test_builder_rejects_changed_pins_and_existing_destination(completed, tmp_path):
    root, pin = completed; destination = tmp_path / 'portable'
    with pytest.raises(ValueError): build_package(root, destination, '0' * 64)
    assert not destination.exists()
    build_package(root, destination, pin)
    before = (destination / 'manifest.json').read_bytes()
    with pytest.raises((ValueError, FileExistsError)): build_package(root, destination, pin)
    assert (destination / 'manifest.json').read_bytes() == before


@pytest.mark.parametrize('mutation', ['symlink', 'extra', 'traversal'])
def test_package_paths_are_exact_bounded_and_non_symlink(completed, tmp_path, mutation):
    root, pin = completed; destination = tmp_path / 'portable'
    build_package(root, destination, pin)
    if mutation == 'symlink':
        content = destination / 'weights.pt'; target = tmp_path / 'weights.pt'
        content.rename(target); content.symlink_to(target)
    elif mutation == 'extra': (destination / 'unexpected').write_text('private')
    else:
        manifest = json.loads((destination / 'manifest.json').read_bytes())
        manifest['files'][0]['path'] = '../model.py'
        (destination / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)): validate_package(destination)


def reseal_package(destination, name, value):
    raw = json.dumps(value, allow_nan=False).encode()
    (destination / name).write_bytes(raw)
    manifest = json.loads((destination / 'manifest.json').read_bytes())
    entry = next(e for e in manifest['files'] if e['path'] == name)
    entry.update(bytes=len(raw), sha256=sha(raw))
    (destination / 'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('mutation', ['subset', 'reorder', 'row-width', 'fractional-target', 'fitted-prefix', 'fitted-state', 'count-bound'])
def test_resealed_package_cannot_change_protected_population_or_fitted_contract(completed, tmp_path, mutation):
    root, pin = completed; destination = tmp_path / 'portable'
    build_package(root, destination, pin, include_test_rows=True)
    filename = 'input-contract.json' if mutation in ('fitted-prefix', 'count-bound') else ('fitted-parameters.json' if mutation == 'fitted-state' else 'test-input.json')
    value = json.loads((destination / filename).read_bytes())
    if mutation == 'subset':
        value['rows'].pop(); value['observationIDs'].pop()
    elif mutation == 'reorder':
        value['rows'].reverse(); value['observationIDs'].reverse()
    elif mutation == 'row-width': value['rows'][0].append(1)
    elif mutation == 'fractional-target': value['rows'][0][-1] = 0.5
    elif mutation == 'fitted-prefix': value['appliedPrefix'] = ['0' * 64]
    elif mutation == 'fitted-state': value[0]['sourceContentDigest'] = '0' * 64
    else: value['testCount'] = 100001
    reseal_package(destination, filename, value)
    with pytest.raises(ValueError): validate_package(destination)


@pytest.mark.parametrize('name', ['request.json', 'input/snapshot.json', 'input/training/artifacts/checkpoint.pt', 'artifacts/summary.json', 'artifacts/predictions.jsonl'])
def test_builder_checks_captured_and_result_bytes(completed, tmp_path, name):
    root, pin = completed; destination = tmp_path / 'portable'
    with (root / name).open('ab') as stream: stream.write(b' ')
    with pytest.raises((ValueError, OSError)): build_package(root, destination, pin)
    assert not destination.exists()


def test_cancelled_builder_never_publishes_completed_package(completed, tmp_path):
    root, pin = completed; destination = tmp_path / 'portable'; calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 8: raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError): build_package(root, destination, pin, check_cancelled=cancel)
    assert not destination.exists()
    assert calls == 8


def captured_evaluation(tmp_path, preparation, model_graph, mapping, endpoint, **input_options):
    """Build the same pinned worker input from a real prepared/trained model."""
    import uuid
    from runner.engine import run
    from runner.prepared_data import load_prepared
    from runner.tests.test_frozen_evaluation_cli import write, ROOT
    prepared = load_prepared(preparation, *mapping, **input_options)
    trained = tmp_path / 'trained'; run(model_graph, trained, epochs=1, device='cpu', prepared=prepared)
    root = tmp_path / 'evaluation'; inputs = root / 'input'; inputs.mkdir(parents=True)
    shutil.copytree(preparation.parent, inputs / 'preparation/artifacts')
    (inputs / 'training/artifacts').mkdir(parents=True)
    for name in ('manifest.json', 'checkpoint.pt'): shutil.copyfile(trained / name, inputs / 'training/artifacts' / name)
    shutil.copyfile(ROOT / 'Sources/KwipMLCore/Resources/components.json', inputs / 'model-registry.json')
    shutil.copyfile(ROOT / 'runner/dataflow-operators.json', inputs / 'data-registry.json')
    parent = dict(projectID=str(uuid.uuid4()).upper(), directoryID=str(uuid.uuid4()).upper(), runID=str(uuid.uuid4()).upper(), requestIdentity='a'*64,
        modelManifestSHA256=sha((trained/'manifest.json').read_bytes()), preparationManifestSHA256=sha(preparation.read_bytes()),checkpointSHA256=sha((trained/'checkpoint.pt').read_bytes()))
    write(inputs/'capture.json', {k:parent[k] for k in ('projectID','directoryID','runID','requestIdentity')})
    write(inputs/'workflow.json', {'schemaVersion':5});write(inputs/'runtime-evidence.json', {'schemaVersion':1})
    parent['outcomeSHA256']=sha(write(inputs/'outcome.json',dict(status='completed',**parent)))
    entries=[dict(path=str(p.relative_to(inputs)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(inputs.rglob('*')) if p.is_file()]
    receipt=write(inputs/'snapshot.json',dict(schemaVersion=1,reference=parent,testEndpoint=dict(node=endpoint[0],port=endpoint[1]),files=entries))
    raw=write(root/'request.json',dict(schemaVersion=1,kind='frozenEvaluation',evaluationID=str(uuid.uuid4()).upper(),projectID=parent['projectID'],requestIdentity='b'*64,snapshotReceiptSHA256=sha(receipt),settings=dict(device='cpu',batchSize=5,previewLimit=2)))
    execute_request(root/'request.json',sha(raw),lambda _:None)
    return root,sha((root/'artifacts/manifest.json').read_bytes())


def columns_config(columns, *, missing='error', rename=None):
    return dict(rows=[dict(source=name,included=True,outputName=(rename or {}).get(name,name),targetType=dtype)
                      for name,dtype in columns],castFailure='error',trimText=False,newColumns='drop',missingPolicy=missing)


def time_partition_config():
    # Preserve the original scientific populations: times0..30 /31..45 /46..59.
    return dict(strategy='time',timeColumn='time',timeEncoding='epochMilliseconds',trainingEnd=30,
                validationEnd=45,embargoMilliseconds=0,entityColumns=[],entityPolicy='allowAcrossTime')


@pytest.mark.parametrize('strategy,prefix,rename', [('mean',0,False),('median',1,False),('mostFrequent',3,False),('mean',3,True)])
def test_full_fitted_chain_and_selected_prefix_never_refit(tmp_path, monkeypatch, strategy, prefix, rename):
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_runner import graph
    from runner import dataflow_fitted, dataflow_preprocess
    source=tmp_path/'PRIVATE-SOURCE.csv'
    source.write_text('time,a,b,kind,note,label\n'+''.join(f'{i},{"" if i%7==0 else i%9},{i},{"red" if i%2 else "blue"},{"PRIVATE-TRAIN-ONLY" if i<=30 else "held"},{i%2}\n' for i in range(60)))
    from runner.preprocess_parameters import rule_defaults
    # Author current tools directly. Each fitted stage owns all three role ports;
    # no legacy Fit/Transform execution or automatic migration is exercised here.
    columns=[('time','int64'),('a','float64'),('b','int64'),('kind','string'),('note','string'),('label','int64')]
    nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={}),
           dict(id='columns',operator='kwip.data.columns',config=columns_config(columns,missing='preserve')),
           dict(id='split',operator='kwip.data.partition',config=time_partition_config())]
    def edge(a,ap,b,bp='table'):return {'from':{'node':a,'port':ap},'to':{'node':b,'port':bp}}
    edges=[edge('source','table','columns'),edge('columns','table','split')]
    previous={r:('split',r) for r in ('training','validation','test')};selected=previous['test']
    rules=[
        [dict(rule_defaults('a'),missing=strategy,tiePolicy='smallest' if strategy=='mostFrequent' else 'error')],
        [dict(rule_defaults(c),scaling='standard') for c in ['renamed' if rename else 'a','b']],
        [dict(rule_defaults('kind'),encoding='oneHot',unknownPolicy='ignore',maxCategories=10)],
    ]
    for index,column_rules in enumerate(rules):
        fit='fit'+str(index)
        nodes.append(dict(id=fit,operator='kwip.data.preprocess',config={'mode':'partitions','columns':column_rules}))
        edges.extend(edge(*previous[role],fit,role) for role in previous)
        previous={role:(fit,role) for role in previous}
        if index==0 and rename:
            for role in previous:
                name=role+'Rename'
                nodes.append(dict(id=name,operator='kwip.data.columns',config=columns_config(columns,rename={'a':'renamed'})))
                edges.append(edge(*previous[role],name));previous[role]=(name,'table')
        if index+1==prefix:selected=previous['test']
    flow=tmp_path/'flow.json';flow.write_text(json.dumps(dict(schemaVersion=1,nodes=nodes,edges=edges)))
    preparation=tmp_path/'preparation';run_dataflow(flow,{'source':str(source)},preparation,lambda _:None)
    mapping=(*previous['training'],*previous['validation'],['b','onehot:["kind","red"]'],'label')
    root,pin=captured_evaluation(tmp_path,preparation/'manifest.json',graph(),mapping,selected)
    source.unlink()
    refuse_fit=lambda *a,**k:pytest.fail('Export/replay must never refit')
    monkeypatch.setattr(dataflow_fitted,'fit',refuse_fit)
    monkeypatch.setattr(dataflow_preprocess,'_fit',refuse_fit)
    destination=tmp_path/'portable';build_package(root,destination,pin,include_test_rows=True)
    contract=json.loads((destination/'input-contract.json').read_bytes());assert len(contract['appliedPrefix'])==prefix
    parameters=json.loads((destination/'fitted-parameters.json').read_bytes())
    assert len(parameters)==3
    assert all(p['operator']=='kwip.data.preprocess' for p in parameters)
    # Independent source-value oracle: learned replacement and population scales
    # use only times0..30, even when the selected test already has a fitted prefix.
    import numpy as np
    from collections import Counter
    observed=[i%9 for i in range(31) if i%7]
    fill=(sum(observed)/len(observed) if strategy=='mean' else
          float(np.median(observed)) if strategy=='median' else
          min(v for v,count in Counter(observed).items() if count==max(Counter(observed).values())))
    assert parameters[0]['rules'][0]['fill']==pytest.approx(fill)
    filled=np.array([fill if i%7==0 else i%9 for i in range(31)])
    for rule,values in zip(parameters[1]['rules'],[filled,np.arange(31)]):
        assert rule['mean']==pytest.approx(float(values.mean()))
        assert rule['scale']==pytest.approx(float(values.std()))
    assert parameters[2]['rules'][0]['vocabulary']==['blue','red']
    from runner.prepared_data import load_prepared
    tensors,provenance,_=load_prepared(preparation/'manifest.json',*mapping)
    for x,y,times in [(tensors[0],tensors[1],range(31)),(tensors[2],tensors[3],range(31,46))]:
        expected=torch.tensor([[(i-15)/float(np.arange(31).std()),float(i%2)] for i in times])
        torch.testing.assert_close(x,expected)
        assert y.tolist()==[i%2 for i in times]
    assert len(provenance['trainingRowIDs'])==31 and len(provenance['validationRowIDs'])==15
    raw=b''.join(p.read_bytes() for p in destination.iterdir() if p.suffix!='.pt')
    assert b'PRIVATE-TRAIN-ONLY' not in raw and b'PRIVATE-SOURCE' not in raw
    result=evaluate_package(destination,replay=True,batch_size=3);assert result['evaluatedCount']==14
    saved=json.loads((root/'artifacts/summary.json').read_bytes())
    assert result['lossMean']==pytest.approx(saved['lossMean'],abs=1e-6)
    assert result['confusionMatrix']==saved['confusionMatrix']


def test_integer_embedding_topology_preserves_exact_int64_tables(tmp_path):
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_embedding_graph import graph
    source=tmp_path/'indices.csv';source.write_text('time,a,b,boundary,label\n'+''.join(f'{i},{i%5},{(i+1)%5},{2**63-1 if i%2 else -2**63},{i%2}\n' for i in range(60)))
    columns=[(c,'int64') for c in ['time','a','b','boundary','label']]
    nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={}),
           dict(id='columns',operator='kwip.data.columns',config=columns_config(columns)),
           dict(id='split',operator='kwip.data.partition',config=time_partition_config())]
    edges=[{'from':{'node':a,'port':'table'},'to':{'node':b,'port':'table'}} for a,b in [('source','columns'),('columns','split')]]
    flow=tmp_path/'flow.json';flow.write_text(json.dumps(dict(schemaVersion=1,nodes=nodes,edges=edges)))
    preparation=tmp_path/'preparation';run_dataflow(flow,{'source':str(source)},preparation,lambda _:None)
    from runner.prepared_data import load_prepared
    tensors,_,_=load_prepared(preparation/'manifest.json','split','training','split','validation',['b','a'],'label',input_dtype='int64',sample_shape=[2])
    for x,y,times in [(tensors[0],tensors[1],range(31)),(tensors[2],tensors[3],range(31,46))]:
        assert x.dtype==torch.int64
        assert x.tolist()==[[(i+1)%5,i%5] for i in times]
        assert y.tolist()==[i%2 for i in times]
    root,pin=captured_evaluation(tmp_path,preparation/'manifest.json',graph(),('split','training','split','validation',['b','a'],'label'),('split','test'),input_dtype='int64',sample_shape=[2])
    destination=tmp_path/'portable';build_package(root,destination,pin,include_test_rows=True)
    table=json.loads((destination/'test-input.json').read_bytes());assert {r[3] for r in table['rows']}=={-(2**63),2**63-1}
    assert json.loads((destination/'input-contract.json').read_bytes())['dtype']=='int64'
    shutil.rmtree(root);source.unlink()
    assert evaluate_package(destination,replay=True,batch_size=3)['evaluatedCount']==14


@pytest.mark.parametrize('phase', ['files', 'manifest'])
def test_cancellation_during_publication_cleans_only_owned_destination(completed,tmp_path,phase):
    root,pin=completed;destination=tmp_path/'portable';reached=False
    def cancel():
        nonlocal reached
        trigger=(destination/'manifest.json').exists() if phase=='manifest' else destination.exists() and any(destination.iterdir())
        if trigger:reached=True;raise InterruptedError('cancelled publication')
    with pytest.raises(InterruptedError):build_package(root,destination,pin,check_cancelled=cancel)
    assert reached and not destination.exists()


@pytest.mark.parametrize('field,value', [('evaluatedCount',1),('device','cuda'),('batchSize',0),('lossMean',-1),('lossSum',1e309),('accuracy',2),('confusionMatrix',[[1,0],[0,0]]),('perClass',[])])
def test_default_aggregate_summary_is_validated_without_optional_rows(completed,tmp_path,field,value):
    root,pin=completed;destination=tmp_path/'portable';build_package(root,destination,pin)
    summary=json.loads((destination/'saved-summary.json').read_bytes());summary[field]=value
    if value==float('inf'):
        raw=json.dumps(summary).encode();(destination/'saved-summary.json').write_bytes(raw)
        manifest=json.loads((destination/'manifest.json').read_bytes());entry=next(e for e in manifest['files'] if e['path']=='saved-summary.json');entry.update(bytes=len(raw),sha256=sha(raw));(destination/'manifest.json').write_text(json.dumps(manifest))
    else:reseal_package(destination,'saved-summary.json',summary)
    with pytest.raises(ValueError):validate_package(destination)


def test_saved_cross_entropy_is_stable_under_large_common_logit_offset():
    from runner.final_evaluation import evaluate_classifier
    from runner.portable_evaluation import predictions_valid, encoded
    class Offset(torch.nn.Module):
        def forward(self,x):return torch.full((len(x),2),1e20,dtype=torch.float32)
    result=evaluate_classifier(Offset(),torch.ones(3,2),torch.tensor([0,1,1]),['one','two','three'],batch_size=2,device='cpu',preview_limit=0)
    assert result['lossMean']==pytest.approx(0.6931471805599453)
    summary={k:v for k,v in result.items() if k not in ('preview','predictions')}
    raw=b''.join(encoded(row)+b'\n' for row in result['predictions'])
    assert predictions_valid(raw,summary)==result['predictions']


@pytest.mark.parametrize('field,value', [('classIndex',False),('classIndex',0.0),('support',False),('precision',False)])
def test_per_class_fields_require_numeric_types_not_boolean_equivalence(field,value):
    from runner.final_evaluation import evaluate_classifier
    from runner.portable_evaluation import summary_valid
    class Positive(torch.nn.Module):
        def forward(self,x):return torch.tensor([[0.,1.]]).repeat(len(x),1)
    result=evaluate_classifier(Positive(),torch.ones(1,2),torch.tensor([1]),['one'],batch_size=1,device='cpu',preview_limit=0)
    summary={k:v for k,v in result.items() if k not in ('preview','predictions')}
    # class0 has zero support, classIndex0, unavailable precision. Use recall
    # or precision on class1 (one) where bool True equals1, to isolate typing.
    index=0
    if field=='precision':index=1;value=True
    summary['perClass'][index][field]=value
    with pytest.raises(ValueError):summary_valid(summary)


@pytest.mark.skipif(sys.platform!='darwin',reason='Darwin sandbox directory authority')
def test_directory_darwin_opens_leaf_without_ancestor_read_permission(tmp_path,monkeypatch):
    import os
    from runner import portable_evaluation as portable
    leaf=tmp_path.resolve()/'allowed';leaf.mkdir()
    original=os.open;calls=[]
    def restricted(path,flags,*args,**kwargs):
        calls.append((os.fspath(path),flags,kwargs))
        if os.fspath(path)!=str(leaf) or kwargs.get('dir_fd') is not None:
            raise PermissionError(1,'Ancestor read permission intentionally denied',os.fspath(path))
        return original(path,flags,*args,**kwargs)
    monkeypatch.setattr(portable.os,'open',restricted)
    fd=portable.directory(leaf)
    try:
        assert os.fstat(fd).st_ino==leaf.stat().st_ino
        assert calls==[(str(leaf),os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC|os.O_NOFOLLOW_ANY,{})]
        assert not os.get_inheritable(fd)
    finally:os.close(fd)


@pytest.mark.skipif(sys.platform!='darwin',reason='Darwin no-follow-any traversal')
def test_directory_darwin_rejects_actual_intermediate_symlink(tmp_path):
    import os
    from runner.portable_evaluation import directory
    real=tmp_path.resolve()/'real';(real/'leaf').mkdir(parents=True)
    alias=tmp_path.resolve()/'alias';alias.symlink_to(real,target_is_directory=True)
    with pytest.raises(OSError):directory(alias/'leaf')
    fd=directory(real/'leaf')
    try:assert os.fstat(fd).st_ino==(real/'leaf').stat().st_ino
    finally:os.close(fd)


def test_new_evaluation_and_portable_manifests_mark_corrected_device_contract(completed,tmp_path):
    root,pin=completed
    assert json.loads((root/'artifacts/manifest.json').read_bytes())['schemaVersion']==3
    receipt=build_package(root,tmp_path/'portable',pin)
    assert receipt['schemaVersion']==2


def legacy_evaluation(root,device):
    path=root/'artifacts/manifest.json';manifest=json.loads(path.read_bytes());manifest['schemaVersion']=1;manifest['runtime']['device']=device
    manifest['artifacts']=[v for v in manifest['artifacts'] if v['path']!='prediction-table.json']
    summary=root/'artifacts/summary.json';value=json.loads(summary.read_bytes());value['device']=device;raw=json.dumps(value).encode();summary.write_bytes(raw)
    entry=next(v for v in manifest['artifacts'] if v['path']=='summary.json');entry.update(bytes=len(raw),sha256=sha(raw))
    raw=json.dumps(manifest).encode();path.write_bytes(raw);return sha(raw)


def test_prior_mps_evaluation_is_explicitly_refused_without_rewriting(completed,tmp_path):
    root,_=completed;pin=legacy_evaluation(root,'mps');before=(root/'artifacts/manifest.json').read_bytes()
    with pytest.raises(ValueError,match='(?i)(earlier|prior|legacy).*MPS.*(evaluat|conversion)'):
        build_package(root,tmp_path/'portable',pin)
    assert (root/'artifacts/manifest.json').read_bytes()==before and not (tmp_path/'portable').exists()


def test_prior_cpu_evaluation_remains_readable_and_exports_corrected_package(completed,tmp_path):
    root,_=completed;pin=legacy_evaluation(root,'cpu')
    receipt=build_package(root,tmp_path/'portable',pin,include_test_rows=True)
    assert receipt['schemaVersion']==2
    assert evaluate_package(tmp_path/'portable',replay=True)['evaluatedCount']==14


def test_legacy_portable_mps_is_flagged_but_legacy_cpu_stays_readable(completed,tmp_path):
    root,pin=completed;destination=tmp_path/'portable';build_package(root,destination,pin)
    path=destination/'manifest.json';manifest=json.loads(path.read_bytes());manifest['schemaVersion']=1;path.write_text(json.dumps(manifest))
    validate_package(destination)
    summary=json.loads((destination/'saved-summary.json').read_bytes());summary['device']='mps';reseal_package(destination,'saved-summary.json',summary)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='(?i)(earlier|prior|legacy).*MPS.*(evaluat|conversion)'):
        validate_package(destination)
    assert path.read_bytes()==before

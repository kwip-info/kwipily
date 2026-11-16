import copy
import hashlib
import json


def test_shared_kernel_source_changes_invalidate_materializations(monkeypatch):
    from pathlib import Path
    from runner.dataflow_cache import implementation_digest
    before=implementation_digest();read=Path.read_bytes
    def changed(path):
        content=read(path)
        return content+b'\n# changed shared kernel\n' if path.name=='dataflow_shared.py' else content
    monkeypatch.setattr(Path,'read_bytes',changed)
    assert implementation_digest()!=before
from pathlib import Path
import pytest
from runner.dataflow_cli import run_dataflow

BASE=Path(__file__).resolve().parents[1]/'dataflow_examples'

def prepare(tmp_path,policy='materialize',prior=None,graph=None,emit=lambda e:None):
    graph_path=BASE/'preparation.json'
    if graph is not None:
        graph_path=tmp_path.with_suffix('.graph.json');graph_path.write_text(json.dumps(graph))
    options={'cache_policy':policy}
    if prior:options.update(reuse_manifest=prior/'manifest.json',reuse_manifest_sha256=hashlib.sha256((prior/'manifest.json').read_bytes()).hexdigest())
    return run_dataflow(graph_path,{'source':str(BASE/'observations.csv')},tmp_path,emit,**options)


def test_reuse_skips_real_fitting_and_preserves_all_artifact_outputs(tmp_path,monkeypatch):
    first=tmp_path/'first';a=prepare(first)
    def forbidden(*a,**k):raise AssertionError('Fitted operator executed on cache hit')
    monkeypatch.setattr('runner.dataflow_preprocess._fit',forbidden)
    second=tmp_path/'second';b=prepare(second,'reuse',first)
    assert b['cache']['executedNodeIDs']==['source']
    assert len(b['cache']['reusedNodeIDs'])==6
    assert a['artifacts']==b['artifacts']


def test_changed_configuration_invalidates_descendants_but_preserves_prefix(tmp_path):
    first=tmp_path/'first';prepare(first)
    graph=json.loads((BASE/'preparation.json').read_text())
    next(n for n in graph['nodes'] if n['id']=='prepare')['config']['columns'][0]['scaling']='none'
    result=prepare(tmp_path/'second','reuse',first,graph)
    assert 'columns' in result['cache']['reusedNodeIDs'] and 'partition' in result['cache']['reusedNodeIDs']
    assert 'prepare' in result['cache']['executedNodeIDs']
    assert 'training' in result['cache']['executedNodeIDs'] and 'validation' in result['cache']['executedNodeIDs']


def test_actual_source_change_invalidates_all_and_does_not_scan_other_sources(tmp_path):
    first=tmp_path/'first';prepare(first)
    source=tmp_path/'changed.csv';source.write_text((BASE/'observations.csv').read_text().replace('0.0,0.5','0.1,0.5'))
    result=run_dataflow(BASE/'preparation.json',{'source':str(source)},tmp_path/'next',lambda e:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest())
    assert result['cache']['reusedNodeIDs']==[]


def test_corrupt_reused_artifact_fails_without_publishing_success(tmp_path):
    first=tmp_path/'first';manifest=prepare(first)
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='columns')
    (first/entry['path']).write_text('{}')
    events=[];output=tmp_path/'next'
    with pytest.raises(ValueError,match='cache.*integrity|Cache.*integrity'):prepare(output,'reuse',first,emit=events.append)
    assert not output.exists() and not any(e['event']=='dataflowCompleted' for e in events)


def test_cancelled_materialization_never_becomes_reusable(tmp_path):
    output=tmp_path/'cancelled'
    def cancel(event):
        if event['event']=='dataflowPreview':raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):prepare(output,emit=cancel)
    assert not output.exists() and not list(tmp_path.glob('.dataflow-*'))


def test_reuse_requires_explicit_pin_and_current_cache_version(tmp_path):
    first=tmp_path/'first';prepare(first)
    with pytest.raises(ValueError,match='pin'):run_dataflow(BASE/'preparation.json',{'source':str(BASE/'observations.csv')},tmp_path/'bad',lambda e:None,cache_policy='reuse',reuse_manifest=first/'manifest.json')
    manifest=json.loads((first/'manifest.json').read_text());manifest['cache']['implementationDigest']='0'*64;(first/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='stale'):prepare(tmp_path/'stale','reuse',first)


def test_cancellation_during_completion_delivery_removes_success_manifest(tmp_path):
    output=tmp_path/'cancelled-final'
    def cancel(event):
        if event['event']=='dataflowCompleted':raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):prepare(output,emit=cancel)
    assert not (output/'manifest.json').exists()


def test_cache_pin_symlink_and_resource_bounds(tmp_path):
    from runner.dataflow_cache import Materialization
    first=tmp_path/'first';manifest=prepare(first);original=(first/'manifest.json').read_bytes()
    with pytest.raises(ValueError,match='integrity'):Materialization('reuse',first/'manifest.json','0'*64)
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='columns')
    file=first/entry['path'];saved=first/'saved-artifact.json';file.rename(saved);file.symlink_to(saved)
    with pytest.raises(ValueError,match='regular file'):prepare(tmp_path/'linked','reuse',first)
    oversized=copy.deepcopy(manifest);oversized['artifacts'][0]['bytes']=50_000_001
    (first/'manifest.json').write_text(json.dumps(oversized))
    with pytest.raises(ValueError,match='size bound'):prepare(tmp_path/'oversized','reuse',first)


def test_cache_disabled_never_reuses_and_unmaterialized_runs_are_rejected(tmp_path):
    first=tmp_path/'first';result=prepare(first,'off')
    assert result['cache']['reusedNodeIDs']==[] and result['cache']['nodes']==[]
    with pytest.raises(ValueError,match='not materialized|no reusable outputs'):prepare(tmp_path/'next','reuse',first)


def test_equivalent_parent_configuration_change_still_invalidates_descendants(tmp_path):
    first=tmp_path/'first';prepare(first)
    graph=json.loads((BASE/'preparation.json').read_text())
    next(n for n in graph['nodes'] if n['id']=='source')['config']={}
    result=prepare(tmp_path/'next','reuse',first,graph)
    assert result['cache']['reusedNodeIDs']==[]
    assert 'columns' in result['cache']['executedNodeIDs'] and 'partition' in result['cache']['executedNodeIDs']


def test_identical_explicit_source_at_new_path_can_reuse_without_path_leak(tmp_path):
    first=tmp_path/'first';prepare(first)
    source=tmp_path/'PRIVATE-source-name.csv';source.write_bytes((BASE/'observations.csv').read_bytes())
    out=tmp_path/'next'
    result=run_dataflow(BASE/'preparation.json',{'source':str(source)},out,lambda e:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest())
    assert result['cache']['executedNodeIDs']==['source']
    assert 'PRIVATE-source-name' not in (out/'manifest.json').read_text()+(out/'prep.py').read_text()


def test_rehashed_cache_with_wrong_partition_role_is_not_trusted(tmp_path):
    first=tmp_path/'first';manifest=prepare(first)
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='columns')
    table=json.loads((first/entry['path']).read_text());table['lineage']['role']='validation'
    content=json.dumps(table).encode();(first/entry['path']).write_bytes(content)
    entry.update(sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))
    manifest['cache']['materializedBytes']=sum(e['bytes'] for e in manifest['artifacts'] if e['kind'] in ('table','state'))
    (first/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='partition lineage'):prepare(tmp_path/'next','reuse',first)


def test_materialization_budget_includes_exported_support_code(tmp_path,monkeypatch):
    import runner.dataflow_cli as adapter
    first=prepare(tmp_path/'first')
    data_bytes=sum(entry['bytes'] for entry in first['artifacts'] if entry['kind'] not in ('preparationCode','registry'))
    monkeypatch.setattr(adapter,'MAX_ARTIFACT_BYTES',data_bytes+1)
    with pytest.raises(ValueError,match='size limit|index artifact allowance'):prepare(tmp_path/'too-large')
    assert not (tmp_path/'too-large').exists()

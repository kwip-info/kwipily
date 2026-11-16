"""Cooperative preparation cancellation must not publish a usable result."""
import json
from pathlib import Path
import pytest
from runner import dataflow_cli
from runner.tests.test_dataflow import fixture

class PreparationCancelled(Exception):
    pass

def setup_flow(tmp_path):
    graph, bindings = fixture(tmp_path)
    path = tmp_path/'graph.json'
    path.write_text(json.dumps(graph))
    return path, {key: str(value) for key, value in bindings.items()}

def test_cancel_before_loading_graph_or_cache(tmp_path, monkeypatch):
    path, bindings = setup_flow(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail('cancelled preparation loaded graph/cache')
    monkeypatch.setattr(dataflow_cli, 'read_json', unexpected)
    def check(): raise PreparationCancelled()
    with pytest.raises(PreparationCancelled):
        dataflow_cli.run_dataflow(path, bindings, tmp_path/'out', lambda _: None, check_cancelled=check)
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('boundary', ['node', 'rename'])
def test_cancel_stops_nodes_and_invalidates_post_rename_manifest(tmp_path, monkeypatch, boundary):
    path, bindings = setup_flow(tmp_path)
    events=[]; cancelled=False
    def emit(event):
        nonlocal cancelled
        events.append(event)
        if boundary=='node' and event['event']=='dataflowPreview': cancelled=True
    original = dataflow_cli.os.replace
    def replace(*args, **kwargs):
        nonlocal cancelled
        original(*args, **kwargs)
        if boundary=='rename': cancelled=True
    monkeypatch.setattr(dataflow_cli.os, 'replace', replace)
    def check():
        if cancelled: raise PreparationCancelled()
    with pytest.raises(PreparationCancelled):
        dataflow_cli.run_dataflow(path, bindings, tmp_path/'out', emit, check_cancelled=check)
    assert not (tmp_path/'out/manifest.json').exists()
    assert not any(e['event']=='dataflowCompleted' for e in events)
    assert not list(tmp_path.glob('.dataflow-*'))
    if boundary=='node': assert len([e for e in events if e['event']=='dataflowPreview'])==1

def test_cli_passes_retained_authority_into_preparation(tmp_path, monkeypatch):
    import sys
    from runner import main as cli
    path, bindings = setup_flow(tmp_path)
    binding_path=tmp_path/'bindings.json';binding_path.write_text(json.dumps(bindings))
    monkeypatch.syspath_prepend(str(Path(cli.__file__).parent))
    monkeypatch.setattr(sys,'argv',['main.py','--execute-dataflow',str(path),'--source-bindings',str(binding_path),'--output',str(tmp_path/'out')])
    monkeypatch.setattr(cli.signal,'signal',lambda *args:None)
    events=[];monkeypatch.setattr(cli,'emit',events.append)
    def check():
        if any(e['event']=='dataflowPreview' for e in events): raise cli.Cancelled()
    assert cli.main(check_cancelled=check)==130
    assert not (tmp_path/'out/manifest.json').exists()
    assert not any(e['event']=='dataflowCompleted' for e in events)

"""Cancellation must prevent inference work and completed-preview publication."""
import pytest
from runner.preview import preview
from runner.tests.test_runner import graph

class PreviewCancelled(Exception):
    pass

def test_cancelled_preview_does_not_load_data_or_create_output(tmp_path, monkeypatch):
    import runner.preview as module
    def unexpected(*args, **kwargs):
        pytest.fail('cancelled preview loaded data')
    monkeypatch.setattr(module, 'load_data', unexpected)
    def check():
        raise PreviewCancelled('cancelled')
    with pytest.raises(PreviewCancelled):
        preview(graph(), tmp_path/'out', device='cpu', check_cancelled=check)
    assert not (tmp_path/'out').exists()

def test_cancel_after_component_stops_before_next_and_never_completes(tmp_path):
    executed=[];events=[]
    def check():
        if executed:raise PreviewCancelled('cancelled after one component')
    with pytest.raises(PreviewCancelled):
        preview(graph(),tmp_path/'out',device='cpu',check_cancelled=check,on_execute=executed.append,emit=events.append)
    assert executed==['input']
    assert not (tmp_path/'out/preview.json').exists()
    assert not any(event['event']=='previewCompleted' for event in events)

def test_cancel_during_cache_write_never_publishes_completion(tmp_path,monkeypatch):
    import runner.preview as module
    original=module.torch.save;cancelled=False;events=[]
    def save(*args,**kwargs):
        nonlocal cancelled
        original(*args,**kwargs);cancelled=True
    def check():
        if cancelled:raise PreviewCancelled('cancelled while serializing')
    monkeypatch.setattr(module.torch,'save',save)
    with pytest.raises(PreviewCancelled):
        preview(graph(),tmp_path/'out',device='cpu',check_cancelled=check,emit=events.append)
    assert not (tmp_path/'out/preview.json').exists()
    assert not any(event['event']=='previewCompleted' for event in events)


def test_cli_passes_cancellation_to_preview_and_returns_cancelled(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from runner import main as cli
    monkeypatch.syspath_prepend(str(Path(cli.__file__).parent))
    source=tmp_path/'graph.json';source.write_text(json.dumps(graph()))
    monkeypatch.setattr(sys,'argv',['main.py','--preview','--stop-at','output','--device','cpu','--graph',str(source),'--output',str(tmp_path/'out')])
    monkeypatch.setattr(cli.signal,'signal',lambda *args:None)
    events=[]
    monkeypatch.setattr(cli,'emit',events.append)
    def check():
        if any(event['event']=='previewStarted' for event in events):raise cli.Cancelled('cancelled at first component')
    assert cli.main(check_cancelled=check)==130
    assert events[-1]['event']=='error'
    assert not any(event['event']=='previewCompleted' for event in events)
    assert not (tmp_path/'out/preview.json').exists()

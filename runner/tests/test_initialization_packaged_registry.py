"""Packaged lessons must use the supplied registry, never development paths."""
import json
from pathlib import Path
import pytest
from runner import engine,learning_initialization
from runner.tests.test_initialization_learning_evidence import TASKS,measured

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('task',TASKS)
def test_real_dispatch_uses_captured_registry_without_development_sources(tmp_path,monkeypatch,task):
    captured=tmp_path/'captured-components.json'
    captured.write_bytes((ROOT/'Sources/KwipMLCore/Resources/components.json').read_bytes())
    monkeypatch.setattr(engine,'REGISTRY',tmp_path/'runtime/Sources/KwipMLCore/Resources/components.json')
    observed=[];original=learning_initialization.GraphModel
    def model(*args,**kwargs):
        observed.append(kwargs.get('registry'))
        return original(*args,**kwargs)
    monkeypatch.setattr(learning_initialization,'GraphModel',model)
    result=measured(task,registry=captured)
    assert not result.get('backendError'),result
    assert observed==[captured]*4 and len(result['fixtures'])==4
    for entry in result['fixtures']:
        assert set(entry['tensors'])=={'startingSettings','predictions','firstAdjustment','repeatIsolation','savedReuse'}
        for name,tensor in entry['tensors'].items():
            assert tensor['values']==pytest.approx([0]*len(tensor['values']),abs=0 if name in ('startingSettings','repeatIsolation') else 2e-6)


def test_captured_registry_missing_operator_never_uses_development_fallback(tmp_path):
    value=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())
    value['components']=[c for c in value['components'] if c['id']!='leaky_relu']
    captured=tmp_path/'captured-components.json';captured.write_text(json.dumps(value))
    result=measured(TASKS[0],registry=captured)
    assert result.get('backendError') and not result['fixtures']

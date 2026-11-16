"""Demand flags cannot disappear through an unrelated mode's early return."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

@pytest.mark.parametrize('mode', [['--capabilities'], ['--inspect-dataflow','missing.json'], ['--dataflow-registry'], ['--capabilities','--execute-dataflow','missing.json']])
def test_demand_flags_reject_other_modes_before_action(mode):
    result = subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),*mode,
        '--dataflow-complete-nodes','[]'],capture_output=True,text=True,timeout=30)
    assert result.returncode != 0
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]['event']=='error'
    assert 'require only --execute-dataflow mode' in events[-1]['message']
    assert not any(item.get('event') in ('capabilities','dataflowPreview') for item in events)

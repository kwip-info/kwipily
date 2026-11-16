"""Worker options must have the same unambiguous spelling as launch intents."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('arguments', [
    ['--capabil'],
    ['--capabilities', '--out=unused'],
    ['--capabilities', '--grap=unused'],
])
def test_abbreviated_options_are_rejected(arguments):
    result = subprocess.run([sys.executable, str(ROOT / 'runner/main.py'), *arguments],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    event = json.loads(result.stdout)
    assert event['event'] == 'error'
    assert 'unrecognized arguments' in event['message']

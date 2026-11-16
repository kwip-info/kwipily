"""Exported metrics must run without the authoring engine or repository."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest


@pytest.mark.parametrize("named",[False,True])
def test_metric_helper_runs_from_relocated_standalone_files(tmp_path,named):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / 'final_evaluation.py', tmp_path / 'final_evaluation.py')
    device = root / 'execution_device.py'
    if device.exists():
        shutil.copyfile(device, tmp_path / device.name)
    program = '''
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
from final_evaluation import evaluate_classifier
class NamedIdentity(torch.nn.Module):
    def forward(self,x):return x['scores']
named=sys.argv[2]=='True'
x=torch.tensor([[2., 0.], [0., 2.], [0., 0.]])
result = evaluate_classifier(NamedIdentity() if named else torch.nn.Identity(), {'scores':x} if named else x, torch.tensor([0, 1, 1]), ['a', 'b', 'c'], batch_size=2, device='cpu', preview_limit=0)
assert 'engine' not in sys.modules and 'runner.engine' not in sys.modules
print(json.dumps(result))
'''
    process = subprocess.run([sys.executable, '-I', '-c', program, str(tmp_path),str(named)], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result['confusionMatrix'] == [[1, 0], [1, 1]]
    assert result['evaluatedCount'] == 3 and result['preview'] == []
    assert [row['observationID'] for row in result['predictions']] == ['a', 'b', 'c']

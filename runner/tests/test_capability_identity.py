import torch
from runner.engine import capabilities


def test_capabilities_reports_exact_loaded_torch_version():
    result = capabilities()
    assert result['torchVersion'] == str(torch.__version__)
    assert result['defaultDevice'] in result['devices']

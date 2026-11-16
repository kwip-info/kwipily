"""Exercise preview infrastructure with a real two-tensor sorting operation.

The test adapter is not a registered product component or pooling coverage claim.
"""
import json
import pytest
import torch
from runner import preview as module
from runner.engine import GraphModel
from runner.tests.test_runner import graph


class SortingOutputs(GraphModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.layouts['relu']={'out':self.layouts['relu'],'indices':self.layouts['relu']}
        self.dtypes['relu']={'out':'float32','indices':'int64'}
    def execute_node(self,ident,values,x):
        result=super().execute_node(ident,values,x)
        if ident=='relu':
            sorted_values,indices=torch.sort(result,dim=-1)
            return {'out':sorted_values,'indices':indices}
        return result


def test_preview_roundtrip_preserves_named_outputs_and_resumes(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'GraphModel',SortingOutputs)
    first=module.preview(graph(),tmp_path/'first',device='cpu')
    assert set(first['previews']['relu']['outputs'])=={'out','indices'}
    assert first['previews']['relu']['outputs']['indices']['dtype']=='int64'
    meta=json.loads((tmp_path/'first/activation-cache.json').read_text())
    assert meta['schemaVersion']==3
    payload=torch.load(first['cachePath'],weights_only=True)
    assert payload['activations']['relu']['indices'].dtype==torch.int64
    second=module.preview(graph(),tmp_path/'second',device='cpu',resume_from='merge',cache=first['cachePath'])
    assert second['previews']==first['previews']
    assert second['executedNodeIDs']==['merge','head','output']


def test_cache_rejects_mistyped_named_tensor_even_with_matching_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'GraphModel',SortingOutputs)
    first=module.preview(graph(),tmp_path/'first',device='cpu')
    payload=torch.load(first['cachePath'],weights_only=True)
    payload['activations']['relu']['indices']=payload['activations']['relu']['indices'].float()
    torch.save(payload,first['cachePath'])
    path=tmp_path/'first/activation-cache.json';meta=json.loads(path.read_text())
    meta['fileHash']=module.file_hash(first['cachePath'],32_000_000);path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='cache.*invalid'):
        module.preview(graph(),tmp_path/'second',device='cpu',resume_from='merge',cache=first['cachePath'])

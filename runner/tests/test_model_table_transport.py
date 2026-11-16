"""New owned table transport extends only versioned imported Train authority."""
import hashlib
import json
from pathlib import Path
import pytest
from runner.model_asset_operation_contract import parse_contract, verify_inputs


def operation(tmp_path, *, version=2, action='train', size=8_000_001):
    project='A39507A9-44EC-4E41-BAF5-D3A932E15365'
    operation='52EF2B86-F41D-4786-A628-487D9522E397'
    root=tmp_path/'workflow-operations'/project/operation
    (root/'input').mkdir(parents=True)
    (root/'source').mkdir()
    files={f'source/{name}':b'{}' for name in ('config.json','tokenizer.json','model.safetensors')}
    files['input/training.table.json']=b' '*(size-2)+b'{}'
    files['input/validation.table.json']=b'{}'
    inner=dict(schemaVersion=version,projectID=project,operationID=operation,requestIdentity='a'*64,action=action,settings={},sourceIdentity='b'*64)
    files['input/model-request.json']=json.dumps(inner).encode()
    for path,data in files.items():(root/path).write_bytes(data)
    request=dict(projectID=project,operationID=operation,requestIdentity='a'*64,arguments=['--model-asset',str(root/'input/model-request.json')],inputFiles=[dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for path,data in files.items()])
    return root,request


def test_owned_training_table_above_old_inline_limit_is_verified(tmp_path):
    root,request=operation(tmp_path)
    assert verify_inputs(request,root,{})['schemaVersion']==2


def test_legacy_inline_capture_keeps_its_original_limit(tmp_path):
    root,request=operation(tmp_path,version=1)
    with pytest.raises(ValueError):verify_inputs(request,root,{})


@pytest.mark.parametrize('change',['missing-validation','nontraining','corrupted','oversized','unknown-version'])
def test_new_table_authority_refuses_invalid_capture(tmp_path,change):
    root,request=operation(tmp_path,action='inspect' if change=='nontraining' else 'train',version=4 if change=='unknown-version' else 2)
    if change=='missing-validation':request['inputFiles']=[f for f in request['inputFiles'] if f['path']!='input/validation.table.json']
    if change=='corrupted':(root/'input/training.table.json').write_bytes(b'changed')
    if change=='oversized':next(f for f in request['inputFiles'] if f['path']=='input/training.table.json')['bytes']=50_000_001
    with pytest.raises((ValueError,OSError)):verify_inputs(request,root,{})

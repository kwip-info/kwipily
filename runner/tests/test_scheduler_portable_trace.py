"""Portable inference pins optional trace data without executing training code."""
import hashlib
import json
import pytest
import torch
from runner.tests.test_portable_inference import bundle

def add_trace(target,raw):
    trace=target/'rate-trace.json';trace.write_bytes(raw)
    path=target/'export-manifest.json';manifest=json.loads(path.read_text())
    manifest['files'].append(dict(path=trace.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    path.write_text(json.dumps(manifest));return trace

def test_optional_trace_is_pinned_inert_data_at_exact_limit(tmp_path):
    target,helper,model=bundle(tmp_path)
    add_trace(target,b'{}'+b' '*(4_000_000-2))
    sample=torch.zeros(2,2)
    torch.testing.assert_close(helper.predict(sample),model(sample).detach(),rtol=0,atol=0)

@pytest.mark.parametrize('mutation',['changed','oversized','missing','symlink'])
def test_trace_integrity_and_size_refuse_before_model_execution(tmp_path,mutation):
    target,helper,_=bundle(tmp_path)
    trace=add_trace(target,b'{}')
    if mutation=='changed':trace.write_bytes(b'[]')
    elif mutation=='oversized':
        trace.unlink();manifest=json.loads((target/'export-manifest.json').read_text());manifest['files'].pop();(target/'export-manifest.json').write_text(json.dumps(manifest));add_trace(target,b'{}'+b' '*(4_000_001-2))
    elif mutation=='missing':trace.unlink()
    else:
        outside=tmp_path/'outside.json';outside.write_bytes(b'{}');trace.unlink();trace.symlink_to(outside)
    with pytest.raises((ValueError,FileNotFoundError)):helper.predict(torch.zeros(1,2))

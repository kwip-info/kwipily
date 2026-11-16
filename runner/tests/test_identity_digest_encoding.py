"""Bounded identity metadata hashes need no recursive token generator."""
import hashlib,json,pytest
from runner import dataflow as df

@pytest.mark.parametrize('value',[
 ['kwip.join.row.v1','a:00001','b:00002'],
 ['kwip.union.row.v1','table1','😀:é\\\"\n'],
 ['flag',None,True,False,-2**63,2**63-1,-0.0,1e-12],
])
def test_identity_digest_uses_same_canonical_bytes_without_recursive_stream(value,monkeypatch):
    expected=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=True).encode()).hexdigest()
    def forbidden(*args,**kwargs):raise AssertionError('Recursive JSON walker used for a bounded identity vector')
    monkeypatch.setattr(df,'iter_json',forbidden)
    assert df.digest(value)==expected


def test_complex_or_large_values_keep_streamed_boundary(monkeypatch):
    visits=[];original=df.iter_json
    def observed(value):visits.append(True);yield from original(value)
    monkeypatch.setattr(df,'iter_json',observed)
    assert df.digest({'nested':[[1,2],[3]]})
    assert visits
    with pytest.raises(ValueError):df.digest(['bad',float('nan')])

"""Bounded pinned-range inspection, distinct from complete scientific validation."""
import hashlib,json
import pytest
from runner import segmented_table_capture as capture
from runner.segmented_table_store import SegmentWriter,SegmentResolver
from runner.artifact_storage import ArtifactStorage
from runner.tests.test_segmented_table_capture import table


def recorded(tmp_path,rows):
    original=table(rows)
    with ArtifactStorage(tmp_path,maximum_bytes=8_000_000) as storage:
        with SegmentWriter(storage,maximum_bytes=8_000_000) as writer:
            ref=capture.capture_table(writer,original)
    path=tmp_path/'view.table.json';raw=json.dumps(capture.envelope(original,ref)).encode();path.write_bytes(raw)
    return path,hashlib.sha256(raw).hexdigest(),ref


def test_page_does_not_validate_or_index_the_whole_population(tmp_path,monkeypatch):
    from runner import table_view
    path,pin,ref=recorded(tmp_path,['value'+str(i) for i in range(10000)])
    def refuse(*a,**k):raise AssertionError('A bounded page scanned the whole table')
    monkeypatch.setattr(capture,'read_capture',refuse)
    monkeypatch.setattr(SegmentResolver,'audit',refuse)
    monkeypatch.setattr(table_view,'build_index',refuse)
    request={'schemaVersion':3,'inputKind':'table','offset':9870,'limit':3,'search':'','sourceSHA256':pin,'indexSHA256':pin}
    request_path=tmp_path/'view-query.json';request_path.write_text(json.dumps(request));events=[]
    table_view.run_query(request_path,events.append)
    page=events[-1]
    assert page['rows']==[['value9870'],['value9871'],['value9872']]
    assert page['positions']==[9870,9871,9872] and page['totalRows']==10000
    assert page['verification']=='pinnedRanges' and 'materializedIndex' not in page
    assert not (tmp_path.parent/'artifacts/view.sqlite3').exists()


def test_page_preserves_long_cell_paging_and_cancel(tmp_path):
    from runner import table_view
    value='😀abc'*10000
    path,pin,ref=recorded(tmp_path,[value])
    request={'schemaVersion':5,'inputKind':'table','offset':0,'limit':1,'search':'','sourceSHA256':pin,'indexSHA256':pin,'fullCellColumn':0,'fullCellStart':16000}
    q=tmp_path/'view-query.json';q.write_text(json.dumps(request));events=[]
    table_view.run_query(q,events.append)
    assert events[-1]['rows']==[[value[16000:32384]]]
    assert events[-1]['cellTotalCharacters']==len(value)
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):table_view.run_query(q,events.append,cancel)


def test_selected_range_tamper_is_refused(tmp_path):
    from runner import table_view
    path,pin,ref=recorded(tmp_path,['secret-original'])
    pack=tmp_path/ref['pack']['path'];raw=pack.read_bytes();assert b'secret-original' in raw
    pack.write_bytes(raw.replace(b'secret-original',b'secret-modified'))
    q=tmp_path/'view-query.json';q.write_text(json.dumps({'schemaVersion':3,'inputKind':'table','offset':0,'limit':1,'search':'','sourceSHA256':pin,'indexSHA256':pin}))
    with pytest.raises(ValueError):table_view.run_query(q,lambda _:None)

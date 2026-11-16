"""Metadata paging, stable spans, bounded lookup and corrupt geometry refusal."""
import pytest
from runner.artifact_storage import ArtifactStorage
from runner.table_artifact_pack import TableArtifactPack,TableArtifactPackReader
from runner.packed_ordered_index import build_index,OrderedIndex


@pytest.mark.parametrize('count',[0,1,127,128,129,16385])
def test_paged_index_preserves_order_and_span_lookup_without_flattening(tmp_path,count):
    records=[(1+i%3,{'ordinal':i}) for i in range(count)]
    with ArtifactStorage(tmp_path,maximum_bytes=20_000_000) as storage:
        with TableArtifactPack(storage,maximum_bytes=20_000_000) as pack:root=build_index(pack,iter(records))
    with TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=20_000_000) as reader:
        reader.audit();index=OrderedIndex(reader,root,maximum_count=max(1,count*3))
        assert list(index.records())==records
        if count:
            value,offset=index.locate(0);assert value=={'ordinal':0} and offset==0
            value,offset=index.locate(root['count']-1);assert value==records[-1][1] and offset==records[-1][0]-1
        for invalid in (-1,root['count'],True):
            with pytest.raises(IndexError):index.locate(invalid)


def test_valid_pins_cannot_hide_inconsistent_child_geometry(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=10000) as storage:
        with TableArtifactPack(storage,maximum_bytes=10000) as pack:
            leaf=pack.add_json({'schemaVersion':1,'kind':'orderedIndex','height':0,'count':1,'entries':[{'count':1,'value':'record'}]})
            reference=pack.add_json({'schemaVersion':1,'kind':'orderedIndex','height':1,'count':2,'entries':[{'count':2,'reference':leaf}]})
    with TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=10000) as reader:
        index=OrderedIndex(reader,{'reference':reference,'count':2,'height':1},maximum_count=10)
        with pytest.raises(ValueError):index.locate(1)
        with pytest.raises(ValueError):list(index.records())

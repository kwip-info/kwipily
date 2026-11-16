"""Current Partition storage, diagnostics, calendar and publication bounds."""
from dataclasses import replace
import json
import pytest
from runner import dataflow as df, dataflow_disk, dataflow_cli as cli
from runner import dataflow_partition as partition
from runner.tests.test_dataflow_partition import group, temporal, partition_table, current_config
from runner.tests.test_dataflow_join import table


def test_output_record_bytes_rejected_before_derive(monkeypatch):
    source=table([(i,'é'*500) for i in range(10)],dtypes=['int64','string'])
    monkeypatch.setattr(dataflow_disk,'MAX_ROW_BYTES',1000)
    monkeypatch.setattr(df,'derive',lambda *a,**k:pytest.fail('derive before record byte preflight'))
    with pytest.raises(ValueError,match='record budget'):partition_table(source,'groupSplit',group())


def test_diagnostics_bounded_before_derive(monkeypatch):
    source=replace(table([(i,i) for i in range(10)]),diagnostics=tuple({'n':i} for i in range(16384)))
    monkeypatch.setattr(df,'derive',lambda *a,**k:pytest.fail('derive before diagnostics bound'))
    with pytest.raises(ValueError,match='diagnostic'):partition_table(source,'groupSplit',group())


def test_epoch_calendar_bounds_and_empty_embargo_partition():
    from runner.dataflow_partition_tool import validate_config
    with pytest.raises(ValueError):validate_config(current_config('temporalSplit',temporal(trainingEnd=-(2**63))))
    with pytest.raises(ValueError,match='nonempty'):
        partition_table(table([(1,1),(11,2),(21,3)]),'temporalSplit',temporal(embargoMilliseconds=2))


def test_exact_publication_byte_boundary_includes_ids_schema_diagnostics_and_code(tmp_path,monkeypatch):
    source=tmp_path/'source.csv';source.write_text('key,value\n'+''.join(f'{i},é\n' for i in range(10)))
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{}},{'id':'p','operator':'kwip.data.partition','config':current_config('groupSplit',group())}],
           'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'p','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    first=cli.run_dataflow(path,{'s':str(source)},tmp_path/'first',lambda _:None)
    boundary=sum(entry['bytes'] for entry in first['artifacts'])
    monkeypatch.setattr(cli,'MAX_ARTIFACT_BYTES',boundary)
    exact=cli.run_dataflow(path,{'s':str(source)},tmp_path/'exact',lambda _:None)
    assert sum(entry['bytes'] for entry in exact['artifacts'])==boundary
    monkeypatch.setattr(cli,'MAX_ARTIFACT_BYTES',boundary-1)
    with pytest.raises(ValueError,match='size limit'):cli.run_dataflow(path,{'s':str(source)},tmp_path/'overflow',lambda _:None)
    assert not (tmp_path/'overflow'/'manifest.json').exists()
    assert not list(tmp_path.rglob('rows.sqlite3'))

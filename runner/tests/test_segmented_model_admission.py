"""Small envelopes must not bypass the existing model allocation boundary."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from runner.dataflow_cli import run_dataflow
from runner.workload_policy import LEGACY
from runner.dataflow_storage import completed_inventory,admit_model_ports,prepared_table_document


def test_small_segment_envelope_refuses_oversized_population_before_rows_or_ids(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[2]
    out=tmp_path/'prepared'
    run_dataflow(repo/'examples/dataflow/scaler.json',{'source':str(repo/'examples/dataflow/observations.csv')},out,lambda _:None,workload_policy=LEGACY.to_document())
    manifest,files,_=completed_inventory(out/'manifest.json',lazy=True)
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='training' and e.get('port')=='table')
    assert entry['bytes']<1_000_000 and json.loads(files[entry['path']])['kind']=='segmentedTable'
    class Unreadable:
        store=SimpleNamespace(close=lambda:None)
        def __len__(self):return 100001
        def __iter__(self):pytest.fail('Oversized selected model rows/IDs were materialized')
    oversized=SimpleNamespace(rows=Unreadable(),row_ids=Unreadable(),columns=[{'name':'x'}])
    monkeypatch.setattr('runner.dataflow_table_reader.read_table_artifact',lambda *a,**k:oversized)
    with pytest.raises(ValueError,match='Row Sample'):
        admit_model_ports(files,manifest,{('training','table')})
    with pytest.raises(ValueError,match='Row Sample'):
        prepared_table_document(files,entry)

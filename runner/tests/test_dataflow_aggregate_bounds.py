from dataclasses import replace
import pytest
from runner import dataflow as df
from runner import dataflow_aggregate as aggregate
from runner.tests.test_dataflow_join import table
from runner.tests.test_dataflow_aggregate import config


def test_group_count_budget_precedes_output_derivation(monkeypatch):
    source=table([(1,2),(2,3)])
    monkeypatch.setattr(df,'derive',lambda *a,**k:pytest.fail('derived before bound'))
    with pytest.raises(ValueError,match='row limit'):aggregate.aggregate_table(source,config(maxRows=1))


def test_escaped_payload_and_metadata_byte_budget(monkeypatch):
    source=table([('é'*1000,2)],dtypes=['string','int64'])
    monkeypatch.setattr(aggregate,'MAX_ARTIFACT_BYTES',5000)
    monkeypatch.setattr(df,'derive',lambda *a,**k:pytest.fail('derived before bound'))
    with pytest.raises(ValueError,match='byte'):aggregate.aggregate_table(source,config())
    source=replace(table([(1,2)]),diagnostics=({'note':'é'*1000},))
    with pytest.raises(ValueError,match='byte'):aggregate.aggregate_table(source,config())


def test_output_column_budget():
    fields=[{'output':f'n{i}','op':'countRows'} for i in range(128)]
    with pytest.raises(ValueError,match='column'):aggregate.validate_config(config(aggregations=fields))

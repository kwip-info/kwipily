"""Generic Table Input inspection keeps the pinned, bounded transport contract."""
import hashlib
import json
import pytest
from runner.data import inspect_data
from runner.source_inspection_operation_contract import parse_contract


def request(path, raw, config):
    return {'arguments': ['--inspect-data', str(path), '--table-options', json.dumps(config)],
            'inputFiles': [{'path': 'input/data/source.json', 'bytes': len(raw),
                            'sha256': hashlib.sha256(raw).hexdigest()}]}


def test_table_inspection_admits_normalized_local_format(tmp_path):
    source = tmp_path / 'input/data/source.json'
    source.parent.mkdir(parents=True)
    raw = b'[{"value":1}]'; source.write_bytes(raw)
    assert parse_contract(request(source, raw, {'format': 'json', 'options': {}}), tmp_path)


@pytest.mark.parametrize('options', [
    {'format': 'json', 'options': {'source': '/tmp/secret'}},
    {'format': 'parquet', 'options': {'storage_options': {'key': 'secret'}}},
    {'format': 'ipc', 'options': {}},
    {'format': 'unknown', 'options': {}},
    {'format': 'json', 'options': {}, 'credential': 'secret'},
])
def test_table_inspection_rejects_unsafe_or_unknown_configuration(tmp_path, options):
    source = tmp_path / 'input/data/source.json'
    with pytest.raises(ValueError):
        parse_contract(request(source, b'[]', options), tmp_path)


def test_table_inspection_rejects_duplicate_json_keys(tmp_path):
    source = tmp_path / 'input/data/source.json'
    value = request(source, b'[]', {'format': 'json', 'options': {}})
    value['arguments'][-1] = '{"format":"json","format":"parquet","options":{}}'
    with pytest.raises(ValueError):
        parse_contract(value, tmp_path)


def test_only_generic_options_have_the_larger_transport_budget(tmp_path):
    source = tmp_path / 'input/data/source.json'
    value = request(source, b'[]', {'format': 'json', 'options': {}})
    value['arguments'][-1] += ' ' * 5000
    assert parse_contract(value, tmp_path)
    value['arguments'][-1] += ' ' * 32768
    with pytest.raises(ValueError): parse_contract(value, tmp_path)
    value['arguments'][-2:] = ['--csv-options', '{}' + ' ' * 4096]
    with pytest.raises(ValueError): parse_contract(value, tmp_path)


@pytest.mark.parametrize('separator', [':', '§'])
def test_legacy_separator_transport_matches_the_native_single_scalar_contract(tmp_path, separator):
    source = tmp_path / 'input/data/source.json'
    value = request(source, b'x', {'format': 'json', 'options': {}})
    value['arguments'][-2:] = ['--csv-separator', separator]
    assert parse_contract(value, tmp_path)


def test_polars_csv_uses_generic_transport_and_preserves_typed_scalar_preview(tmp_path):
    source = tmp_path / 'input/data/source.json'; source.parent.mkdir(parents=True)
    raw = b'value,enabled,name\n2,true,A\n3,false,B\n'; source.write_bytes(raw)
    options = {'parserProfile': 'polars', 'inferSchema': True}
    value = request(source, raw, options)
    value['arguments'][-1] += ' ' * 5000
    assert parse_contract(value, tmp_path)
    result = inspect_data(source, table_options=options)
    assert result['rows'] == [['2', 'true', 'A'], ['3', 'false', 'B']]
    assert [column['dtype'] for column in result['columnSchema']] == ['int64', 'bool', 'string']
    assert result['parsing']['options']['inferSchema'] is True
    assert result['sourceSHA256'] == hashlib.sha256(raw).hexdigest()


def test_polars_csv_can_report_an_empty_but_named_table(tmp_path):
    source = tmp_path / 'empty.csv'; source.write_text('value\n')
    result = inspect_data(source, table_options={'parserProfile': 'polars', 'raiseIfEmpty': False})
    assert result['columns'] == ['value'] and result['rows'] == [] and result['rowCount'] == 0
    assert result['columnSchema'][0]['dtype'] == 'string' and result['parsing']['sampleOnly']


def test_generic_json_preview_preserves_scalar_display_nulls_and_hash(tmp_path):
    source = tmp_path / 'rows.json'
    raw = json.dumps([{'value': i, 'enabled': i % 2 == 0, 'name': None if i == 0 else 'row'}
                      for i in range(20)]).encode()
    source.write_bytes(raw)
    result = inspect_data(source, table_options={'format': 'json', 'options': {}})
    assert result['columns'] == ['value', 'enabled', 'name']
    assert result['rows'][0] == ['0', 'true', '']
    assert result['rows'][1] == ['1', 'false', 'row']
    assert len(result['rows']) == result['rowCount'] == 8
    assert result['nullCells'] == [[0, 2]]
    assert result['sourceSHA256'] == hashlib.sha256(raw).hexdigest()
    assert result['parsing']['sampleOnly'] is True
    assert result['parsing']['options']['format'] == 'json'
    assert result['columnSchema'] == [
        {'name': 'value', 'dtype': 'int64', 'nullable': None},
        {'name': 'enabled', 'dtype': 'bool', 'nullable': None},
        {'name': 'name', 'dtype': 'string', 'nullable': None}]


def test_generic_preview_echoes_only_validated_authored_options_not_inactive_engine_defaults(tmp_path):
    import polars as pl
    from runner.source_inspection_operation_contract import parse_table_options_json
    source = tmp_path / 'rows.parquet'; pl.DataFrame({'x': [1, 2]}).write_parquet(source)
    requested = {'format': 'parquet', 'options': {'columns': [], 'use_pyarrow': False}}
    decoded = parse_table_options_json(json.dumps(requested))
    assert decoded == requested
    result = inspect_data(source, table_options=decoded)
    assert result['parsing']['options'] == requested
    assert 'pyarrow_options' not in result['parsing']['options']['options']
    assert result['rows'] == [['1'], ['2']]


def test_generic_inspection_cancellation_prevents_evidence(tmp_path):
    source = tmp_path / 'rows.json'; source.write_text('[{"x":1}]')
    def cancel():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        inspect_data(source, table_options={'format': 'json', 'options': {}}, check_cancelled=cancel)


def test_csv_and_table_options_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError):
        inspect_data(tmp_path / 'absent', options={}, table_options={'format': 'json', 'options': {}})


@pytest.mark.parametrize('fmt', ['jsonl', 'parquet', 'excel', 'markdown', 'xml', 'html'])
def test_all_local_format_inspections_share_bounded_transport(tmp_path, fmt):
    import polars as pl
    path = tmp_path / ('source.' + fmt)
    if fmt == 'jsonl':
        path.write_text('\n'.join(json.dumps({'x': i, 'name': 'row'}) for i in range(20)))
    elif fmt == 'parquet':
        pl.DataFrame({'x': list(range(20)), 'name': ['row'] * 20}).write_parquet(path)
    elif fmt == 'excel':
        import openpyxl
        book = openpyxl.Workbook(); sheet = book.active
        sheet.append(['x', 'name'])
        for i in range(20): sheet.append([i, 'row'])
        book.save(path)
    elif fmt == 'markdown':
        path.write_text('| x | name |\n|---|---|\n' + ''.join(f'| {i} | row |\n' for i in range(20)))
    elif fmt == 'xml':
        path.write_text('<rows>' + ''.join(f'<row><x>{i}</x><name>row</name></row>' for i in range(20)) + '</rows>')
    else:
        path.write_text('<table><tr><th>x</th><th>name</th></tr>' + ''.join(f'<tr><td>{i}</td><td>row</td></tr>' for i in range(20)) + '</table>')
    result = inspect_data(path, table_options={'format': fmt, 'options': {}})
    assert result['columns'] == ['x', 'name']
    assert result['rows'][0] == ['0', 'row']
    assert result['rowCount'] == 8 and len(result['rows']) == 8
    assert result['sourceSHA256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['parsing']['options']['format'] == fmt


def test_database_inspection_only_reads_the_brokers_typed_snapshot(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from runner import database_source
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary inspection must not open a database connection')
    monkeypatch.setattr(database_source, 'selected_cursor', forbidden)
    path = tmp_path / 'capture.arrow'
    table=pa.table({'x': list(range(20)), 'name': ['row'] * 20})
    table=table.replace_schema_metadata({b'kwip.database.selection.v1':database_source.capture_selection_identity({'table':'observations'})})
    with ipc.new_file(path,table.schema) as writer:writer.write_table(table)
    result = inspect_data(path, table_options={'format': 'database', 'options': {'table': 'observations'}})
    assert result['columns'] == ['x', 'name'] and result['rows'][0] == ['0', 'row']
    assert result['rowCount'] == 8 and result['parsing']['options']['format'] == 'database'
    assert result['sourceSHA256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Reconnect or refresh'):
        inspect_data(path,table_options={'format':'database','options':{'table':'different'}})


def test_generic_admission_does_not_import_a_data_engine_before_worker_authority():
    import subprocess
    import sys
    from pathlib import Path
    script = '''import sys
from runner.source_inspection_operation_contract import parse_table_options_json
parse_table_options_json('{"format":"json","options":{}}')
parse_table_options_json('{"format":"database","options":{"table":"observations"}}')
parse_table_options_json('{"parserProfile":"polars","inferSchema":true}')
assert not any(name in sys.modules for name in ('polars','numpy','torch','pyarrow','psycopg','pymysql'))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_sqlite_database_inspection_selects_table_and_bounds_sample(tmp_path):
    import sqlite3
    path=tmp_path/'source.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE observations (id INTEGER, name TEXT)')
        connection.executemany('INSERT INTO observations VALUES (?,?)',[(i,'row') for i in range(100)])
        connection.execute('CREATE TABLE private (secret TEXT)')
    result=inspect_data(path,table_options={'format':'database','options':{'table':'observations','columns':['id'],'orderBy':['-id']}})
    assert result['columns']==['id'] and result['rows']==[[str(i)] for i in range(99,91,-1)]
    assert result['rowCount']==8 and result['sourceSHA256']==hashlib.sha256(path.read_bytes()).hexdigest()

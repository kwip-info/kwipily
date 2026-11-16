"""The public two-tool lesson's claims are backed by its actual bundled observations."""
import json
from pathlib import Path
from runner.dataflow_cli import run_dataflow
from runner.table_view import query_index


def test_bundled_table_view_practice_complete_demand_and_source_only_comparison(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    source = repository / 'Sources/KwipMLApp/Resources/workflow-observations.csv'
    graph = {'schemaVersion': 1, 'nodes': [
        {'id': 'observations', 'operator': 'kwip.data.sourceCSV', 'config': {'parserProfile': 'polars'}},
        {'id': 'view', 'operator': 'kwip.data.view', 'config': {}}],
        'edges': [{'from': {'node': 'observations', 'port': 'table'}, 'to': {'node': 'view', 'port': 'table'}}]}
    path = tmp_path / 'practice.json'; path.write_text(json.dumps(graph))
    output = tmp_path / 'complete'
    manifest = run_dataflow(path, {'observations': str(source)}, output, lambda _: None,
                            sampling={'mode': 'rowCount', 'rowCount': 4})
    coverage = manifest['executionCoverage']['nodes']
    assert coverage['observations']['rowCounts']['table'] == coverage['view']['rowCounts']['table'] == 64
    assert coverage['view']['completeness'] == 'complete'
    receipt = manifest['tableViews']['view']['index']
    page = query_index(output / receipt['path'], receipt['sha256'], offset=0, limit=10,
                       source_sha256=receipt['sourceSHA256'])
    assert page['totalRows'] == 64 and len(page['rows']) == 10 and page['nextOffset'] == 10
    assert [column['name'] for column in page['columns']] == ['signal', 'context', 'label']
    graph['nodes'] = graph['nodes'][:1]; graph['edges'] = []
    path.write_text(json.dumps(graph))
    manifest = run_dataflow(path, {'observations': str(source)}, tmp_path / 'sampled', lambda _: None,
                            sampling={'mode': 'rowCount', 'rowCount': 4})
    coverage = manifest['executionCoverage']['nodes']['observations']
    assert coverage['rowCounts']['table'] == 4 and coverage['completeness'] == 'sampled'

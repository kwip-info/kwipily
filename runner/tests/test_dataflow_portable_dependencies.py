"""Exported preparation must carry its dependencies beyond the repository."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_sample import graph_fixture


def replay(export, bindings, output, cwd):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    environment['PYTHONNOUSERSITE'] = '1'
    result = subprocess.run(
        [sys.executable, str(export / 'prep.py'), '--source-bindings', str(bindings),
         '--output', str(output)], cwd=cwd, env=environment,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout.splitlines()[-1])['event'] == 'dataflowCompleted'
    return json.loads((output / 'manifest.json').read_text())


def test_sampling_export_replays_outside_repository_and_exports_again(tmp_path):
    graph, sources = graph_fixture(tmp_path, after_split=True)
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(graph))
    bindings = {key: str(path) for key, path in sources.items()}
    bindings_path = tmp_path / 'private-bindings.json'
    bindings_path.write_text(json.dumps(bindings))
    exported = tmp_path / 'export'
    original = run_dataflow(graph_path, bindings, exported, lambda event: None)
    unrelated = tmp_path / 'unrelated-working-directory'
    unrelated.mkdir()

    # This invokes the exported script, not an imported in-repository adapter.
    reproduced = tmp_path / 'reproduced'
    first = replay(exported, bindings_path, reproduced, unrelated)
    second = replay(reproduced, bindings_path, tmp_path / 'reexported', unrelated)
    assert original['artifacts'] == first['artifacts'] == second['artifacts']
    sampled = next(entry for entry in original['artifacts']
                   if entry.get('nodeID') == 'sample' and entry.get('port') == 'table')
    table = json.loads((reproduced / sampled['path']).read_text())
    assert len(table['rows']) == 20
    assert table['lineage']['role'] == 'training'


def test_export_inventory_pins_sampling_and_join_helpers(tmp_path):
    graph, sources = graph_fixture(tmp_path)
    path = tmp_path / 'graph.json'
    path.write_text(json.dumps(graph))
    output = tmp_path / 'export'
    manifest = run_dataflow(path, {key: str(value) for key, value in sources.items()},
                            output, lambda event: None)
    inventory = {entry['path']: entry for entry in manifest['artifacts']}
    for name in ('dataflow_sample.py', 'dataflow_join.py', 'dataflow_aggregate.py', 'dataflow_partition.py', 'formula_decimal.py'):
        assert name in inventory, f'Missing portable dependency: {name}'
        payload = (output / name).read_bytes()
        assert inventory[name]['kind'] == 'preparationCode'
        assert inventory[name]['bytes'] == len(payload)
        assert inventory[name]['sha256'] == hashlib.sha256(payload).hexdigest()
        assert payload == Path(__file__).parents[1].joinpath(name).read_bytes()


def test_table_input_nondefault_csv_replays_with_nulls_and_empty_text(tmp_path):
    raw='metadata\r\nkey^note^empty^missing\r\n001^§café, §§quoted§§§^^NA\r\n002^§two\nlines§^^NULL\r\n'.encode('cp1252')
    source=tmp_path/'source.csv';source.write_bytes(raw)
    config=dict(separator='^',quoteChar='§',encoding='cp1252',recordEndings='crlf',
                skipRecords=1,emptyField='text',nullMarkers=['NA','NULL'])
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':config}],'edges':[]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    binding=tmp_path/'bindings.json';binding.write_text(json.dumps({'s':str(source)}))
    exported=tmp_path/'export';manifest=run_dataflow(path,{'s':str(source)},exported,lambda _:None)
    unrelated=tmp_path/'elsewhere';unrelated.mkdir()
    output=tmp_path/'replayed';reproduced=replay(exported,binding,output,unrelated)
    artifact=next(item for item in manifest['artifacts'] if item.get('nodeID')=='s' and item.get('port')=='table')
    assert manifest['artifacts']==reproduced['artifacts']
    captured=json.loads((output/artifact['path']).read_text())
    assert captured['rows']==[['001','café, §quoted§','',None],['002','two\nlines','',None]]
    assert captured['columns']==[{'name':name,'dtype':'string','nullable':name=='missing'} for name in ['key','note','empty','missing']]
    assert (exported/'csv_parser.py').read_bytes()==Path(__file__).parents[1].joinpath('csv_parser.py').read_bytes()

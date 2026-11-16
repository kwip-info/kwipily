from pathlib import Path
import csv
import io
import math
import pytest
from scripts import measured_diagnostic_training_fixtures as corpus


def test_plan_has_six_bounded_real_training_cases_and_frozen_growth_pair():
    plan = corpus.fixture_plan()
    assert [case['id'] for case in plan] == [
        'ordinary-baseline', 'gradient-growth', 'lowered-rate-control',
        'tiny-holdout', 'insufficient-gradient-window', 'exact-continuation'
    ]
    assert sum(case['trainedEpochs'] for case in plan) == 69
    positive, control = plan[1:3]
    assert (positive['depth'], positive['learningRate'], positive['epochs']) == (8, .3, 20)
    assert (control['depth'], control['learningRate'], control['epochs']) == (8, .03, 20)
    assert all(case['device'] == 'cpu' and case['seed'] == 42 and case['batchSize'] == 32 for case in plan)
    assert plan[-1]['initialEpoch'] == 20 and plan[-1]['epochs'] == 23
    assert plan[-2]['maxCaptures'] == 2


def test_output_location_refuses_external_paths_and_existing_destinations(tmp_path):
    with pytest.raises(ValueError, match='ignored .local'):
        corpus.validate_destination(tmp_path / 'external')
    with pytest.raises(ValueError, match='existing'):
        corpus.validate_destination(corpus.ROOT / '.local')


def test_independent_growth_oracle_checks_windows_and_continuation_boundary():
    rows = [{'epoch': i, 'batches': [{'l2Norm': 1. if i <= 5 else 200.}]} for i in range(1, 11)]
    facts = corpus.gradient_growth_facts({'summaries': rows}, continuation=False)
    assert facts['triggered'] and facts['firstMean'] == 1. and facts['lastMean'] == 200.
    assert corpus.gradient_growth_facts({'summaries': rows}, continuation=True)['withheld'] == 'continuation'
    assert corpus.gradient_growth_facts({'summaries': rows[:2]}, continuation=False)['withheld'] == 'insufficient-window'
    constant = {'summaries': [{'epoch': i, 'batches': [{'l2Norm': 500.}]} for i in range(1, 11)]}
    assert not corpus.gradient_growth_facts(constant, continuation=False)['triggered']


def test_authored_populations_preserve_original_order_and_reserve_separate_test_rows():
    original=list(csv.DictReader(corpus.SOURCE.read_text().splitlines()))
    for percent,(training,validation) in corpus.AUTHORED_POPULATIONS.items():
        authored=list(csv.DictReader(io.StringIO(corpus.authored_source(percent))))
        assert len(authored)==80 and sorted(training+validation)==list(range(64))
        for actual,index in zip(authored[:64],training+validation):
            assert actual['original_row']==str(index)
            assert {key:actual[key] for key in ('signal','context','label')}==original[index]
        assert [row['original_row'] for row in authored[64:]]==[f'reserved-test-{i}' for i in range(16)]
        graph=corpus.preparation_graph(percent)
        assert {node['operator'] for node in graph['nodes']}=={'kwip.data.sourceCSV','kwip.data.columns','kwip.data.partition','kwip.data.preprocess','kwip.data.view'}
        assert next(node for node in graph['nodes'] if node['id']=='split')['config']['strategy']=='time'


def test_actual_training_corpus_pins_captures_and_refuses_modified_evidence():
    import json
    import tempfile
    local = corpus.ROOT / '.local'
    local.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='diagnostic-corpus-test-', dir=local) as container:
        destination = Path(container) / 'corpus'
        manifest = corpus.generate(destination, maximum_seconds=120)
        assert manifest['runtime']['device'] == 'cpu'
        assert manifest['runtime']['threads'] == 1
        assert len(manifest['cases']) == 6
        by_id = {case['id']: case for case in manifest['cases']}
        assert by_id['gradient-growth']['expectations']['growth']['ratio'] > 100
        assert by_id['lowered-rate-control']['expectations']['growth']['triggered'] is False
        assert by_id['tiny-holdout']['expectations']['validationRows'] == 6
        assert by_id['tiny-holdout']['expectations']['accuracyStep'] == 1/6
        assert by_id['insufficient-gradient-window']['expectations']['capturedBatches'] == 2
        assert by_id['exact-continuation']['expectations']['growth']['withheld'] == 'continuation'
        for case in manifest['cases']:
            artifact = destination / case['gradientArtifact']
            assert corpus.digest(artifact) == case['pins']['gradientSHA256']
            events = json.loads((artifact.parent / 'events.json').read_text())
            captured = json.loads(artifact.read_text())
            assert [e for e in events if e.get('event') == 'gradientSummary'] == captured['summaries']
        # Verify real captured role matrices and fit statistics against the public
        # original rows, independently of the generator's diagnostic-growth oracle.
        from runner.dataflow_table_reader import read_table_artifact
        original=list(csv.DictReader(corpus.SOURCE.read_text().splitlines()))
        for percent,(training,validation) in corpus.AUTHORED_POPULATIONS.items():
            folder=destination/f'preparation-{percent}'
            inventory=json.loads((folder/'manifest.json').read_text())
            tables={}
            for role,indices in [('training',training),('validation',validation),('test',None)]:
                entry=next(entry for entry in inventory['artifacts'] if entry.get('nodeID')==role and entry.get('port')=='table')
                table=read_table_artifact(folder/entry['path'],entry['sha256']);tables[role]=table
                assert table.role==role and len(table.rows)==(len(indices) if indices is not None else 16)
                names=[column['name'] for column in table.columns]
                assert [row[names.index('original_row')] for row in table.rows]==([str(i) for i in indices] if indices is not None else [f'reserved-test-{i}' for i in range(16)])
            identities=[set(tables[role].row_ids) for role in ('training','validation','test')]
            assert len(set.union(*identities))==80 and all(not left&right for i,left in enumerate(identities) for right in identities[i+1:])
            state_entry=next(entry for entry in inventory['artifacts'] if entry.get('nodeID')=='fit' and entry.get('port')=='state')
            state=json.loads((folder/state_entry['path']).read_text())
            assert state['fitEvidence']['observationIDs']==list(tables['training'].row_ids)
            for rule,name in zip(state['rules'],['signal','context']):
                values=[float(original[index][name]) for index in training]
                mean=math.fsum(values)/len(values)
                scale=math.sqrt(math.fsum((value-mean)**2 for value in values)/len(values))
                assert math.isclose(rule['mean'],mean,rel_tol=1e-12,abs_tol=1e-12)
                assert math.isclose(rule['scale'],scale,rel_tol=1e-12,abs_tol=1e-12)
        corpus.verify_inventory(destination, manifest)
        artifact = destination / by_id['gradient-growth']['gradientArtifact']
        original = artifact.read_bytes()
        artifact.write_bytes(original + b'\n')
        with pytest.raises(ValueError, match='hash/size'):
            corpus.verify_inventory(destination, manifest)
        artifact.write_bytes(original)
        corpus.verify_inventory(destination, manifest)

import json
import math
import tempfile
from pathlib import Path

import pytest
from scripts import generalization_training_fixtures as corpus


def test_plan_has_fresh_fixed_budget_positive_and_consistent_relation_control():
    plan = corpus.fixture_plan()
    assert [case['id'] for case in plan] == ['heldout-label-shift', 'consistent-relation-control']
    assert all(case['epochs'] == 20 and case['seed'] == 42 and case['device'] == 'cpu'
               and case['batchSize'] == 48 and case['learningRate'] == .03 for case in plan)
    assert all(case['sourceRows'] == 80 and case['trainingRows'] == 48 and case['validationRows'] == 16
               and case['reservedTestRows'] == 16 for case in plan)
    assert plan[0]['invertHeldoutLabels'] and not plan[1]['invertHeldoutLabels']


def test_independent_trend_arithmetic_and_negative_controls():
    metrics = [dict(epoch=i, trainLoss=1-i*.03, validationLoss=.5+i*.03, accuracy=.5) for i in range(1,21)]
    facts = corpus.generalization_facts(metrics)
    assert facts['triggered']
    assert math.isclose(facts['trainStart'], .91) and math.isclose(facts['trainEnd'], .46)
    assert math.isclose(facts['validationStart'], .59) and math.isclose(facts['validationEnd'], 1.04)
    assert math.isclose(facts['gapIncrease'], .9)
    healthy = [{**row, 'validationLoss': row['trainLoss']} for row in metrics]
    assert not corpus.generalization_facts(healthy)['triggered']
    assert corpus.generalization_facts(metrics[:3])['withheld'] == 'insufficient-window'
    with pytest.raises(ValueError):
        corpus.generalization_facts([{**row, 'trainLoss': float('nan')} for row in metrics])


def test_destination_refuses_external_and_existing_paths(tmp_path):
    with pytest.raises(ValueError, match='ignored .local'):
        corpus.generate(tmp_path / 'outside')
    local = corpus.ROOT / '.local'
    local.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match='existing'):
        corpus.generate(local)


def test_actual_cpu_training_has_unmodified_trends_scaler_boundaries_and_integrity_refusal():
    local = corpus.ROOT / '.local'; local.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='generalization-corpus-test-', dir=local) as container:
        destination = Path(container) / 'corpus'
        receipt = corpus.generate(destination, maximum_seconds=90)
        assert receipt['runtime']['device'] == 'cpu' and receipt['runtime']['threads'] == 1
        assert len(receipt['cases']) == 2
        by_id = {case['id']: case for case in receipt['cases']}
        assert by_id['heldout-label-shift']['expectations']['triggered']
        assert not by_id['consistent-relation-control']['expectations']['triggered']
        assert 'label shift' in receipt['limitations'].lower()
        assert 'memorization' in receipt['limitations'].lower()
        sources = []
        training_series = []
        for case in receipt['cases']:
            metrics = json.loads((destination / case['metricsArtifact']).read_text())
            training_series.append([row['trainLoss'] for row in metrics])
            epochs = json.loads((destination / case['eventsArtifact']).read_text())
            assert [row for row in epochs if row.get('event') == 'epoch'] == metrics
            # Recompute both windows from original emitted metrics, independent of helper facts.
            assert math.isclose(case['expectations']['trainStart'], sum(row['trainLoss'] for row in metrics[:5])/5)
            assert math.isclose(case['expectations']['validationEnd'], sum(row['validationLoss'] for row in metrics[-5:])/5)
            manifest = json.loads((destination / case['modelManifest']).read_text())
            assert manifest['lineage']['mode'] == 'newTraining'
            assert manifest['provenance']['trainingSamples'] == 48 and manifest['provenance']['validationSamples'] == 16
            preparation = destination / case['preparationManifest']
            inventory = json.loads(preparation.read_text())
            state_entry = next(entry for entry in inventory['artifacts'] if entry.get('nodeID') == 'fit' and entry.get('port') == 'state')
            state = json.loads((preparation.parent / state_entry['path']).read_text())
            assert len(state['fitEvidence']['observationIDs']) == 48
            assert math.isclose(state['rules'][0]['mean'], 0, abs_tol=1e-12)
            assert math.isclose(state['rules'][1]['mean'], 0, abs_tol=1e-12)
            assert math.isclose(state['rules'][0]['scale'], math.sqrt(7.5), rel_tol=1e-12)
            assert math.isclose(state['rules'][1]['scale'], .25, rel_tol=1e-12)
            sources.append((destination / case['sourceArtifact']).read_text().splitlines())
        assert training_series[0] == training_series[1], "Only held-out labels differ; fresh training must be identical"
        assert len(sources[0]) == 81
        assert sources[0][:49] == sources[1][:49] and sources[0][65:] == sources[1][65:]
        for shifted, healthy in zip(sources[0][49:65], sources[1][49:65]):
            assert shifted.rsplit(',',1)[0] == healthy.rsplit(',',1)[0]
            assert int(shifted.rsplit(',',1)[1]) == 1-int(healthy.rsplit(',',1)[1])
        corpus.verify_inventory(destination, receipt)
        artifact = destination / receipt['cases'][0]['metricsArtifact']
        original = artifact.read_bytes(); artifact.write_bytes(original+b'\n')
        with pytest.raises(ValueError, match='hash/size'):
            corpus.verify_inventory(destination, receipt)

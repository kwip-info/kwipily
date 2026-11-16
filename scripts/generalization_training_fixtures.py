#!/usr/bin/env python3
"""Fresh CPU generalization-trend positive and consistent-relation control.

80 public synthetic source rows: 48 train, 16 validation, 16 reserved test.
The positive deliberately inverts validation labels. This is label shift, not
proof of memorization or an ordinary model-overfitting benchmark. Raw worker
metrics, gradients, weights and manifests are never adjusted after execution.
"""
import argparse
import datetime
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measured_diagnostic_training_fixtures import digest, write_json, validate_destination, verify_inventory, model_graph, preparation_graph

CORPUS_ID = 'kwip.cpu-trained-generalization-controls.v1'
RULE_ID = 'trend.generalization-gap.v1'


def fixture_plan():
    return [dict(id=name, invertHeldoutLabels=inverted, epochs=20, seed=42, device='cpu',
                 batchSize=48, learningRate=.03, sourceRows=80, trainingRows=48,
                 validationRows=16, reservedTestRows=16)
            for name, inverted in [('heldout-label-shift', True), ('consistent-relation-control', False)]]


def generalization_facts(metrics):
    """Independent arithmetic for the documented legacy rule's latest-20 window.

    This does not claim that the legacy rule itself authenticates split provenance.
    The generator separately retains actual preparation membership and state.
    """
    if any(type(row.get('epoch')) is not int or row['epoch'] <= 0 or
           any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
               for key in ('trainLoss', 'validationLoss', 'accuracy')) or
           row['trainLoss'] < 0 or row['validationLoss'] < 0 or not 0 <= row['accuracy'] <= 1
           for row in metrics):
        raise ValueError('Trend metrics require positive epochs and finite valid losses/accuracy.')
    if any(b['epoch'] != a['epoch']+1 for a, b in zip(metrics, metrics[1:])):
        raise ValueError('Designed fresh fixture requires contiguous untouched epoch metrics.')
    if len(metrics) < 10:
        return dict(triggered=False, withheld='insufficient-window')
    window = metrics[-20:]
    train_start = math.fsum(row['trainLoss'] for row in window[:5])/5
    train_end = math.fsum(row['trainLoss'] for row in window[-5:])/5
    val_start = math.fsum(row['validationLoss'] for row in window[:5])/5
    val_end = math.fsum(row['validationLoss'] for row in window[-5:])/5
    gap = (val_end-train_end)-(val_start-train_start)
    triggered = (train_start-train_end >= max(.02, train_start*.1) and
                 val_end-val_start >= max(.02, val_start*.1) and gap >= .05)
    return dict(triggered=triggered, withheld=None, trainStart=train_start, trainEnd=train_end,
                validationStart=val_start, validationEnd=val_end, gapIncrease=gap,
                observedEpochs=len(window), evaluatedRuleID=RULE_ID)


def chronological_preparation():
    graph = preparation_graph(25)
    split = next(node for node in graph['nodes'] if node['id'] == 'split')
    split.update(operator='kwip.data.partition', config=dict(strategy='time',timeColumn='observed_at', timeEncoding='iso8601Offset',
        trainingEnd='2026-01-01T00:47:00Z', validationEnd='2026-01-01T01:03:00Z',
        embargoMilliseconds=0, entityColumns=[], entityPolicy='allowAcrossTime'))
    return graph


def public_source(invert_heldout_labels):
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    pattern = [-4., -3., -2., -1., 1., 2., 3., 4.]
    rows = ['observed_at,signal,context,label']
    for index in range(80):
        signal = pattern[index % 8]
        if index >= 48:
            signal += math.copysign(.05 if index < 64 else .1, signal)
        label = int(signal > 0)
        if invert_heldout_labels and 48 <= index < 64:
            label = 1-label
        stamp = (start+datetime.timedelta(minutes=index)).isoformat().replace('+00:00', 'Z')
        context = -.25 if index % 2 == 0 else .25
        rows.append(f'{stamp},{signal},{context},{label}')
    return '\n'.join(rows)+'\n'


def generate(destination, *, maximum_seconds=90):
    destination = validate_destination(destination)
    if type(maximum_seconds) is not int or not 1 <= maximum_seconds <= 300:
        raise ValueError('Maximum corpus time must be 1–300 seconds.')
    source_paths = [Path(__file__), ROOT/'scripts/measured_diagnostic_training_fixtures.py',
                    ROOT/'Sources/KwipMLApp/RunDiagnostics.swift', ROOT/'Sources/KwipMLApp/DiagnosticRuleRegistry.swift',
                    ROOT/'Sources/KwipMLCore/Resources/components.json', ROOT/'runner/dataflow-operators.json']
    source_paths += sorted((ROOT/'runner').glob('*.py'))
    source_pins = [dict(path=str(path.relative_to(ROOT)), sha256=digest(path)) for path in source_paths]
    import torch
    from runner.dataflow_cli import run_dataflow
    from runner.prepared_data import load_prepared
    from runner.engine import run
    started = time.monotonic()
    def budget():
        if time.monotonic()-started > maximum_seconds:
            raise TimeoutError('Bounded fixture time exhausted; no completed corpus published.')
    destination.mkdir(parents=True)
    records = []
    for case in fixture_plan():
        budget()
        directory = destination/case['id']; directory.mkdir()
        source = directory/'observations.csv'; source.write_text(public_source(case['invertHeldoutLabels']))
        graph_file = directory/'preparation.json'; write_json(graph_file, chronological_preparation())
        preparation = directory/'prepared'
        run_dataflow(graph_file, {'source': str(source)}, preparation, lambda _: None, check_cancelled=budget)
        prep_manifest = preparation/'manifest.json'; prep_document = json.loads(prep_manifest.read_text())
        prepared = load_prepared(prep_manifest, 'training', 'table', 'validation', 'table', ['signal', 'context'], 'label',
            expected_manifest_sha256=digest(prep_manifest), expected_graph_digest=prep_document['graphDigest'], sample_shape=[2])
        data, provenance, _ = prepared
        if len(data[0]) != 48 or len(data[2]) != 16:
            raise ValueError('Authored chronological partition did not retain the declared row counts.')
        events = []
        output = directory/'trained'
        run(model_graph(1), output, epochs=20, seed=42, device='cpu', batch_size=48, learning_rate=.03,
            prepared=prepared, gradient_capture=dict(schemaVersion=1, everyEpochs=1, maxBatchesPerEpoch=1, maxCaptures=20),
            emit=events.append, check_cancelled=budget)
        write_json(output/'events.json', events)
        manifest = json.loads((output/'manifest.json').read_text())
        metrics = json.loads((output/'metrics.json').read_text())
        facts = generalization_facts(metrics)
        # Retain failed hypotheses as raw runs, but never label them accepted controls.
        if facts['triggered'] != case['invertHeldoutLabels']:
            write_json(directory/'unmet-expectation.json', dict(configuration=case, observed=facts))
            raise ValueError(f"First designed attempt did not meet its declared trend expectation: {case['id']}; raw metrics retained.")
        if manifest['lineage']['mode'] != 'newTraining':
            raise ValueError('This corpus requires fresh training, not continuation.')
        records.append(dict(id=case['id'], configuration=case, runID=manifest['runID'],
            sourceArtifact=str(source.relative_to(destination)), preparationGraph=str(graph_file.relative_to(destination)),
            preparationManifest=str(prep_manifest.relative_to(destination)), modelManifest=str((output/'manifest.json').relative_to(destination)),
            metricsArtifact=str((output/'metrics.json').relative_to(destination)), eventsArtifact=str((output/'events.json').relative_to(destination)),
            gradientArtifact=str((output/'gradient-summary.json').relative_to(destination)), expectations=facts,
            scopedExpectedFindings=[RULE_ID] if facts['triggered'] else [],
            pins=dict(sourceSHA256=digest(source), preparationGraphSHA256=digest(graph_file),
                preparationManifestSHA256=digest(prep_manifest), modelGraphSHA256=digest(output/'graph.json'),
                modelManifestSHA256=digest(output/'manifest.json'), metricsSHA256=digest(output/'metrics.json'),
                gradientSHA256=digest(output/'gradient-summary.json'))))
        print(json.dumps(dict(case=case['id'], expectations=facts)), flush=True)
    budget()
    if any(digest(ROOT/pin['path']) != pin['sha256'] for pin in source_pins):
        raise ValueError('Source changed during generation; no completed corpus published.')
    files = [dict(path=str(path.relative_to(destination)), bytes=path.stat().st_size, sha256=digest(path))
             for path in sorted(destination.rglob('*')) if path.is_file()]
    receipt = dict(schemaVersion=1, corpusID=CORPUS_ID, authoredAt='2026-10-08',
        runtime=dict(python=platform.python_version(), pytorch=str(torch.__version__), device='cpu', threads=torch.get_num_threads()),
        limits=dict(maximumSeconds=maximum_seconds, actualSeconds=time.monotonic()-started, trainedEpochs=40),
        limitations='Two designed fresh runs: 80 public source rows per case, 48 training + 16 validation rows used, 16 future test rows reserved and unevaluated. The positive deliberately inverts held-out labels: label shift is the known confounder. It demonstrates the falling-training/rising-validation heuristic, not memorization, proof of overfitting or field calibration. The control only lacks this specific trend. Other diagnostic rules, App archive/native replay and release acceptance are outside this corpus.',
        sourcePins=source_pins, cases=records, files=files)
    verify_inventory(destination, receipt)
    write_json(destination/'corpus.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--maximum-seconds', type=int, default=90)
    arguments = parser.parse_args()
    result = generate(arguments.destination, maximum_seconds=arguments.maximum_seconds)
    print(json.dumps(dict(corpusID=result['corpusID'], cases=len(result['cases']), seconds=result['limits']['actualSeconds'])))

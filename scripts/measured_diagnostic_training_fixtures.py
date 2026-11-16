#!/usr/bin/env python3
"""Bounded real CPU training corpus for measured diagnostic replay.

This is a designed, single-source case corpus, not population/field calibration.
Artifacts remain under ignored .local; no gradients, metrics or weights are patched.
"""
import argparse
import csv
import datetime
import io
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ID = 'kwip.cpu-trained-diagnostic-controls.v1'
SOURCE = ROOT / 'Sources/KwipMLApp/Resources/workflow-observations.csv'


def fixture_plan():
    names = ['ordinary-baseline', 'gradient-growth', 'lowered-rate-control',
             'tiny-holdout', 'insufficient-gradient-window', 'exact-continuation']
    records = []
    for name in names:
        final = 23 if name == 'exact-continuation' else 3 if name in names[3:5] else 20
        initial = 20 if name == 'exact-continuation' else 0
        records.append(dict(id=name, depth=8 if name in names[1:3] else 1,
                            learningRate=.3 if name == 'gradient-growth' else .03,
                            epochs=final, initialEpoch=initial, trainedEpochs=final-initial,
                            maxCaptures=2 if name == 'insufficient-gradient-window' else final-initial,
                            validationPercent=10 if name == 'tiny-holdout' else 25,
                            device='cpu', seed=42, historicalPopulationSeed=42, partitionStrategy='time',
                            authoredPopulationVersion=1, reservedTestRows=16, batchSize=32))
    return records


def validate_destination(destination):
    destination = Path(destination).resolve()
    local = (ROOT / '.local').resolve()
    if destination.exists():
        raise ValueError('Refusing existing destination; use a new fixture directory.')
    if not destination.is_relative_to(local):
        raise ValueError('Corpus artifacts must remain under ignored .local.')
    return destination


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def gradient_growth_facts(artifact, *, continuation):
    if continuation:
        return dict(triggered=False, withheld='continuation')
    groups = {}
    for summary in artifact['summaries']:
        norms = [batch['l2Norm'] for batch in summary['batches']]
        if norms:
            if any(n is None or not math.isfinite(n) for n in norms):
                return dict(triggered=False, withheld='nonfinite-or-absent-norm')
            groups[summary['epoch']] = math.fsum(norms) / len(norms)
    epochs = sorted(groups)[-20:]
    if len(epochs) < 10:
        return dict(triggered=False, withheld='insufficient-window', capturedEpochs=len(epochs))
    first = math.fsum(groups[e] for e in epochs[:5]) / 5
    last = math.fsum(groups[e] for e in epochs[-5:]) / 5
    return dict(triggered=last >= max(first, 1e-6)*100 and last-first >= 1,
                withheld=None, firstMean=first, lastMean=last, ratio=last/max(first, 1e-6),
                capturedEpochs=len(epochs))


# Frozen public fixture row orders from the original checker-owned numerical
# controls. These are authored populations, not a legacy split implementation.
AUTHORED_POPULATIONS = {
    25: ([41,4,27,43,34,56,45,23,35,28,58,19,51,48,31,7,8,12,6,42,22,40,0,39,2,61,9,24,57,33,29,46,32,16,18,49,1,26,63,30,47,5,20,25,60,59,55,50],
         [21,14,38,37,3,10,11,52,44,54,17,36,53,62,13,15]),
    10: ([36,51,57,49,58,34,9,10,54,61,12,29,41,22,31,17,7,8,27,6,45,63,24,42,16,3,20,46,0,30,26,44,40,23,28,35,59,43,25,55,39,48,4,56,2,11,60,19,18,53,37,52,1,50,47,5,33,32],
         [15,14,38,62,21,13]),
}


def authored_source(validation_percent):
    """Retain original ordered training/validation matrices; reserve 16 new rows.

    The additional public test observations are never used to fit or select the
    trained diagnostic controls. Current time Partition reads explicit roles from
    chronological boundaries; no old random split is executed or upgraded.
    """
    training, validation = AUTHORED_POPULATIONS[validation_percent]
    original = list(csv.DictReader(SOURCE.read_text().splitlines()))
    if len(original) != 64 or set(training).intersection(validation) or sorted(training+validation) != list(range(64)):
        raise ValueError('Frozen public diagnostic population no longer matches its 64-row source.')
    stream = io.StringIO(); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['observed_at','original_row','signal','context','label'])
    start = datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc)
    values = [(str(index), original[index]) for index in training+validation]
    for index in range(16):
        signal=(-1 if index%2==0 else 1)*(2+index/16)
        values.append((f'reserved-test-{index}',dict(signal=signal,context=3+index/16,label=int(signal>0))))
    for position,(identity,row) in enumerate(values):
        stamp=(start+datetime.timedelta(minutes=position)).isoformat().replace('+00:00','Z')
        writer.writerow([stamp,identity,row['signal'],row['context'],row['label']])
    return stream.getvalue()


def preparation_graph(validation_percent):
    from runner.preprocess_parameters import rule_defaults
    training, _ = AUTHORED_POPULATIONS[validation_percent]
    end=(datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc)+datetime.timedelta(minutes=len(training)-1)).isoformat().replace('+00:00','Z')
    columns=dict(rows=[dict(source=name,included=True,outputName=name,targetType=dtype) for name,dtype in
        [('signal','float64'),('context','float64'),('label','int64')]],castFailure='error',trimText=False,newColumns='keep',missingPolicy='error')
    specs = [('source','sourceCSV',{'separator':','}),('cast','columns',columns),
             ('split','partition',dict(strategy='time',timeColumn='observed_at',timeEncoding='iso8601Offset',
                 trainingEnd=end,validationEnd='2026-01-01T01:03:00Z',embargoMilliseconds=0,entityColumns=[],entityPolicy='allowAcrossTime')),
             ('fit','preprocess',dict(mode='partitions',columns=[dict(rule_defaults(name),missing='error',scaling='standard') for name in ['signal','context']])),
             ('training','view',{}),('validation','view',{}),('test','view',{})]
    wires=[('source','table','cast','table'),('cast','table','split','table')]
    for role in ['training','validation','test']:
        wires += [('split',role,'fit',role),('fit',role,role,'table')]
    return {'schemaVersion':1,'nodes':[{'id':i,'operator':'kwip.data.'+op,'config':config} for i,op,config in specs],
            'edges':[{'from':{'node':a,'port':b},'to':{'node':c,'port':d}} for a,b,c,d in wires]}


def model_graph(depth):
    widths = [2] + [16]*(depth-1) + [2]
    nodes = [{'id': 'input', 'type': 'input', 'name': 'Two prepared features', 'x': 0, 'y': 0, 'config': {'features': 2}}]
    for i in range(depth):
        nodes.append({'id': f'affine-{i}', 'type': 'linear', 'name': f'Affine stage {i+1}', 'x': 0, 'y': 0,
                      'config': {'in_features': widths[i], 'out_features': widths[i+1]}})
    nodes.append({'id': 'output', 'type': 'output', 'name': 'Class scores', 'x': 0, 'y': 0, 'config': {}})
    return {'schemaVersion': 1, 'name': 'Designed affine training control', 'nodes': nodes,
            'edges': [{'id': f'wire-{i}', 'source': nodes[i]['id'], 'sourcePort': 'out',
                       'target': nodes[i+1]['id'], 'targetPort': 'in'} for i in range(len(nodes)-1)]}


def observed_expectations(manifest, gradients, metrics):
    """Independent arithmetic checks; Swift registry replay remains root-owned."""
    continued = manifest['lineage']['mode'] != 'newTraining'
    growth = gradient_growth_facts(gradients, continuation=continued)
    batches = [b for s in gradients['summaries'] for b in s['batches']]
    assert batches and all(b['presentElementCount'] > 0 and b['missingParameterCount'] == 0 and b['nonfiniteElementCount'] == 0 for b in batches)
    assert all(b['l2Norm'] is not None and math.isfinite(b['l2Norm']) for b in batches)
    rows = manifest['provenance']['validationSamples']
    findings = ['data.coarse-holdout.v1'] if rows < 20 else []
    if growth['triggered']:
        findings.append('gradient.growth.v1')
    if len({s['epoch'] for s in gradients['summaries'] if s['batches']}) >= 3 and all(b['zeroElementCount'] == b['presentElementCount'] for b in batches):
        findings.append('gradient.zero.v1')
    plateau = dict(withheld='continuation' if continued else 'insufficient-window', triggered=False)
    if not continued and len(metrics) >= 20:
        window = metrics[-20:]
        assert all(math.isfinite(m['trainLoss']) and m['trainLoss'] >= 0 for m in window)
        assert all(b['epoch'] == a['epoch']+1 for a, b in zip(window, window[1:]))
        first = math.fsum(m['trainLoss'] for m in window[:5])/5
        last = math.fsum(m['trainLoss'] for m in window[-5:])/5
        plateau = dict(withheld=None, triggered=last >= .5 and abs(last-first) <= max(first, 1e-6)*.01, firstMean=first, lastMean=last)
        if plateau['triggered']:
            findings.append('optimization.plateau.v1')
    return dict(expectedFindings=sorted(findings), growth=growth, plateau=plateau,
                validationRows=rows, accuracyStep=1/rows, capturedBatches=len(batches),
                withheldAlways=['activation.zero-sample.v1', 'activation.saturation-sample.v1', 'runtime.memory.v1'],
                limitations='Linear-only models have no ReLU/Tanh activation evidence; no runtime error is supplied. Healthy means only the explicitly checked symptoms were absent.')


def verify_inventory(destination, manifest):
    destination = Path(destination).resolve()
    for entry in manifest['files']:
        path = destination / entry['path']
        if path.is_symlink() or not path.resolve().is_relative_to(destination) or not path.is_file():
            raise ValueError('Unsafe/missing fixture artifact')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Fixture artifact hash/size mismatch')


def generate(destination, *, maximum_seconds=120):
    destination = validate_destination(destination)
    if not 1 <= maximum_seconds <= 300:
        raise ValueError('Maximum corpus time must be 1–300 seconds.')
    source_paths = [Path(__file__), SOURCE, ROOT/'Sources/KwipMLApp/MeasuredDiagnosticRules.swift', ROOT/'Sources/KwipMLCore/Resources/components.json']
    source_paths += sorted((ROOT/'runner').glob('*.py')) + [ROOT/'runner/dataflow-operators.json']
    source_pins = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in source_paths]
    sys.path.insert(0, str(ROOT))
    import torch
    from runner.dataflow_cli import run_dataflow
    from runner.prepared_data import load_prepared
    from runner.engine import run
    started = time.monotonic()
    def budget():
        if time.monotonic()-started > maximum_seconds:
            raise TimeoutError('Bounded corpus time exhausted; no completed corpus published.')
    destination.mkdir(parents=True)
    source = destination / 'observations.csv'
    source.write_bytes(SOURCE.read_bytes())
    prepared = {}
    prepared_sources = {}
    records = []
    for case in fixture_plan():
        budget()
        percent = case['validationPercent']
        if percent not in prepared:
            source = destination / f'authored-observations-{percent}.csv'
            source.write_text(authored_source(percent))
            prepared_sources[percent] = source
            graph_path = destination / f'preparation-{percent}.json'
            write_json(graph_path, preparation_graph(percent))
            output = destination / f'preparation-{percent}'
            run_dataflow(graph_path, {'source': str(source)}, output, lambda _: None, check_cancelled=budget)
            prepared[percent] = load_prepared(output/'manifest.json', 'training', 'table', 'validation', 'table',
                                            ['signal', 'context'], 'label', expected_manifest_sha256=digest(output/'manifest.json'))
        source = prepared_sources[percent]
        graph = model_graph(case['depth'])
        output = destination / case['id']
        events = []
        options = dict(epochs=case['epochs'], seed=42, device='cpu', batch_size=32, learning_rate=case['learningRate'],
                       prepared=prepared[percent], gradient_capture={'schemaVersion': 1, 'everyEpochs': 1,
                       'maxBatchesPerEpoch': 1, 'maxCaptures': case['maxCaptures']}, emit=events.append, check_cancelled=budget)
        if case['initialEpoch']:
            options['resume_checkpoint'] = destination/'ordinary-baseline/checkpoint.pt'
        run(graph, output, **options)
        write_json(output/'events.json', events)
        manifest = json.loads((output/'manifest.json').read_text())
        gradients = json.loads((output/'gradient-summary.json').read_text())
        metrics = json.loads((output/'metrics.json').read_text())
        facts = observed_expectations(manifest, gradients, metrics)
        assert manifest['settings']['initialEpoch'] == case['initialEpoch']
        assert manifest['settings']['epochsExecuted'] == case['trainedEpochs']
        assert facts['growth']['triggered'] == (case['id'] == 'gradient-growth'), (case['id'], facts)
        assert facts['expectedFindings'] == sorted(['data.coarse-holdout.v1'] + (['gradient.growth.v1'] if case['id'] == 'gradient-growth' else [])), (case['id'], facts)
        records.append({'id': case['id'], 'configuration': case, 'runID': manifest['runID'],
                        'sourceArtifact':str(source.relative_to(destination)),
                        'authoredPopulations':dict(trainingOriginalRows=AUTHORED_POPULATIONS[percent][0],validationOriginalRows=AUTHORED_POPULATIONS[percent][1],reservedTestRows=16),
                        'modelManifest': str((output/'manifest.json').relative_to(destination)),
                        'gradientArtifact': str((output/'gradient-summary.json').relative_to(destination)),
                        'metricsArtifact': str((output/'metrics.json').relative_to(destination)),
                        'pins': {'gradientSHA256': digest(output/'gradient-summary.json'),
                                 'modelManifestSHA256': digest(output/'manifest.json'),
                                 'metricsSHA256': digest(output/'metrics.json'),
                                 'preparationManifestSHA256': digest(destination/f'preparation-{percent}/manifest.json'),
                                 'sourceSHA256': digest(source), 'modelGraphSHA256': digest(output/'graph.json'),
                                 'captureConfigurationSHA256': hashlib.sha256(json.dumps(options['gradient_capture'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()},
                        'expectations': facts})
        print(json.dumps({'case': case['id'], 'expectations': facts}), flush=True)
    budget()
    if any(digest(ROOT/pin['path']) != pin['sha256'] for pin in source_pins):
        raise ValueError('Source changed during generation; no completed corpus published.')
    files = [{'path': str(p.relative_to(destination)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(destination.rglob('*')) if p.is_file()]
    corpus = {'schemaVersion': 1, 'corpusID': CORPUS_ID, 'authoredAt': '2026-10-08',
              'runtime': {'python': platform.python_version(), 'pytorch': str(torch.__version__), 'device': 'cpu', 'threads': torch.get_num_threads()},
              'limits': {'maximumSeconds': maximum_seconds, 'actualSeconds': time.monotonic()-started, 'trainedEpochs': 69},
              'limitations': 'Six deliberately chosen runs retain the original ordered training/validation matrices from one 64-row synthetic source. Each authored current source adds 16 distinct reserved test observations; time Partition preserves explicit population boundaries. Reserved tests are not evaluated. These are trained numerical controls, not field precision, cross-dataset generalization, App archive/native acceptance or Gate F. Exact continuation retains historical metrics but fresh-run growth/plateau checks must be withheld.',
              'sourcePins': source_pins, 'cases': records, 'files': files}
    verify_inventory(destination, corpus)
    write_json(destination/'corpus.json', corpus)
    return corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--maximum-seconds', type=int, default=120)
    arguments = parser.parse_args()
    result = generate(arguments.destination, maximum_seconds=arguments.maximum_seconds)
    print(json.dumps({'corpusID': result['corpusID'], 'cases': len(result['cases']), 'seconds': result['limits']['actualSeconds']}))

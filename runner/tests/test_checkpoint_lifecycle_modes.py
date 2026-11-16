"""One real sealed-data lifecycle matrix; expected inference uses scalar arithmetic."""
import copy
import hashlib
import json
import math
import shutil

import pytest
import torch

from runner import engine
from runner.optimizer_configuration import canonical_configuration as optimizer_configuration
from runner.scheduler_configuration import canonical_configuration as scheduler_configuration
from runner.stateful_normalization_configuration import normalize_stateful_normalization
from runner.tests.test_checkpoint_lifecycle import assert_nested_equal, rewrite_checkpoint, state
from runner.tests.test_loss_engine import prepared


@pytest.fixture(autouse=True)
def restore_global_policy():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn)


def fixture(root, device='cpu', lr=.02):
    graph, objective, mapping, data = prepared(root, 'loss_mse')
    # Current shared preparation is an explicit time partition of all 16 rows:
    # 0..7 train, 8..11 validation, 12..15 untouched test. Pin that population
    # independently so lifecycle step expectations cannot silently follow a
    # changed loader count or lose heldout observations.
    manifest=json.loads((root/'preparation/manifest.json').read_bytes())
    observed_ids={}
    for role,positions in [('training',range(8)),('validation',range(8,12)),('test',range(12,16))]:
        artifact=next(entry for entry in manifest['artifacts']
                      if entry.get('nodeID')=='p' and entry.get('port')==role)
        table=json.loads((root/'preparation'/artifact['path']).read_bytes())
        names=[column['name'] for column in table['columns']]
        assert [row[names.index('fixture_order')] for row in table['rows']]==list(positions)
        observed_ids[role]=table['lineage']['observationIDs']
        assert len(observed_ids[role])==len(positions)
    assert len(set().union(*map(set,observed_ids.values())))==16
    for a,b in [('training','validation'),('training','test'),('validation','test')]:
        assert set(observed_ids[a]).isdisjoint(observed_ids[b])
    assert data[1]['trainingRowIDs']==observed_ids['training']
    assert data[1]['validationRowIDs']==observed_ids['validation']
    assert len(data[0][0])==8 and len(data[0][2])==4
    graph['nodes'][1:1] = [
        dict(id='norm', type='batch_norm1d', config=normalize_stateful_normalization('batch_norm1d', {})),
        dict(id='drop', type='dropout', config={'p': .25}),
    ]
    graph['edges'] = [dict(id=f'e{i}', source=a['id'], sourcePort='out', target=b['id'], targetPort='in')
                      for i, (a, b) in enumerate(zip(graph['nodes'], graph['nodes'][1:]))]
    optimizer = dict(schemaVersion=1, componentID='optimizer_sgd', configurationVersion=1,
                     config=optimizer_configuration('optimizer_sgd', dict(lr=lr, momentum=.5, weight_decay=.01)))
    scheduler = dict(schemaVersion=1, componentID='scheduler_step_lr', configurationVersion=1,
                     config=scheduler_configuration('scheduler_step_lr', dict(step_size=1, gamma=.5)))
    options = dict(seed=41, batch_size=5, device=device, prepared=data, objective_recipe=objective,
                   target_mapping=mapping, optimizer_recipe=optimizer, scheduler_recipe=scheduler)
    return graph, options


def inventory(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}


def scalar_prediction(weights, rows):
    # No adapter/native normalization or matrix multiplication in the oracle.
    mean = weights['layers.norm.running_mean'].tolist()
    variance = weights['layers.norm.running_var'].tolist()
    scale = weights['layers.norm.weight'].tolist()
    offset = weights['layers.norm.bias'].tolist()
    matrix = weights['layers.learn.weight'].tolist()
    bias = weights['layers.learn.bias'].tolist()
    result = []
    for row in rows:
        normalized = [scale[c] * (row[c] - mean[c]) / math.sqrt(variance[c] + 1e-5) + offset[c] for c in range(2)]
        result.append([bias[o] + sum(matrix[o][c] * normalized[c] for c in range(2)) for o in range(2)])
    return torch.tensor(result, dtype=torch.float32)


def test_continuation_save_and_cpu_restore_all_actual_training_state(tmp_path):
    graph, options = fixture(tmp_path/'source')
    def consume(event):
        if event['event'] == 'epoch': torch.rand(7)
    engine.run(graph, tmp_path/'full', epochs=4, emit=consume, **options)
    engine.run(graph, tmp_path/'parent', epochs=2, emit=consume, **options)
    pins = inventory(tmp_path/'parent')
    engine.run(graph, tmp_path/'resumed', epochs=4, resume_checkpoint=tmp_path/'parent/checkpoint.pt', emit=consume, **options)
    full, resumed = state(tmp_path/'full'), state(tmp_path/'resumed')
    for key in ('modelState', 'optimizerState', 'optimizerUpdates', 'rngState', 'objectiveState', 'scheduler'):
        assert_nested_equal(full[key], resumed[key])
    for name in ('metrics.json', 'rate-trace.json'):
        assert json.loads((tmp_path/'full'/name).read_text()) == json.loads((tmp_path/'resumed'/name).read_text())
    assert full['modelState']['layers.norm.num_batches_tracked'].item() == 8  # four epochs × two batches
    lineage = json.loads((tmp_path/'resumed/manifest.json').read_text())['lineage']
    assert {'parameters', 'buffers', 'optimizer', 'randomGenerator', 'epoch', 'metrics', 'objective', 'objectiveBuffers', 'scheduler', 'rateTrace'} == set(lineage['restoredState'])
    assert lineage['guarantee'] == 'epochBoundaryExactCPU'
    assert inventory(tmp_path/'parent') == pins


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_weights_save_and_read_are_portable_eval_without_training_state_restore(tmp_path, device):
    if device == 'mps' and not torch.backends.mps.is_available(): pytest.skip('MPS unavailable')
    graph, options = fixture(tmp_path/'source', device)
    engine.run(graph, tmp_path/'saved', epochs=2, **options)
    saved = state(tmp_path/'saved'); pins = inventory(tmp_path/'saved')
    # A fresh portable model receives only learned model state, never optimizer or RNG.
    namespace = {}; exec((tmp_path/'saved/model.py').read_text(), namespace)
    model = namespace['Model']().to(device)
    model.load_state_dict(saved['modelState'], strict=True); model.eval()
    rows = options['prepared'][0][2].tolist()
    expected = scalar_prediction(saved['modelState'], rows)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        actual = model(torch.tensor(rows, device=device)).cpu()
        micro = torch.cat([model(torch.tensor([row], device=device)).cpu() for row in rows])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(micro, expected, rtol=2e-5, atol=2e-5)
    assert torch.equal(rng, torch.get_rng_state())
    assert saved['optimizerState']['state'] and saved['scheduler'] and saved['objectiveState'] == {}
    assert inventory(tmp_path/'saved') == pins


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_finetune_transfers_parameters_and_buffers_then_resets_training_state(tmp_path, monkeypatch, device):
    if device == 'mps' and not torch.backends.mps.is_available(): pytest.skip('MPS unavailable')
    graph, options = fixture(tmp_path/'source', device)
    engine.run(graph, tmp_path/'parent', epochs=2, **options)
    parent = state(tmp_path/'parent'); pins = inventory(tmp_path/'parent'); transferred = []
    original = engine.GraphModel.load_state_dict
    def observe(self, values, *args, **kwargs):
        result = original(self, values, *args, **kwargs)
        transferred.append({key: value.detach().cpu().clone() for key, value in self.state_dict().items()})
        return result
    monkeypatch.setattr(engine.GraphModel, 'load_state_dict', observe)
    child_options = copy.deepcopy(options)
    child_options['seed'] = 77
    child_options['optimizer_recipe']['config']['lr'] = 0.
    engine.run(graph, tmp_path/'child', epochs=1, finetune_checkpoint=tmp_path/'parent/checkpoint.pt', **child_options)
    assert len(transferred) == 1
    assert_nested_equal(transferred[0], parent['modelState'])
    child = state(tmp_path/'child')
    for key in ('layers.norm.weight', 'layers.norm.bias', 'layers.learn.weight', 'layers.learn.bias'):
        assert torch.equal(child['modelState'][key], parent['modelState'][key])
    assert parent['modelState']['layers.norm.num_batches_tracked'].item() == 4
    assert child['modelState']['layers.norm.num_batches_tracked'].item() == 6  # restored four + two new batches
    assert child['epoch'] == 1 and child['seed'] == 77
    assert not torch.equal(child['rngState'], parent['rngState'])
    assert child['optimizerState']['state'].keys() == parent['optimizerState']['state'].keys()
    assert any(not torch.equal(child['optimizerState']['state'][key]['momentum_buffer'], parent['optimizerState']['state'][key]['momentum_buffer']) for key in child['optimizerState']['state'])
    assert all(torch.isfinite(entry['momentum_buffer']).all() for entry in child['optimizerState']['state'].values())
    assert all(value == 2 for value in child['optimizerUpdates'])
    assert all(value == 4 for value in parent['optimizerUpdates'])
    trace = json.loads((tmp_path/'child/rate-trace.json').read_text())
    assert trace['initialization']['tick'] == 0 and len(trace['records']) == 1
    lineage = json.loads((tmp_path/'child/manifest.json').read_text())['lineage']
    assert set(lineage['restoredState']) == {'parameters', 'buffers'}
    assert {'optimizer', 'randomGenerator', 'epoch', 'metrics', 'objective', 'objectiveBuffers', 'scheduler', 'rateTrace'} == set(lineage['resetState'])
    assert lineage['guarantee'] == 'weightsOnly'
    assert inventory(tmp_path/'parent') == pins


@pytest.mark.parametrize('change', ['seed', 'objective', 'scheduler', 'device', 'state', 'integrity'])
def test_incompatible_artifact_refusal_preserves_real_parent(tmp_path, change):
    graph, options = fixture(tmp_path/'source')
    engine.run(graph, tmp_path/'parent', epochs=2, **options)
    pins = inventory(tmp_path/'parent'); supplied = copy.deepcopy(options)
    checkpoint = tmp_path/'parent/checkpoint.pt'
    if change == 'seed': supplied['seed'] = 99
    elif change == 'objective': supplied['objective_recipe']['config']['reduction'] = 'sum'
    elif change == 'scheduler': supplied['scheduler_recipe']['config']['gamma'] = .25
    elif change == 'device': supplied['device'] = 'mps'
    else:
        shutil.copytree(tmp_path/'parent', tmp_path/'altered'); checkpoint = tmp_path/'altered/checkpoint.pt'
        if change == 'state':
            rewrite_checkpoint(tmp_path/'altered', lambda value, _: value['modelState'].__setitem__('layers.norm.running_mean', value['modelState']['layers.norm.running_mean'].double()))
        else:
            checkpoint.write_bytes(checkpoint.read_bytes() + b'altered')
    with pytest.raises(ValueError):
        engine.run(graph, tmp_path/'refused', epochs=3, resume_checkpoint=checkpoint, **supplied)
    assert not (tmp_path/'refused/manifest.json').exists()
    assert inventory(tmp_path/'parent') == pins

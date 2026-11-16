"""Independent inference-only binary classification metric contracts."""
import math

import pytest
import torch

from runner.final_evaluation import evaluate_classifier


class FrozenLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2), requires_grad=True)
        self.register_buffer('offset', torch.zeros(2))
        self.calls = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return x @ self.weight + self.offset


def evaluate(model=None, x=None, y=None, ids=None, **options):
    x = torch.tensor([[2., 0.], [0., 1.], [1., 1.], [-1., 2.], [3., 0.]]) if x is None else x
    y = torch.tensor([0, 0, 1, 1, 0]) if y is None else y
    return evaluate_classifier(model or FrozenLogits(), x, y,
                               ['row-a', 'row-b', 'row-c', 'row-d', 'row-e'] if ids is None else ids,
                               batch_size=options.pop('batch_size', 2), device=options.pop('device', 'cpu'), **options)


def test_uneven_batches_independent_loss_confusion_rows_and_no_training():
    model = FrozenLogits()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = evaluate(model, preview_limit=2)
    losses = [math.log1p(math.exp(-2)), math.log1p(math.exp(1)), math.log(2),
              math.log1p(math.exp(-3)), math.log1p(math.exp(-3))]
    assert result['schemaVersion'] == 1 and result['evaluatedCount'] == 5
    assert result['device'] == 'cpu' and result['batchSize'] == 2
    assert result['lossSum'] == pytest.approx(math.fsum(losses), abs=1e-6)
    assert result['lossMean'] == pytest.approx(math.fsum(losses) / 5, abs=1e-6)
    assert result['accuracy'] == 3 / 5
    assert result['confusionMatrix'] == [[2, 1], [1, 1]]
    assert result['perClass'] == [
        dict(classIndex=0, precision=2/3, precisionUnavailableReason=None,
             recall=2/3, recallUnavailableReason=None, support=3),
        dict(classIndex=1, precision=1/2, precisionUnavailableReason=None,
             recall=1/2, recallUnavailableReason=None, support=2)]
    assert [row['observationID'] for row in result['predictions']] == ['row-a', 'row-b', 'row-c', 'row-d', 'row-e']
    assert [row['predicted'] for row in result['predictions']] == [0, 1, 0, 1, 0]
    assert result['predictions'][2] == dict(observationID='row-c', logits=[1., 1.], predicted=0, target=1)
    assert result['preview'] == result['predictions'][:2]
    assert model.calls == [(False, False, True)] * 3
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training, "Restore the caller model mode after evaluation"


def test_batch_and_preview_sizes_do_not_change_metric_population():
    expected = evaluate(batch_size=1, preview_limit=0)
    for size in [2, 3, 5, 20]:
        actual = evaluate(batch_size=size, preview_limit=1)
        for key in ['lossSum', 'lossMean', 'accuracy']:
            assert actual[key] == pytest.approx(expected[key], abs=1e-6)
        for key in ['confusionMatrix', 'perClass', 'predictions', 'evaluatedCount']:
            assert actual[key] == expected[key]
    assert expected['preview'] == []


def test_single_class_and_undefined_denominators_are_explicit():
    result = evaluate(x=torch.tensor([[2., 0.], [3., 0.]]), y=torch.tensor([0, 0]), ids=['a', 'b'])
    assert result['accuracy'] == 1 and result['confusionMatrix'] == [[2, 0], [0, 0]]
    assert result['perClass'][1] == dict(classIndex=1, precision=None,
        precisionUnavailableReason='noPredictedObservations', recall=None,
        recallUnavailableReason='noActualObservations', support=0)


@pytest.mark.parametrize('options', [dict(batch_size=0), dict(batch_size=True), dict(batch_size=1.5),
    dict(batch_size=100001), dict(preview_limit=-1), dict(preview_limit=True), dict(preview_limit=101),
    dict(device='cuda'), dict(device=None)])
def test_invalid_configuration_rejects_before_forward(options):
    model = FrozenLogits()
    with pytest.raises((ValueError, TypeError)):
        evaluate(model, **options)
    assert not model.calls


@pytest.mark.parametrize('x,y,ids', [
    (torch.empty((0, 2)), torch.empty(0, dtype=torch.int64), []),
    (torch.zeros((2, 2)), torch.tensor([0]), ['a', 'b']),
    (torch.zeros((2, 2)), torch.tensor([0., 1.]), ['a', 'b']),
    (torch.zeros((2, 2)), torch.tensor([False, True]), ['a', 'b']),
    (torch.zeros((2, 2)), torch.tensor([0, 2]), ['a', 'b']),
    (torch.zeros((2, 2)), torch.tensor([0, 1]), ['a', 'a']),
    (torch.zeros((2, 2)), torch.tensor([0, 1]), ['a', '']),
    (torch.zeros((2, 2)), torch.tensor([0, 1]), ['a', 'x' * 201]),
    (torch.zeros((2, 2)), torch.tensor([0, 1]), ['a', 1]),
    (torch.tensor([[float('nan'), 0.]]), torch.tensor([0]), ['a']),
    (torch.tensor([[float('inf'), 0.]]), torch.tensor([0]), ['a']),
    (torch.zeros((100001, 2)), torch.zeros(100001, dtype=torch.int64), ['a'] * 100001),
])
def test_invalid_population_rejects_before_forward(x, y, ids):
    model = FrozenLogits()
    with pytest.raises((ValueError, TypeError)):
        evaluate(model, x=x, y=y, ids=ids)
    assert not model.calls


@pytest.mark.parametrize('output', [torch.zeros((2, 3)), torch.zeros(2), torch.zeros((1, 2)),
    torch.full((2, 2), float('inf')), torch.full((2, 2), float('nan')), torch.zeros((2, 2), dtype=torch.int64)])
def test_invalid_logits_never_publish_metrics(output):
    class Invalid(torch.nn.Module):
        def forward(self, _): return output
    with pytest.raises((ValueError, TypeError)):
        evaluate(Invalid())


@pytest.mark.parametrize('mutation', ['parameter', 'buffer', 'nonpersistent'])
def test_accidental_forward_mutation_prevents_verified_metrics(mutation):
    class Mutating(FrozenLogits):
        def __init__(self):
            super().__init__()
            self.register_buffer('scratch', torch.zeros(1), persistent=False)
        def forward(self, x):
            {'parameter': self.weight, 'buffer': self.offset, 'nonpersistent': self.scratch}[mutation].add_(1)
            return super().forward(x)
    with pytest.raises(ValueError, match='(?i)(mutat|chang|state)'):
        evaluate(Mutating())


def test_integer_features_are_preserved_for_embedding_architectures():
    class IntegerInput(torch.nn.Module):
        def forward(self, x):
            assert x.dtype == torch.int64
            return torch.stack((x[:, 0], -x[:, 0]), dim=1).float()
    result = evaluate(IntegerInput(), x=torch.tensor([[2], [-1]]), y=torch.tensor([0, 1]), ids=['a', 'b'])
    assert result['accuracy'] == 1


def test_float64_loss_matches_independent_math_and_preserves_existing_eval_mode():
    model = FrozenLogits().double().eval()
    x = torch.tensor([[100., -100.], [-100., 100.], [0., 0.]], dtype=torch.float64)
    result = evaluate(model, x=x, y=torch.tensor([1, 1, 0]), ids=['a', 'b', 'c'])
    assert result['lossSum'] == pytest.approx(200 + math.log1p(math.exp(-200)) * 2 + math.log(2), abs=1e-12)
    assert not model.training


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS is unavailable')
def test_mps_transfers_logits_to_host_before_float64_metric_promotion():
    # Explicit logits retain a positive margin in either direction plus one tie;
    # an all-zero device transfer must fail decisions, logits and loss together.
    x=torch.tensor([[2.,0.],[0.,1.],[1.,1.],[-1.,2.],[3.,0.]],dtype=torch.float32)
    y=torch.tensor([0,0,1,1,0],dtype=torch.int64)
    cpu=evaluate(x=x,y=y,device='cpu',batch_size=2,preview_limit=5)
    mps=evaluate(x=x,y=y,device='mps',batch_size=2,preview_limit=5)
    assert mps['device']=='mps'
    assert [p['logits'] for p in mps['predictions']]==x.tolist()
    assert [p['predicted'] for p in mps['predictions']]==[0,1,0,1,0]
    assert mps['confusionMatrix']==cpu['confusionMatrix']==[[2,1],[1,1]]
    assert mps['accuracy']==cpu['accuracy']==3/5
    expected=math.fsum([math.log1p(math.exp(-2)),math.log1p(math.exp(1)),math.log(2),math.log1p(math.exp(-3)),math.log1p(math.exp(-3))])
    assert mps['lossSum']==pytest.approx(expected,abs=1e-12)
    assert mps['perClass']==cpu['perClass']

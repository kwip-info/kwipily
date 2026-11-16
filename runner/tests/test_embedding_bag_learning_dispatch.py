"""Typed bag lessons through the existing pinned learning-evidence boundary."""
import copy
import hashlib
import json
import pytest
from runner import learning_evidence as worker
from runner.tests.test_learning_embedding_bag import fixture, submission

TASKS = ('bag-fixed-sum', 'bag-packed-mean', 'bag-packed-max',
         'bag-fixed-weighted-sum', 'bag-packed-weighted-sum', 'bag-padding-norm')


def case(task):
    layout = 'fixed' if task.startswith('bag-fixed') else 'packed'
    weighted = 'weighted' in task
    mode = 'mean' if task.endswith('mean') else 'max' if task.endswith('max') else 'sum'
    changes = dict(padding_idx=1, max_norm=1.) if task == 'bag-padding-norm' else {}
    construction = submission(layout=layout, mode=mode, weighted=weighted, **changes)
    fixtures = []
    for index in range(4):
        value = fixture(weighted=weighted, layout=layout)
        value['id'] = f'private-{index}'
        fixtures.append(value)
    request = dict(assessmentID='assessment.' + task,
        assessmentVersion=dict(major=1, minor=0, patch=0), fixtureDigest='a' * 64,
        runtime=dict(backend='pytorch', version=worker.runtime_versions()['pytorch'],
            capabilities={key:dict(major=1, minor=0, patch=0) for key in ('bag_input', 'embedding_bag')}),
        budget=dict(milliseconds=60000, maximumTensorElements=4096), fixtures=fixtures)
    return request, construction


@pytest.mark.parametrize('task', TASKS)
def test_six_policies_route_real_measurements_through_pinned_files(task, tmp_path):
    request, construction = case(task)
    raw = json.dumps(construction, sort_keys=True, separators=(',', ':')).encode()
    request_path, submission_path = tmp_path/'request.json', tmp_path/'submission.json'
    request_path.write_text(json.dumps(request)); submission_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    measured = worker.execute_request(request_path, submission_path, digest)
    assert measured['assessmentID'] == request['assessmentID']
    assert measured['submissionDigest'] == digest
    assert measured['runtime'] == request['runtime']
    assert measured['cancelled'] is False
    assert 'backendError' not in measured
    assert [value['fixtureID'] for value in measured['fixtures']] == [f'private-{i}' for i in range(4)]
    for value in measured['fixtures']:
        tensors = value['tensors']
        assert tensors['output']['shape'] == [3, 2]
        assert tensors['tableAfterForward']['shape'] == [4, 2]
        assert tensors['tableGradient']['shape'] == [4, 2]
        assert ('sourceWeightsGradient' in tensors) == ('weighted' in task)
        assert not {'expected', 'indicesGradient', 'offsetsGradient', 'inputGradient'} & tensors.keys()
        if task == 'bag-fixed-sum':
            assert tensors['output']['values'] == pytest.approx([5, 1, 0, 0, -1, 7])
        elif task == 'bag-packed-mean':
            assert tensors['output']['values'] == pytest.approx([5/3, 1/3, 0, 0, -2, 4])
    with pytest.raises(ValueError, match='digest changed'):
        worker.execute_request(request_path, submission_path, '0' * 64)


@pytest.mark.parametrize('task', TASKS)
def test_runtime_mismatch_and_cancellation_never_gain_measurements(task):
    request, construction = case(task)
    stale = copy.deepcopy(request); stale['runtime']['version'] = '0.0.0'
    result = worker.generate_evidence(stale, construction, submission_digest='b'*64)
    assert result['fixtures'] == []
    cancelled = worker.generate_evidence(request, construction, submission_digest='b'*64, cancel=lambda:True)
    assert cancelled['cancelled'] is True and cancelled['fixtures'] == []


@pytest.mark.parametrize('task', TASKS)
def test_wrong_role_is_bounded_diagnostic_not_success(task):
    request, construction = case(task)
    construction['graph']['edges'][0]['sourcePort'] = 'foreign'
    result = worker.generate_evidence(request, construction, submission_digest='b'*64)
    assert result['fixtures'] == []
    assert result.get('backendError')

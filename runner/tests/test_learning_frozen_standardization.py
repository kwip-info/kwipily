"""Frozen preprocessing expressed as a learner-authored affine graph."""
import pytest
import torch
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_evidence import affine_request, affine_submission, tensor


def request():
    value = affine_request()
    value['assessmentID'] = 'assessment.frozen-standardization'
    value['runtime']['capabilities']['linear'] = {'major':1,'minor':1,'patch':0}
    value['fixtures'] = [{'id':str(i), 'inputs':{'x':tensor([2,2], rows)}} for i,rows in enumerate([
        [2,12,6,4], [202,-388,0,0], [-2,20,3,10], [2,12,202,-388]])]
    return value


def submission(two):
    value = affine_submission(two)
    value['schemaVersion'] = 2; value['graph']['schemaVersion'] = 2
    for node in value['graph']['nodes']:
        if node['type'] == 'linear': node['config']['bias'] = True
    if two:
        weights = {'first':([1,0,0,1],[-2,-12]), 'second':([.5,0,0,.25],[0,0])}
    else: weights = {'first':([.5,0,0,.25],[-1,-3])}
    value['parameters'] = {key:{'weight':tensor([2,2],w),'bias':tensor([2],b)} for key,(w,b) in weights.items()}
    return value


@pytest.mark.parametrize('two',[False,True])
def test_single_or_composed_map_applies_saved_statistics_to_new_rows(two):
    r = request()
    result = generate_evidence(r, submission(two), submission_digest='a'*64)
    assert not result.get('backendError'), result
    assert len(result['fixtures']) == 4
    for measured, fixture in zip(result['fixtures'],r['fixtures']):
        x = torch.tensor(fixture['inputs']['x']['values']).reshape(2,2)
        expected = (x-torch.tensor([2,12]))/torch.tensor([2,4])
        actual = torch.tensor(measured['tensors']['output']['values']).reshape(2,2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The same observation has the same answer with a radically different batch companion.
    assert result['fixtures'][0]['tensors']['output']['values'][:2] == result['fixtures'][3]['tensors']['output']['values'][:2]


def test_cancellation_and_wrong_capability_cannot_claim_measurements():
    cancelled = generate_evidence(request(), submission(False), submission_digest='a'*64, cancel=lambda:True)
    assert cancelled['cancelled'] and not cancelled['fixtures']
    r=request(); r['runtime']['capabilities']['linear']['minor']=0
    mismatch=generate_evidence(r, submission(False), submission_digest='a'*64)
    assert not mismatch['fixtures']

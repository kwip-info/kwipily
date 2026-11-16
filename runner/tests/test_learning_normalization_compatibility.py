"""Retired lesson versions stay non-executable; current Preprocess owns the math."""
import copy
import numpy as np
import pytest
from runner.learning_evidence import generate_evidence
from runner.tests.test_learning_evidence import tensor, V

def normalization():
    request={'assessmentID':'assessment.training-standardization','assessmentVersion':dict(V),'fixtureDigest':'a'*64,'runtime':{'backend':'numpy','version':np.__version__,'capabilities':{'kwip.data.standardScalerFit':dict(V),'kwip.data.transform':dict(V)}},'budget':{'milliseconds':5000,'maximumTensorElements':4096},'fixtures':[{'id':'heldout-outlier','inputs':{'training':tensor([3,2],[0,10,2,12,4,14]),'validation':tensor([1,2],[12,22])}}]}
    specs=[('s','sourceCSV',{}),('c','cast',{'columns':{'x':'float64','y':'float64'}}),('split','split',{'seed':42,'validationPercent':25}),('fit','standardScalerFit',{'columns':['x','y']}),('val','transform',{})]
    graph={'schemaVersion':1,'nodes':[{'id':i,'operator':'kwip.data.'+o,'config':c} for i,o,c in specs],'edges':[{'from':{'node':a,'port':ap},'to':{'node':b,'port':bp}} for a,ap,b,bp in [('s','table','c','table'),('c','table','split','table'),('split','training','fit','training'),('split','validation','val','table'),('fit','state','val','state')]]}
    return request,{'schemaVersion':1,'kind':'normalizationGraph','graph':graph,'outputs':{'state':{'node':'fit','port':'state'},'validation':{'node':'val','port':'table'},'training':{'node':'split','port':'training'}}}


@pytest.mark.parametrize('minor', [0, 1, 2])
@pytest.mark.parametrize('forge_current_capabilities', [False, True])
def test_retired_normalization_versions_never_execute_or_publish_new_results(minor, forge_current_capabilities, monkeypatch):
    from runner import dataflow
    request, submission = normalization()
    request['assessmentVersion']['minor'] = minor
    if forge_current_capabilities:
        request['runtime']['capabilities'] = {}
    def forbidden(*args, **kwargs):
        pytest.fail('Retired normalization attempted a new dataflow execution')
    monkeypatch.setattr(dataflow, 'execute', forbidden)
    original = copy.deepcopy((request, submission))
    evidence = generate_evidence(request, submission, submission_digest='b'*64)
    assert evidence['fixtures'] == []
    assert evidence['runtime']['capabilities'] == {}
    assert evidence['assessmentVersion'] == request['assessmentVersion']
    assert (request, submission) == original


def test_retired_normalization_does_not_accept_an_unreviewed_version():
    request, submission = normalization()
    request['assessmentVersion']['minor'] = 3
    with pytest.raises(ValueError):
        generate_evidence(request, submission, submission_digest='b'*64)


@pytest.mark.parametrize('training,heldout', [
    ([[0,10],[2,12],[4,14]],[[12,22]]),
    ([[-10,2],[-6,8],[-2,14]],[[8,-4]]),
    ([[7,-3],[7,0],[7,3]],[[12,9]]),
    ([[3,9],[4,9],[5,9]],[[-2,7]]),
])
def test_current_preprocess_preserves_varied_two_column_population_oracles(training, heldout):
    from runner import dataflow as df
    from runner.dataflow_preprocess import apply, rule_defaults
    columns=tuple({'name':name,'dtype':'float64','nullable':False} for name in ['x','y'])
    def table(rows, role):
        return df.derive(df.Table(columns,tuple(map(tuple,rows)),tuple(f'{role}-{i}' for i in range(len(rows))),'a'*64,role,partition_digest='b'*64))
    train=table(training,'training');validation=table(heldout,'validation')
    config={'mode':'partitions','columns':[dict(rule_defaults(name),scaling='standard') for name in ['x','y']]}
    output=apply({'training':train,'validation':validation},config,'c'*64,lambda:None)
    a=np.array(training,dtype=float);mean=a.mean(axis=0);scale=a.std(axis=0);scale[scale==0]=1
    rules=output['state']['rules']
    np.testing.assert_allclose([rule['mean'] for rule in rules],mean,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose([rule['scale'] for rule in rules],scale,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(list(output['training'].rows),(a-mean)/scale,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(list(output['validation'].rows),(np.array(heldout)-mean)/scale,rtol=1e-10,atol=1e-10)
    assert output['training'].row_ids == train.row_ids
    assert output['validation'].row_ids == validation.row_ids
    assert output['state']['fitEvidence']['observationIDs'] == list(train.row_ids)
    assert set(train.row_ids).isdisjoint(validation.row_ids)


def test_version_mutation_is_private_to_each_request_and_preserves_actual_affine_execution():
    from runner.tests.test_learning_evidence import affine_request, affine_submission
    expected={'major':1,'minor':0,'patch':0}
    changed,_=normalization();sibling,_=normalization();existing_affine=affine_request()
    changed['assessmentVersion']['minor']=3
    changed['runtime']['capabilities']['kwip.data.standardScalerFit']['minor']=2
    assert V==expected
    assert changed['runtime']['capabilities']['kwip.data.transform']==expected
    assert sibling['assessmentVersion']==expected
    assert all(version==expected for version in sibling['runtime']['capabilities'].values())
    assert existing_affine['assessmentVersion']==expected
    assert existing_affine['runtime']['capabilities']['linear']==expected
    future_affine=affine_request()
    evidence=generate_evidence(future_affine,affine_submission(),submission_digest='b'*64)
    assert evidence['fixtures'][0]['tensors']['output']['values']==[.25,-6.5,2.25,2.0]
    # Affine's own assessment and capability versions must not alias either.
    future_affine['assessmentVersion']['minor']=4
    assert future_affine['runtime']['capabilities']['linear']==expected
    assert V==expected and affine_request()['assessmentVersion']==expected
    assert normalization()[0]['assessmentVersion']==expected

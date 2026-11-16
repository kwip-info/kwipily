"""Independent calibration support, held-out groups and pre-load resource admission."""
import hashlib
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
from runner import laya_training as training, laya_training_worker as worker
from runner.tests.test_laya_models import model_folder
from runner.tests.test_laya_training import settings, rows
from runner import dataflow as df
from runner.dataflow_disk import disk_scope


def test_repeated_calibration_rows_do_not_manufacture_independent_support():
    with pytest.raises(ValueError, match='independent|groups'):
        training.fit_calibration([[2., 0.]] * 100, [[1., 0.]] * 100, .1, groups=['one-case'] * 100)


def test_calibration_freezes_policy_before_independent_audit_and_never_retries_cutoff():
    groups = ['case-' + str(i) for i in range(200)]
    order = sorted(groups); random.Random(7).shuffle(order)
    audit = set(order[len(order)//2:])
    logits = [[2., 0.]] * len(groups)
    good = [[1., 0.]] * len(groups)
    passed = training.fit_calibration(logits, good, .1, groups=groups, seed=7)
    # Audit labels cannot influence fitted temperature or selected cutoff.
    bad = [[0., 1.] if group in audit else [1., 0.] for group in groups]
    refused = training.fit_calibration(logits, bad, .1, groups=groups, seed=7)
    assert passed['acceptEnabled'] and passed['acceptedSupport'] == 100
    assert not refused['acceptEnabled']
    assert passed['temperature'] == refused['temperature']
    assert passed['threshold'] == refused['threshold']
    assert passed['selectionGroupsDigest'] == refused['selectionGroupsDigest']
    assert passed['auditGroupsDigest'] == refused['auditGroupsDigest']
    assert passed['selectionGroupsDigest'] != passed['auditGroupsDigest']
    assert refused['observedError'] == 1
    assert passed['supportUnit'] == 'independentSourceGroup'
    # Repeating each identical observation does not add support or change the policy.
    repeated = training.fit_calibration(np.repeat(logits, 3, axis=0), np.repeat(good, 3, axis=0), .1,
                                        groups=np.repeat(groups, 3).tolist(), seed=7)
    for key in ('support', 'acceptedSupport', 'errorUpperBound', 'temperature', 'threshold'):
        assert repeated[key] == pytest.approx(passed[key])


def test_soft_targets_fit_temperature_without_binary_error_acceptance():
    fit = training.fit_calibration([[2., 0.]] * 100, [[.9, .1]] * 100, .2,
                                   groups=[str(i) for i in range(100)])
    assert .5 <= fit['temperature'] <= 5
    assert not fit['acceptEnabled'] and fit['acceptanceReason'] == 'softTargetsRequireReview'
    assert fit['errorUpperBound'] is None


def write_table(root, name, role, group, observation):
    columns = [dict(name=n, dtype='string', nullable=False) for n in ('case_id', 'text', 'team')]
    values = [[group, 'technical crash', 'technical']]; ids = [observation]
    value = dict(schemaVersion=1, kind='table', columns=columns, rows=values,
                 lineage=dict(role=role, sourceDigest='a'*64, partitionDigest='', observationIDs=ids,
                              membershipDigest=df.membership_digest(ids),
                              dataVersion=df.digest(dict(columns=columns, rows=values, ids=ids))))
    path = root / ('input/' + name + '.table.json'); path.parent.mkdir(exist_ok=True)
    raw = json.dumps(value).encode(); path.write_bytes(raw)
    return dict(path='input/' + name + '.table.json', sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('overlap', [True, False])
def test_worker_evaluation_uses_explicit_group_identity_before_disclosure(tmp_path, monkeypatch, overlap):
    files = [write_table(tmp_path, 'training', 'training', 'customer-a', 'train-row'),
             write_table(tmp_path, 'evaluation', 'test', 'customer-a' if overlap else 'customer-b', 'test-row'),
             write_table(tmp_path, 'evaluationPopulation', 'test', 'customer-a' if overlap else 'customer-b', 'test-row')]
    authority = SimpleNamespace(_path=tmp_path, request=dict(inputFiles=files, executionBudget=dict(resources={})))
    called = []
    def evaluate(*args):
        called.append(True)
        return dict(kind='typedDecisionEvaluation', rows=[], loss=0.)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    written = []
    def run():
        with disk_scope(tmp_path):
            worker.execute(authority, dict(action='evaluate', settings=dict(settings(), evaluationRole='test')),
                           dict(identity='b'*64), tmp_path/'output', [], lambda *x: written.append(x), lambda _: None, lambda: None)
    if overlap:
        with pytest.raises(ValueError, match='group'):
            run()
        assert not called and not written
    else:
        run(); assert called and [x[0] for x in written] == ['tokens.json', 'table.json']


@pytest.mark.parametrize('operation', ['train', 'prepare', 'evaluate'])
def test_resource_refusal_precedes_loading_real_weights(model_folder, tmp_path, monkeypatch, operation):
    def forbidden(*args, **kwargs):
        pytest.fail('Resident model constructed before resource refusal')
    monkeypatch.setattr(training, 'Laya', forbidden)
    c = dict(settings(), maximumResidentBytes=100)
    with pytest.raises(ValueError, match='memory|allowance|budget'):
        if operation == 'train': training.fine_tune(model_folder, tmp_path/'out', rows('a', 2), rows('b', 2), c)
        elif operation == 'prepare': training.prepared_preview(model_folder, rows('a', 2), rows('b', 2), c)
        else: training.evaluate(model_folder, rows('b', 2), c)


def test_evaluate_reuses_one_loaded_model(model_folder, monkeypatch):
    original = training.Laya; created = []
    def counted(*args, **kwargs):
        created.append(True); return original(*args, **kwargs)
    monkeypatch.setattr(training, 'Laya', counted)
    report = training.evaluate(model_folder, rows('test-', 2), settings())
    assert len(created) == 1
    assert report['scoredItems'] == 2 and len(report['rows']) == 2


def test_independent_calibration_package_roundtrip_and_legacy_reads(model_folder, tmp_path):
    from runner.laya_models import Laya, inspect_model, export_model
    from runner.typed_decisions import question_set
    q=settings()['questionSet'];engine=Laya(model_folder)
    raw,_=engine.logits([{'text':'technical crash'}],q)
    logits=raw[0,:2].tolist();target=[float(i==int(raw[0,:2].argmax())) for i in range(2)]
    fit=training.fit_calibration([logits]*100,[target]*100,.1,groups=[str(i) for i in range(100)])
    assert fit['acceptEnabled']
    metadata=dict(schemaVersion=1,questionSet=q,calibration=dict(questionHash=question_set(q)['hash'],questions={'team':fit}))
    path=model_folder/'kwip-decisions.json';path.write_text(json.dumps(metadata))
    expected=Laya(model_folder).predict([{'text':'technical crash'}],q)
    assert expected[0][0]['status']=='accepted'
    export_model(model_folder,tmp_path/'export')
    assert Laya(tmp_path/'export').predict([{'text':'technical crash'}],q)==expected
    # Historical packages remain readable without silently rewriting their pins.
    metadata['calibration']['questions']['team']=dict(temperature=1.,threshold=0.,support=100,acceptedSupport=100,acceptEnabled=True,method='temperature-grid-and-Wilson-upper95-v1')
    path.write_text(json.dumps(metadata));before=path.read_bytes()
    info=inspect_model(model_folder)
    assert any('Legacy' in warning for warning in info['calibrationWarnings'])
    assert Laya(model_folder).predict([{'text':'technical crash'}],q)[0][0]['status']=='needsReview'
    assert path.read_bytes()==before


def test_calibration_group_failures_cannot_be_diluted_by_correct_duplicate_rows():
    groups=[str(i) for i in range(200)]
    # One error plus99 correct rows from every case still fails each independent case.
    # Keep the shared complete-population bound;100 groups suffice for this negative.
    repeated=[g for g in groups[:100] for _ in range(100)]
    fit=training.fit_calibration([[2.,0.]]*len(repeated),[[0.,1.] if i%100==0 else [1.,0.] for i in range(len(repeated))],.1,groups=repeated)
    assert not fit['acceptEnabled'] and fit['support']==100


def test_activation_and_population_bounds_are_not_constant(model_folder):
    c=settings()
    small=training._admission(model_folder,c,[rows('a',2)],training=True,check=lambda:None)
    larger=training._admission(model_folder,dict(c,batchSize=8),[rows('a',2)],training=True,check=lambda:None)
    longer=training._admission(model_folder,c,[rows('a',100)],training=True,check=lambda:None)
    shorter=training._admission(model_folder,dict(c,maximumTokens=64),[rows('a',2)],training=True,check=lambda:None)
    assert larger['estimatedResidentBytes']>small['estimatedResidentBytes']
    assert longer['estimatedResidentBytes']>small['estimatedResidentBytes']
    assert shorter['estimatedResidentBytes']<small['estimatedResidentBytes']


def test_prediction_rows_enforce_the_admitted_context_before_forward(model_folder,monkeypatch):
    from runner.laya_models import Laya
    engine=Laya(model_folder,maximum_tokens=8)
    monkeypatch.setattr(engine.model,'forward',lambda *a,**k:pytest.fail('Must refuse context before forward'))
    with pytest.raises(ValueError,match='token budget'):
        engine.predict([{'text':'technical crash'}],settings()['questionSet'])


def test_actual_training_persists_calibration_group_split_and_safe_reload(model_folder, tmp_path):
    from runner.laya_models import Laya, inspect_model, export_model
    c=dict(settings(),epochs=1,calibrationFraction=.5,targetError=.2)
    report=training.fine_tune(model_folder,tmp_path/'trained',rows('train-',80),rows('valid-',4),c)
    package=tmp_path/'trained/model'
    metadata=json.loads((package/'kwip-decisions.json').read_text())
    fit=metadata['calibration']['questions']['team']
    assert report['calibrationGroups']==40 and report['trainingGroups']==40
    assert fit['support']==40 and fit['selectionSupport']==fit['auditSupport']==20
    assert fit['method']==training.CALIBRATION_METHOD
    before=Laya(package).predict([{'text':'technical crash'}],c['questionSet'])
    export_model(package,tmp_path/'standalone')
    after=Laya(tmp_path/'standalone').predict([{'text':'technical crash'}],c['questionSet'])
    assert before==after
    assert inspect_model(package)['identity']==inspect_model(tmp_path/'standalone')['identity']


def test_duplicate_calibration_groups_refused_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(training,'Laya',lambda *a,**k:pytest.fail('Must refuse group support before load'))
    c=dict(settings(),calibrationFraction=.5)
    duplicated=rows('case-',2)*100
    with pytest.raises(ValueError,match='independent.*groups'):
        training.fine_tune(tmp_path/'unused-model',tmp_path/'out',duplicated,rows('valid-',2),c)

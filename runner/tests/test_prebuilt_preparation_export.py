"""Real fitted Preprocess -> model package -> unseen, unlabelled inputs."""
import hashlib
import json
import copy
from pathlib import Path

import numpy as np
import pytest

from runner import prebuilt_models as pb, prebuilt_preparation as frozen
from runner.dataflow_cli import run_dataflow
from runner.prebuilt_workflow import run_prepared
from runner.tests.test_preprocess import practice_graph


def fitted_request(tmp_path, *, steps=1, after=None):
    graph = practice_graph()
    if steps==2:
        from runner.dataflow_preprocess import rule_defaults
        graph['nodes'].append(dict(id='again',operator='kwip.data.preprocess',config=dict(
            mode='partitions',columns=[dict(rule_defaults('x'),scaling='standard')])))
        graph['edges'] += [dict(**{'from':dict(node='prep',port=role),'to':dict(node='again',port=role)})
                           for role in ('training','validation','test')]
    if after:
        from runner.tests.test_formula import operation
        for role in ('training','validation'):
            graph['nodes'].append(dict(id=role,operator='kwip.data.'+after,config=(
                {'operations':[operation('[x] * 2',('x','float64'),write='replace')]} if after=='formula' else {})))
            graph['edges'].append({'from':dict(node='prep',port=role),'to':dict(node=role,port='table')})
    gp = tmp_path / 'graph.json'
    gp.write_text(json.dumps(graph))
    source = Path(__file__).resolve().parents[2] / 'Sources/KwipMLCore/Resources/preprocess-practice-observations.csv'
    prep = tmp_path / 'prep'
    run_dataflow(gp, {'s': str(source)}, prep, lambda _: None)
    raw = (prep / 'manifest.json').read_bytes()
    config = pb.defaults('ridge')
    result=dict(schemaVersion=1, backendID='sklearn', configuration=dict(
        task='prebuilt', algorithm='ridge', operation='fit', settings=config['options']),
        prepared=dict(manifest=str(prep / 'manifest.json'), manifestSHA256=hashlib.sha256(raw).hexdigest(),
                      graphDigest=json.loads(raw)['graphDigest'], trainingNode='again' if steps==2 else 'prep', trainingPort='training',
                      validationNode='again' if steps==2 else 'prep', validationPort='validation',
                      featureColumns=['x', 'onehot:["city","a"]', 'onehot:["city","b"]'], targetColumn='target'))
    if after:result['prepared'].update(trainingNode='training',trainingPort='table',validationNode='validation',validationPort='table')
    return result


@pytest.mark.parametrize('steps',[1,2])
@pytest.mark.parametrize('nullable',[True,False])
def test_real_fit_attaches_model_bound_rules_and_reuses_without_labels(tmp_path, monkeypatch,steps,nullable):
    request = fitted_request(tmp_path,steps=steps)
    output = tmp_path / 'fit'
    receipt = run_prepared(request, output, lambda _: None)
    attachment = receipt['learnedPreparation']
    raw = (output / attachment['path']).read_bytes()
    assert attachment['sha256'] == hashlib.sha256(raw).hexdigest()
    assert attachment['bytes'] == len(raw)
    model_raw = (output / 'model-state.json').read_bytes()
    envelope = frozen.validate(json.loads(raw), model_raw)
    assert len(envelope['parameters'])==steps
    assert [c['name'] for c in envelope['inputSchema']] == ['x', 'city']
    assert attachment['contentDigest'] == envelope['contentDigest']
    # The portable directory is the exact source copied by native Save model.
    package = output / 'portable-prepared'
    manifest = json.loads((package / 'manifest.json').read_bytes())
    for item in manifest['artifacts']:
        content = (package / item['path']).read_bytes()
        assert len(content) == item['bytes']
        assert hashlib.sha256(content).hexdigest() == item['sha256']
    assert (package / 'state.json').read_bytes() == model_raw
    assert b'observationIDs' not in raw and b'fitEvidence' not in raw
    from runner import dataflow as df, dataflow_preprocess as dp
    monkeypatch.setattr(dp, '_fit', lambda *a, **k: pytest.fail('Frozen reuse must not fit'))
    schema=tuple({**c,'nullable':nullable} for c in envelope['inputSchema'])
    first=None if nullable else envelope['parameters'][0]['rules'][0]['fill']
    table = df.Table(schema, ((first, 'never-seen'), (9., 'a')), ('new-1', 'new-2'), 'd' * 64, 'unassigned')
    applied = frozen.apply(table, envelope, model_raw)
    try:
        assert list(applied.row_ids) == ['new-1', 'new-2']
        values = np.asarray(list(applied.rows), dtype=np.float64)
        assert values[0].tolist() == pytest.approx([0., 0., 0.],abs=1e-14)
        prediction = pb.predict(pb.load(model_raw), values)
        assert len(prediction['values']) == 2
    finally:
        applied.rows.store.close(); applied.row_ids.store.close()


def test_no_learned_attachment_is_invented_for_untransformed_input(tmp_path):
    r=fitted_request(tmp_path)
    r['prepared'].update(trainingNode='p',validationNode='p',featureColumns=['time'])
    output = tmp_path / 'fit'
    receipt = run_prepared(r, output, lambda _: None)
    assert 'learnedPreparation' not in receipt
    assert not (output / 'portable-prepared').exists()


def test_attachment_refuses_other_model_and_altered_rules(tmp_path):
    r=fitted_request(tmp_path);out=tmp_path/'fit';run_prepared(r,out,lambda _:None)
    raw=(out/'model-state.json').read_bytes()
    envelope=json.loads((out/'portable-prepared/preparation.json').read_bytes())
    altered=pb.load(raw);altered['featureColumns']=list(reversed(altered['featureColumns']))
    with pytest.raises(ValueError,match='different model'):
        frozen.validate(envelope,pb.canonical(altered))
    envelope['parameters'][0]['rules'][0]['mean']+=1
    with pytest.raises(ValueError,match='content digest'):
        frozen.validate(envelope,raw)


def test_frozen_input_cannot_widen_nonnullable_requirement(tmp_path):
    from runner import dataflow as df
    r=fitted_request(tmp_path);out=tmp_path/'fit';run_prepared(r,out,lambda _:None)
    raw=(out/'model-state.json').read_bytes();envelope=json.loads((out/'portable-prepared/preparation.json').read_bytes())
    # A legitimate nonnullable numeric input contract, still paired to the same
    # model and mean/scaling rules. Output nullability remains False.
    for schema in [envelope['inputSchema'],envelope['parameters'][0]['inputSchema']]:schema[0]['nullable']=False
    envelope['contentDigest']=frozen.digest({k:v for k,v in envelope.items() if k!='contentDigest'})
    frozen.validate(envelope,raw)
    schema=copy.deepcopy(envelope['inputSchema']);schema[0]['nullable']=True
    table=df.Table(tuple(schema),((None,'a'),),('new',),'e'*64,'unassigned')
    with pytest.raises(ValueError,match='Input schema differs'):frozen.apply(table,envelope,raw)


@pytest.mark.parametrize('after',['formula','view'])
def test_only_identity_views_may_follow_saved_learned_rules(tmp_path,after):
    request=fitted_request(tmp_path,after=after);out=tmp_path/'fit'
    receipt=run_prepared(request,out,lambda _:None)
    assert ('learnedPreparation' in receipt)==(after=='view')
    assert (out/'portable-prepared').exists()==(after=='view')

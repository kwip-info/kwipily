import hashlib
import json
import pytest
import torch
from runner.prediction_table import prediction_table
from runner.dataflow_table_reader import read_table_artifact


def test_all_rows_have_identity_population_and_stable_classification_columns(tmp_path):
    logits=torch.tensor([[0., 2.], [3., 1.], [1., 1.]]).repeat(101,1)
    ids=[f'heldout:{i}' for i in range(len(logits))]
    table=prediction_table(logits,torch.tensor([1,0,1]).repeat(101),ids,'validation','a'*64,classification=True)
    assert len(table['rows'])==303
    names=[c['name'] for c in table['columns']]
    assert names==['observation_id','partition','target','predicted','probability_0','probability_1','output']
    assert table['rows'][0][:4]==['heldout:0','validation',1,1]
    assert table['rows'][0][4:6]==pytest.approx([0.119202922,0.880797078])
    assert table['rows'][2][3]==0
    raw=json.dumps(table,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    path=tmp_path/'table.json';path.write_bytes(raw)
    accepted=read_table_artifact(path,hashlib.sha256(raw).hexdigest(),lambda:None)
    assert len(accepted.rows)==303
    assert list(accepted.row_ids)==ids
    assert accepted.role=='validation'


def test_structured_objectives_keep_values_without_inventing_classes():
    table=prediction_table(torch.tensor([[[1.,2.]],[[3.,4.]]]),{'value':torch.tensor([[5.],[6.]])},['a','b'],'validation','b'*64)
    assert [c['name'] for c in table['columns']]==['observation_id','partition','target','output']
    assert json.loads(table['rows'][1][-1])==[[3.,4.]]
    assert json.loads(table['rows'][1][-2])=={'value':[6.]}


@pytest.mark.parametrize('ids,role',[(['a','a'],'validation'),(['a'],'test'),(['a','b'],'training')])
def test_bad_population_or_identity_refused(ids,role):
    with pytest.raises(ValueError):prediction_table(torch.zeros(2,2),torch.zeros(2,dtype=torch.int64),ids,role,'c'*64,classification=True)


def test_captured_prediction_source_runs_real_view_and_rejects_tamper(tmp_path):
    from runner.dataflow_cli import run_dataflow
    table=prediction_table(torch.tensor([[0.,2.],[3.,1.]]),torch.tensor([1,0]),['a','b'],'test','d'*64,classification=True)
    raw=json.dumps(table,sort_keys=True,separators=(',',':')).encode()
    source=tmp_path/'predictions.json';source.write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
    graph={'schemaVersion':1,'nodes':[{'id':'predictions','operator':'kwip.data.capturedTable','config':{'sha256':pin}},{'id':'view','operator':'kwip.data.view','config':{}}],
           'edges':[{'from':{'node':'predictions','port':'table'},'to':{'node':'view','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    run_dataflow(path,{'predictions':str(source)},tmp_path/'output',lambda _:None)
    manifest=json.loads((tmp_path/'output/manifest.json').read_text())
    assert manifest['tableViews']['view']['rowCount']==2
    source.write_bytes(raw+b' ')
    with pytest.raises(ValueError):run_dataflow(path,{'predictions':str(source)},tmp_path/'tampered',lambda _:None)


def test_prepared_training_publishes_complete_table_and_capture_omission(tmp_path):
    from runner.tests.test_prepared_numeric_width import prepared,load,model_graph
    from runner.engine import run
    preparation,_=prepared(tmp_path)
    value=load(preparation,sample_shape=[3])
    policy={'schemaVersion':2,'activationValues':False,'predictionRows':True,'performanceMeasurements':False}
    output=tmp_path/'trained'
    run(model_graph(),output,epochs=1,device='cpu',prepared=value,sample_capture=policy,emit=lambda _:None)
    manifest=json.loads((output/'manifest.json').read_text());meta=manifest['predictionTable']
    raw=(output/meta['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    table=json.loads(raw);assert len(table['rows'])==4
    assert table['lineage']['observationIDs']==value[1]['validationRowIDs']
    assert set(r[1] for r in table['rows'])=={'validation'}
    from runner.engine import GraphModel
    model=GraphModel(model_graph(),bundled_dataset=False);model.load_state_dict(torch.load(output/'checkpoint.pt',weights_only=True,map_location='cpu')['modelState']);model.eval()
    with torch.no_grad():expected=model(value[0][2])
    for row,scores in zip(table['rows'],expected):
        assert json.loads(row[-1])==pytest.approx(scores.tolist())
        assert row[3]==int(scores.argmax())
        assert row[4:6]==pytest.approx(torch.softmax(scores.double(),dim=0).tolist())
    policy['predictionRows']=False
    run(model_graph(),tmp_path/'omitted',epochs=1,device='cpu',prepared=value,sample_capture=policy,emit=lambda _:None)
    assert 'predictionTable' not in json.loads((tmp_path/'omitted/manifest.json').read_text())

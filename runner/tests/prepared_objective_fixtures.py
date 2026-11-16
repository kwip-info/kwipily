"""Purpose-built current prepared populations for loader/gradient scientific tests."""
import json
import torch
from runner.tests.test_loss_engine import model


def prepared_window_fixture(tmp,op,config=None):
    """Current 12/4/4 population retains a full ten-row window + two-row tail.

    Generic loss lessons use 8/4/4; loader contracts additionally require
    original indices8/9, seven selected rows, and ten retained/two dropped rows.
    Gradient contracts require three microbatches and two optimizer updates.
    Author the larger sealed source rather than shrinking those scientific cases.
    """
    import hashlib
    from runner import dataflow
    from runner.dataflow_cli import run_dataflow
    from runner.loss_configuration import validate_recipe
    from runner.loss_prepared_data import load_prepared_objective
    from runner.tests.test_loss_prepared_data import fixture
    tmp.mkdir(parents=True,exist_ok=True)
    table,m,r,prediction=fixture(op)
    if config:r=validate_recipe({**r,'config':{**r['config'],**config}})
    columns=[*table['columns'],dict(name='fixture_order',dtype='int64',nullable=False)]
    rows=[[v+i*.1 if j<len(m['inputColumns']) else v for j,v in enumerate(table['rows'][i%4])]+[i] for i in range(20)]
    source=tmp/'source.csv'
    source.write_text(','.join(c['name'] for c in columns)+'\n'+''.join(','.join(str(v) for v in row)+'\n' for row in rows))
    flow=dict(schemaVersion=1,nodes=[
        dict(id='s',operator='kwip.data.sourceCSV',config={'separator':','}),
        dict(id='c',operator='kwip.data.columns',config=dict(rows=[dict(source=c['name'],included=True,outputName=c['name'],targetType=c['dtype']) for c in columns],castFailure='error',trimText=False,newColumns='drop',missingPolicy='error')),
        dict(id='p',operator='kwip.data.partition',config=dict(strategy='time',timeColumn='fixture_order',timeEncoding='epochMilliseconds',trainingEnd=11,validationEnd=15,embargoMilliseconds=0,entityColumns=[],entityPolicy='allowAcrossTime'))],
        edges=[{'from':{'node':a,'port':'table'},'to':{'node':b,'port':'table'}} for a,b in [('s','c'),('c','p')]])
    path=tmp/'dataflow.json';path.write_text(json.dumps(flow));out=tmp/'preparation'
    run_dataflow(path,{'s':str(source)},out,lambda _:None)
    raw=(out/'manifest.json').read_bytes()
    data=load_prepared_objective(out/'manifest.json','p','training','p','validation',m,r,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_graph_digest=dataflow.digest(flow),sample_shape=[len(m['inputColumns'])],prediction_shape=prediction)
    # Independent original-value/order oracle for every objective, including all
    # six triplet inputs. IDs remain those from the actual current preparation.
    x,_,validation,_=data[0]
    expected=lambda times:torch.tensor([[float(i%4+j)+i*.1 for j in range(len(m['inputColumns']))] for i in times])
    torch.testing.assert_close(x,expected(range(12)));torch.testing.assert_close(validation,expected(range(12,16)))
    assert len(data[1]['trainingRowIDs'])==12 and len(data[1]['validationRowIDs'])==4
    assert set(data[1]['trainingRowIDs']).isdisjoint(data[1]['validationRowIDs'])
    manifest=json.loads(raw)
    entry=next(a for a in manifest['artifacts'] if a.get('nodeID')=='p' and a.get('port')=='test')
    test=json.loads((out/entry['path']).read_bytes())
    assert len(test['rows'])==4 and [row[-1] for row in test['rows']]==list(range(16,20))
    return model(op),r,m,data


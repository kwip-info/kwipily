import hashlib
import json
import subprocess
import sys
import torch
import pytest
from runner.dataflow_cli import run_dataflow
from runner.prepared_data import load_prepared
from runner.tests.test_embedding_graph import graph,registry


def preparation(tmp_path,large=False,dtype='int64'):
    tmp_path.mkdir(parents=True,exist_ok=True)
    source=tmp_path/'rows.csv'
    source.write_text('a,b,label\n'+''.join(f'{2**53+1 if large else i%5},{i%3},{i%2}\n' for i in range(40)))
    nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={'separator':','}),
           dict(id='cast',operator='kwip.data.columns',config={'rows':[dict(source=c,included=True,outputName=c,targetType=t) for c,t in [('a',dtype),('b',dtype),('label','int64')]],'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}),
           dict(id='split',operator='kwip.data.partition',config={'strategy':'random','seed':42,'validationPercent':20,'testPercent':20,'stratifyColumn':'label'})]
    edges=[{'from':{'node':a,'port':'table'},'to':{'node':b,'port':'table'}} for a,b in [('source','cast'),('cast','split')]]
    flow=tmp_path/'flow.json';flow.write_text(json.dumps(dict(schemaVersion=1,nodes=nodes,edges=edges)))
    out=tmp_path/'prepared';run_dataflow(flow,{'source':str(source)},out,lambda _:None)
    return out


def load(path,**kwargs):
    return load_prepared(path/'manifest.json','split','training','split','validation',['a','b'],'label',input_dtype='int64',**kwargs)


def test_exact_integer_mapping_and_multiaxis_shape(tmp_path):
    path=preparation(tmp_path,large=True)
    data,provenance,_=load(path,sample_shape=[1,2])
    assert data[0].dtype==torch.int64 and data[0].shape==(24,1,2)
    assert torch.all(data[0][:,0,0]==2**53+1)
    assert provenance['modelInput']=={'dtype':'int64','sampleShape':[1,2],'columnOrder':['a','b']}
    assert set(provenance['trainingRowIDs']).isdisjoint(provenance['validationRowIDs'])
    # Equal category combinations across partitions are legitimate; row identities remain separate.
    assert set(map(tuple,data[0].flatten(1).tolist())) & set(map(tuple,data[2].flatten(1).tolist()))
    with pytest.raises(ValueError,match='shape|columns'):load(path,sample_shape=[3])


def test_float_schema_requires_explicit_integer_cast(tmp_path):
    with pytest.raises(ValueError,match='int64'):
        load(preparation(tmp_path,dtype='float64'),sample_shape=[2])


def test_real_cli_prepared_embedding_training(tmp_path,registry):
    path=preparation(tmp_path/'data');model=tmp_path/'model.json';model.write_text(json.dumps(graph()))
    raw=(path/'manifest.json').read_bytes();manifest=json.loads(raw)
    command=[sys.executable,'runner/main.py','--graph',str(model),'--registry',str(registry),'--output',str(tmp_path/'run'),'--device','cpu','--epochs','2','--prepared-dataflow',str(path/'manifest.json'),'--prepared-training-node','split','--prepared-training-port','training','--prepared-validation-node','split','--prepared-validation-port','validation','--feature-columns','a,b','--target-column','label','--prepared-manifest-sha256',hashlib.sha256(raw).hexdigest(),'--prepared-graph-digest',manifest['graphDigest']]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    saved=json.loads((tmp_path/'run/manifest.json').read_text())
    assert saved['provenance']['modelInput']['dtype']=='int64'
    assert (tmp_path/'run/checkpoint.pt').is_file()


@pytest.mark.parametrize('invalid',[None,True,1.5,2**63,-2**63-1])
def test_rehashed_invalid_integer_cells_rejected(tmp_path,invalid):
    from runner.dataflow import digest
    path=preparation(tmp_path)
    manifest_path=path/'manifest.json';manifest=json.loads(manifest_path.read_text())
    entry=next(item for item in manifest['artifacts'] if item.get('nodeID')=='split' and item.get('port')=='training')
    table_path=path/entry['path'];table=json.loads(table_path.read_text())
    table['rows'][0][0]=invalid
    table['lineage']['dataVersion']=digest({'columns':table['columns'],'rows':table['rows'],'ids':table['lineage']['observationIDs']})
    raw=json.dumps(table).encode();table_path.write_bytes(raw)
    entry.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='int64'):
        load(path,sample_shape=[2])

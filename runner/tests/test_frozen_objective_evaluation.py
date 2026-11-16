"""Actual sealed objective training, frozen evaluation and portable replay contracts."""
import hashlib,json,shutil,uuid
from pathlib import Path
import pytest,torch
from runner import engine,dataflow
from runner.named_prepared_data import load_named_prepared
from runner.tests.test_loss_engine import prepared
from runner.tests.test_frozen_evaluation_cli import write,sha,ROOT
from runner.frozen_evaluation_cli import execute_request


def objective_fixture(tmp_path,op='loss_mse',*,role='test',named=True):
    graph,recipe,mapping,data=prepared(tmp_path/'source',op)
    preparation=tmp_path/'source/preparation/manifest.json'
    if named:
        plan=dict(schemaVersion=1,inputs=[dict(name='observations',nodeID='input',columns=mapping['inputColumns'],sampleShape=[len(mapping['inputColumns'])],dtype='float32',representation='numeric')],roles=[dict(name=r['role'],columns=r['columns'],sampleShape=r['sampleShape'],dtype=r['dtype'],representation='numeric') for r in mapping['roles']],maximumBatchBytes=67108864)
        data=load_named_prepared(preparation,'p','training','p','validation',plan,graph,expected_manifest_sha256=sha(preparation.read_bytes()),expected_graph_digest=json.loads(preparation.read_bytes())['graphDigest'],objective_recipe=recipe,target_mapping=mapping)
    trained=tmp_path/'trained'
    engine.run(graph,trained,epochs=1,batch_size=3,device='cpu',prepared=data,objective_recipe=recipe,target_mapping=mapping)
    folder=tmp_path/'evaluation';inputs=folder/'input';inputs.mkdir(parents=True)
    shutil.copytree(preparation.parent,inputs/'preparation/artifacts')
    (inputs/'training/artifacts').mkdir(parents=True)
    for name in ['manifest.json','checkpoint.pt']:shutil.copyfile(trained/name,inputs/'training/artifacts'/name)
    shutil.copyfile(ROOT/'Sources/KwipMLCore/Resources/components.json',inputs/'model-registry.json')
    shutil.copyfile(ROOT/'runner/dataflow-operators.json',inputs/'data-registry.json')
    project=str(uuid.uuid4()).upper();run_id=str(uuid.uuid4()).upper();identity='a'*64
    parent=dict(projectID=project,directoryID=str(uuid.uuid4()).upper(),runID=run_id,requestIdentity=identity,modelManifestSHA256=sha((trained/'manifest.json').read_bytes()),preparationManifestSHA256=sha(preparation.read_bytes()),checkpointSHA256=sha((trained/'checkpoint.pt').read_bytes()))
    write(inputs/'capture.json',dict(schemaVersion=5,compilerContractVersion=4,projectID=project,directoryID=parent['directoryID'],runID=run_id,requestIdentity=identity))
    write(inputs/'workflow.json',dict(schemaVersion=23,projectID=project));write(inputs/'runtime-evidence.json',dict(schemaVersion=1))
    parent['outcomeSHA256']=sha(write(inputs/'outcome.json',dict(schemaVersion=2,status='completed',projectID=project,directoryID=parent['directoryID'],runID=run_id,requestIdentity=identity,modelManifestSHA256=parent['modelManifestSHA256'],preparationManifestSHA256=parent['preparationManifestSHA256'])))
    files=[dict(path=str(p.relative_to(inputs)),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))for p in sorted(inputs.rglob('*'))if p.is_file()]
    receipt=dict(schemaVersion=1,reference=parent,testEndpoint=dict(node='p',port=role),files=files)
    settings=dict(device='cpu',batchSize=3,previewLimit=2)
    if role=='validation':receipt.update(schemaVersion=2,partitionRole=role);settings.update(settingsVersion=2,partitionRole=role)
    receipt_raw=write(inputs/'snapshot.json',receipt)
    path=folder/'request.json';raw=write(path,dict(schemaVersion=1,kind='frozenEvaluation',evaluationID=str(uuid.uuid4()).upper(),projectID=project,requestIdentity='b'*64,snapshotReceiptSHA256=sha(receipt_raw),settings=settings))
    return path,sha(raw)


@pytest.mark.parametrize('role',['test','validation'])
def test_explicit_named_vector_mse_frozen_evaluation_uses_archived_objective(tmp_path,role):
    path,pin=objective_fixture(tmp_path,role=role)
    manifest=execute_request(path,pin,lambda _:None)
    summary=json.loads((path.parent/'artifacts/summary.json').read_bytes())
    rows=[json.loads(v)for v in (path.parent/'artifacts/predictions.jsonl').read_bytes().splitlines()]
    assert manifest['schemaVersion']==4 and summary['schemaVersion']==2
    assert summary['evaluatedCount']==len(rows)==4
    assert 'accuracy' not in summary and 'confusionMatrix' not in summary
    assert summary['objective']['componentID']=='loss_mse'
    squared=[(a-b)**2 for row in rows for a,b in zip(row['output'],row['targets']['target'])]
    assert summary['objective']['effectiveScalar']==pytest.approx(sum(squared)/len(squared),rel=2e-6)
    assert summary['preview']==rows[:2]
    table=json.loads((path.parent/'artifacts/prediction-table.json').read_bytes())
    assert len(table['rows'])==4 and table['lineage']['observationIDs']==[r['observationID']for r in rows]

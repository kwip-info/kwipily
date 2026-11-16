"""Explicit frozen Validation, independent from protected Test evidence."""
import json
from pathlib import Path
import pytest
from runner.tests.test_frozen_evaluation_cli import evaluation_fixture,write,sha
from runner.frozen_evaluation_cli import execute_request
from runner.portable_evaluation_export import build_package
from runner.portable_evaluation import evaluate_package,validate_package


@pytest.fixture(scope='module')
def captured(tmp_path_factory):
    path,pin=evaluation_fixture.__wrapped__(tmp_path_factory.mktemp('validation-capture'))
    return tuple((str(p.relative_to(path.parent)),p.read_bytes()) for p in sorted(path.parent.rglob('*')) if p.is_file())


@pytest.fixture
def validation_request(captured,tmp_path):
    folder=tmp_path/'evaluation'
    for name,raw in captured:
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    path=folder/'request.json';receipt_path=folder/'input/snapshot.json'
    receipt=json.loads(receipt_path.read_bytes());receipt.update(schemaVersion=2,partitionRole='validation',testEndpoint={'node':'validation','port':'table'})
    request=json.loads(path.read_bytes());request['settings'].update(settingsVersion=2,partitionRole='validation')
    request['snapshotReceiptSHA256']=sha(write(receipt_path,receipt))
    return path,sha(write(path,request))


def test_validation_full_population_and_receipt_does_not_claim_test(validation_request):
    path,pin=validation_request;manifest=execute_request(path,pin,lambda _:None)
    provenance=manifest['testProvenance'];assert provenance['dataset']=='prepared-validation-v1'
    assert provenance['validationSamples']==15 and 'validationRowIDs' not in provenance
    assert 'testSamples' not in provenance and 'testRowIDs' not in provenance
    parent=json.loads((path.parent/'input/training/artifacts/manifest.json').read_bytes())['provenance']
    predictions=[json.loads(row) for row in (path.parent/'artifacts/predictions.jsonl').read_bytes().splitlines()]
    assert [row['observationID'] for row in predictions]==parent['validationRowIDs']
    assert not set(parent['trainingRowIDs'])&{row['observationID'] for row in predictions}
    assert manifest['schemaVersion']==3 and manifest['settings']['partitionRole']=='validation'
    table=json.loads((path.parent/'artifacts/prediction-table.json').read_bytes())
    assert len(table['rows'])==15 and table['lineage']['role']=='validation'
    assert table['lineage']['observationIDs']==parent['validationRowIDs']


@pytest.mark.parametrize('mutation',['test-endpoint','raw-validation','receipt-role','receipt-version','settings-role','settings-version'])
def test_validation_crosswiring_and_version_tampering_never_publishes(validation_request,mutation):
    path,pin=validation_request;request=json.loads(path.read_bytes());rp=path.parent/'input/snapshot.json';receipt=json.loads(rp.read_bytes())
    if mutation=='test-endpoint':receipt['testEndpoint']={'node':'split','port':'test'}
    elif mutation=='raw-validation':receipt['testEndpoint']={'node':'split','port':'validation'}
    elif mutation=='receipt-role':receipt['partitionRole']='test'
    elif mutation=='receipt-version':receipt['schemaVersion']=1
    elif mutation=='settings-role':request['settings']['partitionRole']='test'
    else:request['settings']['settingsVersion']=1
    request['snapshotReceiptSHA256']=sha(write(rp,receipt));pin=sha(write(path,request))
    with pytest.raises(ValueError):execute_request(path,pin,lambda _:None)
    assert not (path.parent/'artifacts/manifest.json').exists()


def test_validation_portable_replay_role_minimization_and_no_test_rows(validation_request,tmp_path):
    path,pin=validation_request;manifest=execute_request(path,pin,lambda _:None)
    destination=tmp_path/'portable';receipt=build_package(path.parent,destination,sha((path.parent/'artifacts/manifest.json').read_bytes()),include_test_rows=True)
    assert receipt['schemaVersion']==3 and receipt['partitionRole']=='validation'
    _,_,contract,_=validate_package(destination)
    assert contract['schemaVersion']==2 and contract['partitionRole']=='validation' and contract['testCount']==15
    result=evaluate_package(destination,replay=True,batch_size=4,device='cpu')
    assert result['population']=='originalValidation' and result['evaluatedCount']==15
    parent=json.loads((path.parent/'input/training/artifacts/manifest.json').read_bytes())['provenance']
    rows=json.loads((destination/'test-input.json').read_bytes())
    assert rows['observationIDs']==parent['validationRowIDs']
    assert not set(rows['observationIDs'])&set(parent['trainingRowIDs'])
    prep=path.parent/'input/preparation/artifacts';inventory=json.loads((prep/'manifest.json').read_bytes())
    entry=next(a for a in inventory['artifacts'] if a.get('nodeID')=='split' and a.get('port')=='test')
    protected=json.loads((prep/entry['path']).read_bytes())['lineage']['observationIDs']
    assert not set(protected)&set(rows['observationIDs'])
    exported=b''.join(p.read_bytes() for p in destination.iterdir() if p.suffix!='.pt')
    assert all(identity.encode() not in exported for identity in protected)
    minimized=tmp_path/'minimal';build_package(path.parent,minimized,sha((path.parent/'artifacts/manifest.json').read_bytes()))
    assert not (minimized/'test-input.json').exists()
    texts=b''.join(p.read_bytes() for p in minimized.iterdir() if p.suffix!='.pt')
    assert all(identity.encode() not in texts for identity in parent['trainingRowIDs']+parent['validationRowIDs'])


@pytest.mark.parametrize('target',['manifest','contract'])
def test_portable_resealed_role_crosswiring_is_refused(validation_request,tmp_path,target):
    path,pin=validation_request;execute_request(path,pin,lambda _:None)
    destination=tmp_path/'portable'
    build_package(path.parent,destination,sha((path.parent/'artifacts/manifest.json').read_bytes()),include_test_rows=True)
    mp=destination/'manifest.json';manifest=json.loads(mp.read_bytes())
    if target=='manifest':
        manifest['schemaVersion']=2;del manifest['partitionRole']
    else:
        cp=destination/'input-contract.json';contract=json.loads(cp.read_bytes());contract['schemaVersion']=1;del contract['partitionRole']
        raw=write(cp,contract);entry=next(a for a in manifest['files'] if a['path']=='input-contract.json');entry.update(bytes=len(raw),sha256=sha(raw))
    write(mp,manifest)
    with pytest.raises(ValueError,match='(?i)role'):validate_package(destination)

"""Captured allowances apply at staging, before payload allocation or effects."""
from pathlib import Path
import pytest
from runner.workload_policy import WorkloadPolicy
from runner.preparation_inspection_operation_contract import parse_contract


def inspection(tmp_path,workload=None,size=60_000_000):
    files=[{'path':'input/preparation/manifest.json','bytes':100,'sha256':'a'*64},
           {'path':'input/preparation/table-pack-'+'b'*32+'.kwipdata','bytes':size,'sha256':'c'*64}]
    request={'arguments':['--inspect-dataflow',str(tmp_path/'input/preparation/manifest.json'),'--dataflow-manifest-sha256','a'*64],'inputFiles':files}
    if workload is not None:request['executionBudget']={'resources':{'schemaVersion':2,'cpuThreads':1,'workloadPolicy':workload.to_document()}}
    return request


def test_inspection_uses_captured_segment_and_aggregate_allowance(tmp_path):
    policy=WorkloadPolicy('balanced')
    request=inspection(tmp_path,policy,size=250_000_000)
    assert len(parse_contract(request,tmp_path))==2
    with pytest.raises(ValueError):parse_contract(inspection(tmp_path,size=60_000_000),tmp_path)
    with pytest.raises(ValueError):parse_contract(inspection(tmp_path,policy.overriding({'retainedBytes':1000}),size=1001),tmp_path)
    request['inputFiles'][1]['path']='input/preparation/untrusted.bin'
    with pytest.raises(ValueError):parse_contract(request,tmp_path)


def test_backend_transfer_rejects_larger_segment_without_launch_allowance(tmp_path):
    from runner.backend_fit_operation_contract import parse_contract as parse_backend
    path=str(tmp_path/'input/backend-request.json')
    value={'arguments':['--fit-backend',path,'--backend-request-sha256','a'*64,'--output',str(tmp_path/'artifacts')],
        'inputFiles':[{'path':'input/backend-request.json','bytes':100,'sha256':'a'*64},
                      {'path':'input/preparation/manifest.json','bytes':100,'sha256':'b'*64},
                      {'path':'input/preparation/table-pack-'+'c'*32+'.kwipdata','bytes':250_000_000,'sha256':'c'*64}]}
    with pytest.raises(ValueError):parse_backend(value,tmp_path)
    value['executionBudget']={'resources':{'workloadPolicy':WorkloadPolicy('balanced').to_document()}}
    parse_backend(value,tmp_path)
    value['executionBudget']['resources']['workloadPolicy']=WorkloadPolicy('balanced').overriding({'retainedBytes':1000}).to_document()
    with pytest.raises(ValueError):parse_backend(value,tmp_path)


def test_completed_inventory_cannot_expand_smaller_consumer_allowance(tmp_path):
    import hashlib,json
    from runner.worker_input_contract import verify_completed_inventory
    folder=tmp_path/'input/preparation';folder.mkdir(parents=True)
    name='table-pack-'+'a'*32+'.kwipdata'
    entry={'path':name,'bytes':4096,'sha256':'b'*64,'kind':'tableSegment'}
    document={'schemaVersion':1,'kind':'dataflowExecution','status':'completed','graphDigest':'c'*64,
              'tableStorageVersion':1,'workloadPolicy':WorkloadPolicy('balanced').to_document(),'artifacts':[entry]}
    raw=json.dumps(document).encode();(folder/'manifest.json').write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
    inventory={'input/preparation/manifest.json':{'bytes':len(raw),'sha256':pin},'input/preparation/'+name:entry}
    verify_completed_inventory(tmp_path,'input/preparation',inventory,pin,admission_policy=WorkloadPolicy('balanced'))
    with pytest.raises(ValueError,match='Prepared artifact pin differs'):
        verify_completed_inventory(tmp_path,'input/preparation',inventory,pin,admission_policy=WorkloadPolicy('balanced').overriding({'retainedBytes':1000}))
